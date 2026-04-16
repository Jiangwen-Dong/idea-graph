from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from .fs_utils import read_text_file
from .relation_graph_critic_data import RelationGraphCandidateExample, RelationGraphVocabularies


class TextEmbeddingBackend(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class RelationGraphCommitExample:
    state_id: str
    group_id: str
    split: str
    label: int
    state_text_embedding: np.ndarray
    node_text_embeddings: np.ndarray
    node_type_ids: list[int]
    node_role_ids: list[int]
    node_confidence: list[float]
    node_evidence_count: list[float]
    edge_index: list[tuple[int, int]]
    edge_type_ids: list[int]
    edge_resolved: list[float]
    support_coverage: float
    unresolved_contradiction_ratio: float
    utility: float


@dataclass(frozen=True)
class RelationGraphCommitBatch:
    node_text_embeddings: torch.Tensor
    state_text_embeddings: torch.Tensor
    node_type_ids: torch.Tensor
    node_role_ids: torch.Tensor
    node_scalars: torch.Tensor
    edge_index: torch.Tensor
    edge_type_ids: torch.Tensor
    edge_resolved: torch.Tensor
    edge_mask: torch.Tensor
    graph_mask: torch.Tensor
    graph_features: torch.Tensor
    labels: torch.Tensor
    node_type_vocab_size: int
    role_vocab_size: int
    edge_type_vocab_size: int

    def to(self, device: torch.device | str) -> "RelationGraphCommitBatch":
        return RelationGraphCommitBatch(
            node_text_embeddings=self.node_text_embeddings.to(device),
            state_text_embeddings=self.state_text_embeddings.to(device),
            node_type_ids=self.node_type_ids.to(device),
            node_role_ids=self.node_role_ids.to(device),
            node_scalars=self.node_scalars.to(device),
            edge_index=self.edge_index.to(device),
            edge_type_ids=self.edge_type_ids.to(device),
            edge_resolved=self.edge_resolved.to(device),
            edge_mask=self.edge_mask.to(device),
            graph_mask=self.graph_mask.to(device),
            graph_features=self.graph_features.to(device),
            labels=self.labels.to(device),
            node_type_vocab_size=self.node_type_vocab_size,
            role_vocab_size=self.role_vocab_size,
            edge_type_vocab_size=self.edge_type_vocab_size,
        )


@dataclass(frozen=True)
class RelationGraphTwoHeadDataset:
    edit_train_examples: list[RelationGraphCandidateExample]
    edit_dev_examples: list[RelationGraphCandidateExample]
    commit_train_examples: list[RelationGraphCommitExample]
    commit_dev_examples: list[RelationGraphCommitExample]
    metadata: dict[str, Any]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = read_text_file(path)
    for line_index, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} line {line_index} must contain a JSON object.")
        rows.append(dict(payload))
    return rows


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _load_snapshot(g1_dataset_dir: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    relative_path = str(row.get("before_state_snapshot", "")).strip()
    if not relative_path:
        raise ValueError("Two-head dataset row is missing required before_state_snapshot.")
    snapshot_path = Path(g1_dataset_dir) / relative_path
    payload = json.loads(read_text_file(snapshot_path))
    if not isinstance(payload, dict):
        raise ValueError(f"{snapshot_path} does not contain a JSON object.")
    return dict(payload)


def _intern_id(vocab: dict[str, int], raw_value: object) -> int:
    value = str(raw_value).strip() or "unknown"
    existing = vocab.get(value)
    if existing is not None:
        return existing
    next_id = len(vocab)
    vocab[value] = next_id
    return next_id


def _ensure_unknown_bucket(vocab: dict[str, int]) -> None:
    if "unknown" not in vocab:
        vocab["unknown"] = len(vocab)


def _lookup_vocab_with_fallback(vocab: Mapping[str, int], raw_value: object) -> int:
    value = str(raw_value).strip() or "unknown"
    existing = vocab.get(value)
    if existing is not None:
        return existing
    unknown_id = vocab.get("unknown")
    if unknown_id is None:
        raise ValueError("Vocabulary is missing required 'unknown' bucket.")
    return int(unknown_id)


def _build_vocabularies(
    *,
    edit_rows: Sequence[Mapping[str, Any]],
    commit_rows: Sequence[Mapping[str, Any]],
    g1_dataset_dir: Path,
) -> RelationGraphVocabularies:
    node_type_vocab: dict[str, int] = {}
    role_vocab: dict[str, int] = {}
    edge_type_vocab: dict[str, int] = {}
    candidate_kind_vocab: dict[str, int] = {}

    for row in [*edit_rows, *commit_rows]:
        snapshot = _load_snapshot(g1_dataset_dir, row)
        nodes_payload = snapshot.get("nodes", {})
        if isinstance(nodes_payload, Mapping):
            for node_payload in nodes_payload.values():
                if not isinstance(node_payload, Mapping):
                    continue
                _intern_id(node_type_vocab, node_payload.get("type", "unknown"))
                _intern_id(role_vocab, node_payload.get("role", "unknown"))
        edges_payload = snapshot.get("edges", [])
        if isinstance(edges_payload, list):
            for edge_payload in edges_payload:
                if not isinstance(edge_payload, Mapping):
                    continue
                _intern_id(edge_type_vocab, edge_payload.get("relation", "unknown"))

    for row in edit_rows:
        _intern_id(candidate_kind_vocab, row.get("candidate_kind", "unknown"))

    _ensure_unknown_bucket(node_type_vocab)
    _ensure_unknown_bucket(role_vocab)
    _ensure_unknown_bucket(edge_type_vocab)
    _ensure_unknown_bucket(candidate_kind_vocab)

    return RelationGraphVocabularies(
        node_type_to_id=node_type_vocab,
        role_to_id=role_vocab,
        edge_type_to_id=edge_type_vocab,
        candidate_kind_to_id=candidate_kind_vocab,
    )


def _embedding_cache_for_rows(
    *,
    edit_rows: Sequence[Mapping[str, Any]],
    commit_rows: Sequence[Mapping[str, Any]],
    g1_dataset_dir: Path,
    text_backend: TextEmbeddingBackend,
) -> tuple[dict[str, np.ndarray], int]:
    all_texts: list[str] = []
    seen_texts: set[str] = set()
    for row in [*edit_rows, *commit_rows]:
        state_text = str(row.get("state_text", "")).strip()
        if state_text not in seen_texts:
            all_texts.append(state_text)
            seen_texts.add(state_text)
        candidate_text = str(row.get("candidate_text", "")).strip()
        if candidate_text and candidate_text not in seen_texts:
            all_texts.append(candidate_text)
            seen_texts.add(candidate_text)
        snapshot = _load_snapshot(g1_dataset_dir, row)
        nodes_payload = snapshot.get("nodes", {})
        if isinstance(nodes_payload, Mapping):
            for node_payload in nodes_payload.values():
                if not isinstance(node_payload, Mapping):
                    continue
                node_text = str(node_payload.get("text", "")).strip()
                if node_text not in seen_texts:
                    all_texts.append(node_text)
                    seen_texts.add(node_text)

    encoded = text_backend.encode(all_texts)
    text_dim = int(encoded.shape[1]) if encoded.ndim == 2 else 0
    cache = {text: encoded[index] for index, text in enumerate(all_texts)}
    return cache, text_dim


def _graph_tensors_from_snapshot(
    snapshot: Mapping[str, Any],
    *,
    embedding_cache: Mapping[str, np.ndarray],
    vocabularies: RelationGraphVocabularies,
) -> tuple[np.ndarray, list[int], list[int], list[float], list[float], list[tuple[int, int]], list[int], list[float], dict[str, int]]:
    nodes_payload = snapshot.get("nodes", {})
    if not isinstance(nodes_payload, Mapping):
        nodes_payload = {}
    ordered_node_ids = sorted(str(node_id) for node_id in nodes_payload)
    node_index_lookup = {node_id: index for index, node_id in enumerate(ordered_node_ids)}

    node_text_embeddings: list[np.ndarray] = []
    node_type_ids: list[int] = []
    node_role_ids: list[int] = []
    node_confidence: list[float] = []
    node_evidence_count: list[float] = []
    for node_id in ordered_node_ids:
        node_payload = nodes_payload.get(node_id)
        if not isinstance(node_payload, Mapping):
            node_payload = {}
        node_text = str(node_payload.get("text", "")).strip()
        node_text_embeddings.append(embedding_cache[node_text])
        node_type_ids.append(_lookup_vocab_with_fallback(vocabularies.node_type_to_id, node_payload.get("type", "unknown")))
        node_role_ids.append(_lookup_vocab_with_fallback(vocabularies.role_to_id, node_payload.get("role", "unknown")))
        node_confidence.append(_safe_float(node_payload.get("confidence")))
        evidence_payload = node_payload.get("evidence", [])
        node_evidence_count.append(float(len(evidence_payload)) if isinstance(evidence_payload, list) else 0.0)

    if node_text_embeddings:
        node_embeddings_array = np.stack(node_text_embeddings, axis=0).astype(np.float32)
    else:
        sample = next(iter(embedding_cache.values()), np.zeros((0,), dtype=np.float32))
        node_embeddings_array = np.zeros((0, int(sample.shape[0]) if sample.ndim == 1 else 0), dtype=np.float32)

    edges_payload = snapshot.get("edges", [])
    if not isinstance(edges_payload, list):
        edges_payload = []
    ordered_edges = sorted(
        (edge_payload for edge_payload in edges_payload if isinstance(edge_payload, Mapping)),
        key=lambda payload: str(payload.get("id", "")),
    )
    edge_index: list[tuple[int, int]] = []
    edge_type_ids: list[int] = []
    edge_resolved: list[float] = []
    for edge_payload in ordered_edges:
        source_id = str(edge_payload.get("source_id", "")).strip()
        target_id = str(edge_payload.get("target_id", "")).strip()
        if source_id not in node_index_lookup or target_id not in node_index_lookup:
            continue
        edge_index.append((node_index_lookup[source_id], node_index_lookup[target_id]))
        edge_type_ids.append(
            _lookup_vocab_with_fallback(vocabularies.edge_type_to_id, edge_payload.get("relation", "unknown"))
        )
        edge_resolved.append(1.0 if bool(edge_payload.get("resolved", False)) else 0.0)

    return (
        node_embeddings_array,
        node_type_ids,
        node_role_ids,
        node_confidence,
        node_evidence_count,
        edge_index,
        edge_type_ids,
        edge_resolved,
        node_index_lookup,
    )


def build_relation_graph_two_head_dataset(
    *,
    dataset_dir: Path,
    g1_dataset_dir: Path,
    text_backend: TextEmbeddingBackend,
) -> RelationGraphTwoHeadDataset:
    edit_rows = _load_jsonl(Path(dataset_dir) / "edit_head_rows.jsonl")
    commit_rows = _load_jsonl(Path(dataset_dir) / "commit_head_rows.jsonl")
    vocabularies = _build_vocabularies(
        edit_rows=edit_rows,
        commit_rows=commit_rows,
        g1_dataset_dir=Path(g1_dataset_dir),
    )
    embedding_cache, text_dim = _embedding_cache_for_rows(
        edit_rows=edit_rows,
        commit_rows=commit_rows,
        g1_dataset_dir=Path(g1_dataset_dir),
        text_backend=text_backend,
    )

    edit_train_examples: list[RelationGraphCandidateExample] = []
    edit_dev_examples: list[RelationGraphCandidateExample] = []
    for row in edit_rows:
        snapshot = _load_snapshot(Path(g1_dataset_dir), row)
        (
            node_text_embeddings,
            node_type_ids,
            node_role_ids,
            node_confidence,
            node_evidence_count,
            edge_index,
            edge_type_ids,
            edge_resolved,
            node_index_lookup,
        ) = _graph_tensors_from_snapshot(snapshot, embedding_cache=embedding_cache, vocabularies=vocabularies)
        target_node_indices = [
            node_index_lookup[target_id]
            for target_id in [str(item).strip() for item in row.get("candidate_target_ids", []) if str(item).strip()]
            if target_id in node_index_lookup
        ]
        example = RelationGraphCandidateExample(
            state_id=str(row.get("state_id", "")).strip(),
            candidate_id=str(row.get("candidate_id", "")).strip(),
            group_id=str(row.get("group_id", "")).strip(),
            split=str(row.get("split", "train")).strip() or "train",
            label=1 if bool(row.get("is_logged_selected", False)) else 0,
            is_commit=False,
            candidate_kind_id=_lookup_vocab_with_fallback(
                vocabularies.candidate_kind_to_id,
                row.get("candidate_kind", "unknown"),
            ),
            candidate_text_embedding=embedding_cache[str(row.get("candidate_text", "")).strip()],
            state_text_embedding=embedding_cache[str(row.get("state_text", "")).strip()],
            node_text_embeddings=node_text_embeddings,
            node_type_ids=node_type_ids,
            node_role_ids=node_role_ids,
            node_confidence=node_confidence,
            node_evidence_count=node_evidence_count,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_resolved=edge_resolved,
            target_node_indices=target_node_indices,
        )
        if example.split == "validation":
            edit_dev_examples.append(example)
        else:
            edit_train_examples.append(example)

    commit_train_examples: list[RelationGraphCommitExample] = []
    commit_dev_examples: list[RelationGraphCommitExample] = []
    for row in commit_rows:
        snapshot = _load_snapshot(Path(g1_dataset_dir), row)
        (
            node_text_embeddings,
            node_type_ids,
            node_role_ids,
            node_confidence,
            node_evidence_count,
            edge_index,
            edge_type_ids,
            edge_resolved,
            _node_index_lookup,
        ) = _graph_tensors_from_snapshot(snapshot, embedding_cache=embedding_cache, vocabularies=vocabularies)
        example = RelationGraphCommitExample(
            state_id=str(row.get("state_id", "")).strip(),
            group_id=str(row.get("group_id", "")).strip(),
            split=str(row.get("split", "train")).strip() or "train",
            label=_safe_int(row.get("commit_label", {}).get("label"))
            if isinstance(row.get("commit_label"), Mapping)
            else (
                _safe_int(row.get("commit_label"))
                if row.get("commit_label") is not None
                else _safe_int(
                    dict(row.get("commit_supervision", {})).get("label")
                    if isinstance(row.get("commit_supervision"), Mapping)
                    else 0
                )
            ),
            state_text_embedding=embedding_cache[str(row.get("state_text", "")).strip()],
            node_text_embeddings=node_text_embeddings,
            node_type_ids=node_type_ids,
            node_role_ids=node_role_ids,
            node_confidence=node_confidence,
            node_evidence_count=node_evidence_count,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_resolved=edge_resolved,
            support_coverage=_safe_float(row.get("support_coverage")),
            unresolved_contradiction_ratio=_safe_float(row.get("unresolved_contradiction_ratio")),
            utility=_safe_float(row.get("utility")),
        )
        if example.split == "validation":
            commit_dev_examples.append(example)
        else:
            commit_train_examples.append(example)

    metadata = {
        **vocabularies.as_metadata_counts(),
        "text_dim": text_dim,
        "edit_train_example_count": len(edit_train_examples),
        "edit_dev_example_count": len(edit_dev_examples),
        "commit_train_example_count": len(commit_train_examples),
        "commit_dev_example_count": len(commit_dev_examples),
    }
    return RelationGraphTwoHeadDataset(
        edit_train_examples=edit_train_examples,
        edit_dev_examples=edit_dev_examples,
        commit_train_examples=commit_train_examples,
        commit_dev_examples=commit_dev_examples,
        metadata=metadata,
    )


def collate_relation_graph_commit_examples(
    examples: Sequence[RelationGraphCommitExample],
) -> RelationGraphCommitBatch:
    if not examples:
        raise ValueError("examples must not be empty.")

    batch_size = len(examples)
    text_dim = int(examples[0].node_text_embeddings.shape[1])
    max_nodes = max(len(example.node_type_ids) for example in examples)
    max_edges = max(len(example.edge_index) for example in examples)

    node_text_embeddings = torch.zeros((batch_size, max_nodes, text_dim), dtype=torch.float32)
    state_text_embeddings = torch.zeros((batch_size, text_dim), dtype=torch.float32)
    node_type_ids = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    node_role_ids = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    node_scalars = torch.zeros((batch_size, max_nodes, 2), dtype=torch.float32)
    edge_index = torch.zeros((batch_size, max_edges, 2), dtype=torch.long)
    edge_type_ids = torch.zeros((batch_size, max_edges), dtype=torch.long)
    edge_resolved = torch.zeros((batch_size, max_edges), dtype=torch.float32)
    edge_mask = torch.zeros((batch_size, max_edges), dtype=torch.bool)
    graph_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    graph_features = torch.zeros((batch_size, 3), dtype=torch.float32)
    labels = torch.zeros((batch_size,), dtype=torch.float32)

    node_type_max = 0
    role_max = 0
    edge_type_max = 0
    for batch_index, example in enumerate(examples):
        node_count = len(example.node_type_ids)
        node_text_embeddings[batch_index, :node_count] = torch.from_numpy(example.node_text_embeddings)
        state_text_embeddings[batch_index] = torch.from_numpy(example.state_text_embedding)
        node_type_ids[batch_index, :node_count] = torch.tensor(example.node_type_ids, dtype=torch.long)
        node_role_ids[batch_index, :node_count] = torch.tensor(example.node_role_ids, dtype=torch.long)
        node_scalars[batch_index, :node_count, 0] = torch.tensor(example.node_confidence, dtype=torch.float32)
        node_scalars[batch_index, :node_count, 1] = torch.tensor(example.node_evidence_count, dtype=torch.float32)
        graph_mask[batch_index, :node_count] = True
        labels[batch_index] = float(example.label)
        graph_features[batch_index] = torch.tensor(
            [
                float(example.support_coverage),
                float(example.unresolved_contradiction_ratio),
                float(example.utility),
            ],
            dtype=torch.float32,
        )
        for edge_offset, (source_index, target_index) in enumerate(example.edge_index):
            edge_index[batch_index, edge_offset] = torch.tensor([source_index, target_index], dtype=torch.long)
            edge_type_ids[batch_index, edge_offset] = int(example.edge_type_ids[edge_offset])
            edge_resolved[batch_index, edge_offset] = float(example.edge_resolved[edge_offset])
            edge_mask[batch_index, edge_offset] = True
        node_type_max = max(node_type_max, max(example.node_type_ids, default=0))
        role_max = max(role_max, max(example.node_role_ids, default=0))
        edge_type_max = max(edge_type_max, max(example.edge_type_ids, default=0))

    return RelationGraphCommitBatch(
        node_text_embeddings=node_text_embeddings,
        state_text_embeddings=state_text_embeddings,
        node_type_ids=node_type_ids,
        node_role_ids=node_role_ids,
        node_scalars=node_scalars,
        edge_index=edge_index,
        edge_type_ids=edge_type_ids,
        edge_resolved=edge_resolved,
        edge_mask=edge_mask,
        graph_mask=graph_mask,
        graph_features=graph_features,
        labels=labels,
        node_type_vocab_size=node_type_max + 1,
        role_vocab_size=role_max + 1,
        edge_type_vocab_size=edge_type_max + 1,
    )
