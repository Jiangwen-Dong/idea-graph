from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from .candidate_slate_dataset import state_id_from_transition
from .fs_utils import read_text_file
from .online_text_critic import build_partition_role_lookup, load_partition_manifest_rows


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


class TextEmbeddingBackend(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class HashTextEmbeddingBackend:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.encode_call_count = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.encode_call_count += 1
        rows: list[np.ndarray] = []
        for text in texts:
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            values = np.frombuffer(seed, dtype=np.uint8).astype(np.float32)
            rows.append(np.resize(values, self.dim) / 255.0)
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(rows, axis=0).astype(np.float32)


@dataclass(frozen=True)
class RelationGraphCandidateExample:
    state_id: str
    candidate_id: str
    group_id: str
    split: str
    label: int
    is_commit: bool
    candidate_kind_id: int
    candidate_text_embedding: np.ndarray
    state_text_embedding: np.ndarray
    node_text_embeddings: np.ndarray
    node_type_ids: list[int]
    node_role_ids: list[int]
    node_confidence: list[float]
    node_evidence_count: list[float]
    edge_index: list[tuple[int, int]]
    edge_type_ids: list[int]
    edge_resolved: list[float]
    target_node_indices: list[int]


@dataclass(frozen=True)
class RelationGraphDataset:
    train_examples: list[RelationGraphCandidateExample]
    dev_examples: list[RelationGraphCandidateExample]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RelationGraphBatch:
    node_text_embeddings: torch.Tensor
    state_text_embeddings: torch.Tensor
    candidate_text_embeddings: torch.Tensor
    node_type_ids: torch.Tensor
    node_role_ids: torch.Tensor
    node_scalars: torch.Tensor
    edge_index: torch.Tensor
    edge_type_ids: torch.Tensor
    edge_resolved: torch.Tensor
    edge_mask: torch.Tensor
    candidate_kind_ids: torch.Tensor
    target_mask: torch.Tensor
    neighbor_mask: torch.Tensor
    graph_mask: torch.Tensor
    labels: torch.Tensor
    is_commit: torch.Tensor
    node_type_vocab_size: int
    role_vocab_size: int
    edge_type_vocab_size: int
    candidate_kind_vocab_size: int

    def with_updates(self, **kwargs: object) -> "RelationGraphBatch":
        return replace(self, **kwargs)

    def to(self, device: torch.device | str) -> "RelationGraphBatch":
        return RelationGraphBatch(
            node_text_embeddings=self.node_text_embeddings.to(device),
            state_text_embeddings=self.state_text_embeddings.to(device),
            candidate_text_embeddings=self.candidate_text_embeddings.to(device),
            node_type_ids=self.node_type_ids.to(device),
            node_role_ids=self.node_role_ids.to(device),
            node_scalars=self.node_scalars.to(device),
            edge_index=self.edge_index.to(device),
            edge_type_ids=self.edge_type_ids.to(device),
            edge_resolved=self.edge_resolved.to(device),
            edge_mask=self.edge_mask.to(device),
            candidate_kind_ids=self.candidate_kind_ids.to(device),
            target_mask=self.target_mask.to(device),
            neighbor_mask=self.neighbor_mask.to(device),
            graph_mask=self.graph_mask.to(device),
            labels=self.labels.to(device),
            is_commit=self.is_commit.to(device),
            node_type_vocab_size=self.node_type_vocab_size,
            role_vocab_size=self.role_vocab_size,
            edge_type_vocab_size=self.edge_type_vocab_size,
            candidate_kind_vocab_size=self.candidate_kind_vocab_size,
        )


def _build_snapshot_lookup(g1_dataset_dir: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for relative_name in ("trajectory_examples.jsonl", "terminal_state_manifest.jsonl"):
        path = Path(g1_dataset_dir) / relative_name
        if not path.exists():
            continue
        for row in _load_jsonl(path):
            snapshot_relative = str(row.get("before_state_snapshot", "")).strip()
            if not snapshot_relative:
                continue
            snapshot_path = Path(g1_dataset_dir) / snapshot_relative
            payload = json.loads(read_text_file(snapshot_path))
            if not isinstance(payload, dict):
                raise ValueError(f"{snapshot_path} does not contain a JSON object.")
            lookup[state_id_from_transition(row)] = dict(payload)
    return lookup


def _intern_id(vocab: dict[str, int], raw_value: object) -> int:
    value = str(raw_value).strip() or "unknown"
    existing = vocab.get(value)
    if existing is not None:
        return existing
    next_id = len(vocab)
    vocab[value] = next_id
    return next_id


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_relation_graph_dataset(
    *,
    candidate_dataset_dir: Path,
    g1_dataset_dir: Path,
    partition_manifest_path: Path,
    text_backend: TextEmbeddingBackend,
) -> RelationGraphDataset:
    candidate_rows = _load_jsonl(Path(candidate_dataset_dir) / "candidate_dataset.jsonl")
    snapshot_lookup = _build_snapshot_lookup(Path(g1_dataset_dir))
    partition_lookup = build_partition_role_lookup(
        load_partition_manifest_rows(Path(partition_manifest_path))
    )

    node_type_vocab: dict[str, int] = {}
    role_vocab: dict[str, int] = {}
    edge_type_vocab: dict[str, int] = {}
    candidate_kind_vocab: dict[str, int] = {}

    all_texts: list[str] = []
    seen_texts: set[str] = set()
    for row in candidate_rows:
        snapshot = snapshot_lookup.get(str(row.get("state_id", "")).strip())
        if snapshot is None:
            continue
        candidate_text = str(row.get("candidate_text", "")).strip()
        state_text = str(row.get("state_text", "")).strip()
        if candidate_text not in seen_texts:
            all_texts.append(candidate_text)
            seen_texts.add(candidate_text)
        if state_text not in seen_texts:
            all_texts.append(state_text)
            seen_texts.add(state_text)
        nodes_payload = snapshot.get("nodes", {})
        if isinstance(nodes_payload, Mapping):
            for node_id in sorted(nodes_payload):
                node_payload = nodes_payload.get(node_id)
                if isinstance(node_payload, Mapping):
                    node_text = str(node_payload.get("text", "")).strip()
                    if node_text not in seen_texts:
                        all_texts.append(node_text)
                        seen_texts.add(node_text)

    encoded = text_backend.encode(all_texts)
    text_dim = int(encoded.shape[1]) if encoded.ndim == 2 else 0
    embedding_cache = {
        text: encoded[index]
        for index, text in enumerate(all_texts)
    }

    train_examples: list[RelationGraphCandidateExample] = []
    dev_examples: list[RelationGraphCandidateExample] = []

    for row in candidate_rows:
        group_id = str(row.get("group_id", "")).strip()
        partition_role = partition_lookup.get(group_id)
        if partition_role is None:
            raise ValueError(f"Candidate row references unmapped group_id '{group_id}'.")
        if partition_role == "paper_eval":
            continue

        state_id = str(row.get("state_id", "")).strip()
        snapshot = snapshot_lookup.get(state_id)
        if snapshot is None:
            raise ValueError(f"Missing snapshot for state_id '{state_id}'.")

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
            node_type_ids.append(_intern_id(node_type_vocab, node_payload.get("type", "unknown")))
            node_role_ids.append(_intern_id(role_vocab, node_payload.get("role", "unknown")))
            node_confidence.append(_safe_float(node_payload.get("confidence", 0.0)))
            evidence_payload = node_payload.get("evidence", [])
            evidence_count = len(evidence_payload) if isinstance(evidence_payload, list) else 0
            node_evidence_count.append(float(evidence_count))

        edges_payload = snapshot.get("edges", [])
        if not isinstance(edges_payload, list):
            edges_payload = []
        ordered_edges = sorted(
            (
                edge_payload
                for edge_payload in edges_payload
                if isinstance(edge_payload, Mapping)
            ),
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
            edge_type_ids.append(_intern_id(edge_type_vocab, edge_payload.get("relation", "unknown")))
            edge_resolved.append(1.0 if bool(edge_payload.get("resolved", False)) else 0.0)

        target_node_indices: list[int] = []
        for raw_target_id in row.get("candidate_target_ids", []):
            target_id = str(raw_target_id).strip()
            if target_id in node_index_lookup:
                target_node_indices.append(node_index_lookup[target_id])

        example = RelationGraphCandidateExample(
            state_id=state_id,
            candidate_id=str(row.get("candidate_id", "")).strip(),
            group_id=group_id,
            split=partition_role,
            label=1 if bool(row.get("is_logged_selected", False)) else 0,
            is_commit=bool(row.get("is_commit", False)),
            candidate_kind_id=_intern_id(candidate_kind_vocab, row.get("candidate_kind", "unknown")),
            candidate_text_embedding=embedding_cache[str(row.get("candidate_text", "")).strip()],
            state_text_embedding=embedding_cache[str(row.get("state_text", "")).strip()],
            node_text_embeddings=(
                np.stack(node_text_embeddings, axis=0).astype(np.float32)
                if node_text_embeddings
                else np.zeros((0, text_dim), dtype=np.float32)
            ),
            node_type_ids=node_type_ids,
            node_role_ids=node_role_ids,
            node_confidence=node_confidence,
            node_evidence_count=node_evidence_count,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_resolved=edge_resolved,
            target_node_indices=target_node_indices,
        )
        if partition_role == "critic_train":
            train_examples.append(example)
        elif partition_role == "critic_dev":
            dev_examples.append(example)

    return RelationGraphDataset(
        train_examples=train_examples,
        dev_examples=dev_examples,
        metadata={
            "node_type_count": len(node_type_vocab),
            "role_count": len(role_vocab),
            "edge_type_count": len(edge_type_vocab),
            "candidate_kind_count": len(candidate_kind_vocab),
        },
    )


def collate_relation_graph_examples(
    examples: Sequence[RelationGraphCandidateExample],
) -> RelationGraphBatch:
    if not examples:
        raise ValueError("examples must not be empty.")

    batch_size = len(examples)
    text_dim = int(examples[0].node_text_embeddings.shape[1])
    max_nodes = max(len(example.node_type_ids) for example in examples)
    max_edges = max(len(example.edge_index) for example in examples)

    node_text_embeddings = torch.zeros((batch_size, max_nodes, text_dim), dtype=torch.float32)
    state_text_embeddings = torch.zeros((batch_size, text_dim), dtype=torch.float32)
    candidate_text_embeddings = torch.zeros((batch_size, text_dim), dtype=torch.float32)
    node_type_ids = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    node_role_ids = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    node_scalars = torch.zeros((batch_size, max_nodes, 2), dtype=torch.float32)
    edge_index = torch.zeros((batch_size, max_edges, 2), dtype=torch.long)
    edge_type_ids = torch.zeros((batch_size, max_edges), dtype=torch.long)
    edge_resolved = torch.zeros((batch_size, max_edges), dtype=torch.float32)
    edge_mask = torch.zeros((batch_size, max_edges), dtype=torch.bool)
    candidate_kind_ids = torch.zeros((batch_size,), dtype=torch.long)
    target_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    neighbor_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    graph_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.float32)
    is_commit = torch.zeros((batch_size,), dtype=torch.float32)

    node_type_max = 0
    role_max = 0
    edge_type_max = 0
    candidate_kind_max = 0

    for batch_index, example in enumerate(examples):
        node_count = len(example.node_type_ids)
        edge_count = len(example.edge_index)
        node_text_embeddings[batch_index, :node_count] = torch.from_numpy(example.node_text_embeddings)
        state_text_embeddings[batch_index] = torch.from_numpy(example.state_text_embedding)
        candidate_text_embeddings[batch_index] = torch.from_numpy(example.candidate_text_embedding)
        node_type_ids[batch_index, :node_count] = torch.tensor(example.node_type_ids, dtype=torch.long)
        node_role_ids[batch_index, :node_count] = torch.tensor(example.node_role_ids, dtype=torch.long)
        node_scalars[batch_index, :node_count, 0] = torch.tensor(example.node_confidence, dtype=torch.float32)
        node_scalars[batch_index, :node_count, 1] = torch.tensor(example.node_evidence_count, dtype=torch.float32)
        graph_mask[batch_index, :node_count] = True
        candidate_kind_ids[batch_index] = int(example.candidate_kind_id)
        labels[batch_index] = float(example.label)
        is_commit[batch_index] = float(example.is_commit)
        for target_index in example.target_node_indices:
            target_mask[batch_index, target_index] = True
        for edge_offset, (source_index, target_index) in enumerate(example.edge_index):
            edge_index[batch_index, edge_offset] = torch.tensor([source_index, target_index], dtype=torch.long)
            edge_type_ids[batch_index, edge_offset] = int(example.edge_type_ids[edge_offset])
            edge_resolved[batch_index, edge_offset] = float(example.edge_resolved[edge_offset])
            edge_mask[batch_index, edge_offset] = True
            if target_mask[batch_index, source_index] or target_mask[batch_index, target_index]:
                neighbor_mask[batch_index, source_index] = True
                neighbor_mask[batch_index, target_index] = True
        neighbor_mask[batch_index] = neighbor_mask[batch_index] & (~target_mask[batch_index])
        node_type_max = max(node_type_max, max(example.node_type_ids, default=0))
        role_max = max(role_max, max(example.node_role_ids, default=0))
        edge_type_max = max(edge_type_max, max(example.edge_type_ids, default=0))
        candidate_kind_max = max(candidate_kind_max, int(example.candidate_kind_id))

    return RelationGraphBatch(
        node_text_embeddings=node_text_embeddings,
        state_text_embeddings=state_text_embeddings,
        candidate_text_embeddings=candidate_text_embeddings,
        node_type_ids=node_type_ids,
        node_role_ids=node_role_ids,
        node_scalars=node_scalars,
        edge_index=edge_index,
        edge_type_ids=edge_type_ids,
        edge_resolved=edge_resolved,
        edge_mask=edge_mask,
        candidate_kind_ids=candidate_kind_ids,
        target_mask=target_mask,
        neighbor_mask=neighbor_mask,
        graph_mask=graph_mask,
        labels=labels,
        is_commit=is_commit,
        node_type_vocab_size=node_type_max + 1,
        role_vocab_size=role_max + 1,
        edge_type_vocab_size=edge_type_max + 1,
        candidate_kind_vocab_size=candidate_kind_max + 1,
    )
