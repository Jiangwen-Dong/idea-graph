from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from .action_candidates import flatten_candidate_text
from .candidate_slate_dataset import state_id_from_transition
from .fs_utils import read_text_file
from .models import Edge, IdeaGraph, Node
from .online_text_critic import build_partition_role_lookup, load_partition_manifest_rows


def _strip_leaky_candidate_segments(candidate_text: str) -> str:
    parts = [part.strip() for part in candidate_text.split("|")]
    kept_parts = [
        part
        for part in parts
        if part
        and not part.lower().startswith("source=")
        and not part.lower().startswith("rationale=")
    ]
    return "|".join(kept_parts)


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


class SentenceTransformerEmbeddingBackend:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self.dim: int | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dim = int(self.dim or 0)
            return np.zeros((0, dim), dtype=np.float32)
        model = self._get_model()
        encoded = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(encoded, dtype=np.float32)


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
class RelationGraphVocabularies:
    node_type_to_id: dict[str, int]
    role_to_id: dict[str, int]
    edge_type_to_id: dict[str, int]
    candidate_kind_to_id: dict[str, int]

    def as_metadata_counts(self) -> dict[str, int]:
        return {
            "node_type_count": len(self.node_type_to_id),
            "role_count": len(self.role_to_id),
            "edge_type_count": len(self.edge_type_to_id),
            "candidate_kind_count": len(self.candidate_kind_to_id),
        }


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
    candidate_state_index: torch.Tensor
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
            candidate_state_index=self.candidate_state_index.to(device),
            labels=self.labels.to(device),
            is_commit=self.is_commit.to(device),
            node_type_vocab_size=self.node_type_vocab_size,
            role_vocab_size=self.role_vocab_size,
            edge_type_vocab_size=self.edge_type_vocab_size,
            candidate_kind_vocab_size=self.candidate_kind_vocab_size,
        )


@dataclass(frozen=True)
class RelationGraphRuntimeRowDiagnostics:
    candidate_id: str
    missing_node_types: tuple[str, ...]
    missing_roles: tuple[str, ...]
    missing_edge_types: tuple[str, ...]
    missing_candidate_kind: bool
    missing_target_ids: tuple[str, ...]
    used_vocab_fallback: bool
    requires_heuristic_fallback: bool


@dataclass(frozen=True)
class RelationGraphRuntimeBatch:
    examples: tuple[RelationGraphCandidateExample, ...]
    batch: RelationGraphBatch
    diagnostics: tuple[RelationGraphRuntimeRowDiagnostics, ...]
    fallback_row_mask: torch.Tensor


def _build_snapshot_lookup(g1_dataset_dir: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for relative_name in (
        "trajectory_examples.jsonl",
        "terminal_state_manifest.jsonl",
        "parallel_edit_examples.jsonl",
    ):
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
            explicit_state_id = str(row.get("state_id", "")).strip()
            lookup[explicit_state_id or state_id_from_transition(row)] = dict(payload)
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


def _ensure_unknown_bucket(vocab: dict[str, int]) -> None:
    if "unknown" not in vocab:
        vocab["unknown"] = len(vocab)


def _load_relation_graph_inputs(
    *,
    candidate_dataset_dir: Path,
    g1_dataset_dir: Path,
    partition_manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    candidate_rows = _load_jsonl(Path(candidate_dataset_dir) / "candidate_dataset.jsonl")
    snapshot_lookup = _build_snapshot_lookup(Path(g1_dataset_dir))
    partition_lookup = build_partition_role_lookup(
        load_partition_manifest_rows(Path(partition_manifest_path))
    )
    return candidate_rows, snapshot_lookup, partition_lookup


def _ordered_candidate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("state_id", "")).strip(),
            str(row.get("candidate_id", "")).strip(),
            str(row.get("group_id", "")).strip(),
        ),
    )


def _build_relation_graph_vocabularies_from_inputs(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    snapshot_lookup: Mapping[str, Mapping[str, Any]],
    partition_lookup: Mapping[str, str],
) -> RelationGraphVocabularies:
    node_type_vocab: dict[str, int] = {}
    role_vocab: dict[str, int] = {}
    edge_type_vocab: dict[str, int] = {}
    candidate_kind_vocab: dict[str, int] = {}

    for row in _ordered_candidate_rows(candidate_rows):
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

        _intern_id(candidate_kind_vocab, row.get("candidate_kind", "unknown"))

        nodes_payload = snapshot.get("nodes", {})
        if not isinstance(nodes_payload, Mapping):
            nodes_payload = {}
        for node_id in sorted(str(node_id) for node_id in nodes_payload):
            node_payload = nodes_payload.get(node_id)
            if not isinstance(node_payload, Mapping):
                node_payload = {}
            _intern_id(node_type_vocab, node_payload.get("type", "unknown"))
            _intern_id(role_vocab, node_payload.get("role", "unknown"))

        edges_payload = snapshot.get("edges", [])
        if not isinstance(edges_payload, list):
            edges_payload = []
        ordered_edges = sorted(
            (
                edge_payload
                for edge_payload in edges_payload
                if isinstance(edge_payload, Mapping)
            ),
            key=lambda payload: (
                str(payload.get("source_id", "")).strip(),
                str(payload.get("relation", "")).strip(),
                str(payload.get("target_id", "")).strip(),
                str(payload.get("id", "")).strip(),
            ),
        )
        for edge_payload in ordered_edges:
            _intern_id(edge_type_vocab, edge_payload.get("relation", "unknown"))

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


def build_relation_graph_vocabularies(
    *,
    candidate_dataset_dir: Path,
    g1_dataset_dir: Path,
    partition_manifest_path: Path,
) -> RelationGraphVocabularies:
    candidate_rows, snapshot_lookup, partition_lookup = _load_relation_graph_inputs(
        candidate_dataset_dir=candidate_dataset_dir,
        g1_dataset_dir=g1_dataset_dir,
        partition_manifest_path=partition_manifest_path,
    )
    return _build_relation_graph_vocabularies_from_inputs(
        candidate_rows=candidate_rows,
        snapshot_lookup=snapshot_lookup,
        partition_lookup=partition_lookup,
    )


def _lookup_vocab_with_fallback(vocab: Mapping[str, int], raw_value: object) -> tuple[int, str | None]:
    value = str(raw_value).strip() or "unknown"
    existing = vocab.get(value)
    if existing is not None:
        return existing, None
    unknown_id = vocab.get("unknown")
    if unknown_id is None:
        raise ValueError("Vocabulary is missing required 'unknown' bucket.")
    fallback_id = int(unknown_id)
    return fallback_id, value


def build_relation_graph_dataset(
    *,
    candidate_dataset_dir: Path,
    g1_dataset_dir: Path,
    partition_manifest_path: Path,
    text_backend: TextEmbeddingBackend,
) -> RelationGraphDataset:
    candidate_rows, snapshot_lookup, partition_lookup = _load_relation_graph_inputs(
        candidate_dataset_dir=candidate_dataset_dir,
        g1_dataset_dir=g1_dataset_dir,
        partition_manifest_path=partition_manifest_path,
    )
    vocabularies = _build_relation_graph_vocabularies_from_inputs(
        candidate_rows=candidate_rows,
        snapshot_lookup=snapshot_lookup,
        partition_lookup=partition_lookup,
    )
    node_type_vocab = vocabularies.node_type_to_id
    role_vocab = vocabularies.role_to_id
    edge_type_vocab = vocabularies.edge_type_to_id
    candidate_kind_vocab = vocabularies.candidate_kind_to_id

    all_texts: list[str] = []
    seen_texts: set[str] = set()
    for row in candidate_rows:
        snapshot = snapshot_lookup.get(str(row.get("state_id", "")).strip())
        if snapshot is None:
            continue
        candidate_text = _strip_leaky_candidate_segments(str(row.get("candidate_text", "")).strip())
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
        clean_candidate_text = _strip_leaky_candidate_segments(str(row.get("candidate_text", "")).strip())

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
            node_type_id, _ = _lookup_vocab_with_fallback(node_type_vocab, node_payload.get("type", "unknown"))
            node_role_id, _ = _lookup_vocab_with_fallback(role_vocab, node_payload.get("role", "unknown"))
            node_type_ids.append(node_type_id)
            node_role_ids.append(node_role_id)
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
            edge_type_id, _ = _lookup_vocab_with_fallback(edge_type_vocab, edge_payload.get("relation", "unknown"))
            edge_type_ids.append(edge_type_id)
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
            candidate_kind_id=_lookup_vocab_with_fallback(
                candidate_kind_vocab,
                row.get("candidate_kind", "unknown"),
            )[0],
            candidate_text_embedding=embedding_cache[clean_candidate_text],
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


def _normalized_runtime_candidate_specs(
    candidate_specs: Sequence[Mapping[str, Any]],
    *,
    use_commit: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(candidate_specs):
        kind = str(spec.get("kind", spec.get("candidate_kind", ""))).strip() or "unknown"
        if kind == "commit" and not use_commit:
            continue
        row = dict(spec)
        row["kind"] = kind
        row["candidate_id"] = (
            str(spec.get("candidate_id", f"runtime-candidate:{index:03d}")).strip()
            or f"runtime-candidate:{index:03d}"
        )
        rows.append(row)
    return rows


def _candidate_kind_from_spec(spec: Mapping[str, Any]) -> str:
    return str(spec.get("candidate_kind", spec.get("kind", "unknown"))).strip() or "unknown"


def _active_runtime_nodes(graph: IdeaGraph) -> list[Node]:
    return sorted(graph.active_nodes(), key=lambda node: node.id)


def _active_runtime_edges(graph: IdeaGraph, active_node_ids: set[str]) -> list[Edge]:
    edges = [
        edge
        for edge in graph.edges
        if edge.source_id in active_node_ids and edge.target_id in active_node_ids
    ]
    return sorted(
        edges,
        key=lambda edge: (edge.source_id, edge.relation, edge.target_id, edge.id),
    )


def _flatten_runtime_state_text(nodes: Sequence[Node], edges: Sequence[Edge]) -> str:
    node_fragments = [f"{node.id}:{node.type}:{node.text}" for node in nodes]
    edge_fragments = [f"{edge.source_id}-{edge.relation}->{edge.target_id}" for edge in edges]
    unresolved_contradictions = sum(
        1
        for edge in edges
        if edge.relation == "contradicts" and not edge.resolved
    )
    return (
        f"nodes={len(nodes)};"
        f"edges={len(edges)};"
        f"contradictions={unresolved_contradictions};"
        f"node_details=[{' || '.join(node_fragments)}];"
        f"edge_details=[{' || '.join(edge_fragments)}]"
    )


def build_relation_graph_runtime_batch(
    *,
    graph: IdeaGraph,
    candidate_specs: Sequence[Mapping[str, Any]],
    text_backend: TextEmbeddingBackend,
    vocabularies: RelationGraphVocabularies,
    use_commit: bool,
) -> RelationGraphRuntimeBatch:
    normalized_specs = _normalized_runtime_candidate_specs(candidate_specs, use_commit=use_commit)
    if not normalized_specs:
        raise ValueError("candidate_specs must not be empty after runtime filtering.")

    node_type_vocab = vocabularies.node_type_to_id
    role_vocab = vocabularies.role_to_id
    edge_type_vocab = vocabularies.edge_type_to_id
    candidate_kind_vocab = vocabularies.candidate_kind_to_id

    nodes = _active_runtime_nodes(graph)
    active_node_ids = {node.id for node in nodes}
    edges = _active_runtime_edges(graph, active_node_ids)
    node_index_lookup = {node.id: index for index, node in enumerate(nodes)}
    state_text = _flatten_runtime_state_text(nodes, edges)

    candidate_text_by_id: dict[str, str] = {}
    all_texts: list[str] = [state_text]
    seen_texts: set[str] = {state_text}
    for node in nodes:
        node_text = str(node.text).strip()
        if node_text not in seen_texts:
            all_texts.append(node_text)
            seen_texts.add(node_text)
    for spec in normalized_specs:
        raw_candidate_text = str(spec.get("candidate_text", "")).strip()
        if not raw_candidate_text:
            raw_candidate_text = flatten_candidate_text(graph, dict(spec))
        clean_candidate_text = _strip_leaky_candidate_segments(raw_candidate_text)
        candidate_text_by_id[str(spec["candidate_id"])] = clean_candidate_text
        if clean_candidate_text not in seen_texts:
            all_texts.append(clean_candidate_text)
            seen_texts.add(clean_candidate_text)

    encoded = text_backend.encode(all_texts)
    text_dim = int(encoded.shape[1]) if encoded.ndim == 2 else 0
    embedding_cache = {
        text: encoded[index]
        for index, text in enumerate(all_texts)
    }

    node_type_ids: list[int] = []
    node_role_ids: list[int] = []
    node_confidence: list[float] = []
    node_evidence_count: list[float] = []
    node_text_embeddings: list[np.ndarray] = []
    missing_node_types: set[str] = set()
    missing_roles: set[str] = set()
    for node in nodes:
        node_text = str(node.text).strip()
        node_text_embeddings.append(embedding_cache[node_text])
        node_type_id, missing_node_type = _lookup_vocab_with_fallback(node_type_vocab, node.type)
        role_id, missing_role = _lookup_vocab_with_fallback(role_vocab, node.role)
        node_type_ids.append(node_type_id)
        node_role_ids.append(role_id)
        node_confidence.append(_safe_float(node.confidence))
        node_evidence_count.append(float(len(node.evidence) if isinstance(node.evidence, list) else 0))
        if missing_node_type is not None:
            missing_node_types.add(missing_node_type)
        if missing_role is not None:
            missing_roles.add(missing_role)

    edge_index: list[tuple[int, int]] = []
    edge_type_ids: list[int] = []
    edge_resolved: list[float] = []
    missing_edge_types: set[str] = set()
    for edge in edges:
        source_index = node_index_lookup.get(edge.source_id)
        target_index = node_index_lookup.get(edge.target_id)
        if source_index is None or target_index is None:
            continue
        edge_index.append((source_index, target_index))
        edge_type_id, missing_edge_type = _lookup_vocab_with_fallback(edge_type_vocab, edge.relation)
        edge_type_ids.append(edge_type_id)
        edge_resolved.append(1.0 if bool(edge.resolved) else 0.0)
        if missing_edge_type is not None:
            missing_edge_types.add(missing_edge_type)

    state_embedding = embedding_cache[state_text]
    shared_node_text_embeddings = (
        np.stack(node_text_embeddings, axis=0).astype(np.float32)
        if node_text_embeddings
        else np.zeros((0, text_dim), dtype=np.float32)
    )
    base_missing_node_types = tuple(sorted(missing_node_types))
    base_missing_roles = tuple(sorted(missing_roles))
    base_missing_edge_types = tuple(sorted(missing_edge_types))

    examples: list[RelationGraphCandidateExample] = []
    diagnostics: list[RelationGraphRuntimeRowDiagnostics] = []
    fallback_rows: list[bool] = []
    runtime_state_id = "runtime::live-state"

    for spec in normalized_specs:
        candidate_id = str(spec["candidate_id"])
        candidate_kind = _candidate_kind_from_spec(spec)
        candidate_kind_id, missing_candidate_kind = _lookup_vocab_with_fallback(
            candidate_kind_vocab,
            candidate_kind,
        )

        target_node_indices: list[int] = []
        missing_target_ids: list[str] = []
        raw_target_ids = spec.get("target_ids", [])
        if isinstance(raw_target_ids, Sequence) and not isinstance(raw_target_ids, (str, bytes)):
            for raw_target_id in raw_target_ids:
                target_id = str(raw_target_id).strip()
                if not target_id:
                    continue
                target_index = node_index_lookup.get(target_id)
                if target_index is None:
                    missing_target_ids.append(target_id)
                    continue
                target_node_indices.append(target_index)

        example = RelationGraphCandidateExample(
            state_id=runtime_state_id,
            candidate_id=candidate_id,
            group_id="runtime",
            split="runtime",
            label=0,
            is_commit=str(spec.get("kind", "")).strip() == "commit",
            candidate_kind_id=candidate_kind_id,
            candidate_text_embedding=embedding_cache[candidate_text_by_id[candidate_id]],
            state_text_embedding=state_embedding,
            node_text_embeddings=shared_node_text_embeddings,
            node_type_ids=list(node_type_ids),
            node_role_ids=list(node_role_ids),
            node_confidence=list(node_confidence),
            node_evidence_count=list(node_evidence_count),
            edge_index=list(edge_index),
            edge_type_ids=list(edge_type_ids),
            edge_resolved=list(edge_resolved),
            target_node_indices=target_node_indices,
        )
        examples.append(example)

        missing_candidate_kind_flag = missing_candidate_kind is not None
        used_vocab_fallback = bool(
            base_missing_node_types
            or base_missing_roles
            or base_missing_edge_types
            or missing_candidate_kind_flag
        )
        requires_heuristic_fallback = used_vocab_fallback or bool(missing_target_ids)
        diagnostics.append(
            RelationGraphRuntimeRowDiagnostics(
                candidate_id=candidate_id,
                missing_node_types=base_missing_node_types,
                missing_roles=base_missing_roles,
                missing_edge_types=base_missing_edge_types,
                missing_candidate_kind=missing_candidate_kind_flag,
                missing_target_ids=tuple(missing_target_ids),
                used_vocab_fallback=used_vocab_fallback,
                requires_heuristic_fallback=requires_heuristic_fallback,
            )
        )
        fallback_rows.append(requires_heuristic_fallback)

    batch = collate_relation_graph_examples(examples)
    return RelationGraphRuntimeBatch(
        examples=tuple(examples),
        batch=batch,
        diagnostics=tuple(diagnostics),
        fallback_row_mask=torch.tensor(fallback_rows, dtype=torch.bool),
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
    candidate_state_index = torch.zeros((batch_size,), dtype=torch.long)
    labels = torch.zeros((batch_size,), dtype=torch.float32)
    is_commit = torch.zeros((batch_size,), dtype=torch.float32)

    node_type_max = 0
    role_max = 0
    edge_type_max = 0
    candidate_kind_max = 0

    state_lookup: dict[str, int] = {}
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
        candidate_state_index[batch_index] = state_lookup.setdefault(example.state_id, len(state_lookup))
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
        candidate_state_index=candidate_state_index,
        labels=labels,
        is_commit=is_commit,
        node_type_vocab_size=node_type_max + 1,
        role_vocab_size=role_max + 1,
        edge_type_vocab_size=edge_type_max + 1,
        candidate_kind_vocab_size=candidate_kind_max + 1,
    )
