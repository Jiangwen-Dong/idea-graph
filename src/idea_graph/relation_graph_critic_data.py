from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

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
            node_text_embeddings=np.stack(node_text_embeddings, axis=0).astype(np.float32),
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
