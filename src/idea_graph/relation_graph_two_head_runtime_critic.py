from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .models import IdeaGraph
from .relation_graph_critic_data import (
    RelationGraphBatch,
    RelationGraphRuntimeBatch,
    RelationGraphVocabularies,
    build_relation_graph_runtime_batch,
)
from .relation_graph_runtime_critic import (
    _load_text_backend,
    _load_torch_checkpoint,
    _load_vocabularies_from_snapshot,
    _require_json_object,
    _try_write_vocabularies_snapshot,
    _unsafe_runtime_candidates_from_batch,
)
from .relation_graph_two_head_data import (
    RelationGraphCommitBatch,
    _build_vocabularies,
    _load_jsonl,
)
from .relation_graph_two_head_model import RelationGraphTwoHeadCritic


class LoadedRelationGraphTwoHeadRuntimeCritic:
    def __init__(
        self,
        *,
        model: RelationGraphTwoHeadCritic,
        vocabs: RelationGraphVocabularies,
        text_backend: Any,
        device: torch.device,
    ) -> None:
        self.model = model
        self.vocabs = vocabs
        self.text_backend = text_backend
        self.device = device

    def build_runtime_batch(
        self,
        *,
        graph: IdeaGraph,
        candidate_specs: Sequence[Mapping[str, Any]],
        use_commit: bool,
    ) -> RelationGraphRuntimeBatch:
        return build_relation_graph_runtime_batch(
            graph=graph,
            candidate_specs=candidate_specs,
            text_backend=self.text_backend,
            vocabularies=self.vocabs,
            use_commit=use_commit,
        )

    def runtime_token_status(self, runtime_batch: RelationGraphRuntimeBatch) -> dict[str, object]:
        unsafe_by_id = _unsafe_runtime_candidates_from_batch(runtime_batch)
        if not unsafe_by_id:
            return {"ok": True, "reason": "", "candidate_ids": ()}
        return {
            "ok": False,
            "reason": "unmapped_runtime_token",
            "candidate_ids": tuple(sorted(unsafe_by_id)),
        }

    def score_runtime_batch(self, batch: RelationGraphBatch) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            scores = self.model.score_edit_batch(batch.to(self.device)).detach().cpu().tolist()
        return [float(value) for value in scores]

    def score_commit_graph(self, graph: IdeaGraph, *, snapshot: Any | None = None) -> float:
        if snapshot is None:
            from .engine import maturity_snapshot

            snapshot = maturity_snapshot(graph)
        branch_id = next(iter(graph.branches), "")
        runtime_batch = self.build_runtime_batch(
            graph=graph,
            candidate_specs=[
                {
                    "candidate_id": "runtime-commit-state",
                    "kind": "skip",
                    "target_ids": [],
                    "payload": {"branch_id": branch_id},
                    "rationale": "Score the current post-round graph for commit.",
                }
            ],
            use_commit=False,
        )
        edit_batch = runtime_batch.batch
        graph_features = torch.tensor(
            [
                [
                    float(getattr(snapshot, "support_coverage", 0.0)),
                    float(getattr(snapshot, "unresolved_contradiction_ratio", 0.0)),
                    float(getattr(snapshot, "utility", 0.0)),
                ]
            ],
            dtype=torch.float32,
        )
        commit_batch = RelationGraphCommitBatch(
            node_text_embeddings=edit_batch.node_text_embeddings,
            state_text_embeddings=edit_batch.state_text_embeddings,
            node_type_ids=edit_batch.node_type_ids,
            node_role_ids=edit_batch.node_role_ids,
            node_scalars=edit_batch.node_scalars,
            edge_index=edit_batch.edge_index,
            edge_type_ids=edit_batch.edge_type_ids,
            edge_resolved=edit_batch.edge_resolved,
            edge_mask=edit_batch.edge_mask,
            graph_mask=edit_batch.graph_mask,
            graph_features=graph_features,
            labels=torch.zeros((1,), dtype=torch.float32),
            node_type_vocab_size=edit_batch.node_type_vocab_size,
            role_vocab_size=edit_batch.role_vocab_size,
            edge_type_vocab_size=edit_batch.edge_type_vocab_size,
        )
        self.model.eval()
        with torch.no_grad():
            logit = self.model.score_commit_batch(commit_batch.to(self.device))
            probability = torch.sigmoid(logit).detach().cpu().item()
        return float(probability)


def _load_two_head_vocabularies(
    resolved_model_dir: Path,
    training_config: Mapping[str, Any],
) -> RelationGraphVocabularies:
    snapshot_path = resolved_model_dir / "vocabularies.json"
    if snapshot_path.exists():
        return _load_vocabularies_from_snapshot(snapshot_path)

    dataset_dir = Path(str(training_config.get("dataset_dir", "")).strip())
    g1_dataset_dir = Path(str(training_config.get("g1_dataset_dir", "")).strip())
    if not dataset_dir.exists() or not g1_dataset_dir.exists():
        raise ValueError(
            "Two-head runtime model is missing vocabularies.json and training_config "
            "does not point to existing dataset_dir/g1_dataset_dir."
        )
    edit_rows = _load_jsonl(dataset_dir / "edit_head_rows.jsonl")
    commit_rows = _load_jsonl(dataset_dir / "commit_head_rows.jsonl")
    vocabs = _build_vocabularies(
        edit_rows=edit_rows,
        commit_rows=commit_rows,
        g1_dataset_dir=g1_dataset_dir,
    )
    _try_write_vocabularies_snapshot(snapshot_path, vocabs)
    return vocabs


def load_relation_graph_two_head_runtime_bundle(
    model_dir: Path | str,
) -> LoadedRelationGraphTwoHeadRuntimeCritic:
    resolved = Path(model_dir).resolve()
    training_config = _require_json_object(resolved / "training_config.json")
    metadata = _require_json_object(resolved / "metadata.json")
    vocabs = _load_two_head_vocabularies(resolved, training_config)
    text_backend = _load_text_backend(training_config)
    text_dim = int(training_config.get("embedding_dim", 0) or 0)
    if text_dim <= 0:
        raise ValueError("training_config.embedding_dim must be a positive integer.")
    hidden_dim = int(metadata.get("hidden_dim", training_config.get("hidden_dim", 0) or 0))
    if hidden_dim <= 0:
        raise ValueError("metadata.hidden_dim (or training_config.hidden_dim) must be positive.")

    model = RelationGraphTwoHeadCritic(
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        node_type_count=len(vocabs.node_type_to_id),
        role_count=len(vocabs.role_to_id),
        edge_type_count=len(vocabs.edge_type_to_id),
        candidate_kind_count=len(vocabs.candidate_kind_to_id),
    )
    state_payload = _load_torch_checkpoint(resolved / "model.pt")
    if not isinstance(state_payload, Mapping):
        raise ValueError(f"{resolved / 'model.pt'} must contain a model state dict.")
    state_dict = state_payload.get("model_state_dict", state_payload)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"{resolved / 'model.pt'} does not contain a valid model state dict.")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise ValueError(f"{resolved / 'model.pt'} does not match the two-head architecture: {exc}") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return LoadedRelationGraphTwoHeadRuntimeCritic(
        model=model,
        vocabs=vocabs,
        text_backend=text_backend,
        device=device,
    )


__all__ = [
    "LoadedRelationGraphTwoHeadRuntimeCritic",
    "load_relation_graph_two_head_runtime_bundle",
]
