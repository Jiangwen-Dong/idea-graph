from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn


def _is_positive_label(value: object) -> bool:
    try:
        return float(value) > 0.5
    except (TypeError, ValueError):
        return False


def compute_state_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    state_index: torch.Tensor,
) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D tensor.")
    if labels.shape != scores.shape:
        raise ValueError("labels must have the same shape as scores.")
    if state_index.shape != scores.shape:
        raise ValueError("state_index must have the same shape as scores.")

    losses: list[torch.Tensor] = []
    for state_id in state_index.unique(sorted=True):
        mask = state_index == state_id
        state_scores = scores[mask]
        state_labels = labels[mask]
        positive_positions = torch.nonzero(state_labels > 0.5, as_tuple=False).flatten()
        if positive_positions.numel() != 1:
            continue
        target = positive_positions[:1].long()
        losses.append(nn.functional.cross_entropy(state_scores.unsqueeze(0), target))
    if not losses:
        raise ValueError("At least one valid state with a single positive candidate is required.")
    return torch.stack(losses).mean()


def _ranking_metrics_by_state(
    rows: Sequence[dict[str, Any]],
    *,
    skip_states_without_single_positive: bool,
) -> dict[str, float | int]:
    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_state.setdefault(str(row.get("state_id", "")).strip(), []).append(dict(row))

    top1_hits = 0
    reciprocal_ranks: list[float] = []
    for state_id in sorted(by_state):
        state_rows = by_state[state_id]
        positive_count = sum(1 for row in state_rows if _is_positive_label(row.get("label", 0)))
        if positive_count != 1:
            continue
        ranked = sorted(
            state_rows,
            key=lambda row: (-float(row.get("score", 0.0)), str(row.get("candidate_id", ""))),
        )
        positive_rank = next(
            rank
            for rank, row in enumerate(ranked, start=1)
            if _is_positive_label(row.get("label", 0))
        )
        reciprocal_ranks.append(1.0 / positive_rank)
        if positive_rank == 1:
            top1_hits += 1

    state_count = len(reciprocal_ranks)
    return {
        "state_count": state_count,
        "top1_accuracy": top1_hits / state_count if state_count else 0.0,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / state_count if state_count else 0.0,
    }


def evaluate_relation_graph_rankings(
    state_rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    all_rows = [dict(row) for row in state_rows]
    edit_rows = [dict(row) for row in state_rows if not bool(row.get("is_commit", False))]
    return {
        "all": _ranking_metrics_by_state(
            all_rows,
            skip_states_without_single_positive=False,
        ),
        "edit_only": _ranking_metrics_by_state(
            edit_rows,
            skip_states_without_single_positive=True,
        ),
    }


@dataclass(frozen=True)
class RelationGraphTrainingArtifacts:
    metrics_all: dict[str, float | int]
    metrics_edit_only: dict[str, float | int]
    metadata: dict[str, Any]
    model_path: Path | None = None
