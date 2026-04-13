from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Sequence

import torch
from torch import nn
from torch.optim import AdamW

from .fs_utils import write_text_file
from .relation_graph_critic_data import (
    RelationGraphDataset,
    collate_relation_graph_examples,
)
from .relation_graph_critic_model import RelationGraphCritic


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
        ),
        "edit_only": _ranking_metrics_by_state(
            edit_rows,
        ),
    }


@dataclass(frozen=True)
class RelationGraphTrainingArtifacts:
    metrics_all: dict[str, float | int]
    metrics_edit_only: dict[str, float | int]
    metadata: dict[str, Any]
    model_path: Path | None = None


def _iterate_example_batches(
    examples: Sequence[Any],
    *,
    batch_size: int,
    shuffle: bool,
    rng: random.Random | None = None,
) -> list[list[Any]]:
    return _iterate_state_grouped_batches(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        rng=rng,
    )


def _iterate_state_grouped_batches(
    examples: Sequence[Any],
    *,
    batch_size: int,
    shuffle: bool,
    rng: random.Random | None = None,
) -> list[list[Any]]:
    ordered = list(examples)
    if shuffle:
        active_rng = rng or random
        active_rng.shuffle(ordered)
    grouped_by_state: dict[str, list[Any]] = {}
    state_order: list[str] = []
    for example in ordered:
        state_id = str(example.state_id)
        if state_id not in grouped_by_state:
            grouped_by_state[state_id] = []
            state_order.append(state_id)
        grouped_by_state[state_id].append(example)
    batches: list[list[Any]] = []
    current_batch: list[Any] = []
    current_state_count = 0
    for state_id in state_order:
        state_examples = grouped_by_state[state_id]
        if current_batch and current_state_count >= batch_size:
            batches.append(current_batch)
            current_batch = []
            current_state_count = 0
        current_batch.extend(state_examples)
        current_state_count += 1
    if current_batch:
        batches.append(current_batch)
    return batches


def _score_examples(
    model: RelationGraphCritic,
    examples: Sequence[Any],
    *,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch_examples in _iterate_example_batches(examples, batch_size=batch_size, shuffle=False):
            batch = collate_relation_graph_examples(batch_examples).to(device)
            scores = model(batch).detach().cpu().tolist()
            for example, score in zip(batch_examples, scores, strict=True):
                rows.append(
                    {
                        "state_id": example.state_id,
                        "candidate_id": example.candidate_id,
                        "label": example.label,
                        "score": float(score),
                        "is_commit": example.is_commit,
                    }
                )
    return rows


def train_relation_graph_critic(
    *,
    dataset: RelationGraphDataset,
    output_dir: Path,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    text_backend_name: str,
    text_model_name: str | None = None,
) -> RelationGraphTrainingArtifacts:
    if not dataset.train_examples:
        raise ValueError("dataset.train_examples must not be empty.")
    if not dataset.dev_examples:
        raise ValueError("dataset.dev_examples must not be empty.")

    sample_batch = collate_relation_graph_examples(dataset.train_examples[: min(batch_size, len(dataset.train_examples))])
    text_dim = int(sample_batch.node_text_embeddings.shape[-1])
    model = RelationGraphCritic(
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        node_type_count=sample_batch.node_type_vocab_size,
        role_count=sample_batch.role_vocab_size,
        edge_type_count=sample_batch.edge_type_vocab_size,
        candidate_kind_count=sample_batch.candidate_kind_vocab_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_state_dict = None
    best_metrics: dict[str, dict[str, float | int]] | None = None
    best_dev_mrr = float("-inf")
    shuffle_rng = random.Random(0)

    for _epoch in range(max(epochs, 1)):
        model.train()
        for batch_examples in _iterate_example_batches(
            dataset.train_examples,
            batch_size=batch_size,
            shuffle=True,
            rng=shuffle_rng,
        ):
            batch = collate_relation_graph_examples(batch_examples).to(device)
            optimizer.zero_grad()
            scores = model(batch)
            loss = compute_state_ranking_loss(scores, batch.labels, batch.candidate_state_index)
            loss.backward()
            optimizer.step()

        dev_rows = _score_examples(model, dataset.dev_examples, batch_size=batch_size, device=device)
        metrics = evaluate_relation_graph_rankings(dev_rows)
        dev_mrr = float(metrics["all"]["mean_reciprocal_rank"])
        if dev_mrr >= best_dev_mrr:
            best_dev_mrr = dev_mrr
            best_metrics = metrics
            best_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state_dict is None or best_metrics is None:
        raise ValueError("Failed to produce a best model checkpoint.")

    model.load_state_dict(best_state_dict)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    torch.save(best_state_dict, model_path)

    metadata = {
        "text_backend": text_backend_name,
        "text_model_name": text_model_name,
        "device": device.type,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_example_count": len(dataset.train_examples),
        "validation_example_count": len(dataset.dev_examples),
    }
    write_text_file(output_dir / "metrics_all.json", json.dumps(best_metrics["all"], indent=2))
    write_text_file(output_dir / "metrics_edit_only.json", json.dumps(best_metrics["edit_only"], indent=2))
    write_text_file(output_dir / "metadata.json", json.dumps(metadata, indent=2))
    return RelationGraphTrainingArtifacts(
        metrics_all=best_metrics["all"],
        metrics_edit_only=best_metrics["edit_only"],
        metadata=metadata,
        model_path=model_path,
    )
