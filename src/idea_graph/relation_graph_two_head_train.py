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
from .relation_graph_critic_data import RelationGraphCandidateExample, collate_relation_graph_examples
from .relation_graph_critic_train import (
    _iterate_state_grouped_batches,
    compute_state_ranking_loss,
    evaluate_relation_graph_rankings,
)
from .relation_graph_two_head_data import (
    RelationGraphCommitExample,
    RelationGraphTwoHeadDataset,
    collate_relation_graph_commit_examples,
)
from .relation_graph_two_head_model import RelationGraphTwoHeadCritic


@dataclass(frozen=True)
class RelationGraphTwoHeadTrainingArtifacts:
    edit_metrics: dict[str, float | int]
    commit_metrics: dict[str, float | int]
    metadata: dict[str, Any]
    model_path: Path | None = None


def _iterate_commit_batches(
    examples: Sequence[RelationGraphCommitExample],
    *,
    batch_size: int,
    shuffle: bool,
    rng: random.Random | None = None,
) -> list[list[RelationGraphCommitExample]]:
    ordered = list(examples)
    if shuffle:
        active_rng = rng or random
        active_rng.shuffle(ordered)
    return [ordered[index : index + batch_size] for index in range(0, len(ordered), max(batch_size, 1))]


def _score_edit_examples(
    model: RelationGraphTwoHeadCritic,
    examples: Sequence[RelationGraphCandidateExample],
    *,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch_examples in _iterate_state_grouped_batches(examples, batch_size=batch_size, shuffle=False):
            batch = collate_relation_graph_examples(batch_examples).to(device)
            scores = model.score_edit_batch(batch).detach().cpu().tolist()
            for example, score in zip(batch_examples, scores, strict=True):
                rows.append(
                    {
                        "state_id": example.state_id,
                        "candidate_id": example.candidate_id,
                        "label": example.label,
                        "score": float(score),
                        "is_commit": False,
                    }
                )
    return rows


def _evaluate_commit_examples(
    model: RelationGraphTwoHeadCritic,
    examples: Sequence[RelationGraphCommitExample],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    if not examples:
        return {"example_count": 0, "accuracy": 0.0}

    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch_examples in _iterate_commit_batches(examples, batch_size=batch_size, shuffle=False):
            batch = collate_relation_graph_commit_examples(batch_examples).to(device)
            logits = model.score_commit_batch(batch)
            predictions = (torch.sigmoid(logits) >= 0.5).long().detach().cpu().tolist()
            labels = batch.labels.long().detach().cpu().tolist()
            for prediction, label in zip(predictions, labels, strict=True):
                total += 1
                if int(prediction) == int(label):
                    correct += 1
    return {
        "example_count": total,
        "accuracy": (correct / total) if total else 0.0,
    }


def train_relation_graph_two_head_critic(
    *,
    dataset: RelationGraphTwoHeadDataset,
    output_dir: Path,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    text_backend_name: str,
    text_model_name: str | None = None,
    commit_loss_weight: float = 1.0,
) -> RelationGraphTwoHeadTrainingArtifacts:
    if not dataset.edit_train_examples:
        raise ValueError("dataset.edit_train_examples must not be empty.")
    if not dataset.edit_dev_examples:
        raise ValueError("dataset.edit_dev_examples must not be empty.")
    if not dataset.commit_train_examples:
        raise ValueError("dataset.commit_train_examples must not be empty.")
    if not dataset.commit_dev_examples:
        raise ValueError("dataset.commit_dev_examples must not be empty.")

    text_dim = int(dataset.edit_train_examples[0].node_text_embeddings.shape[1])
    model = RelationGraphTwoHeadCritic(
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        node_type_count=int(dataset.metadata.get("node_type_count", 0) or 0),
        role_count=int(dataset.metadata.get("role_count", 0) or 0),
        edge_type_count=int(dataset.metadata.get("edge_type_count", 0) or 0),
        candidate_kind_count=int(dataset.metadata.get("candidate_kind_count", 0) or 0),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_state_dict = None
    best_edit_metrics: dict[str, float | int] | None = None
    best_commit_metrics: dict[str, float | int] | None = None
    best_score = float("-inf")
    shuffle_rng = random.Random(0)

    for _epoch in range(max(epochs, 1)):
        model.train()
        for batch_examples in _iterate_state_grouped_batches(
            dataset.edit_train_examples,
            batch_size=batch_size,
            shuffle=True,
            rng=shuffle_rng,
        ):
            batch = collate_relation_graph_examples(batch_examples).to(device)
            optimizer.zero_grad()
            scores = model.score_edit_batch(batch)
            loss = compute_state_ranking_loss(scores, batch.labels, batch.candidate_state_index)
            loss.backward()
            optimizer.step()

        for batch_examples in _iterate_commit_batches(
            dataset.commit_train_examples,
            batch_size=batch_size,
            shuffle=True,
            rng=shuffle_rng,
        ):
            batch = collate_relation_graph_commit_examples(batch_examples).to(device)
            optimizer.zero_grad()
            logits = model.score_commit_batch(batch)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch.labels)
            (commit_loss_weight * loss).backward()
            optimizer.step()

        edit_rows = _score_edit_examples(model, dataset.edit_dev_examples, batch_size=batch_size, device=device)
        edit_metrics = evaluate_relation_graph_rankings(edit_rows)["edit_only"]
        commit_metrics = _evaluate_commit_examples(
            model,
            dataset.commit_dev_examples,
            batch_size=batch_size,
            device=device,
        )
        combined_score = float(edit_metrics["mean_reciprocal_rank"]) + float(commit_metrics["accuracy"])
        if combined_score >= best_score:
            best_score = combined_score
            best_edit_metrics = edit_metrics
            best_commit_metrics = commit_metrics
            best_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state_dict is None or best_edit_metrics is None or best_commit_metrics is None:
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
        "edit_train_example_count": len(dataset.edit_train_examples),
        "edit_validation_example_count": len(dataset.edit_dev_examples),
        "commit_train_example_count": len(dataset.commit_train_examples),
        "commit_validation_example_count": len(dataset.commit_dev_examples),
    }
    write_text_file(output_dir / "edit_metrics.json", json.dumps(best_edit_metrics, indent=2))
    write_text_file(output_dir / "commit_metrics.json", json.dumps(best_commit_metrics, indent=2))
    write_text_file(output_dir / "metadata.json", json.dumps(metadata, indent=2))
    return RelationGraphTwoHeadTrainingArtifacts(
        edit_metrics=best_edit_metrics,
        commit_metrics=best_commit_metrics,
        metadata=metadata,
        model_path=model_path,
    )
