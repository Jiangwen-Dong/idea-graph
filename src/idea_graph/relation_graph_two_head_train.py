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


def _append_training_progress_row(output_dir: Path, row: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "training_progress.jsonl"
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    positive_count = sum(1 for label in labels if int(label) == 1)
    if positive_count <= 0:
        return 0.0
    ordered = sorted(
        zip(probabilities, labels, strict=True),
        key=lambda item: float(item[0]),
        reverse=True,
    )
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    for _probability, label in ordered:
        if int(label) == 1:
            true_positives += 1
            recall = true_positives / positive_count
            precision = true_positives / max(true_positives + false_positives, 1)
            area += precision * (recall - previous_recall)
            previous_recall = recall
        else:
            false_positives += 1
    return float(area)


def _binary_commit_metrics(
    *,
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    total = len(labels)
    predictions = [1 if float(probability) >= threshold else 0 for probability in probabilities]
    true_positive = sum(1 for prediction, label in zip(predictions, labels, strict=True) if prediction == 1 and int(label) == 1)
    true_negative = sum(1 for prediction, label in zip(predictions, labels, strict=True) if prediction == 0 and int(label) == 0)
    false_positive = sum(1 for prediction, label in zip(predictions, labels, strict=True) if prediction == 1 and int(label) == 0)
    false_negative = sum(1 for prediction, label in zip(predictions, labels, strict=True) if prediction == 0 and int(label) == 1)
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / positive_count if positive_count else 0.0
    specificity = true_negative / negative_count if negative_count else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "example_count": total,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "predicted_positive_count": sum(predictions),
        "accuracy": ((true_positive + true_negative) / total) if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": ((recall + specificity) / 2.0) if positive_count and negative_count else 0.0,
        "average_precision": _average_precision(labels, probabilities),
        "threshold": threshold,
    }


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

    all_probabilities: list[float] = []
    all_labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch_examples in _iterate_commit_batches(examples, batch_size=batch_size, shuffle=False):
            batch = collate_relation_graph_commit_examples(batch_examples).to(device)
            logits = model.score_commit_batch(batch)
            probabilities = torch.sigmoid(logits).detach().cpu().tolist()
            labels = batch.labels.long().detach().cpu().tolist()
            all_probabilities.extend(float(probability) for probability in probabilities)
            all_labels.extend(int(label) for label in labels)
    return _binary_commit_metrics(labels=all_labels, probabilities=all_probabilities)


def _commit_positive_weight(examples: Sequence[RelationGraphCommitExample]) -> float:
    positive_count = sum(1 for example in examples if int(example.label) == 1)
    negative_count = sum(1 for example in examples if int(example.label) == 0)
    if positive_count <= 0 or negative_count <= 0:
        return 1.0
    return float(negative_count / positive_count)


def _selection_score(
    *,
    edit_metrics: dict[str, float | int],
    commit_metrics: dict[str, float | int],
) -> float:
    return (
        float(edit_metrics["mean_reciprocal_rank"])
        + 0.45 * float(commit_metrics.get("f1", 0.0))
        + 0.35 * float(commit_metrics.get("average_precision", 0.0))
        + 0.20 * float(commit_metrics.get("balanced_accuracy", 0.0))
    )


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
    commit_pos_weight = _commit_positive_weight(dataset.commit_train_examples)
    commit_pos_weight_tensor = torch.tensor(commit_pos_weight, dtype=torch.float32, device=device)

    best_state_dict = None
    best_edit_metrics: dict[str, float | int] | None = None
    best_commit_metrics: dict[str, float | int] | None = None
    best_score = float("-inf")
    shuffle_rng = random.Random(0)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "training_progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    for epoch_index in range(max(epochs, 1)):
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
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                batch.labels,
                pos_weight=commit_pos_weight_tensor,
            )
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
        progress_row = {
            "epoch": epoch_index + 1,
            "edit_state_count": int(edit_metrics["state_count"]),
            "edit_top1_accuracy": float(edit_metrics["top1_accuracy"]),
            "edit_mean_reciprocal_rank": float(edit_metrics["mean_reciprocal_rank"]),
            "commit_example_count": int(commit_metrics["example_count"]),
            "commit_accuracy": float(commit_metrics["accuracy"]),
            "commit_precision": float(commit_metrics.get("precision", 0.0)),
            "commit_recall": float(commit_metrics.get("recall", 0.0)),
            "commit_f1": float(commit_metrics.get("f1", 0.0)),
            "commit_average_precision": float(commit_metrics.get("average_precision", 0.0)),
            "commit_balanced_accuracy": float(commit_metrics.get("balanced_accuracy", 0.0)),
        }
        progress_row["model_selection_score"] = _selection_score(
            edit_metrics=edit_metrics,
            commit_metrics=commit_metrics,
        )
        _append_training_progress_row(output_dir, progress_row)
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    **progress_row,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        combined_score = float(progress_row["model_selection_score"])
        if combined_score >= best_score:
            best_score = combined_score
            best_edit_metrics = edit_metrics
            best_commit_metrics = commit_metrics
            best_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state_dict is None or best_edit_metrics is None or best_commit_metrics is None:
        raise ValueError("Failed to produce a best model checkpoint.")

    model.load_state_dict(best_state_dict)
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
        "commit_positive_weight": commit_pos_weight,
        "commit_loss_weight": commit_loss_weight,
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
