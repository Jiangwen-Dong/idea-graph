from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CandidateExample:
    state_id: str
    candidate_id: str
    split: str
    label: int
    state_text: str
    candidate_text: str
    group_id: str
    weak_value_01: float | None
    native_value_01: float | None


def _strip_leaky_segments(candidate_text: str) -> str:
    parts = [part.strip() for part in candidate_text.split("|")]
    kept_parts = [
        part
        for part in parts
        if part
        and not part.lower().startswith("source=")
        and not part.lower().startswith("rationale=")
    ]
    return "|".join(kept_parts)


def _join_text(example: CandidateExample) -> str:
    return f"{example.state_text} [SEP] {_strip_leaky_segments(example.candidate_text)}"


class TextCriticModel:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def score(self, texts: Sequence[str]) -> list[float]:
        values = list(texts)
        if not values:
            return []
        probabilities = self.pipeline.predict_proba(values)
        class_labels = [int(label) for label in self.pipeline.classes_]
        positive_index = class_labels.index(1) if 1 in class_labels else len(class_labels) - 1
        return [float(row[positive_index]) for row in probabilities]


def build_training_examples(candidate_rows: Sequence[Mapping[str, Any]]) -> list[CandidateExample]:
    examples: list[CandidateExample] = []
    for row in candidate_rows:
        targets = row.get("targets", {})
        if not isinstance(targets, Mapping):
            targets = {}
        examples.append(
            CandidateExample(
                state_id=str(row.get("state_id", "")).strip(),
                candidate_id=str(row.get("candidate_id", "")).strip(),
                split=str(row.get("split", "train")).strip() or "train",
                label=1 if bool(row.get("is_logged_selected", False)) else 0,
                state_text=str(row.get("state_text", "")),
                candidate_text=str(row.get("candidate_text", "")),
                group_id=str(row.get("group_id", "")),
                weak_value_01=_as_optional_float(targets.get("weak_value_01")),
                native_value_01=_as_optional_float(targets.get("native_value_01")),
            )
        )
    return examples


def train_text_critic(train_examples: Sequence[CandidateExample]) -> TextCriticModel:
    if not train_examples:
        raise ValueError("train_examples must not be empty.")
    labels = [example.label for example in train_examples]
    if len(set(labels)) < 2:
        raise ValueError("train_examples must contain both positive and negative labels.")

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)),
        ]
    )
    train_texts = [_join_text(example) for example in train_examples]
    pipeline.fit(train_texts, labels)
    return TextCriticModel(pipeline)


def build_split_audit(
    train_examples: Sequence[CandidateExample],
    validation_examples: Sequence[CandidateExample],
) -> dict[str, int]:
    train_groups = {example.group_id for example in train_examples}
    validation_groups = {example.group_id for example in validation_examples}
    return {
        "train_group_count": len(train_groups),
        "validation_group_count": len(validation_groups),
        "group_overlap_count": len(train_groups.intersection(validation_groups)),
    }


def evaluate_state_rankings(
    model: Any,
    validation_examples: Sequence[CandidateExample],
) -> dict[str, float | int]:
    by_state: dict[str, list[CandidateExample]] = {}
    for example in validation_examples:
        by_state.setdefault(example.state_id, []).append(example)

    top1_hits = 0
    reciprocal_ranks: list[float] = []
    scored_state_count = 0

    for state_id in sorted(by_state):
        state_examples = by_state[state_id]
        positive_count = sum(1 for example in state_examples if example.label == 1)
        if positive_count != 1:
            raise ValueError(
                f"State '{state_id}' must have exactly one positive label; found {positive_count}."
            )
        texts = [_join_text(example) for example in state_examples]
        scores = model.score(texts)
        if len(scores) != len(state_examples):
            raise ValueError(
                f"State '{state_id}' produced {len(scores)} scores for {len(state_examples)} candidates."
            )
        ranked = sorted(
            zip(state_examples, scores),
            key=lambda item: (-item[1], item[0].candidate_id),
        )
        positive_rank: int | None = None
        for rank, (example, _) in enumerate(ranked, start=1):
            if example.label == 1:
                positive_rank = rank
                break
        if positive_rank is None:
            continue
        scored_state_count += 1
        reciprocal_ranks.append(1.0 / positive_rank)
        if positive_rank == 1:
            top1_hits += 1

    mean_reciprocal_rank = (
        sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    )
    top1_accuracy = top1_hits / scored_state_count if scored_state_count else 0.0
    return {
        "state_count": scored_state_count,
        "top1_accuracy": top1_accuracy,
        "mean_reciprocal_rank": mean_reciprocal_rank,
    }
