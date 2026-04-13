from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .candidate_slate_dataset import state_id_from_transition
from .fs_utils import read_text_file, write_text_file
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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class GraphFeatureExample:
    state_id: str
    candidate_id: str
    group_id: str
    split: str
    label: int
    is_commit: bool
    is_commit_positive_state: bool
    feature_dict: dict[str, float]


@dataclass(frozen=True)
class GraphFeatureDataset:
    examples: list[GraphFeatureExample]
    split_counts: dict[str, int]
    group_overlap_count: int


@dataclass(frozen=True)
class GraphFeatureCriticTrainResult:
    metrics: dict[str, float | int]
    model_path: Path
    metadata_path: Path


class GraphFeatureCriticModel:
    def __init__(self, vectorizer: DictVectorizer, classifier: LogisticRegression) -> None:
        self.vectorizer = vectorizer
        self.classifier = classifier

    def score_feature_dicts(self, feature_dicts: Sequence[Mapping[str, float]]) -> list[float]:
        items = [dict(feature_dict) for feature_dict in feature_dicts]
        if not items:
            return []
        matrix = self.vectorizer.transform(items)
        probabilities = self.classifier.predict_proba(matrix)
        class_labels = [int(label) for label in self.classifier.classes_]
        positive_index = class_labels.index(1) if 1 in class_labels else len(class_labels) - 1
        return [float(row[positive_index]) for row in probabilities]


def _build_snapshot_lookup(g1_dataset_dir: Path) -> dict[str, Path]:
    dataset_dir = Path(g1_dataset_dir)
    lookup: dict[str, Path] = {}
    for relative_name in ("trajectory_examples.jsonl", "terminal_state_manifest.jsonl"):
        path = dataset_dir / relative_name
        if not path.exists():
            continue
        for row in _load_jsonl(path):
            snapshot_relative = str(row.get("before_state_snapshot", "")).strip()
            if not snapshot_relative:
                continue
            state_id = state_id_from_transition(row)
            lookup[state_id] = dataset_dir / snapshot_relative
    if not lookup:
        raise ValueError(f"No before_state_snapshot entries found under {dataset_dir}.")
    return lookup


def _load_snapshot(snapshot_path: Path) -> dict[str, Any]:
    payload = json.loads(read_text_file(snapshot_path))
    if not isinstance(payload, dict):
        raise ValueError(f"{snapshot_path} does not contain a JSON object.")
    return dict(payload)


def _categorical_feature(features: dict[str, float], namespace: str, value: object) -> None:
    normalized = str(value).strip() or "unknown"
    features[f"{namespace}={normalized}"] = 1.0


def _counted_feature(features: dict[str, float], namespace: str, value: object, amount: float = 1.0) -> None:
    normalized = str(value).strip() or "unknown"
    key = f"{namespace}__{normalized}"
    features[key] = float(features.get(key, 0.0) + amount)


def _extract_snapshot_feature_dict(
    candidate_row: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, float]:
    features: dict[str, float] = {}
    _categorical_feature(features, "benchmark", candidate_row.get("benchmark"))
    _categorical_feature(features, "role", candidate_row.get("role"))
    _categorical_feature(features, "round_name", candidate_row.get("round_name"))
    _categorical_feature(features, "state_kind", candidate_row.get("state_kind"))
    _categorical_feature(features, "candidate_kind", candidate_row.get("candidate_kind"))

    target_ids = [str(item).strip() for item in candidate_row.get("candidate_target_ids", []) if str(item).strip()]
    nodes_payload = snapshot.get("nodes", {})
    if not isinstance(nodes_payload, Mapping):
        nodes_payload = {}
    edges_payload = snapshot.get("edges", [])
    if not isinstance(edges_payload, list):
        edges_payload = []

    features["state_node_count"] = float(_safe_int(snapshot.get("node_count")))
    features["state_edge_count"] = float(_safe_int(snapshot.get("edge_count")))
    features["state_contradiction_count"] = float(_safe_int(snapshot.get("contradiction_count")))
    features["state_support_edge_count"] = float(_safe_int(snapshot.get("support_edge_count")))
    features["candidate_count"] = float(_safe_int(candidate_row.get("candidate_count")))
    features["candidate_target_count"] = float(len(target_ids))
    features["candidate_is_commit"] = 1.0 if bool(candidate_row.get("is_commit", False)) else 0.0

    total_confidence = 0.0
    total_evidence_items = 0
    for node_id, node_payload in nodes_payload.items():
        if not isinstance(node_payload, Mapping):
            continue
        node_type = node_payload.get("type", "unknown")
        node_role = node_payload.get("role", "unknown")
        _counted_feature(features, "state_node_type", node_type)
        _counted_feature(features, "state_node_role", node_role)
        total_confidence += _safe_float(node_payload.get("confidence"))
        evidence = node_payload.get("evidence", [])
        if isinstance(evidence, list):
            total_evidence_items += len(evidence)
        if str(node_id).strip() in target_ids:
            _counted_feature(features, "target_node_type", node_type)
            _counted_feature(features, "target_node_role", node_role)

    node_count = max(len(nodes_payload), 1)
    features["state_average_node_confidence"] = total_confidence / float(node_count)
    features["state_total_evidence_items"] = float(total_evidence_items)

    resolved_edge_count = 0
    for edge_payload in edges_payload:
        if not isinstance(edge_payload, Mapping):
            continue
        relation = edge_payload.get("relation", "unknown")
        _counted_feature(features, "state_edge_relation", relation)
        if bool(edge_payload.get("resolved", False)):
            resolved_edge_count += 1
    features["state_resolved_edge_fraction"] = resolved_edge_count / float(max(len(edges_payload), 1))
    features["state_contradiction_density"] = features["state_contradiction_count"] / float(node_count)
    features["state_support_density"] = features["state_support_edge_count"] / float(node_count)
    return features


def build_graph_feature_examples(
    *,
    candidate_dataset_dir: Path,
    g1_dataset_dir: Path,
    partition_manifest_path: Path,
) -> GraphFeatureDataset:
    candidate_rows = _load_jsonl(Path(candidate_dataset_dir) / "candidate_dataset.jsonl")
    snapshot_lookup = _build_snapshot_lookup(Path(g1_dataset_dir))
    partition_rows = load_partition_manifest_rows(Path(partition_manifest_path))
    partition_lookup = build_partition_role_lookup(partition_rows)

    examples: list[GraphFeatureExample] = []
    split_counts = {"critic_train": 0, "critic_dev": 0}
    train_groups: set[str] = set()
    dev_groups: set[str] = set()

    for row in candidate_rows:
        group_id = str(row.get("group_id", "")).strip()
        partition_role = partition_lookup.get(group_id)
        if partition_role is None:
            raise ValueError(f"Candidate row references unmapped group_id '{group_id}'.")
        if partition_role == "paper_eval":
            continue

        state_id = str(row.get("state_id", "")).strip()
        snapshot_path = snapshot_lookup.get(state_id)
        if snapshot_path is None:
            raise ValueError(f"Missing snapshot for state_id '{state_id}'.")
        snapshot = _load_snapshot(snapshot_path)
        feature_dict = _extract_snapshot_feature_dict(row, snapshot)
        examples.append(
            GraphFeatureExample(
                state_id=state_id,
                candidate_id=str(row.get("candidate_id", "")).strip(),
                group_id=group_id,
                split=partition_role,
                label=1 if bool(row.get("is_logged_selected", False)) else 0,
                is_commit=bool(row.get("is_commit", False)),
                is_commit_positive_state=bool(row.get("is_commit_positive_state", False)),
                feature_dict=feature_dict,
            )
        )
        split_counts[partition_role] = split_counts.get(partition_role, 0) + 1
        if partition_role == "critic_train":
            train_groups.add(group_id)
        elif partition_role == "critic_dev":
            dev_groups.add(group_id)

    return GraphFeatureDataset(
        examples=examples,
        split_counts=split_counts,
        group_overlap_count=len(train_groups.intersection(dev_groups)),
    )


def evaluate_graph_feature_rankings(
    model: GraphFeatureCriticModel,
    validation_examples: Sequence[GraphFeatureExample],
) -> dict[str, float | int]:
    by_state: dict[str, list[GraphFeatureExample]] = {}
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
        scores = model.score_feature_dicts([example.feature_dict for example in state_examples])
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
    return {
        "state_count": scored_state_count,
        "top1_accuracy": (top1_hits / scored_state_count) if scored_state_count else 0.0,
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        ),
    }


def train_graph_feature_critic(
    *,
    dataset: GraphFeatureDataset,
    output_dir: Path,
    commit_positive_weight: float = 2.0,
) -> GraphFeatureCriticTrainResult:
    train_examples = [example for example in dataset.examples if example.split == "critic_train"]
    dev_examples = [example for example in dataset.examples if example.split == "critic_dev"]
    if not train_examples:
        raise ValueError("critic_train is empty; refusing to train graph-feature critic.")
    if not dev_examples:
        raise ValueError("critic_dev is empty; refusing to evaluate graph-feature critic.")
    if dataset.group_overlap_count != 0:
        raise ValueError(f"group overlap detected: {dataset.group_overlap_count}")

    labels = [example.label for example in train_examples]
    if len(set(labels)) < 2:
        raise ValueError("critic_train examples must contain both positive and negative labels.")

    vectorizer = DictVectorizer(sparse=True)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    feature_matrix = vectorizer.fit_transform([example.feature_dict for example in train_examples])
    sample_weights = [
        float(commit_positive_weight if example.is_commit and example.is_commit_positive_state else 1.0)
        for example in train_examples
    ]
    classifier.fit(feature_matrix, labels, sample_weight=sample_weights)

    model = GraphFeatureCriticModel(vectorizer, classifier)
    metrics = evaluate_graph_feature_rankings(model, dev_examples)
    metrics["train_example_count"] = len(train_examples)
    metrics["validation_example_count"] = len(dev_examples)
    metrics["train_commit_positive_count"] = sum(
        1 for example in train_examples if example.is_commit and example.is_commit_positive_state
    )
    metrics["validation_commit_positive_count"] = sum(
        1 for example in dev_examples if example.is_commit and example.is_commit_positive_state
    )

    metadata = {
        "feature_family": "lightweight structured graph-and-action features with DictVectorizer + logistic regression",
        "split_counts": dict(dataset.split_counts),
        "group_overlap_count": dataset.group_overlap_count,
        "commit_positive_weight": float(commit_positive_weight),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "model.pkl"
    metadata_path = output_path / "metadata.json"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    write_text_file(output_path / "metrics.json", json.dumps(metrics, indent=2))
    write_text_file(metadata_path, json.dumps(metadata, indent=2))

    return GraphFeatureCriticTrainResult(
        metrics=metrics,
        model_path=model_path,
        metadata_path=metadata_path,
    )
