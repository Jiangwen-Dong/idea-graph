from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fs_utils import read_text_file
from .text_critic import (
    CandidateExample,
    build_split_audit,
    evaluate_state_rankings,
    train_text_critic,
)

PARTITION_ROLES = ("critic_train", "critic_dev", "paper_eval")
NAMESPACE_NAMES = ("teacher_logged", "terminal_commit")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_index, raw_line in enumerate(read_text_file(path).splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} line {line_index} must contain a JSON object.")
        rows.append(dict(payload))
    return rows


def load_candidate_rows(candidate_dataset_dir: Path) -> list[dict[str, Any]]:
    return _load_jsonl(Path(candidate_dataset_dir) / "candidate_dataset.jsonl")


def load_partition_manifest_rows(partition_manifest_path: Path) -> list[dict[str, Any]]:
    return _load_jsonl(Path(partition_manifest_path))


def build_partition_role_lookup(
    partition_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in partition_rows:
        group_id = str(row.get("group_id", "")).strip()
        if not group_id:
            raise ValueError("Partition row is missing required group_id.")
        partition_role = str(row.get("partition_role", "")).strip()
        if partition_role not in PARTITION_ROLES:
            raise ValueError(
                f"Unsupported partition_role {partition_role!r} for group_id '{group_id}'."
            )
        if group_id in lookup:
            raise ValueError(f"Duplicate partition assignment for group_id '{group_id}'.")
        lookup[group_id] = partition_role
    return lookup


def build_namespace_support(
    candidate_rows: Sequence[Mapping[str, Any]],
    partition_lookup: Mapping[str, str],
) -> dict[str, dict[str, dict[str, int]]]:
    support = {
        partition_role: {
            namespace: {"row_count": 0, "positive_count": 0}
            for namespace in NAMESPACE_NAMES
        }
        for partition_role in PARTITION_ROLES
    }
    for row in candidate_rows:
        group_id = str(row.get("group_id", "")).strip()
        partition_role = partition_lookup.get(group_id)
        if partition_role is None:
            raise ValueError(f"Candidate row references unmapped group_id '{group_id}'.")

        support[partition_role]["teacher_logged"]["row_count"] += 1
        if bool(row.get("is_logged_selected", False)):
            support[partition_role]["teacher_logged"]["positive_count"] += 1

        if bool(row.get("is_commit", False)):
            support[partition_role]["terminal_commit"]["row_count"] += 1
            if bool(row.get("is_commit_positive_state", False)):
                support[partition_role]["terminal_commit"]["positive_count"] += 1
    return support


def validate_required_namespace_support(
    namespace_support: Mapping[str, Mapping[str, Mapping[str, int]]],
    *,
    partition_role: str,
    required_namespaces: Sequence[str],
) -> None:
    if partition_role not in PARTITION_ROLES:
        raise ValueError(f"Unsupported partition_role {partition_role!r}.")
    for namespace in required_namespaces:
        if namespace not in NAMESPACE_NAMES:
            raise ValueError(f"Unsupported namespace {namespace!r}.")
        positive_count = int(
            namespace_support.get(partition_role, {}).get(namespace, {}).get("positive_count", 0)
        )
        if positive_count <= 0:
            raise ValueError(
                f"Required namespace '{namespace}' has zero positive support in {partition_role}."
            )


def _candidate_example_from_row(
    row: Mapping[str, Any],
    *,
    partition_role: str,
) -> CandidateExample:
    targets = row.get("targets", {})
    if not isinstance(targets, Mapping):
        targets = {}
    weak_value = targets.get("weak_value_01")
    native_value = targets.get("native_value_01")
    return CandidateExample(
        state_id=str(row.get("state_id", "")).strip(),
        candidate_id=str(row.get("candidate_id", "")).strip(),
        split=partition_role,
        label=1 if bool(row.get("is_logged_selected", False)) else 0,
        state_text=str(row.get("state_text", "")),
        candidate_text=str(row.get("candidate_text", "")),
        group_id=str(row.get("group_id", "")).strip(),
        weak_value_01=float(weak_value) if weak_value is not None else None,
        native_value_01=float(native_value) if native_value is not None else None,
    )


def build_partition_examples(
    candidate_rows: Sequence[Mapping[str, Any]],
    partition_lookup: Mapping[str, str],
    *,
    partition_role: str,
    commit_positive_weight: float,
) -> tuple[list[CandidateExample], list[float]]:
    examples: list[CandidateExample] = []
    sample_weights: list[float] = []
    for row in candidate_rows:
        group_id = str(row.get("group_id", "")).strip()
        row_partition_role = partition_lookup.get(group_id)
        if row_partition_role is None:
            raise ValueError(f"Candidate row references unmapped group_id '{group_id}'.")
        if row_partition_role != partition_role:
            continue
        if row_partition_role == "paper_eval":
            continue
        examples.append(_candidate_example_from_row(row, partition_role=row_partition_role))
        is_commit_positive = bool(row.get("is_commit", False)) and bool(
            row.get("is_commit_positive_state", False)
        )
        sample_weights.append(float(commit_positive_weight if is_commit_positive else 1.0))
    return examples, sample_weights


@dataclass(frozen=True)
class WarmstartTrainingBundle:
    train_examples: list[CandidateExample]
    train_sample_weights: list[float]
    dev_examples: list[CandidateExample]
    partition_lookup: dict[str, str]
    namespace_support: dict[str, dict[str, dict[str, int]]]
    split_audit: dict[str, int]


def build_warmstart_training_bundle(
    candidate_rows: Sequence[Mapping[str, Any]],
    partition_rows: Sequence[Mapping[str, Any]],
    *,
    required_namespaces: Sequence[str] = NAMESPACE_NAMES,
    commit_positive_weight: float = 2.0,
) -> WarmstartTrainingBundle:
    partition_lookup = build_partition_role_lookup(partition_rows)
    namespace_support = build_namespace_support(candidate_rows, partition_lookup)
    validate_required_namespace_support(
        namespace_support,
        partition_role="critic_train",
        required_namespaces=required_namespaces,
    )
    validate_required_namespace_support(
        namespace_support,
        partition_role="critic_dev",
        required_namespaces=required_namespaces,
    )

    train_examples, train_sample_weights = build_partition_examples(
        candidate_rows,
        partition_lookup,
        partition_role="critic_train",
        commit_positive_weight=commit_positive_weight,
    )
    dev_examples, _ = build_partition_examples(
        candidate_rows,
        partition_lookup,
        partition_role="critic_dev",
        commit_positive_weight=commit_positive_weight,
    )
    if not dev_examples:
        raise ValueError("critic_dev is empty; refusing to warm-start without a dev split.")

    split_audit = build_split_audit(train_examples, dev_examples)
    if split_audit["group_overlap_count"] != 0:
        raise ValueError(
            f"Train/dev group overlap detected: {split_audit['group_overlap_count']} overlapping groups."
        )
    return WarmstartTrainingBundle(
        train_examples=train_examples,
        train_sample_weights=train_sample_weights,
        dev_examples=dev_examples,
        partition_lookup=dict(partition_lookup),
        namespace_support=namespace_support,
        split_audit=split_audit,
    )


def evaluate_warmstart_text_critic(
    model: Any,
    dev_examples: Sequence[CandidateExample],
) -> dict[str, float | int]:
    return evaluate_state_rankings(model, dev_examples)


def train_warmstart_text_critic(
    candidate_rows: Sequence[Mapping[str, Any]],
    partition_rows: Sequence[Mapping[str, Any]],
    *,
    required_namespaces: Sequence[str] = NAMESPACE_NAMES,
    commit_positive_weight: float = 2.0,
) -> tuple[Any, WarmstartTrainingBundle, dict[str, float | int]]:
    bundle = build_warmstart_training_bundle(
        candidate_rows,
        partition_rows,
        required_namespaces=required_namespaces,
        commit_positive_weight=commit_positive_weight,
    )
    model = train_text_critic(
        bundle.train_examples,
        sample_weights=bundle.train_sample_weights,
    )
    metrics = evaluate_warmstart_text_critic(model, bundle.dev_examples)
    metrics["train_example_count"] = len(bundle.train_examples)
    metrics["validation_example_count"] = len(bundle.dev_examples)
    return model, bundle, metrics
