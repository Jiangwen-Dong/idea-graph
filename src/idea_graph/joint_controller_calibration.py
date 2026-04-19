from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .commit_label_repair import relabel_scored_commit_rows
from .fs_utils import read_text_file, write_text_file


class JointControllerCalibrationError(ValueError):
    """Raised when frozen-dev controller calibration cannot be trusted."""


@dataclass(frozen=True)
class JointControllerCalibration:
    tau_override: float
    tau_commit: float
    gamma_commit: float
    min_commit_round: int
    guard_support_threshold: float
    source: str
    tau_override_by_round: dict[int, float] = field(default_factory=dict)
    gamma_commit_by_round: dict[int, float] = field(default_factory=dict)
    guard_commit_support_threshold: float = 0.0
    guard_commit_utility_floor: float = 0.0
    version: str = "joint_controller_calibration_v1"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_float(value: object, *, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise JointControllerCalibrationError(f"Calibration example has invalid float for {key!r}.") from exc


def _as_int(value: object, *, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise JointControllerCalibrationError(f"Calibration example has invalid integer for {key!r}.") from exc


def _label(value: object) -> int:
    label = _as_int(value, key="label")
    if label not in {0, 1}:
        raise JointControllerCalibrationError("Calibration labels must be binary 0/1 values.")
    return label


def _round_float_map(value: object) -> dict[int, float]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[int, float] = {}
    for key, item in value.items():
        try:
            normalized[int(str(key).strip())] = float(item)
        except (TypeError, ValueError):
            continue
    return normalized


def _feedback_commit_label(example: Mapping[str, object]) -> int:
    explicit_feedback = example.get("feedback_label", example.get("outcome_feedback_label"))
    if explicit_feedback is not None:
        return _label(explicit_feedback)
    label = _label(example.get("label"))
    final_native_delta = example.get("final_native_delta")
    if final_native_delta is None:
        return label
    try:
        delta = float(final_native_delta)
    except (TypeError, ValueError):
        return label
    if label == 0:
        return 0
    return 1 if delta >= 0.0 else 0


def _as_object_dict(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _score_commit_example_local_overall(example: Mapping[str, object]) -> float:
    from .candidate_slate_dataset import graph_from_state_snapshot
    from .engine import select_final_subgraph, synthesize_proposal
    from .evaluation import evaluate_graph

    snapshot = _as_object_dict(example.get("state_snapshot"))
    if not snapshot:
        raise JointControllerCalibrationError("Commit example is missing an inline state_snapshot.")
    literature = [
        str(item).strip()
        for item in example.get("literature", [])
        if str(item).strip()
    ] if isinstance(example.get("literature"), list) else []
    graph = graph_from_state_snapshot(
        snapshot,
        topic=str(example.get("topic", "")).strip(),
        literature=literature,
        role="CommitController",
        round_name=str(example.get("round_name", "")).strip(),
    )
    graph.metadata["benchmark"] = str(example.get("benchmark", "")).strip()
    graph.metadata["instance_name"] = str(example.get("instance_name", "")).strip()
    graph.final_subgraph = select_final_subgraph(graph)
    graph.final_proposal = synthesize_proposal(graph, graph.final_subgraph)
    evaluation = evaluate_graph(graph)
    return float(evaluation.overall_score)


def _attach_snapshot_feedback_to_commit_examples(
    commit_examples: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = [dict(example) for example in commit_examples]
    scored_rows: list[dict[str, object]] = []
    scored_keys: list[tuple[str, int, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row.get("state_snapshot"), Mapping):
            continue
        row_key = (
            str(row.get("run_dir", row.get("group_id", ""))).strip(),
            int(row.get("post_round_state_index", index) or index),
            str(row.get("state_id", row.get("round_name", index))).strip(),
        )
        scored_row = dict(row)
        scored_row.setdefault(
            "commit_supervision",
            {
                "available": True,
                "label": _label(scored_row.get("label")),
                "source": str(scored_row.get("label_source", "logged_commit")).strip()
                or "logged_commit",
            },
        )
        try:
            scored_row["outcome_local_overall"] = _score_commit_example_local_overall(scored_row)
        except Exception as exc:
            rows[index]["outcome_feedback_available"] = False
            rows[index]["outcome_scoring_error"] = str(exc)
            continue
        scored_rows.append(scored_row)
        scored_keys.append(row_key)

    if not scored_rows:
        return rows

    repaired_rows, _audit = relabel_scored_commit_rows(scored_rows)
    repaired_by_key: dict[tuple[str, int, str], dict[str, object]] = {}
    for row_key, repaired_row in zip(scored_keys, repaired_rows, strict=True):
        repaired_by_key[row_key] = dict(repaired_row)

    for index, row in enumerate(rows):
        row_key = (
            str(row.get("run_dir", row.get("group_id", ""))).strip(),
            int(row.get("post_round_state_index", index) or index),
            str(row.get("state_id", row.get("round_name", index))).strip(),
        )
        repaired = repaired_by_key.get(row_key)
        if repaired is None:
            continue
        feedback = _as_object_dict(repaired.get("outcome_grounded_commit_label"))
        supervision = _as_object_dict(repaired.get("commit_supervision"))
        row["outcome_local_overall"] = repaired.get("outcome_local_overall", 0.0)
        row["outcome_feedback_available"] = bool(feedback.get("available", False))
        row["outcome_feedback_label"] = (
            _label(feedback.get("label")) if bool(feedback.get("available", False)) else 0
        )
        row["outcome_feedback_status"] = str(feedback.get("status", "")).strip()
        row["future_best_local_overall"] = _as_float(
            feedback.get("future_best_local_overall", row.get("outcome_local_overall", 0.0)),
            key="future_best_local_overall",
        )
        row["episode_best_local_overall"] = _as_float(
            feedback.get("episode_best_local_overall", row.get("outcome_local_overall", 0.0)),
            key="episode_best_local_overall",
        )
        row["future_improvement"] = _as_float(
            feedback.get("future_improvement", 0.0),
            key="future_improvement",
        )
        row["positive_signal_count"] = _as_int(
            feedback.get("positive_signal_count", 0),
            key="positive_signal_count",
        )
        row["outcome_feedback_logged_label"] = supervision.get("logged_label", row.get("label"))
    return rows


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(read_text_file(path))
    if not isinstance(payload, dict):
        raise JointControllerCalibrationError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_index, line in enumerate(read_text_file(path).splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise JointControllerCalibrationError(
                f"{path} line {line_index} must contain a JSON object."
            )
        rows.append(dict(payload))
    return rows


def _resolve_run_dir(base_dir: Path, repo_root: Path | None, run_dir_text: str) -> Path:
    run_dir = Path(run_dir_text)
    if run_dir.is_absolute():
        return run_dir
    manifest_relative = (base_dir / run_dir).resolve()
    if manifest_relative.exists():
        return manifest_relative
    if repo_root is not None:
        repo_relative = (repo_root / run_dir).resolve()
        if repo_relative.exists():
            return repo_relative
    return manifest_relative


def _native_average(summary_payload: Mapping[str, Any]) -> float | None:
    native_payload = summary_payload.get("benchmark_native_evaluation", {})
    if not isinstance(native_payload, Mapping):
        return None
    summary = native_payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return None
    value = summary.get("available_average_normalized_10")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_index(round_name: object) -> int:
    text = str(round_name or "").strip()
    if not text.startswith("Round"):
        return 0
    try:
        return int(text[5:])
    except ValueError:
        return 0


def _best_threshold(
    scored_examples: Sequence[tuple[float, int]],
    *,
    positive_tie_break: str,
) -> float:
    if not scored_examples:
        raise JointControllerCalibrationError("Calibration requires at least one scored example.")
    labels = {label for _, label in scored_examples}
    if labels != {0, 1}:
        raise JointControllerCalibrationError(
            "Calibration requires both positive and negative labels."
        )

    thresholds = sorted({round(score, 8) for score, _ in scored_examples})
    best_threshold = thresholds[0]
    best_metric: tuple[float, float, float] | None = None
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for score, label in scored_examples:
            prediction = 1 if score >= threshold else 0
            if prediction == 1 and label == 1:
                tp += 1
            elif prediction == 1 and label == 0:
                fp += 1
            elif prediction == 0 and label == 0:
                tn += 1
            else:
                fn += 1
        tpr = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        balanced_accuracy = 0.5 * (tpr + tnr)
        threshold_preference = threshold if positive_tie_break == "high" else -threshold
        metric = (balanced_accuracy, precision, threshold_preference)
        if best_metric is None or metric > best_metric:
            best_metric = metric
            best_threshold = threshold
    return float(best_threshold)


def fit_joint_controller_calibration(
    *,
    edit_examples: Sequence[Mapping[str, object]],
    commit_examples: Sequence[Mapping[str, object]],
    source: str = "critic_dev",
) -> JointControllerCalibration:
    edit_pairs = [
        (
            _as_float(example.get("override_margin"), key="override_margin"),
            _label(example.get("label")),
        )
        for example in edit_examples
    ]
    edit_pairs_by_round: dict[int, list[tuple[float, int]]] = {}
    for example, pair in zip(edit_examples, edit_pairs, strict=True):
        round_index = _as_int(example.get("round_index", 0), key="round_index")
        if round_index <= 0:
            continue
        edit_pairs_by_round.setdefault(round_index, []).append(pair)
    commit_pairs = [
        (
            _as_float(example.get("commit_probability"), key="commit_probability"),
            _feedback_commit_label(example),
        )
        for example in commit_examples
    ]
    positive_commit_rounds = [
        _as_int(example.get("round_index"), key="round_index")
        for example in commit_examples
        if _feedback_commit_label(example) == 1
    ]
    if not positive_commit_rounds:
        raise JointControllerCalibrationError(
            "Commit calibration requires both positive and negative labels."
        )

    has_snapshot_feedback = any(
        "outcome_feedback_label" in example or "outcome_feedback_status" in example
        for example in commit_examples
    )
    min_commit_round = max(1, min(positive_commit_rounds))
    if has_snapshot_feedback:
        min_commit_round = max(3, min_commit_round)

    tau_override_by_round: dict[int, float] = {}
    for round_index, round_pairs in sorted(edit_pairs_by_round.items()):
        labels = {label for _, label in round_pairs}
        if labels != {0, 1}:
            continue
        tau_override_by_round[int(round_index)] = round(
            _best_threshold(round_pairs, positive_tie_break="high"),
            4,
        )

    commit_pairs_by_round: dict[int, list[tuple[float, int]]] = {}
    for example, pair in zip(commit_examples, commit_pairs, strict=True):
        round_index = _as_int(example.get("round_index"), key="round_index")
        if round_index <= 0 or round_index < min_commit_round:
            continue
        commit_pairs_by_round.setdefault(round_index, []).append(pair)
    gamma_commit_by_round: dict[int, float] = {}
    for round_index, round_pairs in sorted(commit_pairs_by_round.items()):
        labels = {label for _, label in round_pairs}
        if labels != {0, 1}:
            continue
        gamma_commit_by_round[int(round_index)] = round(
            _best_threshold(round_pairs, positive_tie_break="high"),
            4,
        )

    return JointControllerCalibration(
        tau_override=round(_best_threshold(edit_pairs, positive_tie_break="high"), 4),
        tau_commit=0.08,
        gamma_commit=round(_best_threshold(commit_pairs, positive_tie_break="high"), 4),
        min_commit_round=min_commit_round,
        guard_support_threshold=0.66,
        tau_override_by_round=tau_override_by_round,
        gamma_commit_by_round=gamma_commit_by_round,
        guard_commit_support_threshold=0.0,
        guard_commit_utility_floor=0.0,
        source=str(source).strip() or "critic_dev",
    )


def build_joint_calibration_examples_from_packet(
    *,
    run_manifest_path: str | Path,
    heuristic_baseline: str,
    critic_baseline: str,
    repo_root: str | Path | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    manifest_rows = _load_jsonl_objects(run_manifest_path)
    manifest_dir = Path(run_manifest_path).resolve().parent
    resolved_repo_root = Path(repo_root).resolve() if repo_root is not None else None
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in manifest_rows:
        group_id = str(row.get("group_id", "")).strip()
        baseline_name = str(row.get("baseline_name", "")).strip()
        if not group_id or not baseline_name:
            continue
        grouped.setdefault(group_id, {})[baseline_name] = row

    edit_examples: list[dict[str, object]] = []
    commit_examples: list[dict[str, object]] = []
    for group_id in sorted(grouped):
        baselines = grouped[group_id]
        heuristic_row = baselines.get(str(heuristic_baseline).strip())
        critic_row = baselines.get(str(critic_baseline).strip())
        if heuristic_row is None or critic_row is None:
            continue

        heuristic_run_dir = _resolve_run_dir(
            manifest_dir,
            resolved_repo_root,
            str(heuristic_row.get("run_dir", "")).strip(),
        )
        critic_run_dir = _resolve_run_dir(
            manifest_dir,
            resolved_repo_root,
            str(critic_row.get("run_dir", "")).strip(),
        )
        heuristic_summary = _load_json_object(heuristic_run_dir / "summary.json")
        critic_summary = _load_json_object(critic_run_dir / "summary.json")
        critic_graph = _load_json_object(critic_run_dir / "graph.json")

        heuristic_native = _native_average(heuristic_summary)
        critic_native = _native_average(critic_summary)
        final_native_delta = (
            float(critic_native - heuristic_native)
            if critic_native is not None and heuristic_native is not None
            else None
        )
        critic_outperformed = bool(final_native_delta is not None and final_native_delta >= 0.0)
        instance_name = str(critic_row.get("instance_name", "")).strip()
        metadata = critic_graph.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}

        runtime_log = metadata.get("runtime_controller_log", [])
        if isinstance(runtime_log, list):
            for entry in runtime_log:
                if not isinstance(entry, Mapping):
                    continue
                nested_decision = entry.get("controller_decision", {})
                if not isinstance(nested_decision, Mapping):
                    nested_decision = {}
                selected_source = str(
                    entry.get("selected_source", nested_decision.get("selected_source", ""))
                ).strip()
                round_name = str(entry.get("round", entry.get("round_name", ""))).strip()
                heuristic_candidate = entry.get("heuristic_candidate", {})
                if not isinstance(heuristic_candidate, Mapping):
                    heuristic_candidate = {}
                selected_candidate = entry.get("selected_candidate", {})
                if not isinstance(selected_candidate, Mapping):
                    selected_candidate = {}
                heuristic_kind = str(
                    entry.get("heuristic_kind", heuristic_candidate.get("kind", ""))
                ).strip()
                selected_kind = str(
                    entry.get("selected_kind", selected_candidate.get("kind", ""))
                ).strip()
                label = 1 if selected_source == "critic" and critic_outperformed else 0
                edit_examples.append(
                    {
                        "group_id": group_id,
                        "instance_name": instance_name,
                        "round_name": round_name,
                        "round_index": _round_index(round_name),
                        "role": str(entry.get("role", "")).strip(),
                        "override_margin": _as_float(
                            entry.get("override_margin", nested_decision.get("override_margin")),
                            key="override_margin",
                        ),
                        "label": label,
                        "selected_source": selected_source or "unknown",
                        "heuristic_candidate_id": str(
                            entry.get(
                                "heuristic_candidate_id",
                                heuristic_candidate.get("candidate_id", ""),
                            )
                        ).strip(),
                        "selected_candidate_id": str(
                            entry.get(
                                "selected_candidate_id",
                                selected_candidate.get("candidate_id", ""),
                            )
                        ).strip(),
                        "heuristic_kind": heuristic_kind,
                        "selected_kind": selected_kind,
                        "heuristic_is_skip": heuristic_kind == "skip",
                        "selected_is_skip": selected_kind == "skip",
                        "heuristic_final_native_average": heuristic_native,
                        "critic_final_native_average": critic_native,
                        "final_native_delta": final_native_delta,
                    }
                )

        raw_commit_rows = metadata.get("post_round_commit_rows", [])
        if isinstance(raw_commit_rows, list):
            graph_topic = str(critic_graph.get("topic", "")).strip()
            graph_literature = [
                str(item).strip()
                for item in critic_graph.get("literature", [])
                if str(item).strip()
            ] if isinstance(critic_graph.get("literature"), list) else []
            for state_index, row in enumerate(raw_commit_rows):
                if not isinstance(row, Mapping):
                    continue
                commit_supervision = row.get("commit_supervision", {})
                if not isinstance(commit_supervision, Mapping):
                    commit_supervision = {}
                if not bool(commit_supervision.get("available", False)):
                    continue
                probability = row.get("commit_probability")
                if probability is None:
                    continue
                round_name = str(row.get("round_name", "")).strip()
                commit_example: dict[str, object] = {
                        "group_id": group_id,
                        "run_dir": str(critic_run_dir),
                        "instance_name": instance_name,
                        "round_name": round_name,
                        "round_index": _round_index(round_name),
                        "post_round_state_index": state_index,
                        "state_id": str(row.get("state_id", "")).strip()
                        or f"{group_id}::{round_name}::post_round_commit",
                        "benchmark": str(
                            row.get("benchmark", metadata.get("benchmark", ""))
                        ).strip(),
                        "topic": str(row.get("topic", graph_topic)).strip(),
                        "literature": graph_literature,
                        "commit_probability": _as_float(probability, key="commit_probability"),
                        "label": _label(commit_supervision.get("label")),
                        "label_source": str(
                            commit_supervision.get("source", row.get("label_source", ""))
                        ).strip(),
                        "support_coverage": _as_float(
                            row.get("support_coverage", 0.0),
                            key="support_coverage",
                        ),
                        "unresolved_contradiction_ratio": _as_float(
                            row.get("unresolved_contradiction_ratio", 0.0),
                            key="unresolved_contradiction_ratio",
                        ),
                        "utility": _as_float(row.get("utility", 0.0), key="utility"),
                        "heuristic_final_native_average": heuristic_native,
                        "critic_final_native_average": critic_native,
                        "final_native_delta": final_native_delta,
                    }
                state_snapshot = row.get("state_snapshot")
                if isinstance(state_snapshot, Mapping):
                    commit_example["state_snapshot"] = dict(state_snapshot)
                commit_examples.append(commit_example)

    return edit_examples, _attach_snapshot_feedback_to_commit_examples(commit_examples)


def load_joint_controller_calibration(path: str | Path) -> JointControllerCalibration:
    payload = json.loads(read_text_file(path))
    if not isinstance(payload, dict):
        raise JointControllerCalibrationError("Calibration artifact must be a JSON object.")
    return JointControllerCalibration(
        tau_override=_as_float(payload.get("tau_override"), key="tau_override"),
        tau_commit=_as_float(payload.get("tau_commit"), key="tau_commit"),
        gamma_commit=_as_float(payload.get("gamma_commit"), key="gamma_commit"),
        min_commit_round=_as_int(payload.get("min_commit_round"), key="min_commit_round"),
        guard_support_threshold=_as_float(
            payload.get("guard_support_threshold"),
            key="guard_support_threshold",
        ),
        tau_override_by_round=_round_float_map(payload.get("tau_override_by_round")),
        gamma_commit_by_round=_round_float_map(payload.get("gamma_commit_by_round")),
        guard_commit_support_threshold=_as_float(
            payload.get("guard_commit_support_threshold", 0.0),
            key="guard_commit_support_threshold",
        ),
        guard_commit_utility_floor=_as_float(
            payload.get("guard_commit_utility_floor", 0.0),
            key="guard_commit_utility_floor",
        ),
        source=str(payload.get("source", "")).strip() or "critic_dev",
        version=str(payload.get("version", "")).strip() or "joint_controller_calibration_v1",
    )


def write_joint_controller_calibration(
    calibration: JointControllerCalibration,
    path: str | Path,
) -> None:
    output_path = Path(path)
    write_text_file(
        output_path,
        json.dumps(calibration.as_dict(), indent=2, ensure_ascii=False),
    )


def apply_joint_controller_calibration(
    metadata: Mapping[str, object],
    calibration: JointControllerCalibration,
) -> dict[str, object]:
    updated = dict(metadata)
    updated["runtime_controller_tau_override"] = float(calibration.tau_override)
    updated["runtime_controller_tau_override_by_round"] = {
        str(key): float(value) for key, value in calibration.tau_override_by_round.items()
    }
    updated["runtime_controller_tau_commit"] = float(calibration.tau_commit)
    updated["runtime_controller_gamma_commit"] = float(calibration.gamma_commit)
    updated["runtime_controller_gamma_commit_by_round"] = {
        str(key): float(value) for key, value in calibration.gamma_commit_by_round.items()
    }
    updated["runtime_controller_min_commit_round"] = int(calibration.min_commit_round)
    updated["runtime_controller_guard_support_threshold"] = float(calibration.guard_support_threshold)
    updated["runtime_controller_guard_commit_support_threshold"] = float(
        calibration.guard_commit_support_threshold
    )
    updated["runtime_controller_guard_commit_utility_floor"] = float(
        calibration.guard_commit_utility_floor
    )
    updated["runtime_controller_calibration_source"] = calibration.source
    updated["runtime_controller_calibration_version"] = calibration.version
    return updated


__all__ = [
    "JointControllerCalibration",
    "JointControllerCalibrationError",
    "apply_joint_controller_calibration",
    "build_joint_calibration_examples_from_packet",
    "fit_joint_controller_calibration",
    "load_joint_controller_calibration",
    "write_joint_controller_calibration",
]
