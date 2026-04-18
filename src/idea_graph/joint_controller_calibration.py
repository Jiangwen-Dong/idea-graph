from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


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


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise JointControllerCalibrationError(f"{path} must contain a JSON object.")
    return dict(payload)


def _load_jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
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
    commit_pairs = [
        (
            _as_float(example.get("commit_probability"), key="commit_probability"),
            _label(example.get("label")),
        )
        for example in commit_examples
    ]
    positive_commit_rounds = [
        _as_int(example.get("round_index"), key="round_index")
        for example in commit_examples
        if _label(example.get("label")) == 1
    ]
    if not positive_commit_rounds:
        raise JointControllerCalibrationError(
            "Commit calibration requires both positive and negative labels."
        )

    return JointControllerCalibration(
        tau_override=round(_best_threshold(edit_pairs, positive_tie_break="high"), 4),
        tau_commit=0.08,
        gamma_commit=round(_best_threshold(commit_pairs, positive_tie_break="high"), 4),
        min_commit_round=max(1, min(positive_commit_rounds)),
        guard_support_threshold=0.66,
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
            for row in raw_commit_rows:
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
                commit_examples.append(
                    {
                        "group_id": group_id,
                        "instance_name": instance_name,
                        "round_name": round_name,
                        "round_index": _round_index(round_name),
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
                )

    return edit_examples, commit_examples


def load_joint_controller_calibration(path: str | Path) -> JointControllerCalibration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
        source=str(payload.get("source", "")).strip() or "critic_dev",
        version=str(payload.get("version", "")).strip() or "joint_controller_calibration_v1",
    )


def write_joint_controller_calibration(
    calibration: JointControllerCalibration,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(calibration.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def apply_joint_controller_calibration(
    metadata: Mapping[str, object],
    calibration: JointControllerCalibration,
) -> dict[str, object]:
    updated = dict(metadata)
    updated["runtime_controller_tau_override"] = float(calibration.tau_override)
    updated["runtime_controller_tau_commit"] = float(calibration.tau_commit)
    updated["runtime_controller_gamma_commit"] = float(calibration.gamma_commit)
    updated["runtime_controller_min_commit_round"] = int(calibration.min_commit_round)
    updated["runtime_controller_guard_support_threshold"] = float(calibration.guard_support_threshold)
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
