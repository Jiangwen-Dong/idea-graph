from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence


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
    "fit_joint_controller_calibration",
    "load_joint_controller_calibration",
    "write_joint_controller_calibration",
]
