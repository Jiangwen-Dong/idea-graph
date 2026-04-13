from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fs_utils import read_text_file, write_text_file


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_10_to_01(value: object) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return parsed / 10.0


def _as_object_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): inner for key, inner in value.items()}


def package_labels_from_manifest_row(manifest_row: Mapping[str, Any]) -> dict[str, Any]:
    weak_available = bool(manifest_row.get("has_local_eval", False))
    native_available = bool(manifest_row.get("has_native_eval", False))
    weak_overall_10 = _safe_float(manifest_row.get("final_local_overall"))
    weak_alignment_10 = _safe_float(manifest_row.get("final_local_alignment"))
    if weak_alignment_10 is None:
        weak_alignment_10 = _safe_float(
            _as_object_dict(manifest_row.get("local_category_scores")).get("benchmark_alignment")
        )
    native_average_10 = _safe_float(manifest_row.get("final_native_average"))
    has_native_average = native_average_10 is not None

    weak_local = {
        "available": weak_available,
        "overall_10": weak_overall_10,
        "overall_01": _normalize_10_to_01(weak_overall_10),
        "benchmark_alignment_10": weak_alignment_10,
        "benchmark_alignment_01": _normalize_10_to_01(weak_alignment_10),
        "category_scores": _as_object_dict(manifest_row.get("local_category_scores")),
    }
    native = {
        "available": native_available,
        "benchmark": str(manifest_row.get("benchmark", "unknown")).strip() or "unknown",
        "average_10": native_average_10,
        "average_01": _normalize_10_to_01(native_average_10),
        "metrics": _as_object_dict(manifest_row.get("native_metric_map")),
    }
    label_availability = {
        "has_weak_local": weak_available,
        "has_native": native_available,
        "has_native_average": has_native_average,
    }
    targets = {
        "weak_value_01": _normalize_10_to_01(weak_overall_10),
        "native_value_01": _normalize_10_to_01(native_average_10),
    }
    return {
        "weak_local": weak_local,
        "native": native,
        "label_availability": label_availability,
        "targets": targets,
    }


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


def load_g1_dataset(dataset_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_path = Path(dataset_dir)
    manifest_rows = _load_jsonl(dataset_path / "run_manifest.jsonl")
    transition_rows = _load_jsonl(dataset_path / "trajectory_examples.jsonl")
    return manifest_rows, transition_rows


def load_split_override_rows(path: Path) -> list[dict[str, Any]]:
    return _load_jsonl(Path(path))


def make_group_id(row: Mapping[str, Any]) -> str:
    benchmark = str(row.get("benchmark", "unknown")).strip() or "unknown"
    instance_name = str(row.get("instance_name", "unknown")).strip() or "unknown"
    return f"{benchmark}::{instance_name}"


def build_group_manifest(
    manifest_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_group_runs: dict[str, list[Mapping[str, Any]]] = {}
    transition_count_by_group: dict[str, int] = {}

    for row in manifest_rows:
        group_id = make_group_id(row)
        by_group_runs.setdefault(group_id, []).append(row)

    for row in transition_rows:
        group_id = make_group_id(row)
        transition_count_by_group[group_id] = transition_count_by_group.get(group_id, 0) + 1

    output_rows: list[dict[str, Any]] = []
    for group_id, runs in by_group_runs.items():
        first = runs[0] if runs else {}
        weak_values = [
            normalized
            for normalized in (_normalize_10_to_01(row.get("final_local_overall")) for row in runs)
            if normalized is not None
        ]
        native_values = [
            normalized
            for normalized in (_normalize_10_to_01(row.get("final_native_average")) for row in runs)
            if normalized is not None
        ]
        output_rows.append(
            {
                "group_id": group_id,
                "benchmark": str(first.get("benchmark", "unknown")).strip() or "unknown",
                "instance_name": str(first.get("instance_name", "unknown")).strip() or "unknown",
                "run_count": len(runs),
                "transition_count": transition_count_by_group.get(group_id, 0),
                "has_any_weak_local": any(bool(row.get("has_local_eval", False)) for row in runs),
                "has_any_native": any(bool(row.get("has_native_eval", False)) for row in runs),
                "mean_weak_value_01": (sum(weak_values) / len(weak_values)) if weak_values else None,
                "mean_native_value_01": (sum(native_values) / len(native_values)) if native_values else None,
            }
        )

    return sorted(output_rows, key=lambda row: str(row["group_id"]))


def assign_group_splits(
    group_rows: Sequence[Mapping[str, Any]],
    validation_fraction: float = 0.2,
    split_override_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    split_overrides: dict[str, str] = {}
    if split_override_rows:
        for row in split_override_rows:
            group_id = str(row.get("group_id", "")).strip()
            if not group_id:
                raise ValueError("Split override row is missing required group_id.")
            split = str(row.get("split", "")).strip()
            if split not in {"train", "validation"}:
                raise ValueError(
                    f"Split override for group_id '{group_id}' has invalid split '{split}'. "
                    "Allowed values are 'train' and 'validation'."
                )
            if group_id in split_overrides:
                raise ValueError(f"Duplicate split override for group_id '{group_id}'.")
            split_overrides[group_id] = split

    known_group_ids = {
        str(row.get("group_id", "")).strip()
        for row in group_rows
        if str(row.get("group_id", "")).strip()
    }
    unknown_override_group_ids = sorted(group_id for group_id in split_overrides if group_id not in known_group_ids)
    if unknown_override_group_ids:
        listed = ", ".join(unknown_override_group_ids)
        raise ValueError(f"Split overrides contain unknown group_id(s): {listed}")

    by_benchmark: dict[str, list[Mapping[str, Any]]] = {}
    for row in group_rows:
        benchmark = str(row.get("benchmark", "unknown")).strip() or "unknown"
        by_benchmark.setdefault(benchmark, []).append(row)

    output_rows: list[dict[str, Any]] = []
    for _, benchmark_rows in by_benchmark.items():
        ordered = sorted(benchmark_rows, key=lambda row: str(row.get("group_id", "")))
        default_split_by_group: dict[str, str] = {}
        validation_count = 0
        if len(ordered) >= 3:
            validation_count = max(1, round(len(ordered) * validation_fraction))
        boundary = len(ordered) - validation_count
        for index, row in enumerate(ordered):
            group_id = str(row.get("group_id", "")).strip()
            default_split_by_group[group_id] = "validation" if index >= boundary else "train"

        for row in ordered:
            group_id = str(row.get("group_id", "")).strip()
            split = split_overrides.get(group_id, default_split_by_group.get(group_id, "train"))
            copied = dict(row)
            copied["split"] = split
            output_rows.append(copied)

    return sorted(output_rows, key=lambda row: str(row.get("group_id", "")))


def _manifest_run_lookup(
    manifest_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, int], dict[str, int]]:
    run_lookup: dict[str, Mapping[str, Any]] = {}
    group_run_count: dict[str, int] = {}
    run_index_by_run_dir: dict[str, int] = {}

    group_seen_runs: dict[str, list[str]] = {}
    for row in manifest_rows:
        run_dir = str(row.get("run_dir", "")).strip()
        if run_dir:
            run_lookup[run_dir] = row
        group_id = make_group_id(row)
        group_seen_runs.setdefault(group_id, [])
        if run_dir and run_dir not in group_seen_runs[group_id]:
            group_seen_runs[group_id].append(run_dir)

    for group_id, run_dirs in group_seen_runs.items():
        ordered_run_dirs = sorted(run_dirs)
        group_run_count[group_id] = len(ordered_run_dirs)
        for index, run_dir in enumerate(ordered_run_dirs):
            run_index_by_run_dir[run_dir] = index

    return run_lookup, group_run_count, run_index_by_run_dir


def build_critic_dataset_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    split_by_group = {}
    for row in split_rows:
        group_id = str(row.get("group_id", "")).strip()
        if not group_id:
            continue
        split_by_group[group_id] = str(row.get("split", "train")).strip() or "train"
    run_lookup, group_run_count, run_index_by_run_dir = _manifest_run_lookup(manifest_rows)

    packaged_rows: list[dict[str, Any]] = []
    for transition in transition_rows:
        transition_copy = dict(transition)
        run_dir = str(transition_copy.get("run_dir", "")).strip()
        if not run_dir:
            raise ValueError("Transition row is missing required run_dir.")
        manifest_row = run_lookup.get(run_dir)
        if manifest_row is None:
            raise ValueError(f"Transition row references run_dir '{run_dir}' missing from manifest rows.")

        transition_group_id = make_group_id(manifest_row)
        split = split_by_group.get(transition_group_id)
        if split is None:
            raise ValueError(f"Missing split assignment for group_id '{transition_group_id}'.")
        run_index = run_index_by_run_dir.get(run_dir)
        if run_index is None:
            raise ValueError(f"Missing group_run_index mapping for run_dir '{run_dir}'.")
        run_count = group_run_count.get(transition_group_id)
        if run_count is None:
            raise ValueError(f"Missing group_run_count mapping for group_id '{transition_group_id}'.")

        label_package = package_labels_from_manifest_row(manifest_row)

        transition_copy["group_id"] = transition_group_id
        transition_copy["split"] = split
        transition_copy["group_run_count"] = run_count
        transition_copy["group_run_index"] = run_index
        transition_copy["weak_local"] = label_package["weak_local"]
        transition_copy["native"] = label_package["native"]
        transition_copy["label_availability"] = label_package["label_availability"]
        transition_copy["targets"] = label_package["targets"]
        packaged_rows.append(transition_copy)

    return packaged_rows


def build_label_schema() -> dict[str, Any]:
    return {
        "weak_local": {
            "available": "bool",
            "overall_10": "float|null",
            "overall_01": "float|null",
            "benchmark_alignment_10": "float|null",
            "benchmark_alignment_01": "float|null",
            "category_scores": "object",
        },
        "native": {
            "available": "bool",
            "benchmark": "str",
            "average_10": "float|null",
            "average_01": "float|null",
            "metrics": "object",
        },
        "label_availability": {
            "has_weak_local": "bool",
            "has_native": "bool",
            "has_native_average": "bool",
        },
        "targets": {
            "weak_value_01": "float|null",
            "native_value_01": "float|null",
        },
    }


def build_dataset_stats(
    critic_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    run_ids = {
        str(row.get("run_dir", "")).strip()
        for row in critic_rows
        if str(row.get("run_dir", "")).strip()
    }
    group_ids = {
        str(row.get("group_id", "")).strip()
        for row in split_rows
        if str(row.get("group_id", "")).strip()
    }
    train_group_count = sum(1 for row in split_rows if str(row.get("split", "train")) == "train")
    validation_group_count = sum(1 for row in split_rows if str(row.get("split", "train")) == "validation")
    train_transition_count = sum(1 for row in critic_rows if str(row.get("split", "train")) == "train")
    validation_transition_count = sum(
        1 for row in critic_rows if str(row.get("split", "train")) == "validation"
    )

    benchmark_group_counts: dict[str, int] = {}
    for row in split_rows:
        benchmark = str(row.get("benchmark", "unknown")).strip() or "unknown"
        benchmark_group_counts[benchmark] = benchmark_group_counts.get(benchmark, 0) + 1

    benchmark_transition_counts: dict[str, int] = {}
    for row in critic_rows:
        benchmark = str(row.get("benchmark", "unknown")).strip() or "unknown"
        benchmark_transition_counts[benchmark] = benchmark_transition_counts.get(benchmark, 0) + 1

    transition_count = len(critic_rows)
    weak_local_count = sum(
        1 for row in critic_rows if bool(_as_object_dict(row.get("label_availability")).get("has_weak_local"))
    )
    native_count = sum(
        1 for row in critic_rows if bool(_as_object_dict(row.get("label_availability")).get("has_native"))
    )
    native_average_count = sum(
        1 for row in critic_rows if bool(_as_object_dict(row.get("label_availability")).get("has_native_average"))
    )

    run_counts = [_safe_float(row.get("run_count")) for row in split_rows]
    run_counts = [count for count in run_counts if count is not None]

    def _fraction(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    return {
        "run_count": len(run_ids),
        "transition_count": transition_count,
        "group_count": len(group_ids),
        "train_group_count": train_group_count,
        "validation_group_count": validation_group_count,
        "train_transition_count": train_transition_count,
        "validation_transition_count": validation_transition_count,
        "benchmark_group_counts": benchmark_group_counts,
        "benchmark_transition_counts": benchmark_transition_counts,
        "label_coverage": {
            "weak_local_fraction": _fraction(weak_local_count, transition_count),
            "native_fraction": _fraction(native_count, transition_count),
            "native_average_fraction": _fraction(native_average_count, transition_count),
        },
        "duplicate_burden": {
            "mean_runs_per_group": (sum(run_counts) / len(run_counts)) if run_counts else 0.0,
            "max_runs_per_group": max(run_counts) if run_counts else 0.0,
        },
    }


def _jsonl_lines(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n" for row in rows)


def _readme_text(dataset_name: str) -> str:
    return "\n".join(
        [
            f"# {dataset_name}",
            "",
            "Graph-critic dataset produced from a G1 trajectory export.",
            "",
            "Files:",
            "- critic_dataset.jsonl",
            "- split_manifest.jsonl",
            "- label_schema.json",
            "- dataset_stats.json",
            "",
        ]
    )


@dataclass(frozen=True)
class CriticDatasetBuildResult:
    dataset_dir: Path
    group_count: int
    transition_count: int


def build_graph_critic_dataset(
    g1_dataset_dir: Path,
    output_dir: Path,
    dataset_name: str,
    validation_fraction: float = 0.2,
    split_overrides_path: Path | None = None,
) -> CriticDatasetBuildResult:
    manifest_rows, transition_rows = load_g1_dataset(Path(g1_dataset_dir))
    group_rows = build_group_manifest(manifest_rows, transition_rows)
    split_override_rows = (
        load_split_override_rows(split_overrides_path) if split_overrides_path is not None else None
    )
    split_rows = assign_group_splits(
        group_rows,
        validation_fraction=validation_fraction,
        split_override_rows=split_override_rows,
    )
    critic_rows = build_critic_dataset_rows(manifest_rows, transition_rows, split_rows)
    label_schema = build_label_schema()
    dataset_stats = build_dataset_stats(critic_rows, split_rows)

    dataset_dir = Path(output_dir) / dataset_name
    write_text_file(dataset_dir / "critic_dataset.jsonl", _jsonl_lines(critic_rows))
    write_text_file(dataset_dir / "split_manifest.jsonl", _jsonl_lines(split_rows))
    write_text_file(
        dataset_dir / "label_schema.json",
        json.dumps(label_schema, indent=2, ensure_ascii=False, default=str),
    )
    write_text_file(
        dataset_dir / "dataset_stats.json",
        json.dumps(dataset_stats, indent=2, ensure_ascii=False, default=str),
    )
    write_text_file(dataset_dir / "README.md", _readme_text(dataset_name))

    return CriticDatasetBuildResult(
        dataset_dir=dataset_dir,
        group_count=len(split_rows),
        transition_count=len(critic_rows),
    )
