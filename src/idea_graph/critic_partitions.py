from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fs_utils import read_text_file, write_text_file


_ROLE_BY_SOURCE_SPLIT = {
    "train": "critic_train",
    "validation": "critic_dev",
}
_KNOWN_ROLES = ("critic_train", "critic_dev", "paper_eval")


def _normalize_str(value: object, *, default: str) -> str:
    normalized = str(value if value is not None else "").strip()
    return normalized or default


def _normalize_holdout_groups(holdout_groups: Sequence[object] | set[object] | None) -> set[str]:
    if holdout_groups is None:
        return set()
    normalized: set[str] = set()
    for raw in holdout_groups:
        candidate = _normalize_str(raw, default="")
        if candidate:
            normalized.add(candidate)
    return normalized


def build_partition_manifest(
    split_rows: Sequence[Mapping[str, Any]],
    *,
    holdout_groups: Sequence[object] | set[object] | None = None,
) -> list[dict[str, object]]:
    holdout_group_ids = _normalize_holdout_groups(holdout_groups)
    rows_by_group: dict[str, dict[str, object]] = {}

    for row in split_rows:
        group_id = _normalize_str(row.get("group_id"), default="")
        if not group_id:
            raise ValueError("split row is missing required group_id.")
        if group_id in rows_by_group:
            raise ValueError(f"Duplicate group_id in split rows: {group_id}")

        benchmark = _normalize_str(row.get("benchmark"), default="unknown")
        instance_name = _normalize_str(row.get("instance_name"), default="unknown")
        source_split = _normalize_str(row.get("split"), default="train")
        if source_split not in _ROLE_BY_SOURCE_SPLIT:
            raise ValueError(
                f"Unsupported split {source_split!r} for group_id '{group_id}'. "
                "Expected one of: train, validation."
            )

        partition_role = (
            "paper_eval"
            if group_id in holdout_group_ids
            else _ROLE_BY_SOURCE_SPLIT[source_split]
        )
        rows_by_group[group_id] = {
            "group_id": group_id,
            "benchmark": benchmark,
            "instance_name": instance_name,
            "source_split": source_split,
            "partition_role": partition_role,
        }

    unknown_holdouts = sorted(group_id for group_id in holdout_group_ids if group_id not in rows_by_group)
    if unknown_holdouts:
        raise ValueError(
            "Holdout groups not present in split rows: " + ", ".join(unknown_holdouts)
        )

    return sorted(
        rows_by_group.values(),
        key=lambda row: (
            str(row["benchmark"]),
            str(row["instance_name"]),
            str(row["group_id"]),
        ),
    )


def build_partition_stats(partition_rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    role_counts = {role: 0 for role in _KNOWN_ROLES}
    benchmark_role_counts: dict[str, dict[str, int]] = {}

    for row in partition_rows:
        benchmark = _normalize_str(row.get("benchmark"), default="unknown")
        role = _normalize_str(row.get("partition_role"), default="")
        if role not in role_counts:
            role_counts[role] = 0
        role_counts[role] += 1

        benchmark_counts = benchmark_role_counts.setdefault(
            benchmark,
            {known_role: 0 for known_role in _KNOWN_ROLES},
        )
        if role not in benchmark_counts:
            benchmark_counts[role] = 0
        benchmark_counts[role] += 1

    ordered_benchmark_role_counts = {
        benchmark: benchmark_role_counts[benchmark]
        for benchmark in sorted(benchmark_role_counts)
    }

    return {
        "group_count": len(partition_rows),
        "role_counts": role_counts,
        "benchmark_role_counts": ordered_benchmark_role_counts,
        "has_paper_eval": role_counts.get("paper_eval", 0) > 0,
    }


def load_split_manifest_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    path = Path(dataset_dir) / "split_manifest.jsonl"
    text = read_text_file(path)
    rows: list[dict[str, Any]] = []
    for line_index, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} line {line_index} must contain a JSON object.")
        rows.append(dict(payload))
    return rows


def write_partition_outputs(
    dataset_dir: Path,
    partition_rows: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
) -> None:
    dataset_path = Path(dataset_dir)
    manifest_text = "".join(
        json.dumps(dict(row), ensure_ascii=False, default=str) + "\n"
        for row in partition_rows
    )
    write_text_file(dataset_path / "partition_manifest.jsonl", manifest_text)
    write_text_file(
        dataset_path / "partition_stats.json",
        json.dumps(dict(stats), indent=2, ensure_ascii=False, default=str),
    )
    write_text_file(dataset_path / "README.md", _partition_readme_text())


def _partition_readme_text() -> str:
    return "\n".join(
        [
            "# Critic Partition Manifest",
            "",
            "Files:",
            "- partition_manifest.jsonl",
            "- partition_stats.json",
            "- README.md",
            "",
            "Roles:",
            "- critic_train: groups mapped from source split train",
            "- critic_dev: groups mapped from source split validation",
            "- paper_eval: explicit holdout groups",
            "",
        ]
    )

