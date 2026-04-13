from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fs_utils import read_text_file, write_text_file

_ROLE_ALLOWED_USAGES = {
    "critic_train": [
        "train_offline_critic",
        "train_online_critic",
        "development_analysis",
    ],
    "critic_dev": [
        "select_checkpoint",
        "calibrate_threshold",
        "development_analysis",
    ],
    "paper_eval": [
        "paper_final_eval",
    ],
}


def _normalize_str(value: object, *, default: str = "") -> str:
    normalized = str(value if value is not None else "").strip()
    return normalized or default


def load_partition_manifest_rows(path: Path) -> list[dict[str, Any]]:
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


def build_split_registry(
    partition_rows: Sequence[Mapping[str, Any]],
    *,
    pool_name: str,
) -> list[dict[str, object]]:
    normalized_pool_name = _normalize_str(pool_name)
    if not normalized_pool_name:
        raise ValueError("pool_name must not be empty.")

    rows_by_group: dict[str, dict[str, object]] = {}
    for row in partition_rows:
        group_id = _normalize_str(row.get("group_id"))
        if not group_id:
            raise ValueError("partition row is missing required group_id.")
        if group_id in rows_by_group:
            raise ValueError(f"Duplicate group_id in partition rows: {group_id}")
        partition_role = _normalize_str(row.get("partition_role"))
        if partition_role not in _ROLE_ALLOWED_USAGES:
            raise ValueError(f"Unsupported partition_role {partition_role!r} for group_id '{group_id}'.")
        benchmark = _normalize_str(row.get("benchmark"), default="unknown")
        instance_name = _normalize_str(row.get("instance_name"), default="unknown")
        source_split = _normalize_str(row.get("source_split"), default="unknown")
        allowed_usages = list(_ROLE_ALLOWED_USAGES[partition_role])
        note = (
            "Development-only benchmark instance."
            if partition_role != "paper_eval"
            else "Frozen final benchmark evaluation only."
        )
        rows_by_group[group_id] = {
            "group_id": group_id,
            "benchmark": benchmark,
            "instance_name": instance_name,
            "pool_name": normalized_pool_name,
            "partition_role": partition_role,
            "source_split": source_split,
            "allowed_usages": allowed_usages,
            "notes": note,
        }
    return sorted(
        rows_by_group.values(),
        key=lambda row: (
            str(row["benchmark"]),
            str(row["instance_name"]),
            str(row["group_id"]),
        ),
    )


def build_split_registry_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    pool_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    benchmark_role_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        pool_name = _normalize_str(row.get("pool_name"), default="unknown")
        partition_role = _normalize_str(row.get("partition_role"), default="unknown")
        benchmark = _normalize_str(row.get("benchmark"), default="unknown")
        pool_counts[pool_name] = pool_counts.get(pool_name, 0) + 1
        role_counts[partition_role] = role_counts.get(partition_role, 0) + 1
        benchmark_counts = benchmark_role_counts.setdefault(benchmark, {})
        benchmark_counts[partition_role] = benchmark_counts.get(partition_role, 0) + 1
    return {
        "row_count": len(rows),
        "pool_counts": dict(sorted(pool_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "benchmark_role_counts": {
            benchmark: dict(sorted(counts.items()))
            for benchmark, counts in sorted(benchmark_role_counts.items())
        },
    }


def write_split_registry_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
) -> None:
    output_path = Path(output_dir)
    registry_text = "".join(
        json.dumps(dict(row), ensure_ascii=False, default=str) + "\n" for row in rows
    )
    write_text_file(output_path / "split_registry.jsonl", registry_text)
    write_text_file(
        output_path / "split_registry_stats.json",
        json.dumps(dict(stats), indent=2, ensure_ascii=False, default=str),
    )
