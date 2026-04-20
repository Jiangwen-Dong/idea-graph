from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _normalize_row(row: dict[str, Any], *, line_index: int) -> dict[str, Any]:
    normalized = dict(row)
    for key in ("group_id", "benchmark", "instance_name"):
        if not str(normalized.get(key, "")).strip():
            raise ValueError(f"Split registry line {line_index} is missing `{key}`.")
    normalized["group_id"] = str(normalized["group_id"]).strip()
    normalized["benchmark"] = str(normalized["benchmark"]).strip()
    normalized["instance_name"] = str(normalized["instance_name"]).strip()
    normalized["partition_role"] = str(normalized.get("partition_role", "")).strip()
    normalized["source_split"] = str(normalized.get("source_split", "")).strip()
    return normalized


def load_split_registry_rows(
    path: str | Path,
    *,
    partition_role: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_role = str(partition_role or "").strip()
    for line_index, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"Split registry line {line_index} must be a JSON object.")
        row = _normalize_row(dict(payload), line_index=line_index)
        if expected_role and row["partition_role"] != expected_role:
            continue
        rows.append(row)
    return rows


def _sample_benchmark_rows(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    target_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if target_count < 0:
        raise ValueError("target counts must be non-negative.")
    available = sorted(
        (dict(row) for row in rows if str(row.get("benchmark", "")).strip() == benchmark),
        key=lambda row: str(row.get("group_id", "")),
    )
    if len(available) < target_count:
        raise ValueError(
            f"Requested {target_count} {benchmark} rows, but only {len(available)} are available."
        )
    generator = random.Random(seed)
    generator.shuffle(available)
    return sorted(available[:target_count], key=lambda row: str(row.get("group_id", "")))


def select_balanced_split_rows(
    rows: list[dict[str, Any]],
    *,
    target_aiib: int,
    target_live: int,
    seed: int,
) -> list[dict[str, Any]]:
    aiib_rows = _sample_benchmark_rows(
        rows,
        benchmark="AI_Idea_Bench_2025",
        target_count=target_aiib,
        seed=seed,
    )
    live_rows = _sample_benchmark_rows(
        rows,
        benchmark="liveideabench",
        target_count=target_live,
        seed=seed + 1,
    )
    return aiib_rows + live_rows


def shard_split_rows(
    rows: list[dict[str, Any]],
    *,
    shard_count: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count.")
    return [dict(row) for index, row in enumerate(rows) if index % shard_count == shard_index]
