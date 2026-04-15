from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

AIIB_PATTERN = re.compile(r"ai-idea-bench-2025-(\d+)$")
LIVE_PATTERN = re.compile(r"liveideabench-(.+)-(\d+)$")


def build_broad_dev_gate(
    manifest_path: Path | str,
    output_root: Path | str,
    manifest_name: str = "broad_dev_gate_59.jsonl",
    stats_name: str = "packet_stats.json",
    readme_name: str = "README.md",
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    output_root = Path(output_root)
    normalized_rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    role_counts: Counter[str] = Counter()
    benchmark_counts: Counter[str] = Counter()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"source manifest not found: {manifest_path}")

    for row in _iter_manifest_rows(manifest_path):
        group_id = row["group_id"]
        if group_id in seen_groups:
            raise ValueError(f"duplicate group_id in manifest: {group_id}")
        seen_groups.add(group_id)
        normalized = dict(row)
        normalized.update(_normalized_selectors(row))
        normalized_rows.append(normalized)
        role_counts[normalized["partition_role"]] += 1
        benchmark_counts[normalized["benchmark"]] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_target = output_root / manifest_name
    with manifest_target.open("w", encoding="utf-8") as writer:
        for normalized in normalized_rows:
            json.dump(normalized, writer)
            writer.write("\n")

    stats = {
        "group_count": len(normalized_rows),
        "role_counts": dict(role_counts),
        "benchmark_counts": dict(benchmark_counts),
    }
    stats_target = output_root / stats_name
    stats_target.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    readme_target = output_root / readme_name
    readme_target.write_text(_build_readme_text(stats, manifest_path), encoding="utf-8")

    return stats


def _iter_manifest_rows(manifest_path: Path) -> Iterable[dict[str, Any]]:
    with manifest_path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalized_selectors(row: dict[str, Any]) -> dict[str, Any]:
    selectors: dict[str, Any] = {}
    instance_name = row["instance_name"]
    benchmark = row["benchmark"]
    if benchmark == "AI_Idea_Bench_2025":
        selectors["benchmark_index"] = _parse_aiib_index(instance_name)
    if benchmark.lower() == "liveideabench":
        keyword, row_index = _parse_live_selectors(instance_name)
        selectors["benchmark_keyword"] = keyword
        selectors["row_index"] = row_index
    return selectors


def _parse_aiib_index(instance_name: str) -> int:
    match = AIIB_PATTERN.search(instance_name)
    if not match:
        raise ValueError(f"cannot parse AI_Idea_Bench_2025 index from {instance_name}")
    return int(match.group(1))


def _parse_live_selectors(instance_name: str) -> tuple[str, int]:
    match = LIVE_PATTERN.search(instance_name)
    if not match:
        raise ValueError(f"cannot parse LiveIdeaBench selectors from {instance_name}")
    keyword = match.group(1).strip()
    return keyword, int(match.group(2))


def _build_readme_text(stats: dict[str, Any], manifest_source: Path) -> str:
    lines = [
        "# Broad Development Gate Manifest",
        "",
        f"Source manifest: {manifest_source}",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Total groups: {stats['group_count']}",
        "",
        "Role counts:",
    ]
    for role, count in stats["role_counts"].items():
        lines.append(f"- {role}: {count}")
    lines.append("")
    lines.append("Benchmark counts:")
    for benchmark, count in stats["benchmark_counts"].items():
        lines.append(f"- {benchmark}: {count}")
    lines.append("")
    lines.append("Normalized selectors include:")
    lines.append("- `benchmark_index` for AI_Idea_Bench_2025 rows")
    lines.append("- `benchmark_keyword` and `row_index` for liveideabench rows")
    lines.append("")
    lines.append("This manifest supports the broad development gate described in the 2026 plan.")
    return "\n".join(lines) + "\n"
