from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.critic_partitions import build_partition_stats, write_partition_outputs
from idea_graph.critic_pool_expansion import build_expansion_partition_rows, make_group_id
from idea_graph.critic_split_registry import (
    build_split_registry,
    build_split_registry_stats,
    write_split_registry_outputs,
)
from idea_graph.fs_utils import read_text_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build overlap-safe critic expansion pool partition and split-registry artifacts."
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        required=True,
        help="Path to candidate_instances.json.",
    )
    parser.add_argument(
        "--blocked-split-registry",
        action="append",
        default=[],
        type=Path,
        help="Path to a blocked split_registry.jsonl (repeatable).",
    )
    parser.add_argument(
        "--blocked-candidate-file",
        action="append",
        default=[],
        type=Path,
        help="Path to blocked candidate_instances.json (repeatable).",
    )
    parser.add_argument(
        "--pool-name",
        type=str,
        required=True,
        help="Pool name for split-registry rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for partition and split-registry artifacts.",
    )
    return parser


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(read_text_file(path))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array.")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"{path} entry {index} must be a JSON object.")
        rows.append(dict(row))
    return rows


def _group_ids_from_row(row: dict[str, Any], *, source_path: Path, row_index: int) -> set[str]:
    group_ids: set[str] = set()
    raw_group_id = str(row.get("group_id", "")).strip()
    if raw_group_id:
        group_ids.add(raw_group_id)

    benchmark = str(row.get("benchmark", "")).strip()
    instance_name = str(row.get("instance_name", "")).strip()
    if benchmark and instance_name:
        try:
            group_ids.add(make_group_id(benchmark, instance_name))
        except ValueError:
            if not raw_group_id:
                raise

    if not group_ids:
        raise ValueError(
            f"{source_path} row {row_index} must include group_id or benchmark+instance_name."
        )
    return group_ids


def _load_blocked_group_ids(
    *,
    blocked_split_registries: list[Path],
    blocked_candidate_files: list[Path],
) -> set[str]:
    blocked_group_ids: set[str] = set()

    for registry_path in blocked_split_registries:
        for row_index, raw_line in enumerate(read_text_file(registry_path).splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{registry_path} line {row_index} must contain a JSON object.")
            blocked_group_ids.update(
                _group_ids_from_row(dict(payload), source_path=registry_path, row_index=row_index)
            )

    for candidate_path in blocked_candidate_files:
        candidate_rows = _load_json_array(candidate_path)
        for row_index, row in enumerate(candidate_rows, start=1):
            blocked_group_ids.update(
                _group_ids_from_row(row, source_path=candidate_path, row_index=row_index)
            )

    return blocked_group_ids


def main() -> None:
    args = build_parser().parse_args()
    candidate_rows = _load_json_array(args.candidate_file)
    blocked_group_ids = _load_blocked_group_ids(
        blocked_split_registries=list(args.blocked_split_registry),
        blocked_candidate_files=list(args.blocked_candidate_file),
    )
    partition_rows = build_expansion_partition_rows(
        candidate_rows,
        blocked_group_ids=blocked_group_ids,
    )
    partition_stats = build_partition_stats(partition_rows)
    write_partition_outputs(args.output_dir, partition_rows, partition_stats)

    registry_rows = build_split_registry(partition_rows, pool_name=args.pool_name)
    registry_stats = build_split_registry_stats(registry_rows)
    write_split_registry_outputs(args.output_dir, registry_rows, registry_stats)

    print(f"Candidate rows: {len(candidate_rows)}")
    print(f"Blocked group ids: {len(blocked_group_ids)}")
    print(f"Expansion groups: {len(partition_rows)}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
