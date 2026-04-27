from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.critic_split_registry import (
    build_split_registry,
    build_split_registry_stats,
    load_partition_manifest_rows,
    write_split_registry_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a human-auditable split registry from a critic partition manifest."
    )
    parser.add_argument(
        "--partition-manifest",
        type=Path,
        required=True,
        help="Path to partition_manifest.jsonl.",
    )
    parser.add_argument(
        "--pool-name",
        type=str,
        required=True,
        help="Pool name such as development_pool_v1 or paper_eval_v1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where split_registry.jsonl and split_registry_stats.json will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    partition_rows = load_partition_manifest_rows(args.partition_manifest)
    registry_rows = build_split_registry(partition_rows, pool_name=args.pool_name)
    stats = build_split_registry_stats(registry_rows)
    write_split_registry_outputs(args.output_dir, registry_rows, stats)
    print(f"Registry rows: {len(registry_rows)}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
