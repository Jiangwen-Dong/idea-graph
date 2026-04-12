from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.critic_partitions import (
    build_partition_manifest,
    build_partition_stats,
    load_split_manifest_rows,
    write_partition_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic group-level partition artifacts for a G2 graph-critic dataset."
    )
    parser.add_argument(
        "--g2-dataset-dir",
        type=Path,
        required=True,
        help="Directory containing split_manifest.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional directory under which the partition artifact directory will be created. "
            "Defaults to the G2 dataset dir."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help=(
            "Optional partition artifact directory name. "
            "Defaults to writing directly into the G2 dataset dir."
        ),
    )
    parser.add_argument(
        "--holdout-group",
        action="append",
        default=[],
        help="Group id to force into paper_eval (repeatable).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split_rows = load_split_manifest_rows(args.g2_dataset_dir)
    partition_rows = build_partition_manifest(
        split_rows,
        holdout_groups=args.holdout_group,
    )
    stats = build_partition_stats(partition_rows)
    output_dataset_dir = Path(args.g2_dataset_dir)
    if args.output_dir is not None:
        output_dataset_dir = Path(args.output_dir)
        if args.dataset_name:
            output_dataset_dir = output_dataset_dir / args.dataset_name
    write_partition_outputs(output_dataset_dir, partition_rows, stats)

    print(f"G2 dataset dir: {args.g2_dataset_dir}")
    print(f"Partition dataset dir: {output_dataset_dir}")
    print(f"Partition groups: {len(partition_rows)}")
    print(f"Has paper_eval: {bool(stats.get('has_paper_eval', False))}")


if __name__ == "__main__":
    main()
