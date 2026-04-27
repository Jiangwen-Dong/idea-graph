from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.controller_eval_packets import build_broad_dev_gate

DEFAULT_MANIFEST = (
    ROOT
    / "outputs"
    / "graph_critic_datasets"
    / "02_active_graph_critic"
    / "development_pool_v3_combined_g2_partitions"
    / "partition_manifest.jsonl"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "controller_eval_packets" / "graph_critic_scaleup_v2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the broad controller evaluation packet manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Source partition manifest for the broad development gate.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where manifest, README, and stats are written.",
    )
    parser.add_argument(
        "--manifest-name",
        default="broad_dev_gate_59.jsonl",
        help="Filename to use for the generated packet manifest.",
    )
    parser.add_argument(
        "--stats-name",
        default="packet_stats.json",
        help="Filename to use for the packet statistics artifact.",
    )
    parser.add_argument(
        "--readme-name",
        default="README.md",
        help="Filename for the README in the output root.",
    )
    args = parser.parse_args()

    stats = build_broad_dev_gate(
        args.manifest,
        args.output_root,
        manifest_name=args.manifest_name,
        stats_name=args.stats_name,
        readme_name=args.readme_name,
    )
    manifest_path = args.output_root / args.manifest_name
    print(f"Wrote {stats['group_count']} groups to {manifest_path}")


if __name__ == "__main__":
    main()
