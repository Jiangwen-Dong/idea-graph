from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.trajectory_dataset import PricingConfig, export_graph_critic_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export graph-critic training trajectories from saved run artifacts."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        required=True,
        help="Input root to recursively scan for run directories containing summary.json and graph.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "graph_critic_datasets",
        help="Directory under which the dataset export will be written.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the exported dataset subdirectory.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Optional baseline-name filter.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Optional benchmark-name filter.",
    )
    parser.add_argument(
        "--prompt-price-per-1m-tokens",
        type=float,
        help="Optional prompt-token price for estimated cost reporting.",
    )
    parser.add_argument(
        "--completion-price-per-1m-tokens",
        type=float,
        help="Optional completion-token price for estimated cost reporting.",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        help="Optional cap on the number of discovered runs to export.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pricing = None
    if (
        args.prompt_price_per_1m_tokens is not None
        and args.completion_price_per_1m_tokens is not None
    ):
        pricing = PricingConfig(
            prompt_price_per_1m_tokens=args.prompt_price_per_1m_tokens,
            completion_price_per_1m_tokens=args.completion_price_per_1m_tokens,
        )

    result = export_graph_critic_dataset(
        input_roots=args.input_root,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        baseline=args.baseline,
        benchmark=args.benchmark,
        pricing=pricing,
        limit_runs=args.limit_runs,
    )

    print(f"Dataset directory: {result.dataset_dir}")
    print(f"Run count: {result.run_count}")
    print(f"Transition count: {result.transition_count}")


if __name__ == "__main__":
    main()
