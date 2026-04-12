from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.candidate_slate_dataset import build_graph_critic_candidate_dataset


def _dataset_name_arg(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise argparse.ArgumentTypeError("dataset name must not be empty")

    path = Path(candidate)
    if path.is_absolute() or path.anchor:
        raise argparse.ArgumentTypeError("dataset name must be a relative single directory name")
    if len(path.parts) != 1:
        raise argparse.ArgumentTypeError("dataset name must not include path separators")
    if path.parts[0] in {".", ".."}:
        raise argparse.ArgumentTypeError("dataset name must not be '.' or '..'")

    return path.parts[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a graph-critic candidate-slate (G2.5) dataset from aligned G1+G2 exports."
    )
    parser.add_argument(
        "--g1-dataset-dir",
        type=Path,
        required=True,
        help="Directory containing G1 trajectory_examples.jsonl and state_snapshots/.",
    )
    parser.add_argument(
        "--g2-dataset-dir",
        type=Path,
        required=True,
        help="Directory containing G2 critic_dataset.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "graph_critic_datasets",
        help="Directory under which the G2.5 dataset subdirectory will be created.",
    )
    parser.add_argument(
        "--dataset-name",
        type=_dataset_name_arg,
        required=True,
        help="Name of the generated G2.5 dataset subdirectory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_graph_critic_candidate_dataset(
        g1_dataset_dir=args.g1_dataset_dir,
        g2_dataset_dir=args.g2_dataset_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
    )
    print(f"Dataset directory: {result.dataset_dir}")
    print(f"State count: {result.state_count}")
    print(f"Candidate count: {result.candidate_count}")


if __name__ == "__main__":
    main()
