from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.candidate_slate_dataset import build_parallel_two_head_dataset_from_export


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
        description="Build a parallel two-head graph-critic dataset from exported replay rows."
    )
    parser.add_argument("--g1-dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "graph_critic_datasets",
    )
    parser.add_argument("--dataset-name", type=_dataset_name_arg, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-overrides-path", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_parallel_two_head_dataset_from_export(
        g1_dataset_dir=args.g1_dataset_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        validation_fraction=args.validation_fraction,
        split_overrides_path=args.split_overrides_path,
    )
    print(f"Dataset directory: {result.dataset_dir}")
    print(f"Edit state count: {result.edit_state_count}")
    print(f"Edit candidate count: {result.edit_candidate_count}")
    print(f"Commit state count: {result.commit_state_count}")


if __name__ == "__main__":
    main()
