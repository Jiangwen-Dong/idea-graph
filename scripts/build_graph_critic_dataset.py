from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.critic_dataset import build_graph_critic_dataset


def _validation_fraction_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid float value: {value!r}"
        ) from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(
            "validation fraction must be between 0.0 and 1.0 (inclusive)"
        )
    return parsed


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
        description="Build a graph-critic (G2) dataset from a G1 trajectory dataset export."
    )
    parser.add_argument(
        "--g1-dataset-dir",
        type=Path,
        required=True,
        help="Directory containing run_manifest.jsonl and trajectory_examples.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "graph_critic_datasets",
        help="Directory under which the G2 dataset subdirectory will be created.",
    )
    parser.add_argument(
        "--dataset-name",
        type=_dataset_name_arg,
        required=True,
        help="Name of the generated G2 dataset subdirectory.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=_validation_fraction_arg,
        default=0.2,
        help="Fraction of groups assigned to validation split (per benchmark).",
    )
    parser.add_argument(
        "--split-overrides",
        type=Path,
        default=None,
        help="Optional JSONL file with explicit group_id -> split assignments.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_graph_critic_dataset(
        g1_dataset_dir=args.g1_dataset_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        validation_fraction=args.validation_fraction,
        split_overrides_path=args.split_overrides,
    )
    print(f"Dataset directory: {result.dataset_dir}")
    print(f"Group count: {result.group_count}")
    print(f"Transition count: {result.transition_count}")


if __name__ == "__main__":
    main()
