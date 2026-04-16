from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.critic_split_overrides import (
    build_split_override_rows,
    load_split_registry_rows,
    write_split_override_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/validation split overrides from a split registry."
    )
    parser.add_argument("--split-registry", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    rows = load_split_registry_rows(args.split_registry)
    overrides = build_split_override_rows(rows)
    write_split_override_rows(args.output_path, overrides)
    print(f"Wrote {len(overrides)} split overrides to {args.output_path}")


if __name__ == "__main__":
    main()
