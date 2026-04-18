from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.joint_controller_calibration import (
    fit_joint_controller_calibration,
    write_joint_controller_calibration,
)


def _load_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain JSON objects per line.")
        rows.append(dict(payload))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit a frozen-dev joint edit+commit controller calibration artifact.",
    )
    parser.add_argument("--edit-examples", required=True, help="JSONL path with override-margin edit examples.")
    parser.add_argument(
        "--commit-examples",
        required=True,
        help="JSONL path with commit-probability examples.",
    )
    parser.add_argument("--output-path", required=True, help="Destination JSON path for the calibration artifact.")
    parser.add_argument("--source", default="critic_dev", help="Artifact provenance label.")
    args = parser.parse_args()

    calibration = fit_joint_controller_calibration(
        edit_examples=_load_jsonl(args.edit_examples),
        commit_examples=_load_jsonl(args.commit_examples),
        source=args.source,
    )
    write_joint_controller_calibration(calibration, args.output_path)
    print(f"Wrote joint controller calibration to {Path(args.output_path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
