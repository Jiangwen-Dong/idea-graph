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
    build_joint_calibration_examples_from_packet,
    fit_joint_controller_calibration,
    write_joint_controller_calibration,
)
from idea_graph.fs_utils import read_text_file, write_text_file


def _load_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in read_text_file(path).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain JSON objects per line.")
        rows.append(dict(payload))
    return rows


def _write_jsonl(rows: list[dict[str, object]], path: str | Path) -> None:
    output_path = Path(path)
    write_text_file(
        output_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit a frozen-dev joint edit+commit controller calibration artifact.",
    )
    parser.add_argument("--edit-examples", help="JSONL path with override-margin edit examples.")
    parser.add_argument(
        "--commit-examples",
        help="JSONL path with commit-probability examples.",
    )
    parser.add_argument(
        "--run-manifest",
        help="Optional packet run_manifest.jsonl used to derive edit+commit examples directly.",
    )
    parser.add_argument("--heuristic-baseline", default="ours-eig")
    parser.add_argument("--critic-baseline", default="ours-eig-critic-graph-twohead")
    parser.add_argument(
        "--repo-root",
        default="",
        help="Optional repository root used to resolve repo-relative run_dir values.",
    )
    parser.add_argument(
        "--prepared-output-dir",
        default="",
        help="Optional directory where derived edit_examples.jsonl and commit_examples.jsonl are written.",
    )
    parser.add_argument("--output-path", required=True, help="Destination JSON path for the calibration artifact.")
    parser.add_argument("--source", default="critic_dev", help="Artifact provenance label.")
    args = parser.parse_args()

    if args.run_manifest:
        edit_examples, commit_examples = build_joint_calibration_examples_from_packet(
            run_manifest_path=args.run_manifest,
            heuristic_baseline=args.heuristic_baseline,
            critic_baseline=args.critic_baseline,
            repo_root=Path(args.repo_root).resolve() if str(args.repo_root).strip() else None,
        )
        if str(args.prepared_output_dir).strip():
            prepared_dir = Path(args.prepared_output_dir)
            _write_jsonl(edit_examples, prepared_dir / "edit_examples.jsonl")
            _write_jsonl(commit_examples, prepared_dir / "commit_examples.jsonl")
    else:
        if not args.edit_examples or not args.commit_examples:
            parser.error("Either --run-manifest or both --edit-examples and --commit-examples are required.")
        edit_examples = _load_jsonl(args.edit_examples)
        commit_examples = _load_jsonl(args.commit_examples)

    calibration = fit_joint_controller_calibration(
        edit_examples=edit_examples,
        commit_examples=commit_examples,
        source=args.source,
    )
    write_joint_controller_calibration(calibration, args.output_path)
    print(
        "Wrote joint controller calibration to "
        f"{Path(args.output_path).resolve()} "
        f"from {len(edit_examples)} edit examples and {len(commit_examples)} commit examples."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
