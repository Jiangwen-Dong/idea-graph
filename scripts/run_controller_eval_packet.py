from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.controller_eval_runtime import (
    build_run_manifest_row,
    execute_packet_run,
    load_packet_rows,
)
from idea_graph.fs_utils import write_text_file
from idea_graph.repo_paths import default_benchmark_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a controller-evaluation packet for one or more baselines."
    )
    parser.add_argument(
        "--packet-manifest",
        required=True,
        type=Path,
        help="JSONL packet manifest to evaluate.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        required=True,
        help="Runtime baseline names to execute, such as ours-eig ours-eig-critic-graph.",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=None,
        help="Optional OpenAI-compatible runtime config.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory where runs/ and run_manifest.jsonl are written.",
    )
    parser.add_argument(
        "--benchmark-root-base",
        type=Path,
        default=default_benchmark_root(ROOT),
        help="Base benchmark-data directory.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum EIG rounds per run.",
    )
    parser.add_argument(
        "--native-eval",
        action="store_true",
        help="Run benchmark-native evaluation after generation.",
    )
    parser.add_argument(
        "--partition-role-filter",
        default=None,
        help="Optional partition role to run, such as critic_dev.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the planned run manifest without launching generation.",
    )
    parser.add_argument(
        "--runtime-controller-calibration-path",
        type=Path,
        default=None,
        help="Optional explicit joint controller calibration artifact for runtime-controller baselines.",
    )
    parser.add_argument(
        "--disable-runtime-calibration",
        action="store_true",
        help="Disable automatic joint controller calibration loading for runtime-controller baselines.",
    )
    return parser


def _jsonl_lines(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _planned_run_row(row: dict[str, object], *, baseline_name: str) -> dict[str, object]:
    run_row = build_run_manifest_row(
        row,
        baseline_name=baseline_name,
        run_dir="",
    )
    run_row["dry_run"] = True
    return run_row


def main() -> None:
    args = build_parser().parse_args()
    packet_rows = load_packet_rows(
        args.packet_manifest,
        partition_role_filter=args.partition_role_filter,
    )

    run_manifest_rows: list[dict[str, object]] = []
    for row in packet_rows:
        for baseline_name in args.baselines:
            if args.dry_run:
                run_manifest_rows.append(_planned_run_row(row, baseline_name=baseline_name))
            else:
                run_manifest_rows.append(
                    execute_packet_run(
                        row,
                        baseline_name=baseline_name,
                        output_root=args.output_root,
                        benchmark_root_base=args.benchmark_root_base,
                        max_rounds=args.max_rounds,
                        native_eval=args.native_eval,
                        llm_config_path=args.llm_config,
                        runtime_controller_calibration_path=args.runtime_controller_calibration_path,
                        disable_runtime_calibration=args.disable_runtime_calibration,
                    )
                )
            write_text_file(args.output_root / "run_manifest.jsonl", _jsonl_lines(run_manifest_rows))

    print(
        f"Wrote {len(run_manifest_rows)} run-manifest rows "
        f"for {len(packet_rows)} packet groups to {args.output_root / 'run_manifest.jsonl'}"
    )


if __name__ == "__main__":
    main()
