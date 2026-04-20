from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_quality_batch import (
    aggregate_rows,
    format_markdown_summary,
    overall_aggregate_rows,
    write_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge sharded quality-batch outputs into one summary.")
    parser.add_argument(
        "--shard-dirs",
        nargs="+",
        required=True,
        type=Path,
        help="Shard directories containing selected_rows.csv and optional raw_rows.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where merged CSV and summary artifacts are written.",
    )
    return parser


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _coerce_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def normalize_selected_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    numeric_float_fields = (
        "overall_score",
        "benchmark_alignment",
        "expert_style_quality",
        "graph_process",
        "llm_call_count",
        "total_tokens",
    )
    for key in numeric_float_fields:
        normalized[key] = float(normalized.get(key, 0.0) or 0.0)
    native_value = _coerce_float(normalized.get("native_average_normalized_10"))
    normalized["native_average_normalized_10"] = native_value
    normalized["executed_round_count"] = _coerce_int(normalized.get("executed_round_count"))
    normalized["action_count"] = _coerce_int(normalized.get("action_count"))
    return normalized


def normalize_raw_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    numeric_float_fields = (
        "overall_score",
        "benchmark_alignment",
        "expert_style_quality",
        "graph_process",
        "llm_call_count",
        "total_tokens",
    )
    for key in numeric_float_fields:
        normalized[key] = float(normalized.get(key, 0.0) or 0.0)
    native_value = _coerce_float(normalized.get("native_average_normalized_10"))
    normalized["native_average_normalized_10"] = native_value
    normalized["executed_round_count"] = _coerce_int(normalized.get("executed_round_count"))
    normalized["action_count"] = _coerce_int(normalized.get("action_count"))
    return normalized


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    shard_dirs = [Path(path) for path in args.shard_dirs]
    for shard_dir in shard_dirs:
        selected_rows.extend(
            normalize_selected_row(row)
            for row in read_csv_rows(shard_dir / "selected_rows.csv")
        )
        raw_rows.extend(
            normalize_raw_row(row)
            for row in read_csv_rows(shard_dir / "raw_rows.csv")
        )

    aggregate_payload = aggregate_rows(selected_rows)
    overall_payload = overall_aggregate_rows(selected_rows)

    payload = {
        "generated_at": output_dir.name,
        "model": "merged-shards",
        "ai_indices": [],
        "live_row_indices": [],
        "method_plans": [],
        "raw_rows": raw_rows,
        "aggregate_rows": aggregate_payload,
        "overall_aggregate_rows": overall_payload,
        "findings": [f"Merged {len(shard_dirs)} shard directories."],
        "next_steps": ["Use the merged pilot outputs for paper-facing comparison and follow-up launches."],
        "merged_shard_dirs": [str(path) for path in shard_dirs],
    }

    write_csv(output_dir / "selected_rows.csv", selected_rows)
    write_csv(output_dir / "aggregate_rows.csv", aggregate_payload)
    write_csv(output_dir / "overall_aggregate_rows.csv", overall_payload)
    (output_dir / "batch_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (output_dir / "batch_summary.md").write_text(
        format_markdown_summary(payload),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
