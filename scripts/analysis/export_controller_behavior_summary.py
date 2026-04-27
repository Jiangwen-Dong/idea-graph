from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


METHOD_LABELS = {
    "ours-eig-critic-graph-twohead": "EIG (Ours)",
    "ours-eig-critic-text": "EIG-Text",
    "ours-eig-critic-no-commit": "EIG-NoCommit",
    "ours-eig-fixed-control": "EIG-Fixed",
    "ours-eig-random-control": "EIG-Random",
}

METHOD_ORDER = ["EIG (Ours)", "EIG-Text", "EIG-NoCommit", "EIG-Fixed", "EIG-Random"]
ROUNDS = [2, 3, 4, 5]


def _safe_path(path: Path) -> str:
    resolved = path.resolve(strict=False)
    if os.name == "nt":
        return "\\\\?\\" + str(resolved)
    return str(resolved)


def _read_json(path: Path) -> dict[str, Any]:
    with open(_safe_path(path), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(_safe_path(path), "r", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} line {line_index} must contain a JSON object.")
            rows.append(payload)
    return rows


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float:
    numeric_values = list(values)
    return statistics.mean(numeric_values) if numeric_values else 0.0


def _resolve_run_dir(run_root: Path, run_dir_value: object) -> Path:
    candidate = Path(str(run_dir_value))
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    rooted = (run_root.parent / candidate).resolve(strict=False)
    if rooted.exists():
        return rooted
    cwd_rooted = (Path.cwd() / candidate).resolve(strict=False)
    if cwd_rooted.exists():
        return cwd_rooted
    return candidate.resolve(strict=False)


def _native_score(run_dir: Path) -> float | None:
    native_path = run_dir / "benchmark_native_evaluation.json"
    if not native_path.exists():
        return None
    payload = _read_json(native_path)
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        return None
    return _float_or_none(summary.get("available_average_normalized_10"))


def _load_run_row(run_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    run_dir = _resolve_run_dir(run_root, row["run_dir"])
    summary = _read_json(run_dir / "summary.json")
    evaluation = _read_json(run_dir / "evaluation.json")
    baseline_name = str(row.get("baseline_name", "")).strip()
    method = METHOD_LABELS.get(baseline_name, baseline_name)
    round_count = int(float(summary.get("executed_round_count", 0) or 0))
    quality = _float_or_none(evaluation.get("overall_score"))
    native = _native_score(run_dir)
    return {
        "method": method,
        "baseline_name": baseline_name,
        "round": round_count,
        "quality": quality,
        "native": native,
    }


def aggregate(run_root: Path, manifest_rows: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    loaded_rows = [_load_run_row(run_root, row) for row in manifest_rows]
    rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in loaded_rows:
        if row["method"] in METHOD_ORDER:
            rows_by_method[row["method"]].append(row)

    summary_rows: list[dict[str, str]] = []
    distribution_rows: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        method_rows = rows_by_method.get(method, [])
        if not method_rows:
            continue
        sample_count = len(method_rows)
        round_counts = Counter(int(row["round"]) for row in method_rows)
        avg_round = _mean(float(row["round"]) for row in method_rows)
        early_stop_count = sum(count for round_id, count in round_counts.items() if round_id < 5)
        quality_values = [float(row["quality"]) for row in method_rows if row["quality"] is not None]
        native_values = [float(row["native"]) for row in method_rows if row["native"] is not None]
        summary_row = {
            "method": method,
            "n": str(sample_count),
            "avg_round": f"{avg_round:.2f}",
            "early_stop_rate": f"{100.0 * early_stop_count / sample_count:.1f}",
            "mean_quality": f"{_mean(quality_values):.2f}",
            "mean_native": f"{_mean(native_values):.2f}",
        }
        for round_id in ROUNDS:
            count = round_counts.get(round_id, 0)
            summary_row[f"round{round_id}_pct"] = f"{100.0 * count / sample_count:.1f}"
            distribution_rows.append(
                {
                    "method": method,
                    "round": str(round_id),
                    "count": str(count),
                    "fraction": f"{count / sample_count:.4f}",
                }
            )
        summary_rows.append(summary_row)
    return summary_rows, distribution_rows


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(_safe_path(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _latex_table(summary_rows: list[dict[str, str]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Controller behavior on 512 held-out paper-evaluation groups. Percentages report the stop-round distribution; early stop is the fraction of runs committing before round 5.}",
        "\\label{tab:controller_behavior_512}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Avg. round & Early stop & R2 & R3 & R4 & R5 \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['method']} & {row['avg_round']} & {row['early_stop_rate']} & "
            f"{row['round2_pct']} & {row['round3_pct']} & {row['round4_pct']} & {row['round5_pct']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export controller-behavior summaries and a paper-ready LaTeX table."
    )
    parser.add_argument("--run-root", required=True, type=Path, help="Directory containing run_manifest.jsonl.")
    parser.add_argument("--paper-dir", required=True, type=Path, help="Path to paper/ideation_2026.")
    args = parser.parse_args()

    manifest_rows = _read_jsonl(args.run_root / "run_manifest.jsonl")
    summary_rows, distribution_rows = aggregate(args.run_root, manifest_rows)

    supporting_dir = args.paper_dir / "supporting"
    tables_dir = args.paper_dir / "tables"
    _write_csv(
        supporting_dir / "controller_behavior_512_summary.csv",
        summary_rows,
        [
            "method",
            "n",
            "avg_round",
            "early_stop_rate",
            "mean_quality",
            "mean_native",
            "round2_pct",
            "round3_pct",
            "round4_pct",
            "round5_pct",
        ],
    )
    _write_csv(
        supporting_dir / "controller_behavior_512_round_distribution.csv",
        distribution_rows,
        ["method", "round", "count", "fraction"],
    )
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "controller_behavior_512_table.tex").write_text(
        _latex_table(summary_rows),
        encoding="utf-8",
    )
    print(f"Exported controller behavior for {len(manifest_rows)} runs to {args.paper_dir}")


if __name__ == "__main__":
    main()
