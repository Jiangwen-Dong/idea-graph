from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_run(
    root: Path,
    *,
    name: str,
    baseline_name: str,
    paper_baseline_name: str,
    benchmark: str,
    instance_name: str,
    round_count: int,
    quality: float,
    native: float,
) -> dict[str, object]:
    run_dir = root / "runs" / baseline_name / name
    _write_json(
        run_dir / "summary.json",
        {
            "executed_round_count": round_count,
            "stop_reason": f"mature_at_Round{round_count}",
        },
    )
    _write_json(
        run_dir / "evaluation.json",
        {
            "overall_score": quality,
            "expert_style_quality": quality + 0.1,
            "benchmark_alignment": quality - 0.1,
        },
    )
    _write_json(
        run_dir / "benchmark_native_evaluation.json",
        {
            "summary": {
                "available_average_normalized_10": native,
            }
        },
    )
    return {
        "baseline_name": baseline_name,
        "paper_baseline_name": paper_baseline_name,
        "benchmark": benchmark,
        "instance_name": instance_name,
        "partition_role": "paper_eval",
        "source_split": "frozen",
        "run_dir": str(run_dir),
    }


def test_export_controller_behavior_summary_writes_csvs_and_table(tmp_path: Path) -> None:
    run_root = tmp_path / "packet"
    rows = [
        _write_run(
            run_root,
            name="ours-a",
            baseline_name="ours-eig-critic-graph-twohead",
            paper_baseline_name="ours-eig-critic-graph-twohead",
            benchmark="AI_Idea_Bench_2025",
            instance_name="ai-idea-bench-2025-10",
            round_count=3,
            quality=6.4,
            native=8.1,
        ),
        _write_run(
            run_root,
            name="ours-b",
            baseline_name="ours-eig-critic-graph-twohead",
            paper_baseline_name="ours-eig-critic-graph-twohead",
            benchmark="liveideabench",
            instance_name="liveideabench-galaxies-163",
            round_count=5,
            quality=6.0,
            native=7.9,
        ),
        _write_run(
            run_root,
            name="fixed-a",
            baseline_name="ours-eig-fixed-control",
            paper_baseline_name="ours-eig-fixed-control",
            benchmark="AI_Idea_Bench_2025",
            instance_name="ai-idea-bench-2025-10",
            round_count=5,
            quality=6.1,
            native=7.7,
        ),
    ]
    (run_root / "run_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    paper_dir = tmp_path / "paper" / "ideation_2026"
    script = ROOT / "scripts" / "analysis" / "export_controller_behavior_summary.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-root",
            str(run_root),
            "--paper-dir",
            str(paper_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary_path = paper_dir / "supporting" / "controller_behavior_512_summary.csv"
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    ours = next(row for row in summary_rows if row["method"] == "EIG (Ours)")
    assert ours["n"] == "2"
    assert ours["avg_round"] == "4.00"
    assert ours["early_stop_rate"] == "50.0"
    assert ours["mean_quality"] == "6.20"
    assert ours["mean_native"] == "8.00"

    table = (paper_dir / "tables" / "controller_behavior_512_table.tex").read_text(encoding="utf-8")
    assert "\\label{tab:controller_behavior_512}" in table
    assert "EIG (Ours)" in table
    assert "50.0" in table
