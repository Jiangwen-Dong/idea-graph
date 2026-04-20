from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_merge_quality_batch_shards_merges_selected_rows(tmp_path: Path) -> None:
    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    shard_a.mkdir()
    shard_b.mkdir()
    (shard_a / "selected_rows.csv").write_text(
        "benchmark,baseline_name,overall_score,benchmark_alignment,expert_style_quality,graph_process,llm_call_count,total_tokens,native_average_normalized_10,run_dir,run_dir_name,stop_reason,runtime_protocol,executed_round_count,action_count\n"
        "AI_Idea_Bench_2025,ours-eig-critic-graph-twohead,8.0,7.0,7.5,8.5,16,100,8.2,run-a,run-a,commit_at_Round3,parallel_graph_v2,3,9\n",
        encoding="utf-8",
    )
    (shard_b / "selected_rows.csv").write_text(
        "benchmark,baseline_name,overall_score,benchmark_alignment,expert_style_quality,graph_process,llm_call_count,total_tokens,native_average_normalized_10,run_dir,run_dir_name,stop_reason,runtime_protocol,executed_round_count,action_count\n"
        "liveideabench,ours-eig-critic-graph-twohead,7.0,6.0,7.1,8.0,17,120,7.4,run-b,run-b,commit_at_Round4,parallel_graph_v2,4,11\n",
        encoding="utf-8",
    )
    (shard_a / "raw_rows.csv").write_text(
        "benchmark,display_selector,baseline_name,overall_score,benchmark_alignment,expert_style_quality,graph_process,llm_call_count,total_tokens,native_average_normalized_10,run_dir,run_dir_name,stop_reason,runtime_protocol,executed_round_count,action_count\n"
        "AI_Idea_Bench_2025,13,ours-eig-critic-graph-twohead,8.0,7.0,7.5,8.5,16,100,8.2,run-a,run-a,commit_at_Round3,parallel_graph_v2,3,9\n",
        encoding="utf-8",
    )
    (shard_b / "raw_rows.csv").write_text(
        "benchmark,display_selector,baseline_name,overall_score,benchmark_alignment,expert_style_quality,graph_process,llm_call_count,total_tokens,native_average_normalized_10,run_dir,run_dir_name,stop_reason,runtime_protocol,executed_round_count,action_count\n"
        "liveideabench,47,ours-eig-critic-graph-twohead,7.0,6.0,7.1,8.0,17,120,7.4,run-b,run-b,commit_at_Round4,parallel_graph_v2,4,11\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "merged"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "merge_quality_batch_shards.py"),
            "--shard-dirs",
            str(shard_a),
            str(shard_b),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "selected_rows.csv").is_file()
    assert (output_dir / "aggregate_rows.csv").is_file()
    assert (output_dir / "overall_aggregate_rows.csv").is_file()
    assert (output_dir / "batch_summary.md").is_file()

    with (output_dir / "overall_aggregate_rows.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["baseline_name"] == "ours-eig-critic-graph-twohead"
    assert float(rows[0]["mean_total_tokens"]) == 110.0
