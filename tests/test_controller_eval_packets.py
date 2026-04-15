
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from idea_graph.controller_eval_packets import build_broad_dev_gate


def _write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            json.dump(row, stream)
            stream.write("\n")


def test_build_broad_dev_gate_creates_stats_and_selectors(tmp_path: Path) -> None:
    rows = [
        {
            "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-10",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "ai-idea-bench-2025-10",
            "source_split": "train",
            "partition_role": "critic_train",
        },
        {
            "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-200",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "ai-idea-bench-2025-200",
            "source_split": "validation",
            "partition_role": "critic_dev",
        },
        {
            "group_id": "liveideabench::liveideabench-earthquakes-70",
            "benchmark": "liveideabench",
            "instance_name": "liveideabench-earthquakes-70",
            "source_split": "train",
            "partition_role": "critic_train",
        },
        {
            "group_id": "liveideabench::liveideabench-global positioning system-191",
            "benchmark": "liveideabench",
            "instance_name": "liveideabench-global positioning system-191",
            "source_split": "validation",
            "partition_role": "critic_dev",
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)
    output_root = tmp_path / "output"

    stats = build_broad_dev_gate(
        manifest,
        output_root,
        manifest_name="test_gate.jsonl",
        stats_name="stats.json",
        readme_name="README.md",
    )

    assert stats["group_count"] == len(rows)
    assert stats["role_counts"] == {"critic_train": 2, "critic_dev": 2}
    assert stats["benchmark_counts"] == {"AI_Idea_Bench_2025": 2, "liveideabench": 2}

    produced_rows = [
        json.loads(line)
        for line in (output_root / "test_gate.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert produced_rows[0]["partition_role"] == rows[0]["partition_role"]
    ai_rows = [row for row in produced_rows if row["benchmark"] == "AI_Idea_Bench_2025"]
    assert ai_rows[0]["benchmark_index"] == 10
    live_row = next(row for row in produced_rows if row["benchmark"] == "liveideabench" and row["instance_name"].endswith("-191"))
    assert live_row["row_index"] == 191
    assert live_row["benchmark_keyword"] == "global positioning system"


def test_build_broad_dev_gate_rejects_duplicate_group_ids(tmp_path: Path) -> None:
    rows = [
        {
            "group_id": "dup::ai-idea-bench-2025-1",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "ai-idea-bench-2025-1",
            "source_split": "train",
            "partition_role": "critic_train",
        },
        {
            "group_id": "dup::ai-idea-bench-2025-1",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "ai-idea-bench-2025-5",
            "source_split": "validation",
            "partition_role": "critic_dev",
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="duplicate"):
        build_broad_dev_gate(manifest, tmp_path / "output")


def test_cli_emits_expected_files(tmp_path: Path) -> None:
    rows = [
        {
            "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-13",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "ai-idea-bench-2025-13",
            "source_split": "train",
            "partition_role": "critic_train",
        },
        {
            "group_id": "liveideabench::liveideabench-energy-3",
            "benchmark": "liveideabench",
            "instance_name": "liveideabench-energy-3",
            "source_split": "validation",
            "partition_role": "critic_dev",
        },
    ]
    manifest = tmp_path / "cli_manifest.jsonl"
    _write_manifest(manifest, rows)
    output_root = tmp_path / "cli_output"
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_controller_eval_packets.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--output-root",
            str(output_root),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert (output_root / "README.md").is_file()
    assert (output_root / "broad_dev_gate_59.jsonl").is_file()
    stats = json.loads((output_root / "packet_stats.json").read_text(encoding="utf-8"))
    assert stats["group_count"] == len(rows)
