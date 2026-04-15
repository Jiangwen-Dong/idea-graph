from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.paper_eval_pool import select_paper_eval_candidates


def _write_aiib_metadata(path: Path, *, indexes: list[int]) -> None:
    entries = []
    for index in indexes:
        entries.append(
            {
                "index": index,
                "summary": {
                    "topic": f"Topic {index}",
                    "revised_topic": f"Revised topic {index}",
                },
            }
        )
    path.write_text(json.dumps(entries, ensure_ascii=False))


def _write_live_csv(path: Path, *, keywords: list[str]) -> None:
    lines = ["keywords,idea"]
    for keyword in keywords:
        lines.append(f"{keyword},idea for {keyword}")
    path.write_text("\n".join(lines))


def test_selects_aiib_and_live_candidates(tmp_path: Path) -> None:
    aiib_json = tmp_path / "aiib.json"
    _write_aiib_metadata(aiib_json, indexes=[1, 2])

    live_csv = tmp_path / "live.csv"
    _write_live_csv(live_csv, keywords=["alpha", "beta"])

    blocked_candidates = tmp_path / "blocked.json"
    blocked_candidates.write_text(
        json.dumps(
            [
                {
                    "benchmark": "AI_Idea_Bench_2025",
                    "instance_name": "ai-idea-bench-2025-1",
                }
            ]
        )
    )

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"benchmark": "liveideabench", "instance_name": "liveideabench-beta-1"}\n'
    )

    rows = select_paper_eval_candidates(
        aiib_metadata=aiib_json,
        live_csv=live_csv,
        target_aiib=1,
        target_live=1,
        blocked_candidate_files=[blocked_candidates],
        blocked_split_registries=[manifest],
    )

    aiib_rows = [row for row in rows if row["benchmark"] == "AI_Idea_Bench_2025"]
    assert len(aiib_rows) == 1
    aiib_row = aiib_rows[0]
    assert aiib_row["instance_name"] == "ai-idea-bench-2025-2"
    assert aiib_row["status"] == "frozen"
    assert aiib_row["intended_role"] == "paper_eval"
    assert "Revised topic 2" in aiib_row["notes"]

    live_rows = [row for row in rows if row["benchmark"] == "liveideabench"]
    assert len(live_rows) == 1
    live_row = live_rows[0]
    assert live_row["instance_name"] == "liveideabench-alpha-0"
    assert "alpha" in live_row["notes"]


def test_selection_errors_when_insufficient_candidates(tmp_path: Path) -> None:
    aiib_json = tmp_path / "aiib.json"
    _write_aiib_metadata(aiib_json, indexes=[5])

    live_csv = tmp_path / "live.csv"
    _write_live_csv(live_csv, keywords=["gamma"])

    with pytest.raises(ValueError, match="AI_Idea_Bench_2025"):
        select_paper_eval_candidates(
            aiib_metadata=aiib_json,
            live_csv=live_csv,
            target_aiib=2,
            target_live=0,
        )


def test_cli_writes_artifacts(tmp_path: Path) -> None:
    aiib_json = tmp_path / "aiib.json"
    _write_aiib_metadata(aiib_json, indexes=[10, 11])

    live_csv = tmp_path / "live.csv"
    _write_live_csv(live_csv, keywords=["delta", "epsilon"])

    blocked_candidates = tmp_path / "blocked.json"
    blocked_candidates.write_text(
        json.dumps(
            [
                {
                    "benchmark": "AI_Idea_Bench_2025",
                    "instance_name": "ai-idea-bench-2025-10",
                },
            ]
        )
    )

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"benchmark": "liveideabench", "instance_name": "liveideabench-epsilon-1"}\n'
    )

    output_root = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_paper_eval_freeze_pool.py"),
            "--output-root",
            str(output_root),
            "--target-aiib",
            "1",
            "--target-live",
            "1",
            "--aiib-metadata",
            str(aiib_json),
            "--live-csv",
            str(live_csv),
            "--blocked-candidate-file",
            str(blocked_candidates),
            "--blocked-split-registry",
            str(manifest),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    assert "Candidate rows: 2" in result.stdout
    candidate_path = output_root / "candidate_instances.json"
    assert candidate_path.exists()
    rows = json.loads(candidate_path.read_text())
    assert len(rows) == 2

    readme = (output_root / "README.md").read_text()
    assert "paper_eval_candidate_pool_v2" in readme
    assert "Status" in readme

    stats = json.loads((output_root / "pool_stats.json").read_text())
    aiib_stats = stats["benchmarks"]["AI_Idea_Bench_2025"]
    assert aiib_stats["selected"] == 1


def test_cli_defaults_block_all_current_development_sources() -> None:
    script_path = ROOT / "scripts" / "build_paper_eval_freeze_pool.py"
    spec = importlib.util.spec_from_file_location("build_paper_eval_freeze_pool", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    candidate_defaults = {
        path.relative_to(ROOT).as_posix() for path in module.DEFAULT_BLOCKED_CANDIDATES
    }
    split_defaults = {
        path.relative_to(ROOT).as_posix() for path in module.DEFAULT_BLOCKED_SPLITS
    }

    assert (
        "outputs/graph_critic_datasets/02_active_graph_critic/"
        "development_pool_v2_candidate_pool_v1/candidate_instances.json"
        in candidate_defaults
    )
    assert (
        "outputs/graph_critic_datasets/02_active_graph_critic/"
        "development_pool_v3_candidate_pool_v1/candidate_instances.json"
        in candidate_defaults
    )
    assert (
        "outputs/graph_critic_datasets/02_active_graph_critic/"
        "paper_eval_candidate_pool_v1/candidate_instances.json"
        in candidate_defaults
    )
    assert (
        "outputs/graph_critic_datasets/01_active_text_critic/"
        "current_benchmarked_ours_eig_full_g35_partitions/partition_manifest.jsonl"
        in split_defaults
    )
    assert (
        "outputs/graph_critic_datasets/02_active_graph_critic/"
        "development_pool_v3_combined_g2_partitions/partition_manifest.jsonl"
        in split_defaults
    )
