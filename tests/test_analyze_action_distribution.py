from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module():
    path = ROOT / "scripts" / "analysis" / "analyze_action_distribution.py"
    spec = importlib.util.spec_from_file_location("analyze_action_distribution", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_graph(
    run_dir: Path,
    *,
    instance_name: str,
    edit_actions: list[str],
    commit_labels: list[int],
    benchmark: str = "liveideabench",
    baseline_name: str = "ours-eig-critic-graph-twohead",
) -> None:
    _write_json(
        run_dir / "graph.json",
        {
            "metadata": {
                "parallel_edit_rows": [
                    {
                        "benchmark": benchmark,
                        "instance_name": instance_name,
                        "baseline_name": baseline_name,
                        "selected_action_kind": action,
                    }
                    for action in edit_actions
                ],
                "post_round_commit_rows": [
                    {
                        "benchmark": benchmark,
                        "instance_name": instance_name,
                        "baseline_name": baseline_name,
                        "commit_supervision": {"label": label},
                    }
                    for label in commit_labels
                ],
                "benchmark": benchmark,
                "instance_name": instance_name,
                "baseline_name": baseline_name,
            },
            "actions": [
                {
                    "kind": action,
                }
                for action in edit_actions
                if action != "skip"
            ],
        },
    )


def _write_graph_with_controller_log(
    run_dir: Path,
    *,
    instance_name: str,
    selected_actions: list[str],
    benchmark: str = "liveideabench",
    baseline_name: str = "ours-eig-critic-graph-twohead",
) -> None:
    _write_json(
        run_dir / "graph.json",
        {
            "metadata": {
                "runtime_controller_log": [
                    {
                        "benchmark": benchmark,
                        "instance_name": instance_name,
                        "baseline_name": baseline_name,
                        "selected_kind": action,
                    }
                    for action in selected_actions
                ],
                "benchmark": benchmark,
                "instance_name": instance_name,
                "baseline_name": baseline_name,
            },
            "actions": [
                {
                    "kind": action,
                }
                for action in selected_actions
                if action != "skip"
            ],
        },
    )


def _write_graph_with_round_signals(
    run_dir: Path,
    *,
    instance_name: str,
    round_signals: list[dict[str, object]],
    benchmark: str = "liveideabench",
    baseline_name: str = "ours-eig-critic-graph-twohead",
) -> None:
    _write_json(
        run_dir / "graph.json",
        {
            "metadata": {
                "post_round_commit_rows": [
                    {
                        "benchmark": benchmark,
                        "instance_name": instance_name,
                        "baseline_name": baseline_name,
                        "round_name": str(item["round"]),
                        "graph_signals": dict(item["signals"]),
                    }
                    for item in round_signals
                ],
                "benchmark": benchmark,
                "instance_name": instance_name,
                "baseline_name": baseline_name,
            }
        },
    )


def _write_graph_with_round_actions(
    run_dir: Path,
    *,
    instance_name: str,
    round_actions: list[tuple[str, str]],
    benchmark: str = "liveideabench",
    baseline_name: str = "ours-eig-critic-graph-twohead",
) -> None:
    _write_json(
        run_dir / "graph.json",
        {
            "metadata": {
                "parallel_edit_rows": [
                    {
                        "benchmark": benchmark,
                        "instance_name": instance_name,
                        "baseline_name": baseline_name,
                        "round_name": round_name,
                        "selected_action_kind": action,
                    }
                    for round_name, action in round_actions
                ],
                "benchmark": benchmark,
                "instance_name": instance_name,
                "baseline_name": baseline_name,
            },
            "actions": [
                {
                    "round_name": round_name,
                    "kind": action,
                }
                for round_name, action in round_actions
                if action != "skip"
            ],
        },
    )


def _write_summary(
    run_dir: Path,
    *,
    stop_reason: str | None = None,
    matured_at_round: str | None = None,
    rounds: list[str] | None = None,
    benchmark: str = "liveideabench",
) -> None:
    payload: dict[str, object] = {
        "idea_evaluation": {
            "benchmark": benchmark,
        }
    }
    if stop_reason is not None:
        payload["stop_reason"] = stop_reason
    if matured_at_round is not None:
        payload["matured_at_round"] = matured_at_round
    if rounds is not None:
        payload["rounds"] = [{"round": round_name} for round_name in rounds]
    _write_json(run_dir / "summary.json", payload)


def _write_selected_rows(batch_root: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "benchmark",
        "instance_name",
        "baseline_name",
        "run_dir",
    ]
    batch_root.mkdir(parents=True, exist_ok=True)
    with (batch_root / "selected_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_collect_action_distribution_filters_category_and_counts_edit_and_commit(tmp_path: Path) -> None:
    module = _load_script_module()
    batch_root = tmp_path / "batch"

    parasites_a = batch_root / "runs" / "parasites-a"
    parasites_b = batch_root / "runs" / "parasites-b"
    viruses = batch_root / "runs" / "viruses-a"
    direct = batch_root / "runs" / "direct-parasites"
    _write_graph(
        parasites_a,
        instance_name="liveideabench-parasites-305",
        edit_actions=["add_support_edge", "skip", "attach_evidence"],
        commit_labels=[0, 1],
    )
    _write_graph(
        parasites_b,
        instance_name="liveideabench-parasites-311",
        edit_actions=["add_support_edge", "request_evidence"],
        commit_labels=[0, 0],
    )
    _write_graph(
        viruses,
        instance_name="liveideabench-viruses-120",
        edit_actions=["mark_overlap"],
        commit_labels=[1],
    )
    _write_graph(
        direct,
        instance_name="liveideabench-parasites-305",
        edit_actions=["add_contradiction_edge"],
        commit_labels=[1],
    )
    _write_selected_rows(
        batch_root,
        [
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-305",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(parasites_a),
            },
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-311",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(parasites_b),
            },
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-viruses-120",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(viruses),
            },
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-305",
                "baseline_name": "direct",
                "run_dir": str(direct),
            },
        ],
    )

    result = module.collect_action_distribution(
        batch_root,
        baseline_name="ours-eig-critic-graph-twohead",
        benchmark="liveideabench",
        category="parasites",
    )

    assert result.run_count == 2
    assert result.edit_counts == {
        "add_support_edge": 2,
        "attach_evidence": 1,
        "request_evidence": 1,
        "skip": 1,
    }
    assert result.commit_counts == {
        "continue": 3,
        "commit": 1,
    }


def test_cli_writes_distribution_csv_and_figure(tmp_path: Path) -> None:
    batch_root = tmp_path / "batch"
    run_dir = batch_root / "runs" / "parasites-a"
    _write_graph(
        run_dir,
        instance_name="liveideabench-parasites-305",
        edit_actions=["add_support_edge", "skip", "attach_evidence"],
        commit_labels=[0, 1],
    )
    _write_selected_rows(
        batch_root,
        [
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-305",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(run_dir),
            },
        ],
    )
    output_dir = tmp_path / "paper" / "supporting"
    figure_path = tmp_path / "paper" / "figures" / "action_distribution.pdf"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analysis" / "analyze_action_distribution.py"),
            "--batch-root",
            str(batch_root),
            "--category",
            "parasites",
            "--output-dir",
            str(output_dir),
            "--figure-path",
            str(figure_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    csv_path = output_dir / "action_distribution_liveideabench_parasites.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert {"action_type": "edit", "action": "skip", "count": "1", "fraction": "0.3333"} in rows
    assert {"action_type": "commit", "action": "commit", "count": "1", "fraction": "0.5000"} in rows
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_collect_action_distribution_can_scan_raw_runs_without_selected_rows(tmp_path: Path) -> None:
    module = _load_script_module()
    batch_root = tmp_path / "batch"
    parasites_a = batch_root / "shards" / "shard-0" / "runs" / "parasites-a"
    parasites_b = batch_root / "shards" / "shard-1" / "runs" / "parasites-b"
    viruses = batch_root / "shards" / "shard-2" / "runs" / "viruses-a"
    direct = batch_root / "shards" / "shard-3" / "runs" / "direct-a"

    _write_graph(
        parasites_a,
        instance_name="liveideabench-parasites-305",
        edit_actions=["add_support_edge", "attach_evidence"],
        commit_labels=[0, 1],
    )
    _write_graph(
        parasites_b,
        instance_name="liveideabench-parasites-311",
        edit_actions=["propose_repair"],
        commit_labels=[0],
    )
    _write_graph(
        viruses,
        instance_name="liveideabench-viruses-120",
        edit_actions=["request_evidence"],
        commit_labels=[1],
    )
    _write_graph(
        direct,
        instance_name="liveideabench-parasites-305",
        edit_actions=["add_support_edge"],
        commit_labels=[1],
        baseline_name="direct",
    )

    result = module.collect_action_distribution(
        batch_root,
        baseline_name="ours-eig-critic-graph-twohead",
        benchmark="liveideabench",
        category="parasites",
    )

    assert result.run_count == 2
    assert result.edit_counts == {
        "add_support_edge": 1,
        "attach_evidence": 1,
        "propose_repair": 1,
    }
    assert result.commit_counts == {
        "continue": 2,
        "commit": 1,
    }


def test_collect_action_distribution_uses_controller_log_when_parallel_rows_are_absent(tmp_path: Path) -> None:
    module = _load_script_module()
    batch_root = tmp_path / "batch"
    run_dir = batch_root / "shards" / "shard-0" / "runs" / "parasites-a"
    _write_graph_with_controller_log(
        run_dir,
        instance_name="liveideabench-parasites-305",
        selected_actions=["add_support_edge", "skip", "propose_repair"],
    )

    result = module.collect_action_distribution(
        batch_root,
        baseline_name="ours-eig-critic-graph-twohead",
        benchmark="liveideabench",
        category="parasites",
    )

    assert result.run_count == 1
    assert result.edit_counts == {
        "add_support_edge": 1,
        "propose_repair": 1,
        "skip": 1,
    }


def test_collect_action_distribution_can_scan_runs_nested_under_baseline_directory(tmp_path: Path) -> None:
    module = _load_script_module()
    batch_root = tmp_path / "batch"
    run_dir = (
        batch_root
        / "shard_0"
        / "runs"
        / "ours-eig-critic-graph-twohead"
        / "parasites-a"
    )
    _write_graph(
        run_dir,
        instance_name="liveideabench-parasites-305",
        edit_actions=["add_support_edge", "skip", "attach_evidence"],
        commit_labels=[0, 1],
    )

    result = module.collect_action_distribution(
        batch_root,
        baseline_name="ours-eig-critic-graph-twohead",
        benchmark="liveideabench",
        category="parasites",
    )

    assert result.run_count == 1
    assert result.edit_counts == {
        "add_support_edge": 1,
        "attach_evidence": 1,
        "skip": 1,
    }
    assert result.commit_counts == {
        "continue": 1,
        "commit": 1,
    }


def test_collect_action_distribution_falls_back_when_glob_misses_raw_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module()
    batch_root = tmp_path / "batch"
    run_dir = (
        batch_root
        / "rerun_retry_patch_missing9"
        / "runs"
        / "ours-eig-critic-graph-twohead"
        / "parasites-a"
    )
    _write_graph(
        run_dir,
        instance_name="liveideabench-parasites-305",
        edit_actions=["add_support_edge", "attach_evidence"],
        commit_labels=[0, 1],
    )

    monkeypatch.setattr(pathlib.Path, "glob", lambda self, pattern: [])

    result = module.collect_action_distribution(
        batch_root,
        baseline_name="ours-eig-critic-graph-twohead",
        benchmark="liveideabench",
        category="parasites",
    )

    assert result.run_count == 1
    assert result.edit_counts == {
        "add_support_edge": 1,
        "attach_evidence": 1,
    }
    assert result.commit_counts == {
        "continue": 1,
        "commit": 1,
    }


def test_collect_comparison_distributions_picks_largest_topic_from_raw_runs(tmp_path: Path) -> None:
    module = _load_script_module()
    batch_root = tmp_path / "batch"
    _write_graph(
        batch_root / "shards" / "shard-0" / "runs" / "parasites-a",
        instance_name="liveideabench-parasites-305",
        edit_actions=["add_support_edge"],
        commit_labels=[0],
    )
    _write_graph(
        batch_root / "shards" / "shard-0" / "runs" / "parasites-b",
        instance_name="liveideabench-parasites-311",
        edit_actions=["attach_evidence"],
        commit_labels=[1],
    )
    _write_graph(
        batch_root / "shards" / "shard-1" / "runs" / "viruses-a",
        instance_name="liveideabench-viruses-120",
        edit_actions=["request_evidence"],
        commit_labels=[0],
    )

    aggregate, topic = module.collect_comparison_distributions(
        batch_root,
        baseline_name="ours-eig-critic-graph-twohead",
        benchmark="liveideabench",
        topic_category="auto",
    )

    assert aggregate.run_count == 3
    assert aggregate.category == "all"
    assert aggregate.edit_counts == {
        "add_support_edge": 1,
        "attach_evidence": 1,
        "request_evidence": 1,
    }
    assert topic.category == "parasites"
    assert topic.run_count == 2
    assert topic.edit_counts == {
        "add_support_edge": 1,
        "attach_evidence": 1,
    }


def test_cli_comparison_mode_writes_aggregate_topic_outputs_and_figure(tmp_path: Path) -> None:
    batch_root = tmp_path / "batch"
    _write_graph(
        batch_root / "shards" / "shard-0" / "runs" / "parasites-a",
        instance_name="liveideabench-parasites-305",
        edit_actions=["add_support_edge", "attach_evidence"],
        commit_labels=[0, 1],
    )
    _write_graph(
        batch_root / "shards" / "shard-0" / "runs" / "parasites-b",
        instance_name="liveideabench-parasites-311",
        edit_actions=["propose_repair"],
        commit_labels=[0],
    )
    _write_graph(
        batch_root / "shards" / "shard-1" / "runs" / "viruses-a",
        instance_name="liveideabench-viruses-120",
        edit_actions=["request_evidence"],
        commit_labels=[1],
    )
    output_dir = tmp_path / "paper" / "supporting"
    figure_path = tmp_path / "paper" / "figures" / "action_distribution_comparison.pdf"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analysis" / "analyze_action_distribution.py"),
            "--batch-root",
            str(batch_root),
            "--benchmark",
            "liveideabench",
            "--comparison-figure-path",
            str(figure_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    aggregate_csv = output_dir / "action_distribution_liveideabench_all.csv"
    topic_csv = output_dir / "action_distribution_liveideabench_parasites.csv"
    comparison_summary = output_dir / "action_distribution_liveideabench_comparison_summary.json"
    assert aggregate_csv.exists()
    assert topic_csv.exists()
    summary = json.loads(comparison_summary.read_text(encoding="utf-8"))
    assert summary["aggregate"]["run_count"] == 3
    assert summary["topic"]["category"] == "parasites"
    assert summary["topic"]["run_count"] == 2
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_collect_action_distribution_accepts_all_benchmarks(tmp_path: Path) -> None:
    module = _load_script_module()
    batch_root = tmp_path / "batch"
    aiib = batch_root / "runs" / "aiib-a"
    live = batch_root / "runs" / "live-a"
    _write_graph(
        aiib,
        benchmark="AI_Idea_Bench_2025",
        instance_name="ai-idea-bench-2025-13",
        edit_actions=["add_support_edge"],
        commit_labels=[0],
    )
    _write_graph(
        live,
        benchmark="liveideabench",
        instance_name="liveideabench-parasites-305",
        edit_actions=["propose_repair"],
        commit_labels=[1],
    )
    _write_selected_rows(
        batch_root,
        [
            {
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "ai-idea-bench-2025-13",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(aiib),
            },
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-305",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(live),
            },
        ],
    )

    result = module.collect_action_distribution(
        batch_root,
        baseline_name="ours-eig-critic-graph-twohead",
        benchmark="all",
        category="all",
    )

    assert result.run_count == 2
    assert result.edit_counts == {"add_support_edge": 1, "propose_repair": 1}


def test_collect_round_action_distribution_counts_actions_by_round(tmp_path: Path) -> None:
    module = _load_script_module()
    batch_root = tmp_path / "batch"
    run_dir = batch_root / "runs" / "case-a"
    _write_json(
        run_dir / "graph.json",
        {
            "metadata": {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-305",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "parallel_edit_rows": [
                    {
                        "round_name": "Round1",
                        "selected_action_kind": "add_support_edge",
                    },
                    {
                        "round_name": "Round1",
                        "selected_action_kind": "skip",
                    },
                    {
                        "round_name": "Round2",
                        "selected_action_kind": "propose_repair",
                    },
                ],
            }
        },
    )
    _write_selected_rows(
        batch_root,
        [
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-305",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(run_dir),
            },
        ],
    )

    result = module.collect_round_action_distribution(
        batch_root,
        baseline_name="ours-eig-critic-graph-twohead",
        benchmark="all",
        category="all",
    )

    assert result.run_count == 1
    assert result.round_edit_counts == {
        "Round1": {"add_support_edge": 1, "skip": 1},
        "Round2": {"propose_repair": 1},
    }


def test_collect_round_signal_distribution_aggregates_post_round_commit_signals(tmp_path: Path) -> None:
    module = _load_script_module()
    batch_root = tmp_path / "batch"
    run_a = batch_root / "runs" / "case-a"
    run_b = batch_root / "runs" / "case-b"
    _write_graph_with_round_signals(
        run_a,
        instance_name="liveideabench-parasites-305",
        round_signals=[
            {
                "round": "Round1",
                "signals": {
                    "grounding": 0.20,
                    "completeness": 0.60,
                    "contradiction_load": 0.80,
                    "maturity": 0.30,
                },
            },
            {
                "round": "Round2",
                "signals": {
                    "grounding": 0.50,
                    "completeness": 0.70,
                    "contradiction_load": 0.30,
                    "maturity": 0.55,
                },
            },
        ],
    )
    _write_graph_with_round_signals(
        run_b,
        instance_name="liveideabench-parasites-311",
        round_signals=[
            {
                "round": "Round1",
                "signals": {
                    "grounding": 0.40,
                    "completeness": 0.50,
                    "contradiction_load": 0.40,
                    "maturity": 0.45,
                },
            },
            {
                "round": "Round2",
                "signals": {
                    "grounding": 0.60,
                    "completeness": 0.80,
                    "contradiction_load": 0.20,
                    "maturity": 0.65,
                },
            },
        ],
    )
    _write_selected_rows(
        batch_root,
        [
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-305",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(run_a),
            },
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-311",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(run_b),
            },
        ],
    )

    result = module.collect_round_signal_distribution(
        batch_root,
        baseline_name="ours-eig-critic-graph-twohead",
        benchmark="all",
        category="all",
    )

    assert result.run_count == 2
    assert result.round_signal_means == {
        "Round1": {
            "grounding": 0.3,
            "completeness": 0.55,
            "contradiction_load": 0.6,
            "maturity": 0.375,
        },
        "Round2": {
            "grounding": 0.55,
            "completeness": 0.75,
            "contradiction_load": 0.25,
            "maturity": 0.6,
        },
    }
    assert result.round_signal_counts == {"Round1": 2, "Round2": 2}


def test_load_round_action_distribution_from_csv_reads_counts(tmp_path: Path) -> None:
    module = _load_script_module()
    csv_path = tmp_path / "round_action.csv"
    csv_path.write_text(
        "\n".join(
            [
                "round,action,count,fraction",
                "Round1,add_support_edge,3,0.6000",
                "Round1,add_contradiction_edge,2,0.4000",
                "Round2,propose_repair,4,1.0000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = module.load_round_action_distribution_from_csv(csv_path)

    assert result.round_edit_counts == {
        "Round1": {
            "add_support_edge": 3,
            "add_contradiction_edge": 2,
        },
        "Round2": {
            "propose_repair": 4,
        },
    }


def test_collect_stop_round_distribution_reads_summary_and_graph_fallback(tmp_path: Path) -> None:
    module = _load_script_module()
    batch_root = tmp_path / "batch"
    run_a = batch_root / "runs" / "a"
    run_b = batch_root / "runs" / "b"
    run_c = batch_root / "runs" / "c"

    _write_graph(
        run_a,
        instance_name="liveideabench-parasites-1",
        edit_actions=["add_support_edge"],
        commit_labels=[0, 0, 1],
    )
    _write_summary(
        run_a,
        stop_reason="commit_at_Round3",
        matured_at_round="Round3",
    )

    _write_graph(
        run_b,
        instance_name="liveideabench-parasites-2",
        edit_actions=["attach_evidence"],
        commit_labels=[0, 0, 0, 1],
    )
    _write_summary(
        run_b,
        matured_at_round="Round4",
    )

    _write_graph(
        run_c,
        instance_name="liveideabench-parasites-3",
        edit_actions=["propose_repair"],
        commit_labels=[0, 1],
    )

    _write_selected_rows(
        batch_root,
        [
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-1",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(run_a),
            },
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-2",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(run_b),
            },
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-3",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(run_c),
            },
        ],
    )

    result = module.collect_stop_round_distribution(
        batch_root,
        baseline_name="ours-eig-critic-graph-twohead",
        benchmark="liveideabench",
        category="parasites",
    )

    assert result.run_count == 3
    assert result.stop_round_counts == {
        "Round2": 1,
        "Round3": 1,
        "Round4": 1,
    }


def test_cli_writes_round_signal_csv_and_figure(tmp_path: Path) -> None:
    batch_root = tmp_path / "batch"
    run_dir = batch_root / "runs" / "case-a"
    _write_graph_with_round_signals(
        run_dir,
        instance_name="liveideabench-parasites-305",
        round_signals=[
            {
                "round": "Round1",
                "signals": {
                    "grounding": 0.20,
                    "completeness": 0.60,
                    "contradiction_load": 0.80,
                    "maturity": 0.30,
                },
            },
            {
                "round": "Round2",
                "signals": {
                    "grounding": 0.50,
                    "completeness": 0.70,
                    "contradiction_load": 0.30,
                    "maturity": 0.55,
                },
            },
        ],
    )
    _write_selected_rows(
        batch_root,
        [
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-305",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(run_dir),
            },
        ],
    )
    output_dir = tmp_path / "paper" / "supporting"
    signal_figure_path = tmp_path / "paper" / "figures" / "round_signal_trajectory.pdf"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analysis" / "analyze_action_distribution.py"),
            "--batch-root",
            str(batch_root),
            "--category",
            "parasites",
            "--output-dir",
            str(output_dir),
            "--signal-figure-path",
            str(signal_figure_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    csv_path = output_dir / "round_signal_trajectory_liveideabench_parasites.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert {
        "round": "Round1",
        "signal": "grounding",
        "mean": "0.2000",
        "std": "0.0000",
        "run_count": "1",
    } in rows
    assert {
        "round": "Round2",
        "signal": "maturity",
        "mean": "0.5500",
        "std": "0.0000",
        "run_count": "1",
    } in rows
    assert signal_figure_path.exists()
    assert signal_figure_path.stat().st_size > 0


def test_cli_writes_round_action_csv_and_figure(tmp_path: Path) -> None:
    batch_root = tmp_path / "batch"
    run_dir = batch_root / "runs" / "parasites-a"
    _write_graph_with_round_actions(
        run_dir,
        instance_name="liveideabench-parasites-305",
        round_actions=[
            ("Round1", "add_support_edge"),
            ("Round1", "add_contradiction_edge"),
            ("Round2", "propose_repair"),
            ("Round3", "attach_evidence"),
        ],
    )
    _write_selected_rows(
        batch_root,
        [
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-305",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(run_dir),
            },
        ],
    )
    output_dir = tmp_path / "paper" / "supporting"
    figure_path = tmp_path / "paper" / "figures" / "round_action_distribution.pdf"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analysis" / "analyze_action_distribution.py"),
            "--batch-root",
            str(batch_root),
            "--category",
            "parasites",
            "--output-dir",
            str(output_dir),
            "--round-action-figure-path",
            str(figure_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    csv_path = output_dir / "round_action_distribution_liveideabench_parasites.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert {"round": "Round1", "action": "add_support_edge", "count": "1", "fraction": "0.5000"} in rows
    assert {"round": "Round2", "action": "propose_repair", "count": "1", "fraction": "1.0000"} in rows
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_cli_writes_stop_round_csv_and_figure(tmp_path: Path) -> None:
    batch_root = tmp_path / "batch"
    run_dir = batch_root / "runs" / "parasites-a"
    _write_graph(
        run_dir,
        instance_name="liveideabench-parasites-305",
        edit_actions=["add_support_edge", "skip", "attach_evidence"],
        commit_labels=[0, 0, 1],
    )
    _write_summary(
        run_dir,
        stop_reason="commit_at_Round3",
    )
    _write_selected_rows(
        batch_root,
        [
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-parasites-305",
                "baseline_name": "ours-eig-critic-graph-twohead",
                "run_dir": str(run_dir),
            },
        ],
    )
    output_dir = tmp_path / "paper" / "supporting"
    figure_path = tmp_path / "paper" / "figures" / "stop_round_distribution.pdf"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analysis" / "analyze_action_distribution.py"),
            "--batch-root",
            str(batch_root),
            "--category",
            "parasites",
            "--output-dir",
            str(output_dir),
            "--stop-round-figure-path",
            str(figure_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    csv_path = output_dir / "stop_round_distribution_liveideabench_parasites.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows == [{"round": "Round3", "count": "1", "fraction": "1.0000"}]
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_create_calibration_comparison_figures_use_two_columns() -> None:
    module = _load_script_module()
    left = module.build_manual_distribution(
        benchmark="all",
        category="all",
        baseline_name="ours-eig-critic-graph-twohead",
        run_count=2,
        edit_counts={"add_support_edge": 2},
        commit_counts={"continue": 2},
    )
    right = module.build_manual_distribution(
        benchmark="all",
        category="all",
        baseline_name="ours-eig-critic-graph-twohead",
        run_count=2,
        edit_counts={"propose_repair": 2},
        commit_counts={"commit": 2},
    )
    left_round = module.RoundActionDistribution(
        benchmark="all",
        category="all",
        baseline_name="ours-eig-critic-graph-twohead",
        run_count=2,
        round_edit_counts={"Round1": {"add_support_edge": 2}},
    )
    right_round = module.RoundActionDistribution(
        benchmark="all",
        category="all",
        baseline_name="ours-eig-critic-graph-twohead",
        run_count=2,
        round_edit_counts={"Round1": {"propose_repair": 2}},
    )

    overall_fig, overall_axes = module.create_overall_calibration_comparison_figure(
        left,
        right,
        left_label="No calibration",
        right_label="Calibration",
    )
    round_fig, round_axes = module.create_round_calibration_comparison_figure(
        left_round,
        right_round,
        left_label="No calibration",
        right_label="Calibration",
    )

    assert len(overall_axes) == 2
    assert len(round_axes) == 2
    assert [axis.get_title() for axis in overall_axes] == ["No calibration", "Calibration"]
    assert [axis.get_title() for axis in round_axes] == ["No calibration", "Calibration"]
    plt.close(overall_fig)
    plt.close(round_fig)


def test_rows_for_csv_include_zero_count_defined_actions() -> None:
    module = _load_script_module()
    distribution = module.build_manual_distribution(
        benchmark="liveideabench",
        category="parasites",
        baseline_name="ours-eig-critic-graph-twohead",
        run_count=2,
        edit_counts={"add_support_edge": 2},
        commit_counts={"continue": 2},
    )

    rows = module._rows_for_csv(distribution)

    assert {"action_type": "edit", "action": "skip", "count": "0", "fraction": "0.0000"} in rows
    assert {
        "action_type": "edit",
        "action": "add_dependency_edge",
        "count": "0",
        "fraction": "0.0000",
    } in rows
    assert not any(row["action"] == "freeze_branch" for row in rows)


def test_create_action_distribution_figure_lists_active_actions_only_and_has_no_footer() -> None:
    module = _load_script_module()
    distribution = module.build_manual_distribution(
        benchmark="liveideabench",
        category="parasites",
        baseline_name="ours-eig-critic-graph-twohead",
        run_count=2,
        edit_counts={"add_support_edge": 2, "attach_evidence": 1},
        commit_counts={"continue": 2},
    )

    fig, axes = module.create_action_distribution_figure(distribution)

    edit_labels = [tick.get_text() for tick in axes[0].get_xticklabels()]
    commit_labels = [tick.get_text() for tick in axes[1].get_xticklabels()]

    assert edit_labels == [
        module.ACTION_LABELS["add_support_edge"],
        module.ACTION_LABELS["attach_evidence"],
    ]
    assert commit_labels == [module.ACTION_LABELS["continue"]]
    assert axes[0].get_title() == "Edit actions"
    assert axes[1].get_title() == "Commit decisions"
    assert axes[0].get_xlabel() == ""
    assert axes[1].get_xlabel() == ""
    edit_rgb = tuple(round(value, 4) for value in axes[0].patches[0].get_facecolor()[:3])
    commit_rgb = tuple(round(value, 4) for value in axes[1].patches[0].get_facecolor()[:3])

    assert edit_rgb == (0.4784, 0.6510, 0.7608)
    assert commit_rgb == (0.7294, 0.7882, 0.8392)
    assert [text.get_text() for text in fig.texts] == []
    plt.close(fig)


def test_create_comparison_figure_uses_row_titles_without_bottom_footer() -> None:
    module = _load_script_module()
    aggregate = module.build_manual_distribution(
        benchmark="liveideabench",
        category="all",
        baseline_name="ours-eig-critic-graph-twohead",
        run_count=3,
        edit_counts={"add_support_edge": 3},
        commit_counts={"continue": 2, "commit": 1},
    )
    topic = module.build_manual_distribution(
        benchmark="liveideabench",
        category="mass spectrometry",
        baseline_name="ours-eig-critic-graph-twohead",
        run_count=2,
        edit_counts={"add_support_edge": 2},
        commit_counts={"continue": 1, "commit": 1},
    )

    fig, _ = module.create_comparison_figure(aggregate, topic)

    assert [text.get_text() for text in fig.texts] == [
        "All LiveIdeaBench topics (n=3)",
        "Largest topic: mass spectrometry (n=2)",
    ]
    plt.close(fig)


def test_create_round_signal_figure_overlays_four_signal_lines() -> None:
    module = _load_script_module()
    distribution = module.RoundSignalDistribution(
        benchmark="all",
        category="all",
        baseline_name="ours-eig-critic-graph-twohead",
        run_count=3,
        round_signal_means={
            "Round1": {
                "grounding": 0.20,
                "completeness": 0.60,
                "contradiction_load": 0.80,
                "maturity": 0.30,
            },
            "Round2": {
                "grounding": 0.50,
                "completeness": 0.70,
                "contradiction_load": 0.30,
                "maturity": 0.55,
            },
        },
        round_signal_stds={
            "Round1": {
                "grounding": 0.05,
                "completeness": 0.07,
                "contradiction_load": 0.10,
                "maturity": 0.04,
            },
            "Round2": {
                "grounding": 0.02,
                "completeness": 0.03,
                "contradiction_load": 0.05,
                "maturity": 0.06,
            },
        },
        round_signal_counts={"Round1": 3, "Round2": 3},
    )

    fig, axes = module.create_round_signal_trajectory_figure(distribution)
    fig.canvas.draw()

    assert len(axes) == 1
    axis = axes[0]
    assert axis.get_title() == ""
    assert axis.get_ylabel() == "Post-round signal"
    assert axis.get_xlabel() == "Round"
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["1", "2"]
    assert len(axis.lines) == 4
    assert [line.get_label() for line in axis.lines] == [
        "Grounding",
        "Completeness",
        "Contradiction load",
        "Maturity",
    ]
    assert [line.get_marker() for line in axis.lines] == ["o", "s", "^", "D"]
    assert [line.get_color() for line in axis.lines] == [
        module.SIGNAL_COLORS["grounding"],
        module.SIGNAL_COLORS["completeness"],
        module.SIGNAL_COLORS["contradiction_load"],
        module.SIGNAL_COLORS["maturity"],
    ]
    legend = axis.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "Grounding",
        "Completeness",
        "Contradiction load",
        "Maturity",
    ]
    plt.close(fig)


def test_create_stop_round_figure_orders_rounds_and_draws_counts() -> None:
    module = _load_script_module()
    distribution = module.StopRoundDistribution(
        benchmark="all",
        category="all",
        baseline_name="ours-eig-critic-graph-twohead",
        run_count=10,
        stop_round_counts={"Round5": 4, "Round3": 2, "Round4": 4},
    )

    fig, ax = module.create_stop_round_distribution_figure(distribution)

    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["Round3", "Round4", "Round5"]
    assert ax.get_title() == "Commit / stop round"
    assert ax.get_ylabel() == "Episodes"
    assert tuple(round(value, 4) for value in ax.patches[0].get_facecolor()[:3]) == (
        0.7294,
        0.7882,
        0.8392,
    )
    plt.close(fig)


def test_create_round_action_count_figure_uses_sparse_nonzero_bars() -> None:
    module = _load_script_module()
    distribution = module.RoundActionDistribution(
        benchmark="all",
        category="all",
        baseline_name="ours-eig-critic-graph-twohead",
        run_count=5,
        round_edit_counts={
            "Round1": {
                "add_support_edge": 3,
                "add_contradiction_edge": 2,
            },
            "Round2": {
                "propose_repair": 4,
            },
            "Round3": {
                "attach_evidence": 1,
                "propose_repair": 2,
            },
        },
    )

    fig, ax = module.create_round_action_count_figure(distribution)

    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["Round1", "Round2", "Round3"]
    assert ax.get_title() == "Per-round edit actions"
    assert ax.get_ylabel() == "Action count"
    assert len(ax.patches) == 5
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        module.ACTION_LABELS["add_support_edge"],
        module.ACTION_LABELS["attach_evidence"],
        module.ACTION_LABELS["propose_repair"],
        module.ACTION_LABELS["add_contradiction_edge"],
    ]
    plt.close(fig)


def test_cli_can_draw_figures_from_existing_csv_files(tmp_path: Path) -> None:
    action_csv = tmp_path / "action_distribution.csv"
    action_csv.write_text(
        "\n".join(
            [
                "action_type,action,count,fraction",
                "edit,add_support_edge,7,0.5000",
                "edit,attach_evidence,3,0.2143",
                "edit,propose_repair,4,0.2857",
                "commit,continue,2,0.5000",
                "commit,commit,2,0.5000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    signal_csv = tmp_path / "round_signal.csv"
    signal_csv.write_text(
        "\n".join(
            [
                "round,signal,mean,std,run_count",
                "Round1,grounding,0.3000,0.0500,4",
                "Round1,completeness,0.6500,0.0400,4",
                "Round1,contradiction_load,0.9000,0.0800,4",
                "Round1,maturity,0.3500,0.0300,4",
                "Round2,grounding,0.5200,0.0600,4",
                "Round2,completeness,0.6800,0.0300,4",
                "Round2,contradiction_load,0.2800,0.0700,4",
                "Round2,maturity,0.5600,0.0400,4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    figure_path = tmp_path / "figures" / "action_distribution_from_csv.pdf"
    signal_figure_path = tmp_path / "figures" / "round_signal_from_csv.pdf"
    stop_round_csv = tmp_path / "stop_round.csv"
    stop_round_csv.write_text(
        "\n".join(
            [
                "round,count,fraction",
                "Round3,9,0.1429",
                "Round4,26,0.4127",
                "Round5,23,0.3651",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stop_round_figure_path = tmp_path / "figures" / "stop_round_from_csv.pdf"
    round_action_csv = tmp_path / "round_action.csv"
    round_action_csv.write_text(
        "\n".join(
            [
                "round,action,count,fraction",
                "Round1,add_support_edge,3,0.6000",
                "Round1,add_contradiction_edge,2,0.4000",
                "Round2,propose_repair,4,1.0000",
                "Round3,attach_evidence,1,0.3333",
                "Round3,propose_repair,2,0.6667",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    round_action_figure_path = tmp_path / "figures" / "round_action_from_csv.pdf"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analysis" / "analyze_action_distribution.py"),
            "--action-csv-path",
            str(action_csv),
            "--round-signal-csv-path",
            str(signal_csv),
            "--stop-round-csv-path",
            str(stop_round_csv),
            "--round-action-csv-path",
            str(round_action_csv),
            "--figure-path",
            str(figure_path),
            "--signal-figure-path",
            str(signal_figure_path),
            "--stop-round-figure-path",
            str(stop_round_figure_path),
            "--round-action-figure-path",
            str(round_action_figure_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
    assert signal_figure_path.exists()
    assert signal_figure_path.stat().st_size > 0
    assert stop_round_figure_path.exists()
    assert stop_round_figure_path.stat().st_size > 0
    assert round_action_figure_path.exists()
    assert round_action_figure_path.stat().st_size > 0
