from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.fs_utils import write_text_file


class ControllerEvalRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        write_text_file(path, text)

    def _make_run_dir(
        self,
        *,
        name: str,
        baseline_name: str,
        benchmark: str,
        instance_name: str,
        partition_role: str,
        native_score: float,
        stop_reason: str,
        executed_round_count: int,
        runtime_log: list[dict[str, object]] | None = None,
    ) -> Path:
        run_dir = self.tmp_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "instance_name": instance_name,
            "executed_round_count": executed_round_count,
            "stop_reason": stop_reason,
            "benchmark_native_evaluation": {
                "benchmark": benchmark,
                "summary": {
                    "available_average_normalized_10": native_score,
                },
            },
        }
        graph_payload = {
            "topic": instance_name,
            "nodes": {},
            "edges": [],
            "branches": {},
            "actions": [
                {
                    "id": f"{name}-action-{index}",
                    "round_name": f"Round{index + 1}",
                    "role": "MechanismProposer",
                    "kind": "add_support_edge",
                    "target_ids": [],
                    "payload": {},
                    "rationale": "",
                }
                for index in range(executed_round_count)
            ],
            "metadata": {
                "baseline_name": baseline_name,
                "instance_name": instance_name,
                "runtime_controller_log": runtime_log or [],
                "stop_reason": stop_reason,
                "executed_round_count": executed_round_count,
                "benchmark": benchmark,
                "partition_role": partition_role,
            },
        }
        write_text_file(run_dir / "summary.json", json.dumps(summary_payload, indent=2))
        write_text_file(run_dir / "graph.json", json.dumps(graph_payload, indent=2))
        return run_dir

    def test_load_packet_rows_supports_partition_role_filter(self) -> None:
        manifest_path = self.tmp_dir / "packet.jsonl"
        self._write_jsonl(
            manifest_path,
            [
                {
                    "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-10",
                    "benchmark": "AI_Idea_Bench_2025",
                    "instance_name": "ai-idea-bench-2025-10",
                    "partition_role": "critic_train",
                    "source_split": "train",
                    "benchmark_index": 10,
                },
                {
                    "group_id": "liveideabench::liveideabench-galaxies-163",
                    "benchmark": "liveideabench",
                    "instance_name": "liveideabench-galaxies-163",
                    "partition_role": "critic_dev",
                    "source_split": "validation",
                    "row_index": 163,
                    "benchmark_keyword": "galaxies",
                },
            ],
        )

        from idea_graph.controller_eval_runtime import load_packet_rows

        filtered = load_packet_rows(manifest_path, partition_role_filter="critic_dev")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["benchmark"], "liveideabench")
        self.assertEqual(filtered[0]["row_index"], 163)
        self.assertEqual(filtered[0]["benchmark_keyword"], "galaxies")

    def test_apply_runtime_controller_overrides_can_set_calibration_path(self) -> None:
        from idea_graph.controller_eval_runtime import apply_runtime_controller_overrides
        from idea_graph.instances import ExperimentInstance

        instance = ExperimentInstance(
            name="demo",
            topic="topic",
            literature=["paper"],
            source_path="demo.json",
            metadata={"runtime_controller_kind": "relation_graph_two_head_critic"},
        )

        updated = apply_runtime_controller_overrides(
            instance,
            runtime_controller_calibration_path=self.tmp_dir / "custom_calibration.json",
            disable_runtime_calibration=False,
        )

        self.assertEqual(
            updated.metadata["runtime_controller_calibration_path"],
            str((self.tmp_dir / "custom_calibration.json").resolve()),
        )
        self.assertNotIn("runtime_controller_disable_calibration", updated.metadata)

    def test_apply_runtime_controller_overrides_can_disable_calibration(self) -> None:
        from idea_graph.controller_eval_runtime import apply_runtime_controller_overrides
        from idea_graph.instances import ExperimentInstance

        instance = ExperimentInstance(
            name="demo",
            topic="topic",
            literature=["paper"],
            source_path="demo.json",
            metadata={
                "runtime_controller_kind": "relation_graph_two_head_critic",
                "runtime_controller_calibration_path": str(self.tmp_dir / "custom_calibration.json"),
            },
        )

        updated = apply_runtime_controller_overrides(
            instance,
            runtime_controller_calibration_path=self.tmp_dir / "ignored.json",
            disable_runtime_calibration=True,
        )

        self.assertTrue(updated.metadata["runtime_controller_disable_calibration"])
        self.assertNotIn("runtime_controller_calibration_path", updated.metadata)

    def test_apply_runtime_controller_overrides_resets_preapplied_calibration(self) -> None:
        from idea_graph.controller_eval_runtime import apply_runtime_controller_overrides
        from idea_graph.instances import ExperimentInstance

        instance = ExperimentInstance(
            name="demo",
            topic="topic",
            literature=["paper"],
            source_path="demo.json",
            metadata={
                "runtime_controller_kind": "relation_graph_two_head_critic",
                "runtime_controller_tau_override": 0.13,
                "runtime_controller_tau_commit": 0.08,
                "runtime_controller_gamma_commit": 0.74,
                "runtime_controller_min_commit_round": 4,
                "runtime_controller_guard_support_threshold": 0.75,
                "runtime_controller_calibration_path": str(self.tmp_dir / "model-default.json"),
                "runtime_controller_calibration_source": "model_dir_default",
                "runtime_controller_calibration_version": "joint_controller_calibration_v1",
            },
        )

        updated = apply_runtime_controller_overrides(
            instance,
            runtime_controller_calibration_path=None,
            disable_runtime_calibration=True,
        )

        self.assertTrue(updated.metadata["runtime_controller_disable_calibration"])
        self.assertEqual(updated.metadata["runtime_controller_tau_override"], 0.05)
        self.assertEqual(updated.metadata["runtime_controller_tau_commit"], 0.08)
        self.assertEqual(updated.metadata["runtime_controller_gamma_commit"], 0.60)
        self.assertEqual(updated.metadata["runtime_controller_min_commit_round"], 2)
        self.assertEqual(updated.metadata["runtime_controller_guard_support_threshold"], 0.66)
        self.assertNotIn("runtime_controller_calibration_path", updated.metadata)
        self.assertNotIn("runtime_controller_calibration_source", updated.metadata)
        self.assertNotIn("runtime_controller_calibration_version", updated.metadata)

    def test_packet_row_to_benchmark_args_handles_aiib_and_liveideabench(self) -> None:
        from idea_graph.controller_eval_runtime import packet_row_to_benchmark_args

        aiib_args = packet_row_to_benchmark_args(
            {
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "ai-idea-bench-2025-110",
                "benchmark_index": 110,
            }
        )
        self.assertEqual(
            aiib_args,
            {
                "benchmark": "ai_idea_bench_2025",
                "benchmark_index": 110,
            },
        )

        live_args = packet_row_to_benchmark_args(
            {
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-global positioning system-191",
                "row_index": 191,
                "benchmark_keyword": "global positioning system",
            }
        )
        self.assertEqual(
            live_args,
            {
                "benchmark": "liveideabench",
                "benchmark_index": 191,
                "benchmark_keyword": "global positioning system",
            },
        )

    def test_summarize_packet_runs_reports_primary_secondary_and_diagnostic_views(self) -> None:
        from idea_graph.controller_eval_runtime import summarize_packet_runs

        dev_ours = self._make_run_dir(
            name="dev-ours",
            baseline_name="ours-eig",
            benchmark="AI_Idea_Bench_2025",
            instance_name="ai-idea-bench-2025-110",
            partition_role="critic_dev",
            native_score=7.0,
            stop_reason="max_rounds_reached",
            executed_round_count=5,
        )
        dev_graph = self._make_run_dir(
            name="dev-graph",
            baseline_name="ours-eig-critic-graph",
            benchmark="AI_Idea_Bench_2025",
            instance_name="ai-idea-bench-2025-110",
            partition_role="critic_dev",
            native_score=8.0,
            stop_reason="mature_at_Round4",
            executed_round_count=4,
            runtime_log=[
                {
                    "round": "Round1",
                    "selected_kind": "add_support_edge",
                    "selected_source": "critic",
                    "used_heuristic_fallback": False,
                },
                {
                    "round": "Round2",
                    "selected_kind": "attach_evidence",
                    "selected_source": "heuristic",
                    "used_heuristic_fallback": True,
                },
            ],
        )
        train_ours = self._make_run_dir(
            name="train-ours",
            baseline_name="ours-eig",
            benchmark="liveideabench",
            instance_name="liveideabench-galaxies-163",
            partition_role="critic_train",
            native_score=6.0,
            stop_reason="max_rounds_reached",
            executed_round_count=5,
        )
        train_graph = self._make_run_dir(
            name="train-graph",
            baseline_name="ours-eig-critic-graph",
            benchmark="liveideabench",
            instance_name="liveideabench-galaxies-163",
            partition_role="critic_train",
            native_score=5.0,
            stop_reason="mature_at_Round3",
            executed_round_count=3,
            runtime_log=[
                {
                    "round": "Round1",
                    "selected_kind": "propose_repair",
                    "selected_source": "critic",
                    "used_heuristic_fallback": False,
                }
            ],
        )

        rows = [
            {
                "baseline_name": "ours-eig",
                "paper_baseline_name": "ours-eig",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "ai-idea-bench-2025-110",
                "partition_role": "critic_dev",
                "run_dir": str(dev_ours),
            },
            {
                "baseline_name": "ours-eig-critic-graph",
                "paper_baseline_name": "ours-eig-graph-critic",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "ai-idea-bench-2025-110",
                "partition_role": "critic_dev",
                "run_dir": str(dev_graph),
            },
            {
                "baseline_name": "ours-eig",
                "paper_baseline_name": "ours-eig",
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-galaxies-163",
                "partition_role": "critic_train",
                "run_dir": str(train_ours),
            },
            {
                "baseline_name": "ours-eig-critic-graph",
                "paper_baseline_name": "ours-eig-graph-critic",
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-galaxies-163",
                "partition_role": "critic_train",
                "run_dir": str(train_graph),
            },
        ]

        summary = summarize_packet_runs(rows)
        primary = summary["readouts"]["critic_dev"]
        pooled = summary["readouts"]["pooled"]
        diagnostic = summary["readouts"]["critic_train"]

        self.assertEqual(primary["instance_count"], 1)
        self.assertEqual(pooled["instance_count"], 2)
        self.assertEqual(diagnostic["instance_count"], 1)
        self.assertEqual(primary["baseline_metrics"]["ours-eig"]["mean_score"], 7.0)
        self.assertEqual(
            primary["baseline_metrics"]["ours-eig-graph-critic"]["mean_score"],
            8.0,
        )
        self.assertEqual(
            primary["paired_against_ours_eig"]["ours-eig-graph-critic"]["mean_delta"],
            1.0,
        )
        self.assertEqual(
            primary["paired_against_ours_eig"]["ours-eig-graph-critic"]["win_rate"],
            1.0,
        )
        self.assertEqual(
            summary["controller_trace_summary"]["ours-eig-graph-critic"]["selected_source_counts"][
                "critic"
            ],
            2,
        )
        self.assertEqual(
            summary["controller_trace_summary"]["ours-eig-graph-critic"]["fallback_count"],
            1,
        )

    def test_summarizer_cli_writes_summary_files(self) -> None:
        run_dir = self._make_run_dir(
            name="dev-graph-cli",
            baseline_name="ours-eig-critic-graph",
            benchmark="AI_Idea_Bench_2025",
            instance_name="ai-idea-bench-2025-110",
            partition_role="critic_dev",
            native_score=8.0,
            stop_reason="mature_at_Round4",
            executed_round_count=4,
            runtime_log=[
                {
                    "round": "Round1",
                    "selected_kind": "add_support_edge",
                    "selected_source": "critic",
                    "used_heuristic_fallback": False,
                }
            ],
        )
        manifest_path = self.tmp_dir / "run_manifest.jsonl"
        self._write_jsonl(
            manifest_path,
            [
                {
                    "baseline_name": "ours-eig-critic-graph",
                    "paper_baseline_name": "ours-eig-graph-critic",
                    "benchmark": "AI_Idea_Bench_2025",
                    "instance_name": "ai-idea-bench-2025-110",
                    "partition_role": "critic_dev",
                    "run_dir": str(run_dir),
                }
            ],
        )

        output_root = self.tmp_dir / "summary_output"
        script_path = ROOT / "scripts" / "summarize_controller_eval_packet.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--input-root",
                str(self.tmp_dir),
                "--write-root",
                str(output_root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertTrue((output_root / "paired_summary.json").exists())
        self.assertTrue((output_root / "paired_summary.md").exists())
        self.assertTrue((output_root / "controller_trace_summary.json").exists())

    def test_runner_cli_dry_run_writes_normalized_manifest(self) -> None:
        packet_path = self.tmp_dir / "packet.jsonl"
        self._write_jsonl(
            packet_path,
            [
                {
                    "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-110",
                    "benchmark": "AI_Idea_Bench_2025",
                    "instance_name": "ai-idea-bench-2025-110",
                    "partition_role": "critic_dev",
                    "source_split": "validation",
                    "benchmark_index": 110,
                }
            ],
        )

        output_root = self.tmp_dir / "runner_output"
        script_path = ROOT / "scripts" / "run_controller_eval_packet.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--packet-manifest",
                str(packet_path),
                "--baselines",
                "ours-eig",
                "ours-eig-critic-graph",
                "--output-root",
                str(output_root),
                "--max-rounds",
                "5",
                "--dry-run",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        manifest_rows = [
            json.loads(line)
            for line in (output_root / "run_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(manifest_rows), 2)
        self.assertEqual(manifest_rows[0]["benchmark"], "AI_Idea_Bench_2025")
        self.assertEqual(manifest_rows[0]["instance_name"], "ai-idea-bench-2025-110")
        self.assertEqual(manifest_rows[0]["partition_role"], "critic_dev")
        self.assertEqual(manifest_rows[1]["paper_baseline_name"], "ours-eig-graph-critic")
        self.assertTrue(manifest_rows[1]["dry_run"])


if __name__ == "__main__":
    unittest.main()
