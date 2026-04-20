from __future__ import annotations

from argparse import Namespace
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from idea_graph.instances import ExperimentInstance
from scripts.run_quality_batch import (
    BenchmarkTarget,
    aggregate_rows,
    build_batch_dir,
    format_markdown_summary,
    load_targets,
    overall_aggregate_rows,
    print_progress,
    summarize_graph_usage,
)


class RunQualityBatchTests(unittest.TestCase):
    def _sample_rows(self) -> list[dict[str, object]]:
        return [
            {
                "benchmark": "AI_Idea_Bench_2025",
                "display_selector": "13",
                "baseline_name": "ours-eig",
                "overall_score": 6.0,
                "benchmark_alignment": 5.2,
                "expert_style_quality": 7.1,
                "graph_process": 9.0,
                "llm_call_count": 12,
                "total_tokens": 1200,
                "native_average_normalized_10": 6.8,
                "run_dir": "run-a",
                "run_dir_name": "run-a",
                "stop_reason": "mature_at_Round4",
                "runtime_protocol": "parallel_graph_v2",
                "executed_round_count": 4,
                "action_count": 18,
            },
            {
                "benchmark": "AI_Idea_Bench_2025",
                "display_selector": "15",
                "baseline_name": "ours-eig",
                "overall_score": 6.4,
                "benchmark_alignment": 5.5,
                "expert_style_quality": 7.3,
                "graph_process": 9.1,
                "llm_call_count": 14,
                "total_tokens": 1500,
                "native_average_normalized_10": 7.0,
                "run_dir": "run-b",
                "run_dir_name": "run-b",
                "stop_reason": "mature_at_Round5",
                "runtime_protocol": "parallel_graph_v2",
                "executed_round_count": 5,
                "action_count": 24,
            },
        ]

    def test_summarize_graph_usage_counts_external_baseline_traces(self) -> None:
        graph = type(
            "Graph",
            (),
            {
                "metadata": {
                    "external_baseline_traces": [
                        {
                            "raw_response": {
                                "usage": {
                                    "prompt_tokens": 1000,
                                    "completion_tokens": 200,
                                    "total_tokens": 1200,
                                }
                            }
                        },
                        {
                            "raw_response": {
                                "usage": {
                                    "prompt_tokens": 1500,
                                    "completion_tokens": 300,
                                    "total_tokens": 1800,
                                }
                            }
                        },
                    ]
                }
            },
        )()

        usage = summarize_graph_usage(graph)

        self.assertEqual(usage["llm_call_count"], 2)
        self.assertEqual(usage["prompt_tokens"], 2500)
        self.assertEqual(usage["completion_tokens"], 500)
        self.assertEqual(usage["total_tokens"], 3000)

    def test_summarize_graph_usage_counts_parallel_round_action_traces(self) -> None:
        graph = type(
            "Graph",
            (),
            {
                "metadata": {
                    "agent_traces": [
                        {
                            "stage": "seed_generation",
                            "role": "MechanismProposer",
                            "raw_response": {
                                "usage": {
                                    "prompt_tokens": 100,
                                    "completion_tokens": 20,
                                    "total_tokens": 120,
                                }
                            },
                        },
                        {
                            "stage": "Round1_action",
                            "role": "MechanismProposer",
                            "raw_response": {
                                "usage": {
                                    "prompt_tokens": 140,
                                    "completion_tokens": 30,
                                    "total_tokens": 170,
                                }
                            },
                        },
                    ],
                    "final_synthesis_trace": {
                        "raw_response": {
                            "usage": {
                                "prompt_tokens": 500,
                                "completion_tokens": 80,
                                "total_tokens": 580,
                            }
                        }
                    },
                }
            },
        )()

        usage = summarize_graph_usage(graph)

        self.assertEqual(usage["llm_call_count"], 3)
        self.assertEqual(usage["prompt_tokens"], 740)
        self.assertEqual(usage["completion_tokens"], 130)
        self.assertEqual(usage["total_tokens"], 870)

    def test_aggregate_rows_tracks_rounds_actions_and_protocols(self) -> None:
        rows = self._sample_rows()

        aggregate = aggregate_rows(rows)

        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["mean_executed_round_count"], 4.5)
        self.assertEqual(aggregate[0]["mean_action_count"], 21.0)
        self.assertEqual(aggregate[0]["runtime_protocols"], ["parallel_graph_v2"])

        overall = overall_aggregate_rows(rows)
        self.assertEqual(len(overall), 1)
        self.assertEqual(overall[0]["mean_executed_round_count"], 4.5)
        self.assertEqual(overall[0]["mean_action_count"], 21.0)
        self.assertEqual(overall[0]["runtime_protocols"], ["parallel_graph_v2"])

    def test_build_batch_dir_shortens_windows_unsafe_paths(self) -> None:
        output_dir = ROOT / "outputs" / "controller_eval_packets" / "two_head_dev_a_matched_critic_only_20260418_fixed_usage"
        batch_dir = build_batch_dir(
            output_dir,
            timestamp="20260418-160159",
            batch_name="critic-twohead-dev-a-matched-8-fixed-usage",
            max_full_path=240,
        )

        projected_output = batch_dir.resolve(strict=False) / "overall_aggregate_rows.csv"
        self.assertLessEqual(len(str(projected_output)), 240)

    def test_print_progress_does_not_crash_on_gbk_stdout(self) -> None:
        class GbkLikeStdout:
            encoding = "gbk"

            def __init__(self) -> None:
                self.text = ""

            def write(self, text: str) -> int:
                text.encode(self.encoding)
                self.text += text
                return len(text)

            def flush(self) -> None:
                return None

        stream = GbkLikeStdout()

        with patch("sys.stdout", stream):
            print_progress("running topic with ö")

        self.assertIn("[batch] running topic with", stream.text)
        self.assertIn("?", stream.text)

    def test_load_targets_can_use_frozen_split_registry_shard(self) -> None:
        registry = ROOT / "outputs" / "tmp-test-split-registry.jsonl"
        rows = [
            {
                "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-1728",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "ai-idea-bench-2025-1728",
                "partition_role": "paper_eval",
                "source_split": "frozen",
            },
            {
                "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-1732",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "ai-idea-bench-2025-1732",
                "partition_role": "paper_eval",
                "source_split": "frozen",
            },
            {
                "group_id": "liveideabench::liveideabench-robotics-8",
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-robotics-8",
                "partition_role": "paper_eval",
                "source_split": "frozen",
                "benchmark_keyword": "robotics",
            },
            {
                "group_id": "liveideabench::liveideabench-vision-12",
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-vision-12",
                "partition_role": "paper_eval",
                "source_split": "frozen",
                "benchmark_keyword": "vision",
            },
        ]
        registry.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        self.addCleanup(lambda: registry.unlink(missing_ok=True))

        def make_target(benchmark: str, selector: int, display_selector: str) -> BenchmarkTarget:
            return BenchmarkTarget(
                benchmark=benchmark,
                selector=selector,
                display_selector=display_selector,
                instance_name=f"{benchmark}-{selector}",
                topic_preview="topic",
                instance=ExperimentInstance(
                    name=f"{benchmark}-{selector}",
                    topic="topic",
                    literature=[],
                    source_path="test",
                    metadata={},
                ),
            )

        args = Namespace(
            ai_benchmark_root=ROOT / "data" / "benchmarks" / "ai_idea_bench_2025",
            live_benchmark_root=ROOT / "data" / "benchmarks" / "liveideabench",
            ai_indices=[],
            live_row_indices=[],
            split_registry=registry,
            partition_role="paper_eval",
            target_aiib=2,
            target_live=2,
            sampling_seed=0,
            shard_count=2,
            shard_index=1,
        )

        with (
            patch(
                "scripts.run_quality_batch.load_ai_target",
                side_effect=lambda _root, index: make_target("AI_Idea_Bench_2025", index, str(index)),
            ),
            patch(
                "scripts.run_quality_batch.load_live_target",
                side_effect=lambda _root, index: make_target("liveideabench", index, str(index)),
            ),
        ):
            targets = load_targets(args)

        self.assertEqual([target.benchmark for target in targets], ["AI_Idea_Bench_2025", "liveideabench"])
        self.assertEqual([target.selector for target in targets], [1732, 12])
        self.assertEqual(targets[0].split_row["group_id"], "AI_Idea_Bench_2025::ai-idea-bench-2025-1732")
        self.assertEqual(targets[1].split_row["group_id"], "liveideabench::liveideabench-vision-12")

    def test_format_markdown_summary_surfaces_parallel_protocol_and_affordability(self) -> None:
        rows = self._sample_rows()
        aggregate = aggregate_rows(rows)
        overall = overall_aggregate_rows(rows)
        payload = {
            "model": "qwen3-8b",
            "generated_at": "2026-04-16T16:30:00",
            "ai_indices": [13, 15],
            "live_row_indices": [],
            "method_plans": [
                {
                    "name": "ours-eig",
                    "baseline_name": "ours-eig",
                    "restarts": 1,
                    "max_rounds": 5,
                    "stop_when_mature": True,
                    "runtime_protocol": "parallel_graph_v2",
                    "rationale": "Parallel heuristic EIG teacher.",
                }
            ],
            "raw_rows": rows,
            "aggregate_rows": aggregate,
            "overall_aggregate_rows": overall,
            "findings": ["Parallel EIG is competitive."],
            "next_steps": ["Scale to broader baselines."],
        }

        markdown = format_markdown_summary(payload)

        self.assertIn("runtime_protocol=`parallel_graph_v2`", markdown)
        self.assertIn("| Benchmark | Selector | Baseline | Protocol | Overall |", markdown)
        self.assertIn("| Benchmark | Baseline | Protocols | Overall |", markdown)
        self.assertIn("| Baseline | Protocols | Overall |", markdown)
        self.assertIn("parallel_graph_v2", markdown)


if __name__ == "__main__":
    unittest.main()
