from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_quality_batch import (
    aggregate_rows,
    build_batch_dir,
    format_markdown_summary,
    overall_aggregate_rows,
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
