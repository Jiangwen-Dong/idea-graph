from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.fs_utils import _windows_safe_path, write_text_file
from idea_graph.trajectory_dataset import (
    PricingConfig,
    aggregate_dataset_profile,
    build_run_manifest_row,
    build_transition_rows,
    discover_run_dirs,
    extract_trace_stats,
    reconstruct_state_before_action,
)


class TrajectoryDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(_windows_safe_path(self.tmp_dir), ignore_errors=True)

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        write_text_file(path, json.dumps(payload, indent=2, ensure_ascii=False))

    def _summary_payload(self, *, benchmark: str = "AI_Idea_Bench_2025") -> dict[str, object]:
        return {
            "instance_name": "toy-instance",
            "topic": "Toy topic",
            "node_count": 3,
            "edge_count": 2,
            "action_count": 3,
            "executed_round_count": 2,
            "stopped_early": False,
            "matured_at_round": None,
            "action_source_counts": {"llm": 3},
            "idea_evaluation": {
                "overall_score": 6.4,
                "category_scores": {
                    "benchmark_alignment": 4.8,
                    "expert_style_quality": 7.1,
                    "graph_process": 7.5,
                },
            },
            "benchmark_native_evaluation": {
                "benchmark": benchmark,
                "summary": {"available_average_normalized_10": 6.9},
            },
        }

    def _graph_payload(self, *, baseline_name: str = "ours-eig") -> dict[str, object]:
        return {
            "topic": "Toy topic",
            "metadata": {
                "benchmark": "AI_Idea_Bench_2025",
                "benchmark_index": 7,
                "baseline_name": baseline_name,
                "agent_traces": [
                    {
                        "stage": "seed_generation",
                        "role": "TaskFramer",
                        "raw_response": {
                            "usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 25,
                                "total_tokens": 125,
                            },
                            "created": 1000,
                        },
                    },
                    {
                        "stage": "round_action",
                        "role": "LiteratureGrounder",
                        "raw_response": {
                            "usage": {
                                "prompt_tokens": 80,
                                "completion_tokens": 20,
                                "total_tokens": 100,
                            },
                            "created": 1010,
                        },
                    },
                ],
                "utility_controller_overrides": [
                    {
                        "round": "Round1",
                        "role": "TaskFramer",
                        "llm_kind": "add_support_edge",
                        "llm_predicted_gain": 0.1,
                        "deterministic_kind": "add_dependency_edge",
                        "deterministic_predicted_gain": 0.3,
                    }
                ],
                "final_synthesis_trace": {
                    "raw_response": {
                        "usage": {
                            "prompt_tokens": 70,
                            "completion_tokens": 30,
                            "total_tokens": 100,
                        },
                        "created": 1030,
                    }
                },
            },
            "nodes": {
                "N1": {
                    "id": "N1",
                    "type": "Problem",
                    "text": "Problem node",
                    "role": "TaskFramer",
                    "branch_id": "B1",
                    "confidence": 0.7,
                    "created_at": "1970-01-01 00:16:41+00:00",
                },
                "N2": {
                    "id": "N2",
                    "type": "Method",
                    "text": "Method node",
                    "role": "MechanismProposer",
                    "branch_id": "B1",
                    "confidence": 0.7,
                    "created_at": "1970-01-01 00:16:42+00:00",
                },
                "N3": {
                    "id": "N3",
                    "type": "Risk",
                    "text": "Risk node",
                    "role": "FeasibilityCritic",
                    "branch_id": "B2",
                    "confidence": 0.6,
                    "created_at": "1970-01-01 00:16:45+00:00",
                },
            },
            "edges": [
                {
                    "id": "E1",
                    "source_id": "N1",
                    "relation": "supports",
                    "target_id": "N2",
                    "role": "TaskFramer",
                    "branch_id": "B1",
                    "resolved": False,
                    "created_at": "1970-01-01 00:16:43+00:00",
                },
                {
                    "id": "E2",
                    "source_id": "N3",
                    "relation": "contradicts",
                    "target_id": "N2",
                    "role": "FeasibilityCritic",
                    "branch_id": "B2",
                    "resolved": True,
                    "created_at": "1970-01-01 00:16:46+00:00",
                },
            ],
            "actions": [
                {
                    "id": "A1",
                    "round_name": "Round1",
                    "role": "TaskFramer",
                    "kind": "add_support_edge",
                    "target_ids": ["N1", "N2"],
                    "payload": {"branch_id": "B1"},
                    "rationale": "Support problem with method.",
                    "source": "llm",
                    "timestamp": "1970-01-01 00:16:44+00:00",
                },
                {
                    "id": "A2",
                    "round_name": "Round2",
                    "role": "FeasibilityCritic",
                    "kind": "propose_repair",
                    "target_ids": ["N2"],
                    "payload": {"branch_id": "B2"},
                    "rationale": "Repair contradiction on the target method.",
                    "source": "llm",
                    "timestamp": "1970-01-01 00:16:47+00:00",
                },
                {
                    "id": "A3",
                    "round_name": "Round2",
                    "role": "LiteratureGrounder",
                    "kind": "attach_evidence",
                    "target_ids": ["N2"],
                    "payload": {"branch_id": "B1", "evidence": "Toy evidence"},
                    "rationale": "Attach evidence after repair.",
                    "source": "llm",
                    "timestamp": "1970-01-01 00:16:48+00:00",
                },
            ],
            "round_summaries": [
                [
                    "Round1",
                    {
                        "support_coverage": 0.5,
                        "unresolved_contradiction_ratio": 1.0,
                        "utility": 4.5,
                        "utility_stable": False,
                        "completeness": False,
                        "is_mature": False,
                    },
                ],
                [
                    "Round2",
                    {
                        "support_coverage": 1.0,
                        "unresolved_contradiction_ratio": 0.0,
                        "utility": 7.5,
                        "utility_stable": True,
                        "completeness": True,
                        "is_mature": True,
                    },
                ],
            ],
        }

    def test_discover_run_dirs_requires_summary_and_graph(self) -> None:
        root = self.tmp_dir / "runs"
        good = root / "good" / "nested"
        only_summary = root / "only_summary"
        only_graph = root / "only_graph"

        self._write_json(good / "summary.json", self._summary_payload())
        self._write_json(good / "graph.json", self._graph_payload())
        self._write_json(only_summary / "summary.json", self._summary_payload())
        self._write_json(only_graph / "graph.json", self._graph_payload())

        discovered = discover_run_dirs([root])

        self.assertEqual(discovered, [good.resolve()])

    def test_extract_trace_stats_and_costs(self) -> None:
        graph_payload = self._graph_payload()

        stats = extract_trace_stats(
            graph_payload,
            pricing=PricingConfig(
                prompt_price_per_1m_tokens=2.0,
                completion_price_per_1m_tokens=4.0,
            ),
        )

        self.assertEqual(stats["llm_call_count"], 3)
        self.assertEqual(stats["prompt_tokens"], 250)
        self.assertEqual(stats["completion_tokens"], 75)
        self.assertEqual(stats["total_tokens"], 325)
        self.assertAlmostEqual(stats["estimated_cost"], (250 * 2.0 + 75 * 4.0) / 1_000_000)
        self.assertEqual(stats["wall_clock_seconds"], 30.0)

    def test_reconstruct_state_before_action_filters_nodes_and_edges_by_timestamp(self) -> None:
        graph_payload = self._graph_payload()

        snapshot = reconstruct_state_before_action(graph_payload, action_index=0)

        self.assertEqual(set(snapshot["nodes"].keys()), {"N1", "N2"})
        self.assertEqual([edge["id"] for edge in snapshot["edges"]], ["E1"])
        self.assertEqual(snapshot["node_count"], 2)
        self.assertEqual(snapshot["edge_count"], 1)
        self.assertEqual(snapshot["support_edge_count"], 1)
        self.assertEqual(snapshot["contradiction_count"], 0)

    def test_reconstruct_state_infers_contradiction_resolution_from_later_repair(self) -> None:
        graph_payload = self._graph_payload()

        before_repair = reconstruct_state_before_action(graph_payload, action_index=1)
        after_repair = reconstruct_state_before_action(graph_payload, action_index=2)

        contradiction_before = next(edge for edge in before_repair["edges"] if edge["id"] == "E2")
        contradiction_after = next(edge for edge in after_repair["edges"] if edge["id"] == "E2")

        self.assertFalse(contradiction_before["resolved"])
        self.assertTrue(contradiction_after["resolved"])
        self.assertEqual(before_repair["contradiction_count"], 1)
        self.assertEqual(after_repair["contradiction_count"], 0)

    def test_build_run_manifest_row_handles_eig_and_non_eig_runs(self) -> None:
        eig_row = build_run_manifest_row(
            self.tmp_dir / "eig_run",
            self._summary_payload(),
            self._graph_payload(baseline_name="ours-eig"),
            pricing=PricingConfig(
                prompt_price_per_1m_tokens=2.0,
                completion_price_per_1m_tokens=4.0,
            ),
        )
        baseline_row = build_run_manifest_row(
            self.tmp_dir / "direct_run",
            {
                **self._summary_payload(benchmark="liveideabench"),
                "action_count": 0,
                "benchmark_native_evaluation": {
                    "benchmark": "liveideabench",
                    "summary": {"available_average_normalized_10": 7.2},
                },
            },
            {
                **self._graph_payload(baseline_name="direct"),
                "metadata": {"benchmark": "liveideabench", "baseline_name": "direct"},
                "actions": [],
            },
        )

        self.assertTrue(eig_row["is_eig_run"])
        self.assertEqual(eig_row["trace_total_tokens"], 325)
        self.assertAlmostEqual(eig_row["final_native_average"], 6.9)
        self.assertTrue(eig_row["has_override_trace"])
        self.assertFalse(baseline_row["is_eig_run"])
        self.assertEqual(baseline_row["action_count"], 0)
        self.assertAlmostEqual(baseline_row["final_native_average"], 7.2)

    def test_build_run_manifest_row_preserves_full_local_and_native_label_payloads(self) -> None:
        summary_payload = {
            **self._summary_payload(),
            "idea_evaluation": {
                "overall_score": 6.4,
                "category_scores": {
                    "benchmark_alignment": 4.8,
                    "expert_style_quality": 7.1,
                    "graph_process": 7.5,
                },
            },
            "benchmark_native_evaluation": {
                "benchmark": "AI_Idea_Bench_2025",
                "metrics": [
                    {"key": "i2i_motivation", "score": 4.0, "max_score": 5.0, "available": True},
                    {"key": "fps", "score": 4.0, "max_score": 5.0, "available": True},
                ],
                "summary": {"available_average_normalized_10": 6.9},
            },
        }
        row = build_run_manifest_row(self.tmp_dir / "eig_run", summary_payload, self._graph_payload())
        self.assertEqual(row["local_category_scores"]["graph_process"], 7.5)
        self.assertEqual(row["native_metric_map"]["i2i_motivation"]["score"], 4.0)
        self.assertEqual(row["native_metric_map"]["fps"]["max_score"], 5.0)

    def test_build_run_manifest_row_preserves_full_local_and_native_label_payloads(self) -> None:
        summary_payload = {
            **self._summary_payload(),
            "idea_evaluation": {
                "overall_score": 6.4,
                "category_scores": {
                    "benchmark_alignment": 4.8,
                    "expert_style_quality": 7.1,
                    "graph_process": 7.5,
                },
            },
            "benchmark_native_evaluation": {
                "benchmark": "AI_Idea_Bench_2025",
                "metrics": [
                    {"key": "i2i_motivation", "score": 4.0, "max_score": 5.0, "available": True},
                    {"key": "fps", "score": 4.0, "max_score": 5.0, "available": True},
                ],
                "summary": {"available_average_normalized_10": 6.9},
            },
        }
        row = build_run_manifest_row(self.tmp_dir / "eig_run", summary_payload, self._graph_payload())
        self.assertEqual(row["local_category_scores"]["graph_process"], 7.5)
        self.assertEqual(row["native_metric_map"]["i2i_motivation"]["score"], 4.0)
        self.assertEqual(row["native_metric_map"]["fps"]["max_score"], 5.0)

    def test_build_transition_rows_only_exports_action_runs(self) -> None:
        snapshot_dir = self.tmp_dir / "state_snapshots"
        eig_rows = build_transition_rows(
            self.tmp_dir / "eig_run",
            self._summary_payload(),
            self._graph_payload(),
            snapshot_dir=snapshot_dir,
        )
        baseline_rows = build_transition_rows(
            self.tmp_dir / "direct_run",
            {**self._summary_payload(), "action_count": 0},
            {
                **self._graph_payload(baseline_name="direct"),
                "metadata": {"benchmark": "AI_Idea_Bench_2025", "baseline_name": "direct"},
                "actions": [],
            },
            snapshot_dir=snapshot_dir,
        )

        self.assertEqual(len(eig_rows), 3)
        self.assertEqual(baseline_rows, [])
        self.assertEqual(eig_rows[0]["selected_action_kind"], "add_support_edge")
        self.assertEqual(eig_rows[1]["selected_action_kind"], "propose_repair")
        self.assertEqual(len(list(snapshot_dir.glob("*.json"))), 3)

    def test_build_dataset_profile_aggregates_counts(self) -> None:
        manifest_rows = [
            {
                "benchmark": "AI_Idea_Bench_2025",
                "baseline_name": "ours-eig",
                "is_eig_run": True,
                "trace_prompt_tokens": 250,
                "trace_completion_tokens": 75,
                "trace_total_tokens": 325,
                "estimated_cost": 0.001,
                "has_agent_traces": True,
                "has_final_synthesis_trace": True,
                "has_override_trace": True,
                "has_local_eval": True,
                "has_native_eval": True,
                "action_count": 3,
            },
            {
                "benchmark": "liveideabench",
                "baseline_name": "direct",
                "is_eig_run": False,
                "trace_prompt_tokens": 0,
                "trace_completion_tokens": 0,
                "trace_total_tokens": 0,
                "estimated_cost": None,
                "has_agent_traces": False,
                "has_final_synthesis_trace": False,
                "has_override_trace": False,
                "has_local_eval": True,
                "has_native_eval": False,
                "action_count": 0,
            },
        ]
        transition_rows = [{"step_index": 0}, {"step_index": 1}, {"step_index": 2}]

        profile = aggregate_dataset_profile(manifest_rows, transition_rows)

        self.assertEqual(profile["run_count"], 2)
        self.assertEqual(profile["usable_eig_run_count"], 1)
        self.assertEqual(profile["transition_count"], 3)
        self.assertEqual(profile["benchmark_counts"]["AI_Idea_Bench_2025"], 1)
        self.assertEqual(profile["baseline_counts"]["ours-eig"], 1)
        self.assertAlmostEqual(profile["trace_coverage"]["agent_traces_fraction"], 0.5)
        self.assertEqual(profile["token_usage"]["total_tokens"], 325)
        self.assertAlmostEqual(profile["token_usage"]["mean_tokens_per_transition"], 325 / 3)


if __name__ == "__main__":
    unittest.main()
