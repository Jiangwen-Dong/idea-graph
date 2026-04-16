from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.engine import run_experiment
from idea_graph.experiment_plans import (
    ABLATION_METHOD_PLANS,
    MAIN_METHOD_PLANS,
    prepare_instance_for_method_plan,
)
from idea_graph.instances import ExperimentInstance


class ExperimentPlanTests(unittest.TestCase):
    def _instance(self) -> ExperimentInstance:
        return ExperimentInstance(
            name="ai-idea-bench-2025-13",
            topic="The topic of this paper is 3D language field modeling for open-vocabulary scene understanding.",
            literature=[
                "LERF: Language Embedded Radiance Fields for Open-Vocabulary 3D Scene Understanding",
                "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
            ],
            source_path="test",
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "benchmark_index": 13,
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "LERF: Language Embedded Radiance Fields for Open-Vocabulary 3D Scene Understanding",
                            "abstract": "LERF studies language-embedded radiance fields for open-vocabulary querying.",
                        },
                        {
                            "resolved_title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
                            "method": "3D Gaussian Splatting enables efficient radiance field rendering.",
                        },
                    ],
                },
                "literature_grounding": {
                    "source": "unit_test",
                    "reference_titles": [
                        "LERF: Language Embedded Radiance Fields for Open-Vocabulary 3D Scene Understanding",
                        "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
                    ],
                    "design_highlights": ["language-aligned 3D querying"],
                    "dataset_items": ["LERF dataset"],
                    "metric_items": ["localization accuracy"],
                    "existing_methods_summary": "Existing methods couple radiance fields with language embeddings.",
                    "experiment_plan_summary": "Evaluate querying quality and localization precision.",
                },
            },
        )

    def test_main_method_plan_preserves_reference_grounding(self) -> None:
        instance = prepare_instance_for_method_plan(
            self._instance(),
            plan=MAIN_METHOD_PLANS["ours-eig"],
        )

        packet = instance.metadata["benchmark_input_packet"]
        self.assertEqual(instance.metadata["method_name"], "ours-eig")
        self.assertEqual(instance.metadata["runner_baseline_name"], "ours-eig")
        self.assertGreater(len(packet["reference_packet"]), 0)
        self.assertGreater(
            len(instance.metadata["paper_grounding"]["reference_paper_snippets"]),
            0,
        )
        self.assertEqual(instance.literature, self._instance().literature)
        self.assertEqual(instance.metadata["runtime_protocol"], "parallel_graph_v2")
        self.assertEqual(
            instance.metadata["idea_graph_protocol_variant"],
            "eig_parallel_v2_heuristic",
        )

    def test_main_method_plan_includes_exact_ai_researcher(self) -> None:
        self.assertIn("ai-researcher", MAIN_METHOD_PLANS)

        instance = prepare_instance_for_method_plan(
            self._instance(),
            plan=MAIN_METHOD_PLANS["ai-researcher"],
        )

        self.assertEqual(instance.metadata["method_name"], "ai-researcher")
        self.assertEqual(instance.metadata["runner_baseline_name"], "ai-researcher")
        self.assertEqual(instance.metadata["baseline_name"], "ai-researcher")
        self.assertGreater(len(instance.metadata["benchmark_input_packet"]["reference_packet"]), 0)
        self.assertEqual(instance.literature, self._instance().literature)

    def test_no_reference_grounding_plan_strips_reference_packet_and_grounding(self) -> None:
        instance = prepare_instance_for_method_plan(
            self._instance(),
            plan=ABLATION_METHOD_PLANS["ours-no-reference-grounding"],
        )

        packet = instance.metadata["benchmark_input_packet"]
        self.assertEqual(packet["reference_packet"], [])
        self.assertIn(
            "Reference packet intentionally removed for this protocol ablation.",
            packet["constraints"],
        )
        self.assertEqual(instance.metadata["paper_grounding"]["reference_paper_snippets"], [])
        self.assertEqual(instance.metadata["reference_titles"], [])
        self.assertEqual(instance.metadata["literature_grounding"]["source"], "protocol_ablation")
        self.assertEqual(instance.metadata["literature_grounding"]["reference_titles"], [])
        self.assertEqual(instance.metadata["method_name"], "ours-no-reference-grounding")
        self.assertEqual(instance.metadata["runner_baseline_name"], "ours-eig")
        self.assertEqual(instance.literature, [instance.topic])

    def test_run_experiment_calls_coverage_safeguard_by_default(self) -> None:
        with patch("idea_graph.engine.ensure_core_node_coverage") as mocked:
            run_experiment(
                topic="graph-based scientific ideation",
                literature=["paper a", "paper b", "paper c", "paper d"],
                max_rounds=1,
                stop_when_mature=False,
            )

        mocked.assert_called_once()

    def test_run_experiment_can_disable_coverage_safeguard(self) -> None:
        with patch("idea_graph.engine.ensure_core_node_coverage") as mocked:
            graph = run_experiment(
                topic="graph-based scientific ideation",
                literature=["paper a", "paper b", "paper c", "paper d"],
                metadata={"idea_graph_disable_core_node_coverage": True},
                max_rounds=1,
                stop_when_mature=False,
            )

        mocked.assert_not_called()
        self.assertTrue(
            any(
                item.get("stage") == "coverage_safeguard"
                and not bool((item.get("details") or {}).get("enabled", True))
                for item in graph.metadata.get("progress_log", [])
                if isinstance(item, dict)
            )
        )


if __name__ == "__main__":
    unittest.main()
