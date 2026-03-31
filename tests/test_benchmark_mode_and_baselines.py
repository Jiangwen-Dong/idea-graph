from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.baselines import BASELINE_SPECS, attach_baseline_metadata, run_baseline_experiment
from idea_graph.external_baselines import load_external_baseline_config
from idea_graph.instances import ExperimentInstance


class BenchmarkModeAndBaselineTests(unittest.TestCase):
    def _ai_idea_bench_instance(self) -> ExperimentInstance:
        return ExperimentInstance(
            name="ai-idea-bench-2025-15",
            topic="The topic of this paper is estimating scaled relative poses in panoramic images.",
            literature=[
                "Unsupervised learning of depth and ego-motion from video",
                "Posenet: A convolutional network for real-time 6-dof camera relocalization",
            ],
            source_path="test",
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "benchmark_index": 15,
                "target_paper": "PanoPose",
                "motivation": "gold motivation",
                "method_summary": "gold method",
                "paper_grounding": {
                    "target_paper_snippet": {"abstract": "gold abstract"},
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "Paper A",
                            "abstract": "Paper A studies self-supervised depth estimation for panoramic scenes.",
                        },
                        {
                            "resolved_title": "Paper B",
                            "method": "Paper B uses a pose network with explicit geometric constraints.",
                        },
                    ],
                },
            },
        )

    def _liveideabench_instance(self) -> ExperimentInstance:
        return ExperimentInstance(
            name="liveideabench-meteorology-0",
            topic="Ideation topic keyword: meteorology",
            literature=["Benchmark keyword: meteorology"],
            source_path="test",
            metadata={
                "benchmark": "liveideabench",
                "keyword": "meteorology",
                "idea": "held-out benchmark idea",
                "full_response": "hidden response",
            },
        )

    def test_attach_baseline_metadata_builds_safe_ai_idea_bench_packet(self) -> None:
        instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-delayed-consensus",
            io_mode="auto",
        )

        packet = instance.metadata["benchmark_input_packet"]
        self.assertEqual(instance.metadata["io_mode"], "benchmark")
        self.assertEqual(packet["benchmark"], "AI_Idea_Bench_2025")
        self.assertEqual(len(packet["reference_packet"]), 2)
        self.assertNotIn("target_paper", packet)
        self.assertNotIn("gold", str(packet).lower())
        self.assertIn("output_schema", packet)

    def test_attach_baseline_metadata_builds_keyword_only_liveideabench_packet(self) -> None:
        instance = attach_baseline_metadata(
            self._liveideabench_instance(),
            baseline_name="direct",
            io_mode="auto",
        )

        packet = instance.metadata["benchmark_input_packet"]
        self.assertEqual(instance.metadata["io_mode"], "benchmark")
        self.assertEqual(packet["benchmark"], "liveideabench")
        self.assertEqual(packet["keyword"], "meteorology")
        self.assertEqual(packet["reference_packet"], [])
        self.assertNotIn("held-out benchmark idea", str(packet))

    def test_direct_baseline_runs_with_zero_rounds(self) -> None:
        instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="direct",
            io_mode="auto",
        )

        graph = run_baseline_experiment(instance, baseline_name="direct")

        self.assertIsNotNone(graph.final_proposal)
        self.assertEqual(graph.metadata["baseline_name"], "direct")
        self.assertEqual(graph.metadata["executed_round_count"], 0)
        self.assertEqual(graph.metadata["stop_reason"], "baseline_direct_complete")
        self.assertEqual(len(graph.round_summaries), 0)
        self.assertNotIn("gold motivation", graph.final_proposal.motivation.casefold())
        self.assertNotIn("gold method", graph.final_proposal.method.casefold())

    def test_self_refine_baseline_runs_with_zero_rounds(self) -> None:
        instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="self-refine",
            io_mode="auto",
        )

        graph = run_baseline_experiment(instance, baseline_name="self-refine")

        self.assertIsNotNone(graph.final_proposal)
        self.assertEqual(graph.metadata["baseline_name"], "self-refine")
        self.assertEqual(graph.metadata["executed_round_count"], 0)
        self.assertEqual(graph.metadata["stop_reason"], "baseline_self_refine_complete")
        self.assertTrue(graph.final_proposal.evaluation)

    def test_proxy_baselines_are_registered(self) -> None:
        self.assertIn("ai-researcher", BASELINE_SPECS)
        self.assertIn("scipip", BASELINE_SPECS)
        self.assertIn("virsci", BASELINE_SPECS)
        self.assertEqual(BASELINE_SPECS["ai-researcher"].strategy, "external")
        self.assertIn("ai-researcher-proxy", BASELINE_SPECS)
        self.assertIn("scipip-proxy", BASELINE_SPECS)
        self.assertIn("virsci-proxy", BASELINE_SPECS)
        self.assertNotIn("research-agent-proxy", BASELINE_SPECS)
        self.assertTrue(BASELINE_SPECS["ai-researcher-proxy"].is_proxy)

    def test_external_baseline_requires_config(self) -> None:
        instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ai-researcher",
            io_mode="auto",
        )

        with self.assertRaises(RuntimeError) as context:
            run_baseline_experiment(instance, baseline_name="ai-researcher")

        self.assertIn("--external-baseline-config", str(context.exception))

    def test_external_baseline_config_loader_filters_non_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "external.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ai-researcher": {"enabled": True, "repo_path": "C:/tmp/AI-Researcher"},
                        "ignored": "not-a-mapping",
                    }
                ),
                encoding="utf-8",
            )

            payload = load_external_baseline_config(config_path)

        self.assertIn("ai-researcher", payload)
        self.assertNotIn("ignored", payload)
        self.assertEqual(payload["ai-researcher"]["repo_path"], "C:/tmp/AI-Researcher")


if __name__ == "__main__":
    unittest.main()
