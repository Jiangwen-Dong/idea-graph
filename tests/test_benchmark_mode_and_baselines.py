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

from idea_graph.agent_backend import (
    OpenAICompatibleCollaborationBackend,
    _baseline_prompt_instruction,
)
from idea_graph.baselines import (
    BASELINE_SPECS,
    _ai_researcher_proxy_postprocess_proposal,
    _ai_researcher_topic_fidelity_score,
    attach_baseline_metadata,
    run_baseline_experiment,
)
from idea_graph.external_baselines import load_external_baseline_config
from idea_graph.instances import ExperimentInstance
from idea_graph.literature_grounding import build_literature_grounding
from idea_graph.models import FinalProposal
from idea_graph.settings import OpenAICompatibleSettings


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

    def _language_field_instance(self) -> ExperimentInstance:
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
                            "evaluation": "The LERF dataset is used for open-vocabulary querying analysis with localization accuracy.",
                        },
                        {
                            "resolved_title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
                            "method": "3D Gaussian Splatting enables efficient radiance field rendering.",
                        },
                    ],
                },
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

    def test_generation_safe_grounding_does_not_leak_target_paper_fields(self) -> None:
        instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="scipip-proxy",
            io_mode="auto",
        )

        grounding = build_literature_grounding(
            literature=instance.literature,
            metadata=instance.metadata["generation_safe_metadata"],
        )

        self.assertNotIn("panopose", grounding.existing_methods_summary.casefold())
        self.assertNotIn("gold method", grounding.existing_methods_summary.casefold())
        self.assertNotIn("target paper", grounding.existing_methods_summary.casefold())

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

    def test_scipip_proxy_baseline_adds_structured_bottleneck_language(self) -> None:
        instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="scipip-proxy",
            io_mode="auto",
        )

        graph = run_baseline_experiment(instance, baseline_name="scipip-proxy")

        self.assertIsNotNone(graph.final_proposal)
        self.assertIn("core bottleneck", graph.final_proposal.method.lower())
        self.assertIn("targeted ablations", graph.final_proposal.method.lower())
        self.assertIn("quantitative metrics", graph.final_proposal.evaluation.lower())
        self.assertNotIn("target paper", graph.final_proposal.existing_methods.lower())
        self.assertNotIn("gold method", graph.final_proposal.existing_methods.lower())

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

    def test_ai_researcher_proxy_falls_back_when_llm_generation_fails(self) -> None:
        class FailingClient:
            def create_chat_completion(self, **kwargs):
                raise RuntimeError("synthetic LLM failure")

        backend = OpenAICompatibleCollaborationBackend(
            OpenAICompatibleSettings.from_mapping(
                {
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key": "test-key",
                    "model": "qwen3-8b",
                    "provider": "dashscope",
                    "reasoning_mode": "auto",
                }
            )
        )
        backend.client = FailingClient()

        instance = attach_baseline_metadata(
            self._language_field_instance(),
            baseline_name="ai-researcher-proxy",
            io_mode="auto",
        )

        graph = run_baseline_experiment(
            instance,
            baseline_name="ai-researcher-proxy",
            collaboration_backend=backend,
        )

        self.assertIsNotNone(graph.final_proposal)
        self.assertEqual(graph.metadata["baseline_name"], "ai-researcher-proxy")
        self.assertIn("baseline_generation_error", graph.metadata)
        self.assertEqual(graph.metadata["stop_reason"], "baseline_candidate_rank_complete")

    def test_ai_researcher_topic_fidelity_prefers_field_modeling_over_generic_reconstruction(self) -> None:
        instance = attach_baseline_metadata(
            self._language_field_instance(),
            baseline_name="ai-researcher-proxy",
            io_mode="auto",
        )
        graph = run_baseline_experiment(instance, baseline_name="direct")

        field_proposal = FinalProposal(
            title="Efficient 3D Language Field Modeling with Gaussian Splatting",
            problem="Current 3D language field models are costly and poorly localized.",
            existing_methods="Radiance field baselines and gaussian splatting approaches remain limited for open-vocabulary queries.",
            motivation="Open-vocabulary 3D language field modeling needs better efficiency and localization.",
            hypothesis="Language field supervision over radiance fields can improve 3D localization.",
            method="Combine gaussian splatting with open-vocabulary language field supervision.",
            evaluation="Evaluate localization accuracy on 3D language field benchmarks.",
            significance="Improves 3D language field modeling.",
            caveats="Needs stable supervision.",
        )
        reconstruction_proposal = FinalProposal(
            title="Hierarchical Text-Guided 3D Reconstruction",
            problem="Sparse textual cues make scene reconstruction hard.",
            existing_methods="Existing reconstruction methods struggle.",
            motivation="Text-guided 3D reconstruction matters for content creation.",
            hypothesis="Hierarchical reconstruction improves scene generation.",
            method="Reconstruct scenes from sparse text prompts.",
            evaluation="Use geometric reconstruction metrics.",
            significance="Improves 3D reconstruction.",
            caveats="May drift with ambiguous prompts.",
        )

        field_score = _ai_researcher_topic_fidelity_score(graph, field_proposal)
        reconstruction_score = _ai_researcher_topic_fidelity_score(graph, reconstruction_proposal)

        self.assertGreater(field_score, reconstruction_score)

    def test_ai_researcher_postprocess_adds_language_field_wording(self) -> None:
        instance = attach_baseline_metadata(
            self._language_field_instance(),
            baseline_name="ai-researcher-proxy",
            io_mode="auto",
        )
        graph = run_baseline_experiment(instance, baseline_name="direct")
        draft = FinalProposal(
            title="Efficient Open-Vocabulary Radiance Field Querying",
            problem="Current methods rely on masks for querying radiance fields.",
            existing_methods="Lerf and Gaussian Splatting are relevant baselines.",
            motivation="Open-vocabulary interaction matters.",
            hypothesis="A better querying mechanism can help.",
            method="Integrate CLIP into Gaussian splatting.",
            evaluation="Compare against strong baselines.",
            significance="Improves querying.",
            caveats="May fail on ambiguity.",
        )

        polished = _ai_researcher_proxy_postprocess_proposal(graph, draft)

        self.assertIn("language field", polished.title.lower())
        self.assertIn("3d language field", polished.problem.lower())
        self.assertIn("3d language field", polished.significance.lower())

    def test_baseline_prompt_instruction_distinguishes_ours_and_virsci(self) -> None:
        ours_instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-delayed-consensus",
            io_mode="auto",
        )
        virsci_instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="virsci-proxy",
            io_mode="auto",
        )

        ours_instruction = _baseline_prompt_instruction(ours_instance.metadata)
        virsci_instruction = _baseline_prompt_instruction(virsci_instance.metadata)

        self.assertIn("typed-graph rigor", ours_instruction.lower())
        self.assertIn("discussion-oriented", virsci_instruction.lower())
        self.assertNotEqual(ours_instruction, virsci_instruction)


if __name__ == "__main__":
    unittest.main()
