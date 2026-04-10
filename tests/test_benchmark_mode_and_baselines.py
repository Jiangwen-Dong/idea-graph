from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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
    _ai_researcher_focus_constraints,
    _ai_researcher_proxy_postprocess_proposal,
    _ai_researcher_topic_fidelity_score,
    _baseline_postprocess_proposal,
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
            baseline_name="ours-eig",
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

    def test_ai_researcher_external_bridge_runs_with_openai_compatible_mode(self) -> None:
        class FakeSettings:
            max_retries = 0
            model = "qwen3-8b"

            def model_for_role(self, role: str) -> str:
                return self.model

        class SequencedClient:
            def __init__(self, payloads: list[dict[str, object]]) -> None:
                self._payloads = list(payloads)

            def create_chat_completion(self, **kwargs):
                payload = self._payloads.pop(0)
                content = json.dumps(payload, ensure_ascii=False)
                return SimpleNamespace(
                    content=content,
                    raw_response={"choices": [{"message": {"content": content}}]},
                )

        fake_backend = SimpleNamespace(
            settings=FakeSettings(),
            client=SequencedClient(
                [
                    {
                        "seed_ideas": [
                            {
                                "idea_name": "Language Field Distillation",
                                "problem_focus": "Open-vocabulary 3D language field modeling remains expensive.",
                                "existing_gap": "Current methods are slow and weakly localized.",
                                "core_mechanism": "Distill hierarchical language features into compact 3D fields.",
                                "evaluation_hint": "Measure localization and query accuracy.",
                            },
                            {
                                "idea_name": "Sparse Gaussian Language Anchors",
                                "problem_focus": "Mask-free language grounding in radiance fields is unstable.",
                                "existing_gap": "Language anchors drift under sparse supervision.",
                                "core_mechanism": "Use sparse Gaussian anchors for stable language grounding.",
                                "evaluation_hint": "Evaluate open-vocabulary grounding stability.",
                            },
                        ]
                    },
                    {
                        "title": "Benchmark-Faithful Language Field Modeling",
                        "problem": "3D language field modeling still struggles with efficient open-vocabulary localization.",
                        "existing_methods": "LERF and Gaussian Splatting are strong nearby baselines.",
                        "motivation": "We need a more efficient and better-grounded language field representation.",
                        "hypothesis": "Compact hierarchical language fields can preserve localization quality.",
                        "method": "Distill language-aware field features into a compact 3D representation with explicit localization heads.",
                        "evaluation": "Evaluate open-vocabulary localization accuracy and retrieval quality against strong baselines.",
                        "significance": "This improves benchmark-faithful 3D language field ideation.",
                        "caveats": "Performance may depend on grounding quality.",
                    },
                    {
                        "title": "Sparse Gaussian Language Anchors for Querying",
                        "problem": "Open-vocabulary 3D querying is unstable under sparse language supervision.",
                        "existing_methods": "Radiance-field and Gaussian baselines lack stable anchor design.",
                        "motivation": "Stable anchor structure could improve 3D querying.",
                        "hypothesis": "Sparse anchor constraints reduce drift.",
                        "method": "Introduce sparse Gaussian anchor nodes for language-to-geometry grounding.",
                        "evaluation": "Evaluate query localization and robustness under sparse supervision.",
                        "significance": "This stabilizes open-vocabulary 3D querying.",
                        "caveats": "Anchor sparsity may trade off recall.",
                    },
                    {
                        "selected_index": 0,
                        "reason": "Candidate 0 is more benchmark-faithful and better aligned with 3D language field modeling.",
                        "scores": [
                            {
                                "index": 0,
                                "topic_fidelity": 5,
                                "novelty": 4,
                                "significance": 4,
                                "feasibility": 4,
                                "clarity": 4,
                                "literature_grounding": 4,
                                "experiment_quality": 4,
                                "overall": 4.3,
                            },
                            {
                                "index": 1,
                                "topic_fidelity": 3,
                                "novelty": 4,
                                "significance": 3,
                                "feasibility": 4,
                                "clarity": 4,
                                "literature_grounding": 3,
                                "experiment_quality": 3,
                                "overall": 3.4,
                            },
                        ],
                    },
                ]
            ),
        )

        instance = attach_baseline_metadata(
            self._language_field_instance(),
            baseline_name="ai-researcher",
            io_mode="auto",
        )

        config = {
            "ai-researcher": {
                "enabled": True,
                "execution_mode": "openai-compatible-bridge",
                "ideas_n": 2,
                "openai_compatible": {
                    "base_url": "https://example.com/v1",
                    "api_key": "test-key",
                    "model": "qwen3-8b",
                    "provider": "dashscope",
                    "reasoning_mode": "auto",
                },
            }
        }

        with patch("idea_graph.external_baselines._build_ai_researcher_bridge_backend", return_value=fake_backend):
            graph = run_baseline_experiment(
                instance,
                baseline_name="ai-researcher",
                external_baseline_config=config,
            )

        self.assertIsNotNone(graph.final_proposal)
        self.assertEqual(graph.final_proposal.title, "Benchmark-Faithful 3D Language Field Modeling")
        self.assertEqual(graph.metadata["baseline_name"], "ai-researcher")
        self.assertEqual(graph.metadata["external_baseline_execution_mode"], "openai-compatible-bridge")
        self.assertEqual(graph.metadata["stop_reason"], "baseline_ai-researcher_complete")
        self.assertEqual(graph.metadata["ai_researcher_proxy_candidate_count"], 2)

    def test_ai_researcher_external_bridge_requires_openai_compatible_settings(self) -> None:
        instance = attach_baseline_metadata(
            self._language_field_instance(),
            baseline_name="ai-researcher",
            io_mode="auto",
        )

        config = {
            "ai-researcher": {
                "enabled": True,
                "execution_mode": "openai-compatible-bridge",
            }
        }

        with self.assertRaises(RuntimeError) as context:
            run_baseline_experiment(
                instance,
                baseline_name="ai-researcher",
                external_baseline_config=config,
            )

        self.assertIn("openai-compatible", str(context.exception).lower())

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

    def test_ai_researcher_focus_constraints_are_generic_for_non_language_field_topics(self) -> None:
        instance = attach_baseline_metadata(
            self._liveideabench_instance(),
            baseline_name="ai-researcher-proxy",
            io_mode="auto",
        )
        graph = run_baseline_experiment(instance, baseline_name="direct")

        constraints = _ai_researcher_focus_constraints(graph)
        joined = " ".join(constraints).lower()

        self.assertIn("meteorology", joined)
        self.assertNotIn("3d language/radiance field", joined)
        self.assertNotIn("radiance field representation", joined)

    def test_ai_researcher_topic_fidelity_prefers_meteorology_over_language_field_drift(self) -> None:
        instance = attach_baseline_metadata(
            self._liveideabench_instance(),
            baseline_name="ai-researcher-proxy",
            io_mode="auto",
        )
        graph = run_baseline_experiment(instance, baseline_name="direct")

        meteorology_proposal = FinalProposal(
            title="Physics-Aware Meteorology Forecasting with Multi-Source Data Fusion",
            problem="Meteorology forecasting still struggles with severe-weather uncertainty and cross-source alignment.",
            existing_methods="Existing numerical weather and neural forecasting systems can be poorly calibrated during extreme events.",
            motivation="Reliable meteorology forecasts matter for disaster response and climate-sensitive planning.",
            hypothesis="Physics-aware calibration plus multi-source fusion can improve meteorology forecasting accuracy and reliability.",
            method="Fuse radar, satellite, and reanalysis signals with a physics-aware calibration module for meteorology prediction.",
            evaluation="Evaluate meteorology forecasting accuracy, calibration, and robustness on benchmark weather datasets.",
            significance="Improves practical meteorology decision support.",
            caveats="May require careful handling of regional shifts.",
        )
        language_field_proposal = FinalProposal(
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

        meteorology_score = _ai_researcher_topic_fidelity_score(graph, meteorology_proposal)
        language_field_score = _ai_researcher_topic_fidelity_score(graph, language_field_proposal)

        self.assertGreater(meteorology_score, language_field_score)

    def test_ai_researcher_postprocess_does_not_inject_language_field_wording_for_liveideabench(self) -> None:
        instance = attach_baseline_metadata(
            self._liveideabench_instance(),
            baseline_name="ai-researcher-proxy",
            io_mode="auto",
        )
        graph = run_baseline_experiment(instance, baseline_name="direct")
        draft = FinalProposal(
            title="Physics-Aware Meteorology Forecasting with Multi-Source Data Fusion",
            problem="Meteorology forecasting still struggles with severe-weather uncertainty.",
            existing_methods="Existing weather models and neural forecasting pipelines have calibration gaps.",
            motivation="More reliable meteorology forecasting matters for early warning systems.",
            hypothesis="Physics-aware fusion can improve meteorology forecasts.",
            method="Fuse radar, satellite, and reanalysis signals with a calibrated forecasting head.",
            evaluation="Measure forecast accuracy and calibration on weather benchmarks.",
            significance="Improves meteorology forecasting.",
            caveats="May need regional adaptation.",
        )

        polished = _baseline_postprocess_proposal(graph, BASELINE_SPECS["ai-researcher-proxy"], draft)
        combined = " ".join(
            [
                polished.title,
                polished.problem,
                polished.existing_methods,
                polished.motivation,
                polished.hypothesis,
                polished.method,
                polished.evaluation,
                polished.significance,
            ]
        ).lower()

        self.assertIn("meteorology", combined)
        self.assertNotIn("language field", combined)
        self.assertNotIn("lerf", combined)
        self.assertNotIn("gaussian splatting", combined)

    def test_baseline_prompt_instruction_distinguishes_ours_and_virsci(self) -> None:
        ours_instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig",
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
