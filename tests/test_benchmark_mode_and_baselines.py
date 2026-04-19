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
    _ai_researcher_expansion_system_prompt,
    _ai_researcher_focus_constraints,
    _ai_researcher_proxy_postprocess_proposal,
    _ai_researcher_topic_fidelity_score,
    _baseline_postprocess_proposal,
    _default_relation_graph_runtime_model_dir,
    _maybe_build_runtime_controller,
    _direct_system_prompt,
    _refine_system_prompt,
    attach_baseline_metadata,
    run_baseline_experiment,
)
from idea_graph.external_baselines import load_external_baseline_config
from idea_graph.external_baselines import _run_virsci
from idea_graph.instances import ExperimentInstance
from idea_graph.literature_grounding import build_literature_grounding
from idea_graph.models import FinalProposal, IdeaGraph
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

    def test_attach_baseline_metadata_filters_noisy_reference_packet_snippets(self) -> None:
        noisy_instance = ExperimentInstance(
            name="ai-idea-bench-2025-3883",
            topic="The topic of this paper is improving GUI grounding and OOD generalization for GUI agents.",
            literature=["SeeClick", "OSWorld"],
            source_path="test",
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "SeeClick",
                            "method": "Use screenshot-grounded interaction instead of structured text for GUI agents.",
                        },
                        {
                            "resolved_title": "OSWorld",
                            "method": (
                                "Task Instruction (See examples above) input Agent (e.g., GPT-4V) a11y-tree "
                                "screenshot keyboardmouse Action Observation input predict OSWorld Environment."
                            ),
                        },
                        {
                            "resolved_title": "UI Control Agents",
                            "abstract": (
                                "Package_name:\"com.google.android.deskclock\" bounds_in_screen { left: 782 top: 1762 } "
                                "class_name: \"android.widget.Button\""
                            ),
                        },
                    ],
                },
            },
        )

        instance = attach_baseline_metadata(
            noisy_instance,
            baseline_name="ours-eig",
            io_mode="auto",
        )

        packet_text = json.dumps(instance.metadata["benchmark_input_packet"], ensure_ascii=False)
        self.assertIn("SeeClick", packet_text)
        self.assertNotIn("Task Instruction (See examples above)", packet_text)
        self.assertNotIn("Package_name", packet_text)

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
        self.assertIn("ours-eig-critic-text", BASELINE_SPECS)
        self.assertIn("ours-eig-critic-graph", BASELINE_SPECS)
        self.assertIn("ours-eig-critic-graph-twohead", BASELINE_SPECS)
        self.assertIn("ours-eig-critic-calibrated", BASELINE_SPECS)
        self.assertIn("ours-eig-critic-no-commit", BASELINE_SPECS)
        self.assertIn("ours-eig-critic-no-edit", BASELINE_SPECS)
        self.assertIn("ours-eig-fixed-control", BASELINE_SPECS)
        self.assertIn("ours-eig-random-control", BASELINE_SPECS)
        self.assertEqual(BASELINE_SPECS["ai-researcher"].strategy, "external")
        self.assertIn("ai-researcher-proxy", BASELINE_SPECS)
        self.assertIn("scipip-proxy", BASELINE_SPECS)
        self.assertIn("virsci-proxy", BASELINE_SPECS)
        self.assertNotIn("research-agent-proxy", BASELINE_SPECS)
        self.assertTrue(BASELINE_SPECS["ai-researcher-proxy"].is_proxy)
        self.assertEqual(
            BASELINE_SPECS["ours-eig-critic-graph"].runtime_controller,
            "relation_graph_critic_rerank",
        )
        self.assertEqual(
            BASELINE_SPECS["ours-eig-critic-graph-twohead"].runtime_controller,
            "relation_graph_two_head_critic",
        )
        self.assertEqual(
            BASELINE_SPECS["ours-eig-fixed-control"].runtime_controller,
            "fixed_control",
        )
        self.assertEqual(
            BASELINE_SPECS["ours-eig-random-control"].runtime_controller,
            "random_control",
        )

    def test_attach_baseline_metadata_enables_relation_graph_runtime_defaults(self) -> None:
        instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-critic-graph",
            io_mode="auto",
        )

        self.assertEqual(instance.metadata["baseline_runtime_controller"], "relation_graph_critic_rerank")
        self.assertEqual(instance.metadata["runtime_controller_kind"], "relation_graph_critic_rerank")
        self.assertFalse(instance.metadata["runtime_controller_use_commit"])
        self.assertIn(
            "development_pool_v3_relation_graph_sanitized_v1",
            instance.metadata["runtime_controller_model_dir"],
        )

    def test_attach_baseline_metadata_enables_two_head_relation_graph_runtime_defaults(self) -> None:
        instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-critic-graph-twohead",
            io_mode="auto",
        )

        self.assertEqual(instance.metadata["baseline_runtime_controller"], "relation_graph_two_head_critic")
        self.assertEqual(instance.metadata["runtime_controller_kind"], "relation_graph_two_head_critic")
        self.assertTrue(instance.metadata["runtime_controller_use_edit"])
        self.assertTrue(instance.metadata["runtime_controller_use_commit"])
        self.assertTrue(instance.metadata["runtime_controller_disable_calibration"])
        self.assertNotIn("runtime_controller_calibration_path", instance.metadata)
        self.assertIn(
            "parallel_v2_twohead_repaired_boundary_st_full_e8_20260418",
            instance.metadata["runtime_controller_model_dir"],
        )
        self.assertEqual(instance.metadata["runtime_protocol"], "parallel_graph_v2")

    def test_attach_baseline_metadata_enables_self_contained_two_head_controller_variants(self) -> None:
        calibrated = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-critic-calibrated",
            io_mode="auto",
        )
        no_commit = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-critic-no-commit",
            io_mode="auto",
        )
        no_edit = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-critic-no-edit",
            io_mode="auto",
        )

        self.assertEqual(calibrated.metadata["runtime_controller_kind"], "relation_graph_two_head_critic")
        self.assertTrue(calibrated.metadata["runtime_controller_use_edit"])
        self.assertTrue(calibrated.metadata["runtime_controller_use_commit"])
        self.assertFalse(calibrated.metadata.get("runtime_controller_disable_calibration", False))
        self.assertTrue(Path(calibrated.metadata["runtime_controller_calibration_path"]).is_file())
        self.assertAlmostEqual(calibrated.metadata["runtime_controller_tau_override"], 0.068)
        self.assertAlmostEqual(calibrated.metadata["runtime_controller_gamma_commit"], 0.6563)

        self.assertTrue(no_commit.metadata["runtime_controller_use_edit"])
        self.assertFalse(no_commit.metadata["runtime_controller_use_commit"])
        self.assertTrue(no_commit.metadata["runtime_controller_disable_calibration"])
        self.assertNotIn("runtime_controller_calibration_path", no_commit.metadata)

        self.assertFalse(no_edit.metadata["runtime_controller_use_edit"])
        self.assertTrue(no_edit.metadata["runtime_controller_use_commit"])
        self.assertFalse(no_edit.metadata.get("runtime_controller_disable_calibration", False))
        self.assertTrue(Path(no_edit.metadata["runtime_controller_calibration_path"]).is_file())

    def test_attach_baseline_metadata_enables_fixed_and_random_control_variants(self) -> None:
        fixed = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-fixed-control",
            io_mode="auto",
        )
        random = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-random-control",
            io_mode="auto",
        )

        self.assertEqual(fixed.metadata["baseline_name"], "ours-eig-fixed-control")
        self.assertEqual(fixed.metadata["runtime_controller_kind"], "fixed_control")
        self.assertTrue(fixed.metadata["runtime_controller_use_edit"])
        self.assertFalse(fixed.metadata["runtime_controller_use_commit"])
        self.assertEqual(fixed.metadata["max_rounds_hint"], 5)
        self.assertTrue(Path(fixed.metadata["runtime_controller_policy_path"]).is_file())

        self.assertEqual(random.metadata["baseline_name"], "ours-eig-random-control")
        self.assertEqual(random.metadata["runtime_controller_kind"], "random_control")
        self.assertTrue(random.metadata["runtime_controller_use_edit"])
        self.assertFalse(random.metadata["runtime_controller_use_commit"])
        self.assertEqual(random.metadata["max_rounds_hint"], 5)
        self.assertEqual(random.metadata["runtime_controller_random_seed"], 0)

    def test_attach_baseline_metadata_uses_parallel_runtime_for_ours_eig_family(self) -> None:
        eig_instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig",
            io_mode="auto",
        )
        text_instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-critic-text",
            io_mode="auto",
        )
        graph_instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-critic-graph",
            io_mode="auto",
        )
        twohead_instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-critic-graph-twohead",
            io_mode="auto",
        )
        fixed_instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-fixed-control",
            io_mode="auto",
        )
        random_instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-random-control",
            io_mode="auto",
        )

        self.assertEqual(eig_instance.metadata["runtime_protocol"], "parallel_graph_v2")
        self.assertEqual(text_instance.metadata["runtime_protocol"], "parallel_graph_v2")
        self.assertEqual(graph_instance.metadata["runtime_protocol"], "parallel_graph_v2")
        self.assertEqual(twohead_instance.metadata["runtime_protocol"], "parallel_graph_v2")
        self.assertEqual(fixed_instance.metadata["runtime_protocol"], "parallel_graph_v2")
        self.assertEqual(random_instance.metadata["runtime_protocol"], "parallel_graph_v2")

    def test_attach_baseline_metadata_overwrites_stale_controller_fields_on_baseline_switch(self) -> None:
        text_instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-critic-text",
            io_mode="auto",
        )
        self.assertEqual(text_instance.metadata["runtime_controller_kind"], "text_critic_rerank")
        self.assertIn("runtime_controller_model_path", text_instance.metadata)

        graph_instance = attach_baseline_metadata(
            text_instance,
            baseline_name="ours-eig-critic-graph",
            io_mode="auto",
        )
        self.assertEqual(graph_instance.metadata["runtime_controller_kind"], "relation_graph_critic_rerank")
        self.assertIn("runtime_controller_model_dir", graph_instance.metadata)
        self.assertNotIn("runtime_controller_model_path", graph_instance.metadata)

        plain_instance = attach_baseline_metadata(
            graph_instance,
            baseline_name="ours-eig",
            io_mode="auto",
        )
        self.assertEqual(plain_instance.metadata["baseline_runtime_controller"], "")
        self.assertNotIn("runtime_controller_enabled", plain_instance.metadata)
        self.assertNotIn("runtime_controller_kind", plain_instance.metadata)
        self.assertNotIn("runtime_controller_model_path", plain_instance.metadata)
        self.assertNotIn("runtime_controller_model_dir", plain_instance.metadata)

    def test_default_relation_graph_runtime_model_dir_prefers_shared_outputs_from_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir) / "idea-graph"
            worktree_root = repo_root / ".worktrees" / "g6-graph-controller-gate"
            expected_dir = (
                repo_root
                / "outputs"
                / "graph_critic_models"
                / "development_pool_v3_relation_graph_sanitized_v1"
            )
            expected_dir.mkdir(parents=True, exist_ok=True)
            worktree_root.mkdir(parents=True, exist_ok=True)

            with patch("idea_graph.baselines.ROOT", worktree_root):
                resolved = _default_relation_graph_runtime_model_dir()

        self.assertEqual(resolved, expected_dir.resolve())

    def test_default_runtime_model_resolution_does_not_walk_arbitrary_ancestors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            parent_root = Path(tmp_dir) / "parent-repo"
            nested_repo_root = parent_root / "nested-repo"
            (parent_root / "outputs" / "graph_critic_models").mkdir(parents=True, exist_ok=True)
            nested_repo_root.mkdir(parents=True, exist_ok=True)

            with patch("idea_graph.baselines.ROOT", nested_repo_root):
                resolved = _default_relation_graph_runtime_model_dir()

        expected = (
            nested_repo_root
            / "outputs"
            / "graph_critic_models"
            / "development_pool_v3_relation_graph_sanitized_v1"
        ).resolve()
        self.assertEqual(resolved, expected)

    def test_runtime_builder_loads_relation_graph_runtime_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "graph-runtime-model"
            model_dir.mkdir(parents=True, exist_ok=True)
            sentinel_bundle = object()
            graph = IdeaGraph(
                topic="runtime test",
                literature=[],
                metadata={
                    "runtime_controller_kind": "relation_graph_critic_rerank",
                    "runtime_controller_model_dir": str(model_dir),
                },
            )
            baseline = BASELINE_SPECS["ours-eig-critic-graph"]

            with patch(
                "idea_graph.baselines.load_relation_graph_runtime_bundle",
                return_value=sentinel_bundle,
            ) as mocked_loader:
                runtime_controller, runtime_metadata = _maybe_build_runtime_controller(graph, baseline)

        mocked_loader.assert_called_once_with(model_dir)
        self.assertIs(runtime_controller, sentinel_bundle)
        self.assertIsNotNone(runtime_metadata)
        assert runtime_metadata is not None
        self.assertEqual(runtime_metadata["kind"], "relation_graph_critic_rerank")
        self.assertFalse(runtime_metadata["use_commit"])

    def test_runtime_builder_loads_two_head_relation_graph_runtime_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "two-head-runtime-model"
            model_dir.mkdir(parents=True, exist_ok=True)
            sentinel_bundle = object()
            graph = IdeaGraph(
                topic="runtime test",
                literature=[],
                metadata={
                    "runtime_controller_kind": "relation_graph_two_head_critic",
                    "runtime_controller_model_dir": str(model_dir),
                },
            )
            baseline = BASELINE_SPECS["ours-eig-critic-graph-twohead"]

            with patch(
                "idea_graph.baselines.load_relation_graph_two_head_runtime_bundle",
                return_value=sentinel_bundle,
            ) as mocked_loader:
                runtime_controller, runtime_metadata = _maybe_build_runtime_controller(graph, baseline)

        mocked_loader.assert_called_once_with(model_dir)
        self.assertIs(runtime_controller, sentinel_bundle)
        self.assertIsNotNone(runtime_metadata)
        assert runtime_metadata is not None
        self.assertEqual(runtime_metadata["kind"], "relation_graph_two_head_critic")
        self.assertTrue(runtime_metadata["use_commit"])

    def test_runtime_builder_loads_fixed_control_policy_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            policy_path = Path(tmp_dir) / "fixed_control_policy.json"
            policy_path.write_text(
                json.dumps(
                    {
                        "Round1": {
                            "MechanismProposer": ["attach_evidence", "add_support_edge", "skip"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            graph = IdeaGraph(
                topic="runtime test",
                literature=[],
                metadata={
                    "runtime_controller_kind": "fixed_control",
                    "runtime_controller_policy_path": str(policy_path),
                },
            )
            baseline = BASELINE_SPECS["ours-eig-fixed-control"]

            runtime_controller, runtime_metadata = _maybe_build_runtime_controller(graph, baseline)

        self.assertIsNotNone(runtime_controller)
        self.assertIsNotNone(runtime_metadata)
        assert runtime_controller is not None
        assert runtime_metadata is not None
        selected = runtime_controller.choose(
            round_name="Round1",
            role="MechanismProposer",
            candidate_specs=[
                {"candidate_id": "c0", "kind": "add_support_edge"},
                {"candidate_id": "c1", "kind": "attach_evidence"},
                {"candidate_id": "c2", "kind": "skip"},
            ],
        )
        self.assertEqual(selected["candidate_id"], "c1")
        self.assertEqual(runtime_metadata["kind"], "fixed_control")
        self.assertEqual(runtime_metadata["policy_path"], str(policy_path.resolve()))
        self.assertFalse(runtime_metadata["use_commit"])

    def test_runtime_builder_loads_random_control_policy_with_seed(self) -> None:
        graph = IdeaGraph(
            topic="runtime test",
            literature=[],
            metadata={
                "runtime_controller_kind": "random_control",
                "runtime_controller_random_seed": 7,
                "batch_restart": 2,
            },
        )
        baseline = BASELINE_SPECS["ours-eig-random-control"]

        runtime_controller, runtime_metadata = _maybe_build_runtime_controller(graph, baseline)

        self.assertIsNotNone(runtime_controller)
        self.assertIsNotNone(runtime_metadata)
        assert runtime_controller is not None
        assert runtime_metadata is not None
        first_pick = runtime_controller.choose(
            round_name="Round1",
            role="MechanismProposer",
            candidate_specs=[
                {"candidate_id": "c0", "kind": "add_support_edge"},
                {"candidate_id": "c1", "kind": "attach_evidence"},
                {"candidate_id": "c2", "kind": "skip"},
            ],
        )
        second_controller, _ = _maybe_build_runtime_controller(graph, baseline)
        assert second_controller is not None
        second_pick = second_controller.choose(
            round_name="Round1",
            role="MechanismProposer",
            candidate_specs=[
                {"candidate_id": "c0", "kind": "add_support_edge"},
                {"candidate_id": "c1", "kind": "attach_evidence"},
                {"candidate_id": "c2", "kind": "skip"},
            ],
        )
        self.assertEqual(first_pick["candidate_id"], second_pick["candidate_id"])
        self.assertEqual(runtime_metadata["kind"], "random_control")
        self.assertEqual(runtime_metadata["seed"], 9)
        self.assertFalse(runtime_metadata["use_commit"])

    def test_controller_baseline_fails_closed_when_runtime_bundle_missing(self) -> None:
        instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="ours-eig-critic-graph",
            io_mode="auto",
        )
        bad_metadata = dict(instance.metadata)
        bad_metadata["runtime_controller_model_dir"] = str(Path(tempfile.gettempdir()) / "missing-graph-runtime-model-dir")
        bad_instance = instance.__class__(
            name=instance.name,
            topic=instance.topic,
            literature=list(instance.literature),
            source_path=instance.source_path,
            metadata=bad_metadata,
        )

        with self.assertRaises(RuntimeError) as context:
            run_baseline_experiment(bad_instance, baseline_name="ours-eig-critic-graph")

        self.assertIn("runtime controller", str(context.exception).lower())
        self.assertIn("failed", str(context.exception).lower())

    def test_generation_prompts_discourage_noisy_fragment_copying(self) -> None:
        direct_prompt = _direct_system_prompt(BASELINE_SPECS["direct"])
        refine_prompt = _refine_system_prompt(BASELINE_SPECS["self-refine"])
        ai_prompt = _ai_researcher_expansion_system_prompt(BASELINE_SPECS["ai-researcher-proxy"])

        self.assertIn("Do not copy raw extraction fragments", direct_prompt)
        self.assertIn("noisy copied snippet fragments", refine_prompt)
        self.assertIn("one coherent proposal with one main mechanism", ai_prompt)

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
        self.assertEqual(graph.metadata["external_baseline_adapter_status"], "paper-faithful-adapter")
        self.assertFalse(graph.metadata.get("external_baseline_proxy_fallback", True))
        self.assertEqual(graph.metadata["stop_reason"], "baseline_ai-researcher_complete")
        self.assertEqual(graph.metadata["ai_researcher_proxy_candidate_count"], 2)

    def test_scipip_external_bridge_runs_with_openai_compatible_mode(self) -> None:
        class FakeSettings:
            max_retries = 0
            model = "qwen3-8b"

            def model_for_role(self, role: str) -> str:
                return self.model

            def sanitized_dict(self) -> dict[str, object]:
                return {"model": self.model}

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
                        "research_problem": "Panoramic relative pose estimation remains fragile under sparse viewpoint overlap.",
                        "rationales": [
                            "Current panorama methods struggle to separate geometry errors from appearance variation.",
                            "Benchmark-safe references suggest geometry-aware constraints but not efficient adaptation.",
                        ],
                        "reference_inspirations": [
                            {
                                "title": "Paper A",
                                "inspiration": "Use self-supervised geometry cues to stabilize relative pose estimation.",
                            }
                        ],
                        "integrated_direction": "Combine overlap-aware geometry cues with lightweight panorama adaptation.",
                        "experiment_axes": ["pose error", "overlap robustness"],
                        "candidate_title": "Overlap-Aware Panorama Pose Adaptation",
                    },
                    {
                        "title": "Overlap-Aware Panorama Pose Adaptation",
                        "problem": "Scaled relative pose estimation in panoramic images remains brittle when overlap is weak.",
                        "existing_methods": "Panorama pose baselines use geometry cues but remain unstable under low overlap.",
                        "motivation": "A structured decomposition of geometry and appearance could improve robustness.",
                        "hypothesis": "Overlap-aware adaptation can reduce pose drift in sparse-overlap panoramas.",
                        "method": "Estimate overlap confidence, then route geometry-aware adaptation modules to stabilize scaled relative pose prediction.",
                        "evaluation": "Measure scaled pose error and robustness across varying panorama overlap regimes.",
                        "significance": "This gives a more benchmark-faithful and testable panoramic pose direction.",
                        "caveats": "Performance may depend on overlap confidence calibration.",
                    },
                ]
            ),
        )

        instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="scipip",
            io_mode="auto",
        )

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "SciPIP"
            (repo_root / "src").mkdir(parents=True)
            (repo_root / "src" / "generator.py").write_text("print('ok')\n", encoding="utf-8")

            config = {
                "scipip": {
                    "enabled": True,
                    "execution_mode": "openai-compatible-bridge",
                    "repo_path": str(repo_root),
                    "openai_compatible": {
                        "base_url": "https://example.com/v1",
                        "api_key": "test-key",
                        "model": "qwen3-8b",
                        "provider": "dashscope",
                        "reasoning_mode": "auto",
                    },
                }
            }

            with patch("idea_graph.external_baselines._build_openai_compatible_bridge_backend", return_value=fake_backend):
                graph = run_baseline_experiment(
                    instance,
                    baseline_name="scipip",
                    external_baseline_config=config,
                )

        self.assertIsNotNone(graph.final_proposal)
        self.assertEqual(graph.final_proposal.title, "Overlap-Aware Panorama Pose Adaptation")
        self.assertEqual(graph.metadata["external_baseline_execution_mode"], "openai-compatible-bridge")
        self.assertEqual(graph.metadata["external_baseline_adapter_status"], "paper-faithful-adapter")
        self.assertEqual(graph.metadata["stop_reason"], "baseline_scipip_complete")
        self.assertIn("external_baseline_decomposition_file", graph.metadata)

    def test_virsci_fixed_topic_bridge_runs_with_openai_compatible_mode(self) -> None:
        class FakeSettings:
            max_retries = 0
            model = "qwen3-8b"

            def model_for_role(self, role: str) -> str:
                return self.model

            def sanitized_dict(self) -> dict[str, object]:
                return {"model": self.model}

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
                        "scientist": "ScientistAlpha",
                        "stance": "Lock the benchmark topic and emphasize the real bottleneck.",
                        "topic_commitment": "Stay focused on 3D language field modeling rather than generic 3D reconstruction.",
                        "mechanism": "Prioritize benchmark-aligned language localization errors.",
                        "novelty_argument": "Current methods do not explicitly optimize benchmark-aligned localization drift.",
                        "risk": "Too much focus on drift could miss representation quality.",
                        "experiment": "Track localization drift and retrieval quality together.",
                    },
                    {
                        "scientist": "ScientistBeta",
                        "stance": "Push one concrete mechanism.",
                        "topic_commitment": "Keep the idea centered on open-vocabulary 3D language fields.",
                        "mechanism": "Distill hierarchical language cues into sparse 3D anchors.",
                        "novelty_argument": "This separates semantic grounding from rendering-heavy field updates.",
                        "risk": "Sparse anchors may underfit fine-grained objects.",
                        "experiment": "Compare anchor density against localization quality.",
                    },
                    {
                        "scientist": "ScientistGamma",
                        "stance": "Stress-test feasibility and evaluation.",
                        "topic_commitment": "Use only task-relevant benchmarks and ablations.",
                        "mechanism": "Add confidence-weighted query heads for stable retrieval.",
                        "novelty_argument": "This makes the method testable under benchmark-native failure modes.",
                        "risk": "Confidence heads may add calibration error.",
                        "experiment": "Run calibration, robustness, and retrieval ablations.",
                    },
                    {
                        "title": "Sparse Anchor Language Fields for Open-Vocabulary 3D Queries",
                        "problem": "3D language field modeling still struggles with stable open-vocabulary localization.",
                        "existing_methods": "Existing language radiance fields and Gaussian variants are strong but drift under sparse supervision.",
                        "motivation": "A discussion-style team synthesis points to stable semantic anchors as the key missing ingredient.",
                        "hypothesis": "Sparse semantic anchors with confidence-weighted query heads can reduce localization drift.",
                        "method": "Use a multi-view language field with sparse semantic anchors and confidence-aware query heads to stabilize open-vocabulary 3D querying.",
                        "evaluation": "Evaluate localization accuracy, retrieval quality, calibration, and sparse-supervision robustness against strong baselines.",
                        "significance": "This preserves the benchmark topic while adding a more discussion-vetted mechanism.",
                        "caveats": "Anchor sparsity and calibration may need careful tuning.",
                    },
                ]
            ),
        )

        instance = attach_baseline_metadata(
            self._language_field_instance(),
            baseline_name="virsci",
            io_mode="auto",
        )

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "Virtual-Scientists" / "sci_platform"
            repo_root.mkdir(parents=True)
            (repo_root / "run.py").write_text("print('ok')\n", encoding="utf-8")

            config = {
                "virsci": {
                    "enabled": True,
                    "execution_mode": "benchmark-fixed-topic-bridge",
                    "repo_path": str(repo_root.parent),
                    "openai_compatible": {
                        "base_url": "https://example.com/v1",
                        "api_key": "test-key",
                        "model": "qwen3-8b",
                        "provider": "dashscope",
                        "reasoning_mode": "auto",
                    },
                }
            }

            with patch("idea_graph.external_baselines._build_openai_compatible_bridge_backend", return_value=fake_backend):
                graph = run_baseline_experiment(
                    instance,
                    baseline_name="virsci",
                    external_baseline_config=config,
                )

        self.assertIsNotNone(graph.final_proposal)
        self.assertEqual(
            graph.final_proposal.title,
            "Sparse Anchor Language Fields for Open-Vocabulary 3D Queries",
        )
        self.assertEqual(graph.metadata["external_baseline_execution_mode"], "benchmark-fixed-topic-bridge")
        self.assertEqual(graph.metadata["external_baseline_adapter_status"], "paper-faithful-adapter")
        self.assertEqual(graph.metadata["external_baseline_discussion_turns"], 3)
        self.assertEqual(graph.metadata["stop_reason"], "baseline_virsci_complete")

    def test_virsci_benchmark_mode_records_no_go_metadata_before_raising(self) -> None:
        instance = attach_baseline_metadata(
            self._ai_idea_bench_instance(),
            baseline_name="virsci",
            io_mode="auto",
        )
        graph = IdeaGraph(
            topic=instance.topic,
            literature=list(instance.literature),
            metadata=dict(instance.metadata),
        )

        with self.assertRaises(RuntimeError) as context:
            _run_virsci(
                graph,
                {"enabled": True, "repo_path": "C:/missing/Virtual-Scientists"},
                progress_callback=None,
            )

        self.assertIn("fixed-topic benchmark", str(context.exception).lower())
        self.assertEqual(graph.metadata["external_baseline_execution_mode"], "upstream-multi-agent")
        self.assertEqual(graph.metadata["external_baseline_adapter_status"], "exclude-until-fixed-topic-adapter")
        self.assertFalse(graph.metadata.get("external_baseline_proxy_fallback", True))

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
