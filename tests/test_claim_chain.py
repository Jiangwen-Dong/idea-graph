from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.claim_chain import select_claim_chain
from idea_graph.engine import create_branch, create_edge, create_node
from idea_graph.models import IdeaGraph


class ClaimChainTests(unittest.TestCase):
    def _build_rich_context_graph(self) -> tuple[IdeaGraph, dict[str, str]]:
        graph = IdeaGraph(
            topic="The topic of this paper is human pose and shape estimation using LiDAR in uncontrolled environments.",
            literature=[
                "LiDAR-Aid Inertial Poser",
                "Learning from Synthetic Humans",
                "SLOPER4D",
            ],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "benchmark_input_packet": {
                    "benchmark": "AI_Idea_Bench_2025",
                    "topic": "Human pose and shape estimation using LiDAR in uncontrolled environments.",
                    "reference_packet": [
                        {"title": "LiDAR-Aid Inertial Poser", "snippet": "Sparse inertial and LiDAR sensing still suffers under occlusion."},
                        {"title": "SLOPER4D", "snippet": "Scene-aware urban evaluation for global 4D human pose estimation."},
                    ],
                },
            },
        )

        problem_branch = create_branch(graph, "ImpactReframer")
        gap_branch = create_branch(graph, "NoveltyExaminer")
        method_branch = create_branch(graph, "MechanismProposer")
        eval_branch = create_branch(graph, "EvaluationDesigner")
        risk_branch = create_branch(graph, "FeasibilityCritic")

        problem = create_node(
            graph,
            node_type="Problem",
            text="LiDAR-based human pose and shape estimation fails under severe self-occlusion and scene occlusion in uncontrolled environments.",
            role="ImpactReframer",
            branch_id=problem_branch.id,
            confidence=0.86,
        )
        gap = create_node(
            graph,
            node_type="NoveltyClaim",
            text="Existing LiDAR-inertial pipelines do not explicitly train against occlusion-driven distribution shift between sparse point clouds and body geometry.",
            role="NoveltyExaminer",
            branch_id=gap_branch.id,
            confidence=0.82,
            evidence=["LiDAR-Aid Inertial Poser remains brittle under occlusion."],
        )
        method = create_node(
            graph,
            node_type="Method",
            text="Use geometry-aware synthetic occlusion generation plus a mesh-aligned distillation module to stabilize LiDAR human pose estimation.",
            role="MechanismProposer",
            branch_id=method_branch.id,
            confidence=0.84,
            evidence=["Learning from Synthetic Humans shows synthetic augmentation can close real-world gaps."],
        )
        hypothesis = create_node(
            graph,
            node_type="Hypothesis",
            text="Explicitly aligning synthetic occlusion patterns with mesh-aware LiDAR supervision will improve robustness in free environments.",
            role="MechanismProposer",
            branch_id=method_branch.id,
            confidence=0.8,
        )
        evaluation = create_node(
            graph,
            node_type="EvalPlan",
            text="Evaluate on SLOPER4D and FreeMotion-style scene data using MPJPE, translation error, and an occlusion-stress ablation against LiDAR-Aid Inertial Poser.",
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
            confidence=0.83,
            evidence=["SLOPER4D offers scene-aware evaluation for global human pose estimation."],
        )
        risk = create_node(
            graph,
            node_type="Risk",
            text="Synthetic occlusion patterns may not match real free-environment sensor noise, so domain-gap ablations are necessary.",
            role="FeasibilityCritic",
            branch_id=risk_branch.id,
            confidence=0.76,
        )

        create_edge(
            graph,
            source_id=gap.id,
            relation="supports",
            target_id=problem.id,
            role="NoveltyExaminer",
            branch_id=gap_branch.id,
            note="Gap sharpens the problem.",
        )
        create_edge(
            graph,
            source_id=method.id,
            relation="supports",
            target_id=gap.id,
            role="MechanismProposer",
            branch_id=method_branch.id,
            note="Method addresses the gap.",
        )
        create_edge(
            graph,
            source_id=hypothesis.id,
            relation="refines",
            target_id=method.id,
            role="MechanismProposer",
            branch_id=method_branch.id,
            note="Hypothesis sharpens the method.",
        )
        create_edge(
            graph,
            source_id=evaluation.id,
            relation="depends_on",
            target_id=method.id,
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
            note="Evaluation directly tests the method.",
        )
        create_edge(
            graph,
            source_id=risk.id,
            relation="contradicts",
            target_id=method.id,
            role="FeasibilityCritic",
            branch_id=risk_branch.id,
            note="Risk exposes the domain-gap failure mode.",
        )

        return graph, {
            "problem": problem.id,
            "gap": gap.id,
            "mechanism": method.id,
            "evaluation": evaluation.id,
            "caveat": risk.id,
        }

    def _build_weak_context_graph(self) -> tuple[IdeaGraph, dict[str, str]]:
        graph = IdeaGraph(
            topic="Ideation topic keyword: meteorology",
            literature=[],
            metadata={
                "benchmark": "liveideabench",
                "keyword": "meteorology",
                "benchmark_input_packet": {
                    "benchmark": "liveideabench",
                    "keyword": "meteorology",
                    "task_instruction": "Generate one structured research idea from the benchmark keyword only.",
                },
            },
        )

        problem_branch = create_branch(graph, "ImpactReframer")
        method_branch = create_branch(graph, "MechanismProposer")
        eval_branch = create_branch(graph, "EvaluationDesigner")
        risk_branch = create_branch(graph, "FeasibilityCritic")

        problem = create_node(
            graph,
            node_type="Problem",
            text="Regional meteorology models still struggle to couple local sensor dynamics with large-scale atmospheric structure during extreme weather.",
            role="ImpactReframer",
            branch_id=problem_branch.id,
            confidence=0.84,
        )
        method = create_node(
            graph,
            node_type="Method",
            text="Build a physics-guided multi-resolution forecast model that fuses satellite fields with sparse local station events for meteorology forecasting.",
            role="MechanismProposer",
            branch_id=method_branch.id,
            confidence=0.85,
        )
        evaluation = create_node(
            graph,
            node_type="EvalPlan",
            text="Evaluate on regional forecast tasks with severe-weather slices, report RMSE and calibration error, and ablate the local-global fusion path.",
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
            confidence=0.82,
        )
        risk = create_node(
            graph,
            node_type="Risk",
            text="The local-global fusion module may overfit to sparse station patterns and fail during distribution shift across seasons.",
            role="FeasibilityCritic",
            branch_id=risk_branch.id,
            confidence=0.75,
        )

        create_edge(
            graph,
            source_id=method.id,
            relation="supports",
            target_id=problem.id,
            role="MechanismProposer",
            branch_id=method_branch.id,
            note="Method addresses the weak-context problem.",
        )
        create_edge(
            graph,
            source_id=evaluation.id,
            relation="depends_on",
            target_id=method.id,
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
            note="Evaluation tests the method.",
        )
        create_edge(
            graph,
            source_id=risk.id,
            relation="contradicts",
            target_id=method.id,
            role="FeasibilityCritic",
            branch_id=risk_branch.id,
            note="Risk keeps the weak-context idea realistic.",
        )

        return graph, {
            "problem": problem.id,
            "mechanism": method.id,
            "evaluation": evaluation.id,
            "caveat": risk.id,
        }

    def test_select_claim_chain_prefers_complete_scientific_path(self) -> None:
        graph, expected = self._build_rich_context_graph()

        chain = select_claim_chain(graph)

        self.assertIsNotNone(chain)
        assert chain is not None
        self.assertEqual(chain["slots"]["problem"], expected["problem"])
        self.assertEqual(chain["slots"]["gap"], expected["gap"])
        self.assertEqual(chain["slots"]["mechanism"], expected["mechanism"])
        self.assertEqual(chain["slots"]["evaluation"], expected["evaluation"])
        self.assertEqual(chain["slots"]["caveat"], expected["caveat"])
        self.assertTrue(chain["coverage"]["is_synthesis_ready"])

    def test_claim_chain_handles_liveideabench_weak_context_without_fake_literature_gap(self) -> None:
        graph, expected = self._build_weak_context_graph()

        chain = select_claim_chain(graph)

        self.assertIsNotNone(chain)
        assert chain is not None
        self.assertTrue(chain["weak_context_mode"])
        self.assertEqual(chain["slots"]["problem"], expected["problem"])
        self.assertEqual(chain["slots"]["mechanism"], expected["mechanism"])
        self.assertEqual(chain["slots"]["evaluation"], expected["evaluation"])
        self.assertEqual(chain["slots"]["caveat"], expected["caveat"])
        self.assertTrue(chain["coverage"]["is_synthesis_ready"])
        self.assertIn("gap", chain["coverage"]["weak_context_relaxed_slots"])


if __name__ == "__main__":
    unittest.main()
