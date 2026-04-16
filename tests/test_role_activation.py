from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.engine import create_branch, create_edge, create_node
from idea_graph.models import IdeaGraph
from idea_graph.role_activation import active_roles_for_round
from idea_graph.schema import ROLE_NAMES


def _build_graph(*, include_eval: bool, include_novelty: bool, include_risk: bool) -> IdeaGraph:
    graph = IdeaGraph(
        topic="The topic of this paper is 3D language field modeling.",
        literature=["LERF", "3D Gaussian Splatting", "Segment Anything"],
        metadata={"benchmark": "AI_Idea_Bench_2025"},
    )
    problem_branch = create_branch(graph, "ImpactReframer")
    method_branch = create_branch(graph, "MechanismProposer")
    novelty_branch = create_branch(graph, "NoveltyExaminer")
    eval_branch = create_branch(graph, "EvaluationDesigner")
    risk_branch = create_branch(graph, "FeasibilityCritic")

    problem = create_node(
        graph,
        node_type="Problem",
        text="3D language field modeling remains slow and ambiguous for open-vocabulary scene queries.",
        role="ImpactReframer",
        branch_id=problem_branch.id,
        confidence=0.84,
        evidence=["LERF grounds CLIP features in radiance fields."],
    )
    hypothesis = create_node(
        graph,
        node_type="Hypothesis",
        text="A language-aware Gaussian field can improve open-vocabulary 3D querying.",
        role="MechanismProposer",
        branch_id=method_branch.id,
        confidence=0.85,
        evidence=["3D Gaussian Splatting improves rendering speed."],
    )
    method = create_node(
        graph,
        node_type="Method",
        text="Attach CLIP language embeddings to Gaussian primitives and refine queries with hierarchical semantics.",
        role="MechanismProposer",
        branch_id=method_branch.id,
        confidence=0.86,
        evidence=["Segment Anything provides hierarchical segmentation cues."],
    )
    create_edge(
        graph,
        source_id=hypothesis.id,
        relation="supports",
        target_id=problem.id,
        role="MechanismProposer",
        branch_id=method_branch.id,
    )
    create_edge(
        graph,
        source_id=method.id,
        relation="supports",
        target_id=hypothesis.id,
        role="MechanismProposer",
        branch_id=method_branch.id,
    )

    if include_novelty:
        novelty = create_node(
            graph,
            node_type="NoveltyClaim",
            text="The visible reference set does not combine Gaussian Splatting with hierarchical language semantics.",
            role="NoveltyExaminer",
            branch_id=novelty_branch.id,
            confidence=0.82,
            evidence=["LERF and Gaussian Splatting appear separately in the visible references."],
        )
        create_edge(
            graph,
            source_id=novelty.id,
            relation="supports",
            target_id=problem.id,
            role="NoveltyExaminer",
            branch_id=novelty_branch.id,
        )

    if include_eval:
        evaluation = create_node(
            graph,
            node_type="EvalPlan",
            text="Evaluate localization accuracy with ablations on CLIP compression and hierarchical semantics.",
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
            confidence=0.83,
            evidence=["Localization accuracy is the visible task-level measurement."],
        )
        create_edge(
            graph,
            source_id=evaluation.id,
            relation="supports",
            target_id=method.id,
            role="EvaluationDesigner",
            branch_id=eval_branch.id,
        )

    if include_risk:
        risk = create_node(
            graph,
            node_type="Risk",
            text="Hierarchical masks may fragment semantic regions under sparse views.",
            role="FeasibilityCritic",
            branch_id=risk_branch.id,
            confidence=0.74,
        )
        create_edge(
            graph,
            source_id=risk.id,
            relation="contradicts",
            target_id=method.id,
            role="FeasibilityCritic",
            branch_id=risk_branch.id,
        )

    return graph


def test_role_activation_keeps_all_roles_in_round1() -> None:
    graph = _build_graph(include_eval=False, include_novelty=False, include_risk=False)

    roles = active_roles_for_round(graph, "Round1")

    assert roles == tuple(ROLE_NAMES)


def test_role_activation_targets_missing_slots_in_later_rounds() -> None:
    graph = _build_graph(include_eval=False, include_novelty=False, include_risk=False)

    roles = active_roles_for_round(graph, "Round4")

    assert "EvaluationDesigner" in roles
    assert "NoveltyExaminer" in roles
    assert "MechanismProposer" in roles
    assert "FeasibilityCritic" not in roles
    assert len(roles) < len(ROLE_NAMES)


def test_role_activation_skips_low_value_roles_for_strong_graph() -> None:
    graph = _build_graph(include_eval=True, include_novelty=True, include_risk=False)

    roles = active_roles_for_round(graph, "Round4")

    assert len(roles) >= 2
    assert len(roles) < len(ROLE_NAMES)
    assert "FeasibilityCritic" not in roles
    assert "ImpactReframer" not in roles
