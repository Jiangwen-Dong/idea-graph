import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.models import Branch, Edge, IdeaGraph, Node
from idea_graph.signal_heuristic_control import (
    SignalHeuristicController,
    compute_graph_signals,
)


def _build_protocol_graph(*, grounded: bool, contradiction: bool, repaired: bool = False) -> IdeaGraph:
    graph = IdeaGraph(topic="topic", literature=["paper a", "paper b"])
    graph.branches["B001"] = Branch(id="B001", role="ImpactReframer")
    graph.branches["B002"] = Branch(id="B002", role="MechanismProposer")
    graph.branches["B003"] = Branch(id="B003", role="FeasibilityCritic")
    graph.branches["B004"] = Branch(id="B004", role="EvaluationDesigner")
    graph.branches["B005"] = Branch(id="B005", role="NoveltyExaminer")

    graph.nodes["N001"] = Node(
        id="N001",
        type="Problem",
        text="Problem node.",
        role="ImpactReframer",
        branch_id="B001",
        confidence=0.8,
        evidence=["problem evidence"] if grounded else [],
    )
    graph.nodes["N002"] = Node(
        id="N002",
        type="Hypothesis",
        text="Hypothesis node.",
        role="MechanismProposer",
        branch_id="B002",
        confidence=0.8,
        evidence=["hypothesis evidence"] if grounded else [],
    )
    graph.nodes["N003"] = Node(
        id="N003",
        type="Method",
        text="Method node.",
        role="MechanismProposer",
        branch_id="B002",
        confidence=0.8,
        evidence=["method evidence"] if grounded else [],
    )
    graph.nodes["N004"] = Node(
        id="N004",
        type="EvalPlan",
        text="Eval node.",
        role="EvaluationDesigner",
        branch_id="B004",
        confidence=0.8,
        evidence=["eval evidence"] if grounded else [],
    )
    graph.nodes["N005"] = Node(
        id="N005",
        type="NoveltyClaim",
        text="Novelty node.",
        role="NoveltyExaminer",
        branch_id="B005",
        confidence=0.8,
        evidence=["novelty evidence"] if grounded else [],
    )
    graph.nodes["N006"] = Node(
        id="N006",
        type="Risk",
        text="Risk node.",
        role="FeasibilityCritic",
        branch_id="B003",
        confidence=0.7,
        evidence=[],
    )

    if grounded:
        graph.edges.append(
            Edge(
                id="E001",
                source_id="N003",
                relation="supports",
                target_id="N002",
                role="MechanismProposer",
                branch_id="B002",
            )
        )
        graph.edges.append(
            Edge(
                id="E002",
                source_id="N002",
                relation="supports",
                target_id="N005",
                role="ImpactReframer",
                branch_id="B001",
            )
        )
        graph.edges.append(
            Edge(
                id="E003",
                source_id="N004",
                relation="depends_on",
                target_id="N003",
                role="EvaluationDesigner",
                branch_id="B004",
            )
        )
    if contradiction:
        graph.edges.append(
            Edge(
                id="E010",
                source_id="N006",
                relation="contradicts",
                target_id="N003",
                role="FeasibilityCritic",
                branch_id="B003",
                resolved=repaired,
            )
        )
        if repaired:
            graph.edges.append(
                Edge(
                    id="E011",
                    source_id="N007",
                    relation="repairs",
                    target_id="N003",
                    role="MechanismProposer",
                    branch_id="B002",
                )
            )
            graph.nodes["N007"] = Node(
                id="N007",
                type="Repair",
                text="Repair node.",
                role="MechanismProposer",
                branch_id="B002",
                confidence=0.7,
                evidence=[],
            )
    return graph


def test_compute_graph_signals_improve_when_the_graph_is_grounded_and_structured() -> None:
    weak_graph = _build_protocol_graph(grounded=False, contradiction=False)
    strong_graph = _build_protocol_graph(grounded=True, contradiction=False)

    weak = compute_graph_signals(weak_graph, round_index=2)
    strong = compute_graph_signals(strong_graph, round_index=2)

    assert {"grounding", "contradiction_load", "completeness", "maturity"} <= set(weak)
    assert strong["grounding"] > weak["grounding"]
    assert strong["completeness"] > weak["completeness"]
    assert strong["maturity"] > weak["maturity"]


def test_compute_graph_signals_soften_contradiction_load_after_repair() -> None:
    unresolved_graph = _build_protocol_graph(grounded=True, contradiction=True, repaired=False)
    repaired_graph = _build_protocol_graph(grounded=True, contradiction=True, repaired=True)

    unresolved = compute_graph_signals(unresolved_graph, round_index=3)
    repaired = compute_graph_signals(repaired_graph, round_index=3)

    assert repaired["contradiction_load"] < unresolved["contradiction_load"]
    assert repaired["contradiction_load"] <= 0.15
    assert repaired["maturity"] > unresolved["maturity"]


def test_compute_graph_signals_grounding_focuses_claim_chain_backbone_over_auxiliary_core_nodes() -> None:
    graph = _build_protocol_graph(grounded=True, contradiction=False)
    for offset in range(20, 24):
        graph.nodes[f"N{offset}"] = Node(
            id=f"N{offset}",
            type="Method" if offset % 2 == 0 else "EvalPlan",
            text=f"Auxiliary node {offset}.",
            role="MechanismProposer" if offset % 2 == 0 else "EvaluationDesigner",
            branch_id="B002" if offset % 2 == 0 else "B004",
            confidence=0.6,
            evidence=[],
        )

    with patch(
        "idea_graph.signal_heuristic_control.select_claim_chain",
        return_value={
            "coverage": {
                "required_slots": ["problem", "gap", "mechanism", "evaluation", "caveat"],
                "slot_count": 5,
            },
            "subgraph": {"core_node_ids": ["N002", "N003", "N004", "N005"]},
        },
    ):
        signals = compute_graph_signals(graph, round_index=3)

    assert signals["grounding"] >= 0.7


def test_signal_heuristic_controller_prefers_dependency_edge_when_structure_is_slot_complete_but_dependency_poor() -> None:
    graph = _build_protocol_graph(grounded=False, contradiction=False)
    graph.edges.append(
        Edge(
            id="E020",
            source_id="N003",
            relation="supports",
            target_id="N002",
            role="MechanismProposer",
            branch_id="B002",
        )
    )
    graph.edges.append(
        Edge(
            id="E021",
            source_id="N002",
            relation="supports",
            target_id="N005",
            role="ImpactReframer",
            branch_id="B001",
        )
    )

    controller = SignalHeuristicController()
    selected = controller.choose(
        round_name="Round1",
        role="EvaluationDesigner",
        candidate_specs=[
            {"candidate_id": "c0", "kind": "add_support_edge", "target_ids": ["N004", "N003"], "payload": {"branch_id": "B004"}},
            {"candidate_id": "c1", "kind": "add_dependency_edge", "target_ids": ["N004", "N003"], "payload": {"branch_id": "B004"}},
            {"candidate_id": "c2", "kind": "attach_evidence", "target_ids": ["N004"], "payload": {"branch_id": "B004", "evidence": "paper a"}},
            {"candidate_id": "c3", "kind": "skip", "target_ids": [], "payload": {"branch_id": "B004"}},
        ],
        graph=graph,
    )

    assert selected["candidate_id"] == "c1"


def test_signal_heuristic_controller_prefers_backbone_grounding_gap_over_auxiliary_core_node() -> None:
    graph = _build_protocol_graph(grounded=False, contradiction=False)
    graph.nodes["N020"] = Node(
        id="N020",
        type="Method",
        text="Auxiliary method node.",
        role="MechanismProposer",
        branch_id="B002",
        confidence=0.6,
        evidence=[],
    )

    controller = SignalHeuristicController()
    with patch(
        "idea_graph.signal_heuristic_control.select_claim_chain",
        return_value={
            "coverage": {
                "required_slots": ["problem", "gap", "mechanism", "evaluation", "caveat"],
                "slot_count": 5,
            },
            "subgraph": {"core_node_ids": ["N002", "N003", "N004", "N005"]},
        },
    ):
        selected = controller.choose(
            round_name="Round4",
            role="EvaluationDesigner",
            candidate_specs=[
                {"candidate_id": "c9", "kind": "attach_evidence", "target_ids": ["N020"], "payload": {"branch_id": "B004", "evidence": "paper a"}},
                {"candidate_id": "c1", "kind": "attach_evidence", "target_ids": ["N004"], "payload": {"branch_id": "B004", "evidence": "paper a"}},
                {"candidate_id": "c3", "kind": "skip", "target_ids": [], "payload": {"branch_id": "B004"}},
            ],
            graph=graph,
        )

    assert selected["candidate_id"] == "c1"


def test_signal_heuristic_controller_prefers_grounding_over_new_dependencies_once_repairs_are_cleared() -> None:
    graph = _build_protocol_graph(grounded=False, contradiction=True, repaired=True)
    graph.edges.append(
        Edge(
            id="E070",
            source_id="N003",
            relation="supports",
            target_id="N002",
            role="MechanismProposer",
            branch_id="B002",
        )
    )
    graph.edges.append(
        Edge(
            id="E071",
            source_id="N002",
            relation="supports",
            target_id="N005",
            role="ImpactReframer",
            branch_id="B001",
        )
    )
    graph.edges.append(
        Edge(
            id="E072",
            source_id="N004",
            relation="depends_on",
            target_id="N003",
            role="EvaluationDesigner",
            branch_id="B004",
        )
    )
    for offset in range(20, 25):
        graph.nodes[f"N{offset}"] = Node(
            id=f"N{offset}",
            type="Method" if offset % 2 == 0 else "EvalPlan",
            text=f"Auxiliary node {offset}.",
            role="MechanismProposer" if offset % 2 == 0 else "EvaluationDesigner",
            branch_id="B002" if offset % 2 == 0 else "B004",
            confidence=0.6,
            evidence=[],
        )

    controller = SignalHeuristicController()
    with patch(
        "idea_graph.signal_heuristic_control.select_claim_chain",
        return_value={
            "coverage": {
                "required_slots": ["problem", "gap", "mechanism", "evaluation", "caveat"],
                "slot_count": 5,
            },
            "subgraph": {"core_node_ids": ["N002", "N003", "N004", "N005"]},
        },
    ):
        selected = controller.choose(
            round_name="Round4",
            role="MechanismProposer",
            candidate_specs=[
                {"candidate_id": "c1", "kind": "attach_evidence", "target_ids": ["N004"], "payload": {"branch_id": "B002", "evidence": "paper a"}},
                {"candidate_id": "c2", "kind": "add_dependency_edge", "target_ids": ["N024", "N002"], "payload": {"branch_id": "B002"}},
                {"candidate_id": "c3", "kind": "add_support_edge", "target_ids": ["N003", "N002"], "payload": {"branch_id": "B002"}},
                {"candidate_id": "c4", "kind": "skip", "target_ids": [], "payload": {"branch_id": "B002"}},
            ],
            graph=graph,
        )

    assert selected["candidate_id"] == "c1"


def test_signal_heuristic_controller_prefers_contradiction_when_graph_is_underdiagnosed_and_not_overloaded() -> None:
    graph = _build_protocol_graph(grounded=False, contradiction=False)
    graph.edges.append(
        Edge(
            id="E030",
            source_id="N003",
            relation="supports",
            target_id="N002",
            role="MechanismProposer",
            branch_id="B002",
        )
    )
    graph.edges.append(
        Edge(
            id="E031",
            source_id="N002",
            relation="supports",
            target_id="N005",
            role="ImpactReframer",
            branch_id="B001",
        )
    )

    controller = SignalHeuristicController()
    selected = controller.choose(
        round_name="Round1",
        role="FeasibilityCritic",
        candidate_specs=[
            {"candidate_id": "c0", "kind": "add_contradiction_edge", "target_ids": ["N006", "N003"], "payload": {"branch_id": "B003"}},
            {"candidate_id": "c1", "kind": "propose_repair", "target_ids": ["N003"], "payload": {"branch_id": "B003", "repair_text": "repair"}},
            {"candidate_id": "c2", "kind": "skip", "target_ids": [], "payload": {"branch_id": "B003"}},
        ],
        graph=graph,
    )

    assert selected["candidate_id"] == "c0"


def test_signal_heuristic_controller_prefers_repair_over_more_contradiction_when_load_is_high() -> None:
    graph = _build_protocol_graph(grounded=True, contradiction=True, repaired=False)
    controller = SignalHeuristicController()
    selected = controller.choose(
        round_name="Round3",
        role="FeasibilityCritic",
        candidate_specs=[
            {"candidate_id": "c0", "kind": "add_contradiction_edge", "target_ids": ["N006", "N003"], "payload": {"branch_id": "B003"}},
            {"candidate_id": "c1", "kind": "propose_repair", "target_ids": ["N003"], "payload": {"branch_id": "B003", "repair_text": "repair"}},
            {"candidate_id": "c2", "kind": "skip", "target_ids": [], "payload": {"branch_id": "B003"}},
        ],
        graph=graph,
    )

    assert selected["candidate_id"] == "c1"


def test_signal_heuristic_controller_prefers_repair_target_with_higher_unresolved_contradiction_burden() -> None:
    graph = _build_protocol_graph(grounded=True, contradiction=True, repaired=False)
    graph.nodes["N008"] = Node(
        id="N008",
        type="Risk",
        text="A second risk node.",
        role="NoveltyExaminer",
        branch_id="B005",
        confidence=0.7,
        evidence=[],
    )
    graph.edges.append(
        Edge(
            id="E040",
            source_id="N008",
            relation="contradicts",
            target_id="N003",
            role="NoveltyExaminer",
            branch_id="B005",
            resolved=False,
        )
    )

    controller = SignalHeuristicController()
    selected = controller.choose(
        round_name="Round3",
        role="FeasibilityCritic",
        candidate_specs=[
            {"candidate_id": "c9", "kind": "propose_repair", "target_ids": ["N002"], "payload": {"branch_id": "B003", "repair_text": "generic repair"}},
            {"candidate_id": "c8", "kind": "propose_repair", "target_ids": ["N001"], "payload": {"branch_id": "B003", "repair_text": "repair problem"}},
            {"candidate_id": "c1", "kind": "propose_repair", "target_ids": ["N003"], "payload": {"branch_id": "B003", "repair_text": "repair method"}},
            {"candidate_id": "c3", "kind": "skip", "target_ids": [], "payload": {"branch_id": "B003"}},
        ],
        graph=graph,
    )

    assert selected["candidate_id"] == "c1"


def test_signal_heuristic_controller_suppresses_new_contradictions_after_round1() -> None:
    graph = _build_protocol_graph(grounded=False, contradiction=False)
    graph.edges.append(
        Edge(
            id="E050",
            source_id="N003",
            relation="supports",
            target_id="N002",
            role="MechanismProposer",
            branch_id="B002",
        )
    )
    graph.edges.append(
        Edge(
            id="E051",
            source_id="N006",
            relation="contradicts",
            target_id="N003",
            role="FeasibilityCritic",
            branch_id="B003",
            resolved=False,
        )
    )

    controller = SignalHeuristicController()
    selected = controller.choose(
        round_name="Round3",
        role="FeasibilityCritic",
        candidate_specs=[
            {"candidate_id": "c0", "kind": "add_contradiction_edge", "target_ids": ["N006", "N002"], "payload": {"branch_id": "B003"}},
            {"candidate_id": "c1", "kind": "propose_repair", "target_ids": ["N003"], "payload": {"branch_id": "B003", "repair_text": "repair method"}},
            {"candidate_id": "c2", "kind": "attach_evidence", "target_ids": ["N003"], "payload": {"branch_id": "B003", "evidence": "paper a"}},
            {"candidate_id": "c3", "kind": "skip", "target_ids": [], "payload": {"branch_id": "B003"}},
        ],
        graph=graph,
    )

    assert selected["candidate_id"] != "c0"


def test_signal_heuristic_controller_avoids_new_contradictions_in_round2_even_when_underdiagnosed() -> None:
    graph = _build_protocol_graph(grounded=False, contradiction=False)
    graph.edges.append(
        Edge(
            id="E060",
            source_id="N003",
            relation="supports",
            target_id="N002",
            role="MechanismProposer",
            branch_id="B002",
        )
    )
    graph.edges.append(
        Edge(
            id="E061",
            source_id="N002",
            relation="supports",
            target_id="N005",
            role="ImpactReframer",
            branch_id="B001",
        )
    )

    controller = SignalHeuristicController()
    selected = controller.choose(
        round_name="Round2",
        role="FeasibilityCritic",
        candidate_specs=[
            {"candidate_id": "c0", "kind": "add_contradiction_edge", "target_ids": ["N006", "N003"], "payload": {"branch_id": "B003"}},
            {"candidate_id": "c1", "kind": "attach_evidence", "target_ids": ["N003"], "payload": {"branch_id": "B003", "evidence": "paper a"}},
            {"candidate_id": "c2", "kind": "propose_repair", "target_ids": ["N003"], "payload": {"branch_id": "B003", "repair_text": "repair method"}},
            {"candidate_id": "c3", "kind": "skip", "target_ids": [], "payload": {"branch_id": "B003"}},
        ],
        graph=graph,
    )

    assert selected["candidate_id"] != "c0"


def test_signal_heuristic_commit_score_tracks_graph_health() -> None:
    controller = SignalHeuristicController()
    weak_graph = _build_protocol_graph(grounded=False, contradiction=True, repaired=False)
    strong_graph = _build_protocol_graph(grounded=True, contradiction=True, repaired=True)

    weak_score = controller.score_commit_graph(weak_graph)
    strong_score = controller.score_commit_graph(strong_graph)

    assert 0.0 <= weak_score <= 1.0
    assert 0.0 <= strong_score <= 1.0
    assert strong_score > weak_score
