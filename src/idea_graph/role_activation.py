from __future__ import annotations

from .collaboration_protocol import round_index_from_name
from .schema import ROLE_NAMES


def _active_nodes(graph, node_type: str):
    return [node for node in graph.active_nodes() if node.type == node_type]


def _node_has_support_or_evidence(graph, node) -> bool:
    if node.evidence:
        return True
    incoming = graph.incoming_edges(node.id)
    return any(edge.relation in {"supports", "repairs"} for edge in incoming)


def _needs_problem_framing(graph) -> bool:
    problems = _active_nodes(graph, "Problem")
    return not problems or all(not _node_has_support_or_evidence(graph, node) for node in problems)


def _needs_mechanism(graph) -> bool:
    hypotheses = _active_nodes(graph, "Hypothesis")
    methods = _active_nodes(graph, "Method")
    if not hypotheses or not methods:
        return True
    return all(not _node_has_support_or_evidence(graph, node) for node in [*hypotheses, *methods])


def _needs_novelty_grounding(graph) -> bool:
    novelty_nodes = _active_nodes(graph, "NoveltyClaim")
    if not novelty_nodes:
        return True
    for node in novelty_nodes:
        if node.evidence:
            return False
        incident_edges = [*graph.incoming_edges(node.id), *graph.outgoing_edges(node.id)]
        if any(edge.relation in {"supports", "overlaps_prior"} for edge in incident_edges):
            return False
    return True


def _needs_evaluation(graph) -> bool:
    eval_nodes = _active_nodes(graph, "EvalPlan")
    if not eval_nodes:
        return True
    for node in eval_nodes:
        if node.evidence:
            return False
        incident_edges = [*graph.incoming_edges(node.id), *graph.outgoing_edges(node.id)]
        if any(edge.relation in {"supports", "depends_on"} for edge in incident_edges):
            return False
    return True


def _has_open_feasibility_work(graph) -> bool:
    if any(edge.relation == "contradicts" and not edge.resolved for edge in graph.edges):
        return True
    active_risks = _active_nodes(graph, "Risk")
    if not active_risks:
        return False
    repaired_target_ids = {edge.target_id for edge in graph.edges if edge.relation == "repairs"}
    return any(node.id not in repaired_target_ids and not node.evidence for node in active_risks)


def active_roles_for_round(graph, round_name: str) -> tuple[str, ...]:
    if round_index_from_name(round_name) <= 1:
        return tuple(ROLE_NAMES)

    selected: list[str] = []

    needs_problem = _needs_problem_framing(graph)
    needs_mechanism = _needs_mechanism(graph)
    needs_novelty = _needs_novelty_grounding(graph)
    needs_eval = _needs_evaluation(graph)
    needs_feasibility = _has_open_feasibility_work(graph)

    if needs_problem:
        selected.append("ImpactReframer")
    if needs_mechanism or needs_novelty or needs_eval:
        selected.append("MechanismProposer")
    if needs_feasibility:
        selected.append("FeasibilityCritic")
    if needs_novelty:
        selected.append("NoveltyExaminer")
    if needs_eval:
        selected.append("EvaluationDesigner")

    if not selected:
        selected = ["MechanismProposer", "NoveltyExaminer"]
    elif len(selected) == 1:
        selected.append("MechanismProposer" if selected[0] != "MechanismProposer" else "NoveltyExaminer")

    selected_set = set(selected)
    return tuple(role for role in ROLE_NAMES if role in selected_set)
