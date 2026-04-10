from __future__ import annotations

from .models import Edge, IdeaGraph, Node


CLAIM_CHAIN_SLOT_TYPES: dict[str, tuple[str, ...]] = {
    "problem": ("Problem",),
    "gap": ("NoveltyClaim", "EvidenceNeed"),
    "mechanism": ("Method", "Hypothesis"),
    "evaluation": ("EvalPlan",),
    "caveat": ("Risk", "Assumption", "Repair"),
}


def _active_nodes(graph: IdeaGraph, node_ids: set[str] | None = None) -> list[Node]:
    nodes = [node for node in graph.active_nodes() if node.status == "active"]
    if node_ids is None:
        return nodes
    return [node for node in nodes if node.id in node_ids]


def _edges_for_node_ids(graph: IdeaGraph, node_ids: set[str]) -> list[Edge]:
    return [
        edge
        for edge in graph.edges
        if edge.source_id in node_ids and edge.target_id in node_ids
    ]


def _incident_edges(graph: IdeaGraph, node_id: str, *, node_ids: set[str] | None = None) -> list[Edge]:
    edges = [edge for edge in graph.edges if edge.source_id == node_id or edge.target_id == node_id]
    if node_ids is None:
        return edges
    return [edge for edge in edges if edge.source_id in node_ids and edge.target_id in node_ids]


def _weak_context_mode(graph: IdeaGraph) -> bool:
    packet = graph.metadata.get("benchmark_input_packet", {})
    if isinstance(packet, dict) and str(packet.get("keyword", "")).strip():
        return True
    return str(graph.metadata.get("benchmark", "")).strip().casefold() == "liveideabench"


def _node_score(
    graph: IdeaGraph,
    node: Node,
    *,
    anchor_ids: set[str] | None = None,
    node_ids: set[str] | None = None,
) -> float:
    anchor_ids = anchor_ids or set()
    incident = _incident_edges(graph, node.id, node_ids=node_ids)
    connected_to_anchor = any(edge.source_id in anchor_ids or edge.target_id in anchor_ids for edge in incident)
    contradiction_penalty = 0.12 * sum(
        1 for edge in incident if edge.relation == "contradicts" and edge.source_id == node.id
    )
    method_bonus = 0.08 if node.type == "Method" else 0.0
    evaluation_bonus = 0.08 if node.type == "EvalPlan" else 0.0
    evidence_bonus = 0.08 if node.evidence else 0.0
    connectivity_bonus = min(0.18, 0.05 * len(incident))
    anchor_bonus = 0.22 if connected_to_anchor else 0.0
    return round(
        float(node.confidence)
        + method_bonus
        + evaluation_bonus
        + evidence_bonus
        + connectivity_bonus
        + anchor_bonus
        - contradiction_penalty,
        4,
    )


def _best_node_for_slot(
    graph: IdeaGraph,
    slot: str,
    *,
    anchor_ids: set[str] | None = None,
    node_ids: set[str] | None = None,
) -> Node | None:
    slot_types = CLAIM_CHAIN_SLOT_TYPES[slot]
    candidates = [node for node in _active_nodes(graph, node_ids=node_ids) if node.type in slot_types]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda node: (_node_score(graph, node, anchor_ids=anchor_ids, node_ids=node_ids), node.id),
    )


def _supporting_mechanism_context(
    graph: IdeaGraph,
    mechanism: Node | None,
    *,
    node_ids: set[str] | None = None,
) -> list[str]:
    if mechanism is None:
        return []
    neighbor_ids: set[str] = {mechanism.id}
    for edge in _incident_edges(graph, mechanism.id, node_ids=node_ids):
        if edge.relation in {"supports", "refines", "depends_on", "repairs"}:
            neighbor_ids.add(edge.source_id)
            neighbor_ids.add(edge.target_id)
    ordered = [
        node_id
        for node_id in graph.nodes
        if node_id in neighbor_ids and (node_ids is None or node_id in node_ids)
    ]
    return ordered


def select_claim_chain(graph: IdeaGraph, node_ids: set[str] | None = None) -> dict[str, object] | None:
    weak_context_mode = _weak_context_mode(graph)
    problem = _best_node_for_slot(graph, "problem", node_ids=node_ids)
    if problem is None:
        return None

    anchor_ids = {problem.id}
    mechanism = _best_node_for_slot(graph, "mechanism", anchor_ids=anchor_ids, node_ids=node_ids)
    if mechanism is not None:
        anchor_ids.add(mechanism.id)

    evaluation = _best_node_for_slot(graph, "evaluation", anchor_ids=anchor_ids, node_ids=node_ids)
    if evaluation is not None:
        anchor_ids.add(evaluation.id)

    caveat = _best_node_for_slot(graph, "caveat", anchor_ids=anchor_ids, node_ids=node_ids)
    if caveat is not None:
        anchor_ids.add(caveat.id)

    gap = _best_node_for_slot(graph, "gap", anchor_ids=anchor_ids, node_ids=node_ids)
    if gap is not None:
        anchor_ids.add(gap.id)

    relaxed_slots = ["gap"] if weak_context_mode and gap is None else []
    slots = {
        "problem": problem.id if problem is not None else None,
        "gap": gap.id if gap is not None else None,
        "mechanism": mechanism.id if mechanism is not None else None,
        "evaluation": evaluation.id if evaluation is not None else None,
        "caveat": caveat.id if caveat is not None else None,
    }

    required_slots = ["problem", "gap", "mechanism", "evaluation", "caveat"]
    missing_slots = [
        slot
        for slot in required_slots
        if slots.get(slot) is None and slot not in relaxed_slots
    ]
    selected_node_ids = {
        node_id
        for node_id in [
            slots["problem"],
            slots["gap"],
            slots["mechanism"],
            slots["evaluation"],
            slots["caveat"],
            *_supporting_mechanism_context(graph, mechanism, node_ids=node_ids),
        ]
        if node_id
    }
    selected_edges = _edges_for_node_ids(graph, selected_node_ids)
    selected_nodes = [graph.nodes[node_id] for node_id in graph.nodes if node_id in selected_node_ids]

    return {
        "slots": slots,
        "weak_context_mode": weak_context_mode,
        "coverage": {
            "required_slots": required_slots,
            "missing_slots": missing_slots,
            "weak_context_relaxed_slots": relaxed_slots,
            "slot_count": 5 - len(missing_slots),
            "is_synthesis_ready": len(missing_slots) == 0,
        },
        "selected_nodes": [
            {
                "id": node.id,
                "type": node.type,
                "role": node.role,
                "text": node.text,
                "confidence": node.confidence,
                "evidence": list(node.evidence),
            }
            for node in selected_nodes
        ],
        "subgraph": {
            "node_ids": list(selected_node_ids),
            "edge_ids": [edge.id for edge in selected_edges],
            "core_node_ids": [slot_id for slot_id in slots.values() if slot_id],
            "selection_mode": "claim_chain",
        },
    }
