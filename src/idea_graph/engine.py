from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from .agent_backend import (
    ActionDecision,
    CollaborationBackend,
    append_agent_trace,
)
from .collaboration_protocol import (
    ACTION_REQUIRED_PAYLOAD_FIELDS,
    ACTION_TARGET_COUNTS,
    build_round_name,
    resolve_round_phase,
)
from .models import (
    Branch,
    Edge,
    FinalProposal,
    GraphAction,
    IdeaGraph,
    MaturitySnapshot,
    Node,
    Provenance,
)
from .schema import ROLE_NAMES, build_seed_template

def normalize_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def find_active_nodes(
    graph: IdeaGraph,
    node_type: str,
    *,
    role: str | None = None,
    exclude_role: str | None = None,
    without_evidence: bool = False,
) -> list[Node]:
    candidates = [node for node in graph.active_nodes() if node.type == node_type]
    if role is not None:
        candidates = [node for node in candidates if node.role == role]
    if exclude_role is not None:
        candidates = [node for node in candidates if node.role != exclude_role]
    if without_evidence:
        candidates = [node for node in candidates if not node.evidence]
    return sorted(candidates, key=lambda node: node.id)


def first_active_node(
    graph: IdeaGraph,
    node_type: str,
    *,
    role: str | None = None,
    exclude_role: str | None = None,
    without_evidence: bool = False,
) -> Node:
    candidates = find_active_nodes(
        graph,
        node_type,
        role=role,
        exclude_role=exclude_role,
        without_evidence=without_evidence,
    )
    if not candidates:
        raise ValueError(
            f"Could not find active node of type={node_type}, role={role}, exclude_role={exclude_role}"
        )
    return candidates[0]


def first_active_node_with_fallback(
    graph: IdeaGraph,
    node_type: str,
    *,
    role: str | None = None,
    exclude_role: str | None = None,
    prefer_without_evidence: bool = False,
) -> Node:
    if prefer_without_evidence:
        candidates = find_active_nodes(
            graph,
            node_type,
            role=role,
            exclude_role=exclude_role,
            without_evidence=True,
        )
        if candidates:
            return candidates[0]
    return first_active_node(
        graph,
        node_type,
        role=role,
        exclude_role=exclude_role,
        without_evidence=False,
    )


def first_available_node(
    graph: IdeaGraph,
    *,
    node_types: tuple[str, ...] | list[str],
    preferred_roles: tuple[str, ...] | list[str] = (),
    exclude_role: str | None = None,
    prefer_without_evidence: bool = False,
) -> Node:
    normalized_types = tuple(node_types)
    normalized_roles = tuple(preferred_roles)

    def _attempt(without_evidence: bool) -> Node | None:
        for node_type in normalized_types:
            for preferred_role in normalized_roles:
                candidates = find_active_nodes(
                    graph,
                    node_type,
                    role=preferred_role,
                    exclude_role=exclude_role,
                    without_evidence=without_evidence,
                )
                if candidates:
                    return candidates[0]
            candidates = find_active_nodes(
                graph,
                node_type,
                exclude_role=exclude_role,
                without_evidence=without_evidence,
            )
            if candidates:
                return candidates[0]
        return None

    candidate = _attempt(prefer_without_evidence)
    if candidate is not None:
        return candidate
    if prefer_without_evidence:
        candidate = _attempt(False)
        if candidate is not None:
            return candidate
    raise ValueError(
        "Could not find an active fallback node for "
        f"types={normalized_types}, preferred_roles={normalized_roles}, exclude_role={exclude_role}"
    )


def edge_exists(
    graph: IdeaGraph,
    *,
    source_id: str,
    relation: str,
    target_id: str,
) -> bool:
    return any(
        edge.source_id == source_id and edge.relation == relation and edge.target_id == target_id
        for edge in graph.edges
    )


def unresolved_contradiction_edges(graph: IdeaGraph) -> list[Edge]:
    return [
        edge
        for edge in graph.edges
        if edge.relation == "contradicts"
        and not edge.resolved
        and edge.source_id in graph.nodes
        and edge.target_id in graph.nodes
        and graph.nodes[edge.source_id].status == "active"
        and graph.nodes[edge.target_id].status == "active"
    ]


REPAIR_TARGET_TYPE_PREFERENCES = {
    "MechanismProposer": ("Method", "Hypothesis", "EvalPlan", "Problem", "NoveltyClaim", "Risk", "Assumption"),
    "FeasibilityCritic": ("EvalPlan", "Method", "Hypothesis", "Problem", "NoveltyClaim", "Risk", "Assumption"),
    "NoveltyExaminer": ("NoveltyClaim", "Hypothesis", "Problem", "Method", "EvalPlan", "Risk", "Assumption"),
    "EvaluationDesigner": ("EvalPlan", "Method", "Hypothesis", "Problem", "NoveltyClaim", "Risk", "Assumption"),
    "ImpactReframer": ("Problem", "Hypothesis", "NoveltyClaim", "Method", "EvalPlan", "Risk", "Assumption"),
}


def choose_repair_target(graph: IdeaGraph, role: str) -> Node | None:
    unresolved = unresolved_contradiction_edges(graph)
    if not unresolved:
        return None

    preferred_types = REPAIR_TARGET_TYPE_PREFERENCES.get(role, tuple())
    targets = [graph.nodes[edge.target_id] for edge in unresolved if edge.target_id in graph.nodes]

    for preferred_type in preferred_types:
        for candidate in targets:
            if candidate.type == preferred_type and candidate.role != role:
                return candidate
        for candidate in targets:
            if candidate.type == preferred_type:
                return candidate

    for candidate in targets:
        if candidate.role != role:
            return candidate
    return targets[0] if targets else None


def contradiction_related_node_ids(graph: IdeaGraph) -> set[str]:
    related: set[str] = set()
    for edge in unresolved_contradiction_edges(graph):
        related.add(edge.source_id)
        related.add(edge.target_id)
    return related


def llm_action_alignment_error(graph: IdeaGraph, round_name: str, action: GraphAction) -> str | None:
    phase = resolve_round_phase(round_name)
    if phase.key != "repair":
        return None

    unresolved = unresolved_contradiction_edges(graph)
    if not unresolved:
        return None

    contradiction_target_ids = {edge.target_id for edge in unresolved}
    contradiction_node_ids = contradiction_related_node_ids(graph)

    if action.kind == "propose_repair":
        if action.target_ids and action.target_ids[0] in contradiction_target_ids:
            return None
        return (
            "Repair-phase action proposed a repair that does not target any unresolved contradiction target. "
            "Use a contradiction target or let deterministic fallback handle the repair."
        )

    if action.kind == "attach_evidence":
        if action.target_ids and action.target_ids[0] in contradiction_node_ids:
            return None
        return (
            "Repair-phase attach_evidence action does not address any unresolved contradiction-related node. "
            "Use a contradiction-related node or let deterministic fallback handle the repair."
        )

    if action.kind == "freeze_branch":
        return (
            "Repair-phase freeze_branch action was rejected because unresolved contradictions still exist."
        )

    if action.kind == "add_support_edge":
        if any(node_id in contradiction_node_ids for node_id in action.target_ids):
            return None
        return (
            "Repair-phase add_support_edge action does not touch any unresolved contradiction-related node. "
            "Use a contradiction-related node or let deterministic fallback handle the repair."
        )

    return None


def branch_for_role(graph: IdeaGraph, role: str) -> Branch:
    return next(branch for branch in graph.branches.values() if branch.role == role)


def literature_item(graph: IdeaGraph, index: int) -> str:
    if not graph.literature:
        return "No literature item available."
    return graph.literature[index % len(graph.literature)]


def create_branch(graph: IdeaGraph, role: str) -> Branch:
    branch = Branch(id=graph.next_branch_id(), role=role)
    graph.branches[branch.id] = branch
    return branch


def create_node(
    graph: IdeaGraph,
    *,
    node_type: str,
    text: str,
    role: str,
    branch_id: str,
    confidence: float,
    evidence: list[str] | None = None,
    source: str = "seed",
) -> Node:
    node = Node(
        id=graph.next_node_id(),
        type=node_type,
        text=text,
        role=role,
        branch_id=branch_id,
        confidence=round(confidence, 2),
        evidence=list(evidence or []),
        provenance=[Provenance(role=role, branch_id=branch_id, source=source)],
    )
    graph.nodes[node.id] = node
    graph.branches[branch_id].node_ids.append(node.id)
    return node


def create_edge(
    graph: IdeaGraph,
    *,
    source_id: str,
    relation: str,
    target_id: str,
    role: str,
    branch_id: str,
    evidence_id: str | None = None,
    note: str = "",
) -> Edge:
    edge = Edge(
        id=graph.next_edge_id(),
        source_id=source_id,
        relation=relation,
        target_id=target_id,
        role=role,
        branch_id=branch_id,
        evidence_id=evidence_id,
        note=note,
    )
    graph.edges.append(edge)
    graph.branches[branch_id].edge_ids.append(edge.id)
    return edge


def build_seed_graphs(graph: IdeaGraph) -> None:
    for role in ROLE_NAMES:
        template = build_seed_template(
            role,
            graph.topic,
            literature=graph.literature,
            metadata=graph.metadata,
        )
        branch = create_branch(graph, role)
        anchor = create_node(
            graph,
            node_type=template.anchor_type,
            text=template.anchor_text,
            role=role,
            branch_id=branch.id,
            confidence=0.72,
        )
        for node_type, text in template.support_nodes:
            support_node = create_node(
                graph,
                node_type=node_type,
                text=text,
                role=role,
                branch_id=branch.id,
                confidence=0.64,
            )
            relation = "contradicts" if node_type == "Risk" else "supports"
            create_edge(
                graph,
                source_id=support_node.id,
                relation=relation,
                target_id=anchor.id,
                role=role,
                branch_id=branch.id,
                note=f"Seed relation from {role}",
            )


def build_seed_graphs_with_backend(graph: IdeaGraph, backend: CollaborationBackend) -> None:
    for role in ROLE_NAMES:
        draft = backend.generate_seed(graph, role)
        append_agent_trace(graph, stage="seed_generation", role=role, trace=draft.trace)

        branch = create_branch(graph, role)
        anchor = create_node(
            graph,
            node_type=draft.anchor_type,
            text=draft.anchor_text,
            role=role,
            branch_id=branch.id,
            confidence=draft.anchor_confidence,
            source="llm_seed",
        )
        for support in draft.support_nodes:
            support_node = create_node(
                graph,
                node_type=support.type,
                text=support.text,
                role=role,
                branch_id=branch.id,
                confidence=support.confidence,
                source="llm_seed",
            )
            relation = support.relation_to_anchor if support.relation_to_anchor in {
                "supports",
                "contradicts",
                "refines",
                "depends_on",
                "requires_evidence",
                "overlaps_prior",
                "repairs",
            } else "supports"
            create_edge(
                graph,
                source_id=support_node.id,
                relation=relation,
                target_id=anchor.id,
                role=role,
                branch_id=branch.id,
                note=draft.rationale or f"LLM seed relation from {role}",
            )


def merge_seed_graphs(graph: IdeaGraph) -> None:
    canonical: dict[tuple[str, str], Node] = {}
    for node in list(graph.nodes.values()):
        key = (node.type, normalize_text(node.text))
        existing = canonical.get(key)
        if existing is None:
            canonical[key] = node
            continue
        existing.provenance.extend(node.provenance)
        node.status = f"merged_into:{existing.id}"

    for branch in graph.branches.values():
        anchor_id = branch.node_ids[0]
        for node_id in branch.node_ids[1:]:
            create_edge(
                graph,
                source_id=node_id,
                relation="refines",
                target_id=anchor_id,
                role=branch.role,
                branch_id=branch.id,
                note="Automatic branch cohesion relation after merge.",
            )

    problems = [node for node in graph.active_nodes() if node.type == "Problem"]
    hypotheses = [node for node in graph.active_nodes() if node.type == "Hypothesis"]
    for problem in problems:
        for hypothesis in hypotheses:
            if problem.branch_id == hypothesis.branch_id:
                continue
            create_edge(
                graph,
                source_id=hypothesis.id,
                relation="supports",
                target_id=problem.id,
                role=hypothesis.role,
                branch_id=hypothesis.branch_id,
                note="Cross-branch support suggestion after merge.",
            )


def focused_view(graph: IdeaGraph, role: str) -> dict[str, object]:
    promising_branches = sorted(
        (branch for branch in graph.branches.values() if not branch.rejected),
        key=lambda branch: len(branch.node_ids),
        reverse=True,
    )[:3]
    low_support_nodes = [
        node
        for node in graph.active_nodes()
        if node.type in {"Hypothesis", "Method", "NoveltyClaim", "EvalPlan"} and not node.evidence
    ][:3]
    unsupported_novelty = [
        node
        for node in graph.active_nodes()
        if node.type == "NoveltyClaim"
        and not any(edge.relation == "supports" for edge in graph.incoming_edges(node.id))
    ][:2]
    unresolved_contradictions = [
        edge for edge in graph.edges if edge.relation == "contradicts" and not edge.resolved
    ][:2]
    return {
        "role": role,
        "promising_branches": [branch.id for branch in promising_branches],
        "unsupported_novelty_claims": [node.id for node in unsupported_novelty],
        "unresolved_contradictions": [edge.id for edge in unresolved_contradictions],
        "low_support_nodes": [node.id for node in low_support_nodes],
    }


def make_action(
    graph: IdeaGraph,
    *,
    round_name: str,
    role: str,
    kind: str,
    target_ids: list[str],
    payload: dict[str, object] | None = None,
    rationale: str = "",
) -> GraphAction:
    return GraphAction(
        id=graph.next_action_id(),
        round_name=round_name,
        role=role,
        kind=kind,
        target_ids=target_ids,
        payload=dict(payload or {}),
        rationale=rationale,
    )


def choose_round_action(graph: IdeaGraph, round_name: str, role: str) -> GraphAction:
    branch = branch_for_role(graph, role)
    phase = resolve_round_phase(round_name)
    _view = focused_view(graph, role)

    if phase.key == "structure":
        if role == "MechanismProposer":
            hypothesis = first_available_node(
                graph,
                node_types=("Hypothesis", "Method", "NoveltyClaim"),
                preferred_roles=("MechanismProposer", "ImpactReframer"),
            )
            problem = first_available_node(
                graph,
                node_types=("Problem", "Hypothesis"),
                preferred_roles=("ImpactReframer",),
            )
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="add_support_edge",
                target_ids=[hypothesis.id, problem.id],
                payload={"branch_id": branch.id},
                rationale="Expose a mechanism-to-problem support relation early.",
            )
        if role == "FeasibilityCritic":
            risk = first_available_node(
                graph,
                node_types=("Risk", "Assumption"),
                preferred_roles=("FeasibilityCritic", "EvaluationDesigner"),
            )
            method = first_available_node(
                graph,
                node_types=("Method", "EvalPlan", "Hypothesis"),
                preferred_roles=("EvaluationDesigner", "MechanismProposer"),
                exclude_role=role,
            )
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="add_contradiction_edge",
                target_ids=[risk.id, method.id],
                payload={"branch_id": branch.id},
                rationale="Surface a concrete feasibility risk against the proposed method.",
            )
        if role == "NoveltyExaminer":
            novelty_claim = first_available_node(
                graph,
                node_types=("NoveltyClaim", "Hypothesis", "Problem"),
                preferred_roles=("NoveltyExaminer", "ImpactReframer"),
            )
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="request_evidence",
                target_ids=[novelty_claim.id],
                payload={
                    "branch_id": branch.id,
                    "query": "Find prior work on graph-based multi-agent scientific ideation.",
                },
                rationale="Require explicit evidence before accepting the novelty claim.",
            )
        if role == "EvaluationDesigner":
            eval_plan = first_available_node(
                graph,
                node_types=("EvalPlan", "Method"),
                preferred_roles=("EvaluationDesigner", "FeasibilityCritic"),
            )
            hypothesis = first_available_node(
                graph,
                node_types=("Hypothesis", "Problem", "Method"),
                preferred_roles=("MechanismProposer", "ImpactReframer"),
            )
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="add_dependency_edge",
                target_ids=[eval_plan.id, hypothesis.id],
                payload={"branch_id": branch.id},
                rationale="Make the evaluation plan explicitly depend on the main hypothesis.",
            )
        hypothesis = first_available_node(
            graph,
            node_types=("Hypothesis", "NoveltyClaim", "Method"),
            preferred_roles=("ImpactReframer", "MechanismProposer"),
        )
        problem = first_available_node(
            graph,
            node_types=("Problem", "Hypothesis"),
            preferred_roles=("ImpactReframer",),
        )
        return make_action(
            graph,
            round_name=round_name,
            role=role,
            kind="add_support_edge",
            target_ids=[hypothesis.id, problem.id],
            payload={"branch_id": branch.id},
            rationale="Clarify how the reframed hypothesis supports the motivating problem.",
        )

    if phase.key == "stress_test":
        if role == "MechanismProposer":
            contradiction_related = contradiction_related_node_ids(graph)
            contradiction_targets = [
                graph.nodes[node_id]
                for node_id in contradiction_related
                if node_id in graph.nodes
                and graph.nodes[node_id].status == "active"
                and graph.nodes[node_id].role != role
                and graph.nodes[node_id].type in {"Method", "EvalPlan", "Hypothesis", "Problem"}
                and not graph.nodes[node_id].evidence
            ]
            if contradiction_targets:
                target = sorted(contradiction_targets, key=lambda node: node.id)[0]
                return make_action(
                    graph,
                    round_name=round_name,
                    role=role,
                    kind="attach_evidence",
                    target_ids=[target.id],
                    payload={"branch_id": branch.id, "evidence": literature_item(graph, 0)},
                    rationale="Ground a contradiction-related node before deciding whether repair is needed.",
                )
            method = first_available_node(
                graph,
                node_types=("Method", "EvalPlan", "Hypothesis"),
                preferred_roles=("EvaluationDesigner", "ImpactReframer", "MechanismProposer"),
                exclude_role=role,
                prefer_without_evidence=True,
            )
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="attach_evidence",
                target_ids=[method.id],
                payload={"branch_id": branch.id, "evidence": literature_item(graph, 0)},
                rationale="Ground a method node from a branch created by another role.",
            )
        if role == "FeasibilityCritic":
            contradiction_targets = [
                graph.nodes[edge.target_id]
                for edge in unresolved_contradiction_edges(graph)
                if edge.target_id in graph.nodes
                and graph.nodes[edge.target_id].status == "active"
                and graph.nodes[edge.target_id].role != role
                and not graph.nodes[edge.target_id].evidence
            ]
            if contradiction_targets:
                target = sorted(contradiction_targets, key=lambda node: node.id)[0]
                return make_action(
                    graph,
                    round_name=round_name,
                    role=role,
                    kind="attach_evidence",
                    target_ids=[target.id],
                    payload={"branch_id": branch.id, "evidence": literature_item(graph, 1)},
                    rationale="Attach evidence to a contradiction target before deciding whether the critique still holds.",
                )
            hypothesis = first_available_node(
                graph,
                node_types=("Hypothesis", "Problem", "Method", "NoveltyClaim"),
                preferred_roles=("ImpactReframer", "MechanismProposer"),
                exclude_role=role,
                prefer_without_evidence=True,
            )
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="attach_evidence",
                target_ids=[hypothesis.id],
                payload={"branch_id": branch.id, "evidence": literature_item(graph, 1)},
                rationale="Attach evidence to a non-self branch before accepting its feasibility.",
            )
        if role == "NoveltyExaminer":
            novelty_claim = first_available_node(
                graph,
                node_types=("NoveltyClaim", "Hypothesis", "Problem"),
                preferred_roles=("ImpactReframer", "NoveltyExaminer"),
                exclude_role=role,
                prefer_without_evidence=True,
            )
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="mark_overlap",
                target_ids=[novelty_claim.id],
                payload={
                    "branch_id": branch.id,
                    "paper_id": "paper-001",
                    "evidence": literature_item(graph, 2),
                },
                rationale="Ground a non-self novelty claim against nearby prior work.",
            )
        if role == "EvaluationDesigner":
            contradiction_targets = [
                graph.nodes[edge.target_id]
                for edge in unresolved_contradiction_edges(graph)
                if edge.target_id in graph.nodes
                and graph.nodes[edge.target_id].status == "active"
                and graph.nodes[edge.target_id].type in {"EvalPlan", "Method", "Hypothesis"}
                and not graph.nodes[edge.target_id].evidence
            ]
            if contradiction_targets:
                target = sorted(contradiction_targets, key=lambda node: node.id)[0]
                return make_action(
                    graph,
                    round_name=round_name,
                    role=role,
                    kind="attach_evidence",
                    target_ids=[target.id],
                    payload={"branch_id": branch.id, "evidence": literature_item(graph, 3)},
                    rationale="Attach evidence to a contradiction target so later repair decisions are better grounded.",
                )
            eval_plan = first_available_node(
                graph,
                node_types=("EvalPlan", "Method", "Hypothesis"),
                preferred_roles=("FeasibilityCritic", "EvaluationDesigner", "MechanismProposer"),
                exclude_role=role,
                prefer_without_evidence=True,
            )
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="attach_evidence",
                target_ids=[eval_plan.id],
                payload={"branch_id": branch.id, "evidence": literature_item(graph, 3)},
                rationale="Attach validation-oriented evidence to another branch's evaluation node.",
            )
        novelty_claim = first_available_node(
            graph,
            node_types=("NoveltyClaim", "Hypothesis", "Problem"),
            preferred_roles=("NoveltyExaminer", "ImpactReframer"),
            exclude_role=role,
            prefer_without_evidence=True,
        )
        return make_action(
            graph,
            round_name=round_name,
            role=role,
            kind="attach_evidence",
            target_ids=[novelty_claim.id],
            payload={"branch_id": branch.id, "evidence": literature_item(graph, 0)},
            rationale="Ground the main novelty branch from a different epistemic perspective.",
        )

    if role == "MechanismProposer":
        repair_target = choose_repair_target(graph, role)
        if repair_target is not None:
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="propose_repair",
                target_ids=[repair_target.id],
                payload={
                    "branch_id": branch.id,
                    "repair_text": (
                        f"Refine the {repair_target.type.lower()} so it directly addresses the unresolved failure mode "
                        f"and becomes more mechanistically specific."
                    ),
                },
                rationale="Resolve an unresolved contradiction by making the target node more mechanistically precise.",
            )
        method = first_available_node(
            graph,
            node_types=("Method", "EvalPlan", "Hypothesis"),
            preferred_roles=("EvaluationDesigner", "MechanismProposer"),
            exclude_role=role,
        )
        return make_action(
            graph,
            round_name=round_name,
            role=role,
            kind="propose_repair",
            target_ids=[method.id],
            payload={
                "branch_id": branch.id,
                "repair_text": "Constrain the method with a compact action schema and one edit per role per round.",
            },
            rationale="Repair the method after feasibility critique.",
        )
    if role == "FeasibilityCritic":
        repair_target = choose_repair_target(graph, role)
        if repair_target is not None:
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="propose_repair",
                target_ids=[repair_target.id],
                payload={
                    "branch_id": branch.id,
                    "repair_text": (
                        f"Adjust the {repair_target.type.lower()} to answer the unresolved feasibility concern "
                        f"with clearer constraints, assumptions, or evaluation details."
                    ),
                },
                rationale="Resolve an unresolved contradiction by tightening feasibility around the target node.",
            )
        eval_plan = first_available_node(
            graph,
            node_types=("EvalPlan", "Method", "Hypothesis"),
            preferred_roles=("EvaluationDesigner", "FeasibilityCritic"),
            exclude_role=role,
        )
        return make_action(
            graph,
            round_name=round_name,
            role=role,
            kind="propose_repair",
            target_ids=[eval_plan.id],
            payload={
                "branch_id": branch.id,
                "repair_text": "Add explicit budget matching and process metrics to stabilize evaluation claims.",
            },
            rationale="Repair the evaluation plan after contradiction exposure.",
        )
    if role == "NoveltyExaminer":
        repair_target = choose_repair_target(graph, role)
        if repair_target is not None:
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="propose_repair",
                target_ids=[repair_target.id],
                payload={
                    "branch_id": branch.id,
                    "repair_text": (
                        f"Narrow the {repair_target.type.lower()} so it is more clearly differentiated from nearby prior work "
                        f"and easier to justify as novel."
                    ),
                },
                rationale="Resolve an unresolved contradiction by making the target node more specific and defensible.",
            )
        novelty_claim = first_available_node(
            graph,
            node_types=("NoveltyClaim", "Hypothesis", "Problem"),
            preferred_roles=("ImpactReframer", "NoveltyExaminer"),
            exclude_role=role,
        )
        return make_action(
            graph,
            round_name=round_name,
            role=role,
            kind="propose_repair",
            target_ids=[novelty_claim.id],
            payload={
                "branch_id": branch.id,
                "repair_text": "Narrow the novelty claim to delayed consensus over typed claim fragments.",
            },
            rationale="Refine the novelty claim after overlap analysis.",
        )
    if role == "EvaluationDesigner":
        repair_target = choose_repair_target(graph, role)
        if repair_target is not None:
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="propose_repair",
                target_ids=[repair_target.id],
                payload={
                    "branch_id": branch.id,
                    "repair_text": (
                        f"Revise the {repair_target.type.lower()} so its evaluation path and success criteria are explicit "
                        f"enough to resolve the current contradiction."
                    ),
                },
                rationale="Resolve an unresolved contradiction by clarifying how the target node will be evaluated or supported.",
            )
        method = first_available_node(
            graph,
            node_types=("Method", "EvalPlan", "Hypothesis"),
            preferred_roles=("MechanismProposer", "EvaluationDesigner"),
        )
        hypothesis = first_available_node(
            graph,
            node_types=("Hypothesis", "Problem", "NoveltyClaim"),
            preferred_roles=("ImpactReframer", "MechanismProposer"),
        )
        return make_action(
            graph,
            round_name=round_name,
            role=role,
            kind="add_support_edge",
            target_ids=[method.id, hypothesis.id],
            payload={"branch_id": branch.id},
            rationale="Reconnect compatible pieces from different branches into one stronger path.",
        )

    repair_target = choose_repair_target(graph, role)
    if repair_target is not None:
        return make_action(
            graph,
            round_name=round_name,
            role=role,
            kind="propose_repair",
            target_ids=[repair_target.id],
            payload={
                "branch_id": branch.id,
                "repair_text": (
                    f"Reframe the {repair_target.type.lower()} so the broader problem framing and significance are clearer "
                    f"while directly addressing the unresolved contradiction."
                ),
            },
            rationale="Resolve an unresolved contradiction by reframing the target node more clearly.",
        )
    problem_branch = branch_for_role(graph, "ImpactReframer")
    return make_action(
        graph,
        round_name=round_name,
        role=role,
        kind="freeze_branch",
        target_ids=[],
        payload={"branch_id": problem_branch.id},
        rationale="Preserve the impact branch as a traceable alternative while final synthesis happens elsewhere.",
    )


def action_from_decision(
    graph: IdeaGraph,
    *,
    round_name: str,
    role: str,
    decision: ActionDecision,
) -> GraphAction:
    action = make_action(
        graph,
        round_name=round_name,
        role=role,
        kind=decision.kind,
        target_ids=list(decision.target_ids),
        payload=dict(decision.payload),
        rationale=decision.rationale,
    )
    validate_action(graph, action)
    return action


def validate_action(graph: IdeaGraph, action: GraphAction) -> None:
    expected_targets = ACTION_TARGET_COUNTS.get(action.kind)
    if expected_targets is None:
        raise ValueError(f"Unsupported action kind: {action.kind}")

    branch_id = str(action.payload.get("branch_id", "")).strip()
    if not branch_id:
        raise ValueError(f"Action {action.id} ({action.kind}) is missing payload.branch_id.")
    if branch_id not in graph.branches:
        raise ValueError(f"Action {action.id} ({action.kind}) referenced unknown branch '{branch_id}'.")

    if len(action.target_ids) != expected_targets:
        raise ValueError(
            f"Action {action.id} ({action.kind}) expected {expected_targets} target ids "
            f"but received {len(action.target_ids)}."
        )

    for node_id in action.target_ids:
        if node_id not in graph.nodes:
            raise ValueError(f"Action {action.id} ({action.kind}) referenced unknown node '{node_id}'.")

    for field_name in ACTION_REQUIRED_PAYLOAD_FIELDS.get(action.kind, ()):
        value = action.payload.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"Action {action.id} ({action.kind}) is missing payload.{field_name}.")


def emit_progress(
    graph: IdeaGraph,
    progress_callback: Callable[[str], None] | None,
    *,
    stage: str,
    message: str,
    details: dict[str, object] | None = None,
) -> None:
    entry: dict[str, object] = {"stage": stage, "message": message}
    if details:
        entry["details"] = details
    log = graph.metadata.setdefault("progress_log", [])
    if isinstance(log, list):
        log.append(entry)
    if progress_callback is not None:
        progress_callback(message)


def apply_action(graph: IdeaGraph, action: GraphAction) -> None:
    validate_action(graph, action)
    graph.actions.append(action)
    branch_id = str(action.payload["branch_id"])

    if action.kind == "add_support_edge":
        create_edge(
            graph,
            source_id=action.target_ids[0],
            relation="supports",
            target_id=action.target_ids[1],
            role=action.role,
            branch_id=branch_id,
            note=action.rationale,
        )
        return

    if action.kind == "add_contradiction_edge":
        create_edge(
            graph,
            source_id=action.target_ids[0],
            relation="contradicts",
            target_id=action.target_ids[1],
            role=action.role,
            branch_id=branch_id,
            note=action.rationale,
        )
        return

    if action.kind == "add_dependency_edge":
        create_edge(
            graph,
            source_id=action.target_ids[0],
            relation="depends_on",
            target_id=action.target_ids[1],
            role=action.role,
            branch_id=branch_id,
            note=action.rationale,
        )
        return

    if action.kind == "request_evidence":
        create_edge(
            graph,
            source_id=action.target_ids[0],
            relation="requires_evidence",
            target_id=action.target_ids[0],
            role=action.role,
            branch_id=branch_id,
            note=str(action.payload.get("query", action.rationale)),
        )
        return

    if action.kind == "attach_evidence":
        node = graph.nodes[action.target_ids[0]]
        evidence = str(action.payload["evidence"])
        if evidence not in node.evidence:
            node.evidence.append(evidence)
        return

    if action.kind == "mark_overlap":
        node = graph.nodes[action.target_ids[0]]
        evidence = str(action.payload.get("evidence", ""))
        if evidence and evidence not in node.evidence:
            node.evidence.append(evidence)
        create_edge(
            graph,
            source_id=action.target_ids[0],
            relation="overlaps_prior",
            target_id=action.target_ids[0],
            role=action.role,
            branch_id=branch_id,
            evidence_id=str(action.payload["paper_id"]),
            note=action.rationale,
        )
        return

    if action.kind == "propose_repair":
        repair = create_node(
            graph,
            node_type="Repair",
            text=str(action.payload["repair_text"]),
            role=action.role,
            branch_id=branch_id,
            confidence=0.61,
            source="collaboration",
        )
        create_edge(
            graph,
            source_id=repair.id,
            relation="repairs",
            target_id=action.target_ids[0],
            role=action.role,
            branch_id=branch_id,
            note=action.rationale,
        )
        for edge in graph.edges:
            if edge.relation == "contradicts" and edge.target_id == action.target_ids[0]:
                edge.resolved = True
        return

    if action.kind == "freeze_branch":
        graph.branches[branch_id].frozen = True
        graph.branches[branch_id].notes.append(action.rationale)
        return

    raise ValueError(f"Unsupported action kind: {action.kind}")


def utility_score(graph: IdeaGraph) -> float:
    active_nodes = graph.active_nodes()
    repair_count = sum(1 for node in active_nodes if node.type == "Repair")
    novelty_count = sum(1 for node in active_nodes if node.type == "NoveltyClaim")
    support_count = sum(1 for edge in graph.edges if edge.relation == "supports")
    contradiction_count = sum(
        1 for edge in graph.edges if edge.relation == "contradicts" and not edge.resolved
    )
    return round((1.5 * repair_count) + (1.2 * novelty_count) + (0.5 * support_count) - (0.8 * contradiction_count), 2)


def maturity_snapshot(graph: IdeaGraph) -> MaturitySnapshot:
    tracked_types = {"Hypothesis", "Method", "NoveltyClaim", "EvalPlan"}
    tracked_nodes = [node for node in graph.active_nodes() if node.type in tracked_types]

    supported_count = 0
    for node in tracked_nodes:
        incoming = graph.incoming_edges(node.id)
        has_support = any(edge.relation == "supports" for edge in incoming)
        has_repair = any(edge.relation == "repairs" for edge in incoming)
        if has_support or has_repair or node.evidence:
            supported_count += 1

    support_coverage = 0.0 if not tracked_nodes else round(supported_count / len(tracked_nodes), 2)
    contradictions = [edge for edge in graph.edges if edge.relation == "contradicts"]
    unresolved = [edge for edge in contradictions if not edge.resolved]
    unresolved_ratio = 0.0 if not contradictions else round(len(unresolved) / len(contradictions), 2)

    active_types = {node.type for node in graph.active_nodes()}
    completeness = {"Problem", "Hypothesis", "Method", "EvalPlan"}.issubset(active_types)

    utility = utility_score(graph)
    graph.utility_history.append(utility)
    utility_stable = False
    if len(graph.utility_history) >= 3:
        a, b, c = graph.utility_history[-3:]
        utility_stable = abs(c - b) < 0.15 and abs(b - a) < 0.15

    return MaturitySnapshot(
        support_coverage=support_coverage,
        unresolved_contradiction_ratio=unresolved_ratio,
        utility=utility,
        utility_stable=utility_stable,
        completeness=completeness,
        is_mature=support_coverage >= 0.6 and unresolved_ratio <= 0.5 and completeness and utility_stable,
    )


def select_final_subgraph(graph: IdeaGraph) -> dict[str, object]:
    selected_nodes = []
    for node_type in ("Problem", "Hypothesis", "Method", "EvalPlan"):
        candidates = [node for node in graph.active_nodes() if node.type == node_type]
        if candidates:
            selected_nodes.append(max(candidates, key=lambda node: node.confidence))

    selected_node_ids = {node.id for node in selected_nodes}
    selected_edges = [
        edge
        for edge in graph.edges
        if edge.source_id in selected_node_ids and edge.target_id in selected_node_ids
    ]
    return {
        "node_ids": [node.id for node in selected_nodes],
        "edge_ids": [edge.id for edge in selected_edges],
        "utility": utility_score(graph),
    }


def synthesize_proposal(graph: IdeaGraph, subgraph: dict[str, object]) -> FinalProposal:
    selected = {graph.nodes[node_id].type: graph.nodes[node_id].text for node_id in subgraph["node_ids"]}
    return FinalProposal(
        problem=selected.get("Problem", ""),
        hypothesis=selected.get("Hypothesis", ""),
        method=selected.get("Method", ""),
        evaluation=selected.get("EvalPlan", ""),
        significance="A typed graph may preserve disagreement long enough to improve proposal selection.",
        caveats="The current scaffold uses deterministic placeholder actions and heuristic deduplication.",
    )


def run_experiment(
    topic: str,
    literature: list[str],
    metadata: dict[str, object] | None = None,
    collaboration_backend: CollaborationBackend | None = None,
    progress_callback: Callable[[str], None] | None = None,
    max_rounds: int = 3,
    stop_when_mature: bool = True,
) -> IdeaGraph:
    graph = IdeaGraph(topic=topic, literature=literature, metadata=dict(metadata or {}))
    graph.metadata["max_rounds_requested"] = max(1, int(max_rounds))
    graph.metadata["stop_when_mature"] = bool(stop_when_mature)
    backend_name = collaboration_backend.name if collaboration_backend is not None else "deterministic"
    emit_progress(
        graph,
        progress_callback,
        stage="start",
        message=(
            f"Initialized idea graph for topic '{topic}' using backend '{backend_name}' "
            f"with up to {max(1, int(max_rounds))} rounds."
        ),
        details={"backend": backend_name, "topic": topic, "max_rounds": max(1, int(max_rounds))},
    )
    if collaboration_backend is None:
        emit_progress(
            graph,
            progress_callback,
            stage="seed_generation",
            message="Building deterministic seed graphs.",
        )
        build_seed_graphs(graph)
    else:
        try:
            emit_progress(
                graph,
                progress_callback,
                stage="seed_generation",
                message="Generating seed graphs with the OpenAI-compatible backend.",
            )
            build_seed_graphs_with_backend(graph, collaboration_backend)
        except Exception as exc:
            graph.metadata["seed_generation_error"] = str(exc)
            graph.metadata["seed_generation_fallback"] = "deterministic"
            emit_progress(
                graph,
                progress_callback,
                stage="seed_generation_fallback",
                message=f"Seed generation failed with the LLM backend, falling back to deterministic seeds: {exc}",
                details={"error": str(exc)},
            )
            build_seed_graphs(graph)
    emit_progress(
        graph,
        progress_callback,
        stage="seed_generation_complete",
        message=f"Seed graphs ready: {len(graph.branches)} branches, {len(graph.nodes)} nodes, {len(graph.edges)} edges.",
        details={"branches": len(graph.branches), "nodes": len(graph.nodes), "edges": len(graph.edges)},
    )
    merge_seed_graphs(graph)
    emit_progress(
        graph,
        progress_callback,
        stage="merge_complete",
        message=f"Seed merge complete: {len(graph.active_nodes())} active nodes, {len(graph.edges)} edges.",
        details={"active_nodes": len(graph.active_nodes()), "edges": len(graph.edges)},
    )

    for round_index in range(1, max(1, int(max_rounds)) + 1):
        round_name = build_round_name(round_index)
        phase = resolve_round_phase(round_name)
        emit_progress(
            graph,
            progress_callback,
            stage="round_start",
            message=f"{round_name} started with phase '{phase.title}'.",
            details={"round": round_name, "phase": phase.key, "phase_title": phase.title},
        )
        for role in ROLE_NAMES:
            action_source = "deterministic"
            if collaboration_backend is None:
                action = choose_round_action(graph, round_name, role)
            else:
                try:
                    decision = collaboration_backend.choose_action(graph, round_name, role)
                    append_agent_trace(graph, stage=f"{round_name}_action", role=role, trace=decision.trace)
                    action = action_from_decision(
                        graph,
                        round_name=round_name,
                        role=role,
                        decision=decision,
                    )
                    alignment_error = llm_action_alignment_error(graph, round_name, action)
                    if alignment_error is not None:
                        raise ValueError(alignment_error)
                    action_source = "llm"
                except Exception as exc:
                    graph.metadata.setdefault("action_errors", []).append(
                        {"round": round_name, "role": role, "error": str(exc)}
                    )
                    emit_progress(
                        graph,
                        progress_callback,
                        stage="action_fallback",
                        message=f"{round_name} {role}: invalid LLM action, using deterministic fallback. Error: {exc}",
                        details={"round": round_name, "role": role, "error": str(exc)},
                    )
                    action = choose_round_action(graph, round_name, role)
                    action_source = "deterministic_fallback"
            try:
                apply_action(graph, action)
            except Exception as exc:
                if action_source == "llm":
                    graph.metadata.setdefault("action_errors", []).append(
                        {
                            "round": round_name,
                            "role": role,
                            "error": f"LLM action application failed: {exc}",
                        }
                    )
                    emit_progress(
                        graph,
                        progress_callback,
                        stage="action_fallback",
                        message=f"{round_name} {role}: LLM action could not be applied, using deterministic fallback. Error: {exc}",
                        details={"round": round_name, "role": role, "error": str(exc)},
                    )
                    action = choose_round_action(graph, round_name, role)
                    apply_action(graph, action)
                    action_source = "deterministic_fallback"
                else:
                    raise RuntimeError(
                        f"Failed to apply {action_source} action for {round_name}/{role}: {exc}"
                    ) from exc
            emit_progress(
                graph,
                progress_callback,
                stage="action_applied",
                message=(
                    f"{round_name} {role}: applied {action.kind} "
                    f"via {'LLM' if action_source == 'llm' else 'deterministic policy'}."
                ),
                details={
                    "round": round_name,
                    "role": role,
                    "action_kind": action.kind,
                    "action_source": action_source,
                    "target_ids": list(action.target_ids),
                },
            )
        snapshot = maturity_snapshot(graph)
        graph.round_summaries.append((round_name, snapshot))
        if snapshot.is_mature and graph.matured_at_round is None:
            graph.matured_at_round = round_name
            graph.metadata["maturity_stop_candidate"] = round_name
        emit_progress(
            graph,
            progress_callback,
            stage="round_complete",
            message=(
                f"{round_name} complete: support={snapshot.support_coverage}, "
                f"contradictions={snapshot.unresolved_contradiction_ratio}, "
                f"utility={snapshot.utility}, mature={snapshot.is_mature}."
            ),
            details={
                "round": round_name,
                "support_coverage": snapshot.support_coverage,
                "unresolved_contradiction_ratio": snapshot.unresolved_contradiction_ratio,
                "utility": snapshot.utility,
                "is_mature": snapshot.is_mature,
            },
        )
        if stop_when_mature and snapshot.is_mature:
            graph.metadata["stopped_early"] = True
            graph.metadata["stop_reason"] = f"mature_at_{round_name}"
            emit_progress(
                graph,
                progress_callback,
                stage="maturity_stop",
                message=f"{round_name} reached maturity, stopping early before additional rounds.",
                details={"round": round_name},
            )
            break

    graph.final_subgraph = select_final_subgraph(graph)
    graph.metadata["executed_round_count"] = len(graph.round_summaries)
    if "stopped_early" not in graph.metadata:
        graph.metadata["stopped_early"] = False
        graph.metadata["stop_reason"] = "max_rounds_reached"
    emit_progress(
        graph,
        progress_callback,
        stage="final_subgraph",
        message=(
            f"Selected final subgraph with {len(graph.final_subgraph['node_ids'])} nodes and "
            f"{len(graph.final_subgraph['edge_ids'])} edges."
        ),
        details=dict(graph.final_subgraph),
    )
    if collaboration_backend is None:
        graph.final_proposal = synthesize_proposal(graph, graph.final_subgraph)
    else:
        try:
            emit_progress(
                graph,
                progress_callback,
                stage="final_synthesis",
                message="Synthesizing the final proposal with the OpenAI-compatible backend.",
            )
            graph.final_proposal = collaboration_backend.synthesize_final_proposal(graph, graph.final_subgraph)
        except Exception as exc:
            graph.metadata["final_synthesis_error"] = str(exc)
            emit_progress(
                graph,
                progress_callback,
                stage="final_synthesis_fallback",
                message=f"Final synthesis failed with the LLM backend, using deterministic synthesis: {exc}",
                details={"error": str(exc)},
            )
            graph.final_proposal = synthesize_proposal(graph, graph.final_subgraph)
    emit_progress(
        graph,
        progress_callback,
        stage="complete",
        message=(
            f"Run complete: {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
            f"{len(graph.actions)} actions."
        ),
        details={"nodes": len(graph.nodes), "edges": len(graph.edges), "actions": len(graph.actions)},
    )
    return graph


def graph_as_dict(graph: IdeaGraph) -> dict[str, object]:
    return {
        "topic": graph.topic,
        "literature": graph.literature,
        "metadata": graph.metadata,
        "nodes": {node_id: asdict(node) for node_id, node in graph.nodes.items()},
        "edges": [asdict(edge) for edge in graph.edges],
        "branches": {branch_id: asdict(branch) for branch_id, branch in graph.branches.items()},
        "actions": [asdict(action) for action in graph.actions],
        "round_summaries": [(name, asdict(snapshot)) for name, snapshot in graph.round_summaries],
        "matured_at_round": graph.matured_at_round,
        "final_subgraph": graph.final_subgraph,
        "final_proposal": asdict(graph.final_proposal) if graph.final_proposal else None,
    }
