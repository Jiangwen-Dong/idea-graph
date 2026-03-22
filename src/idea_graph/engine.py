from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from .agent_backend import (
    ActionDecision,
    CollaborationBackend,
    append_agent_trace,
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


ACTION_TARGET_COUNTS = {
    "add_support_edge": 2,
    "add_contradiction_edge": 2,
    "add_dependency_edge": 2,
    "request_evidence": 1,
    "attach_evidence": 1,
    "mark_overlap": 1,
    "propose_repair": 1,
    "freeze_branch": 0,
}

ACTION_REQUIRED_PAYLOAD_FIELDS = {
    "attach_evidence": ("evidence",),
    "mark_overlap": ("paper_id",),
    "propose_repair": ("repair_text",),
}


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
    _view = focused_view(graph, role)

    if round_name == "Round1":
        if role == "MechanismProposer":
            hypothesis = first_active_node(graph, "Hypothesis", role="MechanismProposer")
            problem = first_active_node(graph, "Problem")
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
            risk = first_active_node(graph, "Risk", role="FeasibilityCritic")
            method = first_active_node(graph, "Method", role="EvaluationDesigner")
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
            novelty_claim = first_active_node(graph, "NoveltyClaim", role="NoveltyExaminer")
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
            eval_plan = first_active_node(graph, "EvalPlan", role="EvaluationDesigner")
            hypothesis = first_active_node(graph, "Hypothesis", role="MechanismProposer")
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="add_dependency_edge",
                target_ids=[eval_plan.id, hypothesis.id],
                payload={"branch_id": branch.id},
                rationale="Make the evaluation plan explicitly depend on the main hypothesis.",
            )
        hypothesis = first_active_node(graph, "Hypothesis", role="ImpactReframer")
        problem = first_active_node(graph, "Problem", role="ImpactReframer")
        return make_action(
            graph,
            round_name=round_name,
            role=role,
            kind="add_support_edge",
            target_ids=[hypothesis.id, problem.id],
            payload={"branch_id": branch.id},
            rationale="Clarify how the reframed hypothesis supports the motivating problem.",
        )

    if round_name == "Round2":
        if role == "MechanismProposer":
            method = first_active_node(graph, "Method", role="EvaluationDesigner", without_evidence=True)
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
            hypothesis = first_active_node(graph, "Hypothesis", role="ImpactReframer", without_evidence=True)
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
            novelty_claim = first_active_node(graph, "NoveltyClaim", role="ImpactReframer", without_evidence=True)
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
            eval_plan = first_active_node(graph, "EvalPlan", role="FeasibilityCritic", without_evidence=True)
            return make_action(
                graph,
                round_name=round_name,
                role=role,
                kind="attach_evidence",
                target_ids=[eval_plan.id],
                payload={"branch_id": branch.id, "evidence": literature_item(graph, 3)},
                rationale="Attach validation-oriented evidence to another branch's evaluation node.",
            )
        novelty_claim = first_active_node(graph, "NoveltyClaim", role="NoveltyExaminer", without_evidence=True)
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
        method = first_active_node(graph, "Method", role="EvaluationDesigner")
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
        eval_plan = first_active_node(graph, "EvalPlan", role="EvaluationDesigner")
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
        novelty_claim = first_active_node(graph, "NoveltyClaim", role="ImpactReframer")
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
        method = first_active_node(graph, "Method", role="MechanismProposer")
        hypothesis = first_active_node(graph, "Hypothesis", role="ImpactReframer")
        return make_action(
            graph,
            round_name=round_name,
            role=role,
            kind="add_support_edge",
            target_ids=[method.id, hypothesis.id],
            payload={"branch_id": branch.id},
            rationale="Reconnect compatible pieces from different branches into one stronger path.",
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
) -> IdeaGraph:
    graph = IdeaGraph(topic=topic, literature=literature, metadata=dict(metadata or {}))
    backend_name = collaboration_backend.name if collaboration_backend is not None else "deterministic"
    emit_progress(
        graph,
        progress_callback,
        stage="start",
        message=f"Initialized idea graph for topic '{topic}' using backend '{backend_name}'.",
        details={"backend": backend_name, "topic": topic},
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

    for round_name in ("Round1", "Round2", "Round3"):
        emit_progress(
            graph,
            progress_callback,
            stage="round_start",
            message=f"{round_name} started.",
            details={"round": round_name},
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

    graph.final_subgraph = select_final_subgraph(graph)
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
