from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from itertools import product
from typing import Any, Callable, Mapping, Sequence

from .agent_backend import (
    ActionDecision,
    CollaborationBackend,
    ROLE_DISPLAY_NAMES,
    append_agent_trace,
)
from .action_candidates import (
    action_spec_from_action as _action_spec_from_action,
    build_action_spec as _build_action_spec,
    dedupe_action_specs as _dedupe_action_specs,
    enumerate_candidate_specs,
)
from .claim_chain import select_claim_chain
from .collaboration_protocol import (
    ACTION_REQUIRED_PAYLOAD_FIELDS,
    ACTION_TARGET_COUNTS,
    build_round_name,
    resolve_round_phase,
)
from .literature_grounding import build_literature_grounding
from .models import (
    Branch,
    Edge,
    FinalProposal,
    GraphAction,
    IdeaGraph,
    MaturitySnapshot,
    Node,
    Provenance,
    UtilityBreakdown,
)
from .relation_graph_runtime_critic import select_relation_graph_critic_candidate
from .runtime_critic import select_text_critic_candidate
from .schema import ROLE_NAMES, build_seed_template

def normalize_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def _generation_metadata(graph: IdeaGraph) -> dict[str, object]:
    payload = graph.metadata.get("generation_safe_metadata", graph.metadata)
    return payload if isinstance(payload, dict) else graph.metadata


def _safe_benchmark_metadata(metadata: dict[str, object]) -> dict[str, object]:
    blocked_keys = {
        "agent_traces",
        "progress_log",
        "action_errors",
        "final_synthesis_trace",
        "final_synthesis_error",
        "seed_generation_error",
        "raw_record",
        "target_paper",
        "target_paper_path",
        "motivation",
        "method_summary",
        "literature_grounding",
    }
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        if key in blocked_keys:
            continue
        if key == "paper_grounding" and isinstance(value, dict):
            snippets = value.get("reference_paper_snippets", [])
            safe[key] = {
                "reference_paper_snippets": [
                    item for item in snippets if isinstance(item, dict)
                ]
                if isinstance(snippets, list)
                else []
            }
            continue
        safe[key] = value
    return safe


def _utility_grounding_metadata(graph: IdeaGraph) -> dict[str, object]:
    metadata = _generation_metadata(graph)
    if graph.metadata.get("benchmark_mode") and metadata is graph.metadata:
        return _safe_benchmark_metadata(graph.metadata)
    return metadata


def _progress_role_name(role: str) -> str:
    return ROLE_DISPLAY_NAMES.get(role, role)


def _weak_context_scaffold(graph: IdeaGraph) -> dict[str, object]:
    grounding = build_literature_grounding(
        literature=graph.literature,
        metadata=_generation_metadata(graph),
    )
    payload = grounding.weak_context_scaffold
    return payload if isinstance(payload, dict) else {}


def _keyword_only_specificity_score(graph: IdeaGraph, node_ids: set[str] | None = None) -> float:
    scaffold = _weak_context_scaffold(graph)
    if not scaffold:
        return 1.0

    active_nodes = _subgraph_nodes(graph, node_ids)
    text = " ".join(
        part
        for node in active_nodes
        for part in [node.text, *node.evidence]
        if part
    )
    normalized_text = normalize_text(text)
    if not normalized_text:
        return 0.0

    keyword = normalize_text(str(scaffold.get("keyword", "")))
    mechanism_terms = [
        normalize_text(str(item))
        for item in scaffold.get("mechanism_terms", [])
        if normalize_text(str(item))
    ]
    evaluation_terms = [
        normalize_text(str(item))
        for item in [
            *scaffold.get("evaluation_assets", []),
            *scaffold.get("metric_items", []),
        ]
        if normalize_text(str(item))
    ]

    keyword_signal = 1.0 if keyword and keyword in normalized_text else 0.0
    mechanism_hits = sum(1 for item in mechanism_terms if item in normalized_text)
    mechanism_signal = min(1.0, mechanism_hits / 2.0) if mechanism_terms else 0.0
    evaluation_hits = sum(1 for item in evaluation_terms if item in normalized_text)
    evaluation_signal = min(1.0, evaluation_hits / 2.0) if evaluation_terms else 0.0
    return round(
        max(
            0.0,
            min(
                1.0,
                (0.40 * keyword_signal) + (0.35 * mechanism_signal) + (0.25 * evaluation_signal),
            ),
        ),
        2,
    )


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
        if action.kind == "propose_repair":
            return (
                "Repair-phase action proposed a repair even though no unresolved contradictions remain. "
                "Prefer freeze_branch, add_support_edge, or attach_evidence on any remaining weak node."
            )
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


def ensure_core_node_coverage(graph: IdeaGraph) -> None:
    generation_metadata = graph.metadata.get("generation_safe_metadata", graph.metadata)
    if not isinstance(generation_metadata, dict):
        generation_metadata = graph.metadata
    grounding = build_literature_grounding(literature=graph.literature, metadata=generation_metadata)
    cleaned_topic = str(graph.topic).strip().rstrip(".")
    prefix = "The topic of this paper is "
    if cleaned_topic.startswith(prefix):
        cleaned_topic = cleaned_topic[len(prefix) :].strip()

    def has_type(node_type: str) -> bool:
        return any(node.type == node_type for node in graph.active_nodes())

    def scaffold_node(*, node_type: str, role: str, text: str) -> Node:
        branch = branch_for_role(graph, role)
        return create_node(
            graph,
            node_type=node_type,
            text=text,
            role=role,
            branch_id=branch.id,
            confidence=0.42,
            source="coverage_scaffold",
        )

    created_problem = None
    created_hypothesis = None
    created_method = None
    created_eval = None

    if not has_type("Problem"):
        created_problem = scaffold_node(
            node_type="Problem",
            role="ImpactReframer",
            text=(
                f"Current methods for {cleaned_topic or graph.topic} still face unresolved limitations in robustness, "
                "efficiency, or generalization under realistic conditions."
            ),
        )

    if not has_type("Hypothesis"):
        anchor = grounding.design_highlights[0] if grounding.design_highlights else ""
        created_hypothesis = scaffold_node(
            node_type="Hypothesis",
            role="MechanismProposer",
            text=(
                f"A more structured representation for {cleaned_topic or graph.topic} should improve reliability and "
                f"testability.{(' ' + anchor) if anchor else ''}"
            ).strip(),
        )

    if not has_type("Method"):
        method_text = grounding.design_highlights[0] if grounding.design_highlights else ""
        if not method_text:
            method_text = (
                f"Use an explicit multi-stage modeling pipeline for {cleaned_topic or graph.topic} with stronger "
                "representation, grounding, and evaluation hooks."
            )
        created_method = scaffold_node(
            node_type="Method",
            role="MechanismProposer",
            text=method_text,
        )

    if not has_type("EvalPlan"):
        datasets = ", ".join(grounding.dataset_items[:2]) or "representative benchmark datasets"
        metrics = ", ".join(grounding.metric_items[:3]) or "task-specific quality metrics"
        created_eval = scaffold_node(
            node_type="EvalPlan",
            role="EvaluationDesigner",
            text=f"Evaluate on {datasets} and report {metrics}.",
        )

    problem_node = created_problem or next((node for node in graph.active_nodes() if node.type == "Problem"), None)
    hypothesis_node = created_hypothesis or next((node for node in graph.active_nodes() if node.type == "Hypothesis"), None)
    method_node = created_method or next((node for node in graph.active_nodes() if node.type == "Method"), None)
    eval_node = created_eval or next((node for node in graph.active_nodes() if node.type == "EvalPlan"), None)

    if hypothesis_node is not None and problem_node is not None and not edge_exists(
        graph,
        source_id=hypothesis_node.id,
        relation="supports",
        target_id=problem_node.id,
    ):
        create_edge(
            graph,
            source_id=hypothesis_node.id,
            relation="supports",
            target_id=problem_node.id,
            role=hypothesis_node.role,
            branch_id=hypothesis_node.branch_id,
            note="Coverage scaffold support edge.",
        )

    if method_node is not None and hypothesis_node is not None and not edge_exists(
        graph,
        source_id=method_node.id,
        relation="supports",
        target_id=hypothesis_node.id,
    ):
        create_edge(
            graph,
            source_id=method_node.id,
            relation="supports",
            target_id=hypothesis_node.id,
            role=method_node.role,
            branch_id=method_node.branch_id,
            note="Coverage scaffold method edge.",
        )

    if eval_node is not None and hypothesis_node is not None and not edge_exists(
        graph,
        source_id=eval_node.id,
        relation="depends_on",
        target_id=hypothesis_node.id,
    ):
        create_edge(
            graph,
            source_id=eval_node.id,
            relation="depends_on",
            target_id=hypothesis_node.id,
            role=eval_node.role,
            branch_id=eval_node.branch_id,
            note="Coverage scaffold evaluation edge.",
        )


def core_node_coverage_enabled(graph: IdeaGraph) -> bool:
    return not bool(graph.metadata.get("idea_graph_disable_core_node_coverage", False))


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


def choose_consolidation_action(graph: IdeaGraph, round_name: str, role: str, branch: Branch) -> GraphAction:
    maturity = _compute_maturity_snapshot(graph, update_history=False)
    if maturity.support_coverage >= 0.9:
        return make_action(
            graph,
            round_name=round_name,
            role=role,
            kind="freeze_branch",
            target_ids=[],
            payload={"branch_id": branch.id},
            rationale="The core path is already well-supported, so freezing the branch preserves it cleanly for final synthesis.",
        )

    evidence_target = None
    try:
        evidence_target = first_available_node(
            graph,
            node_types=("EvalPlan", "Method", "Hypothesis", "NoveltyClaim"),
            preferred_roles=("EvaluationDesigner", "MechanismProposer", "ImpactReframer", "NoveltyExaminer"),
            prefer_without_evidence=True,
        )
    except ValueError:
        evidence_target = None

    if evidence_target is not None:
        return make_action(
            graph,
            round_name=round_name,
            role=role,
            kind="attach_evidence",
            target_ids=[evidence_target.id],
            payload={"branch_id": branch.id, "evidence": literature_item(graph, 0)},
            rationale="No unresolved contradictions remain, so consolidate the strongest branch with one more grounded evidence attachment.",
        )

    if role in {"ImpactReframer", "NoveltyExaminer"}:
        return make_action(
            graph,
            round_name=round_name,
            role=role,
            kind="freeze_branch",
            target_ids=[],
            payload={"branch_id": branch.id},
            rationale="Freeze this branch as a coherent alternative now that no unresolved contradictions remain.",
        )

    support_source = first_available_node(
        graph,
        node_types=("Method", "Hypothesis", "EvalPlan"),
        preferred_roles=("MechanismProposer", "EvaluationDesigner", "ImpactReframer"),
    )
    support_target = first_available_node(
        graph,
        node_types=("Problem", "Hypothesis", "EvalPlan"),
        preferred_roles=("ImpactReframer", "MechanismProposer", "EvaluationDesigner"),
        exclude_role=None,
    )
    if support_source.id != support_target.id and not edge_exists(
        graph,
        source_id=support_source.id,
        relation="supports",
        target_id=support_target.id,
    ):
        return make_action(
            graph,
            round_name=round_name,
            role=role,
            kind="add_support_edge",
            target_ids=[support_source.id, support_target.id],
            payload={"branch_id": branch.id},
            rationale="With contradictions resolved, reinforce one coherent scientific path before synthesis.",
        )

    return make_action(
        graph,
        round_name=round_name,
        role=role,
        kind="freeze_branch",
        target_ids=[],
        payload={"branch_id": branch.id},
        rationale="Freeze the branch because the graph is already coherent enough for synthesis.",
    )


def _legacy_choose_round_action(graph: IdeaGraph, round_name: str, role: str) -> GraphAction:
    branch = branch_for_role(graph, role)
    phase = resolve_round_phase(round_name)
    _view = focused_view(graph, role)

    if phase.key == "repair" and not unresolved_contradiction_edges(graph):
        return choose_consolidation_action(graph, round_name, role, branch)

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
                "repair_text": "Add explicit cost reporting and process metrics so evaluation claims separate quality from coordination overhead.",
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
                    "repair_text": "Narrow the novelty claim to explicit evolving-graph reasoning over typed scientific claim fragments.",
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


def _candidate_nodes(
    graph: IdeaGraph,
    *,
    node_types: tuple[str, ...] | list[str],
    preferred_roles: tuple[str, ...] | list[str] = (),
    exclude_role: str | None = None,
    prefer_without_evidence: bool = False,
    limit: int = 3,
) -> list[Node]:
    results: list[Node] = []
    seen: set[str] = set()
    without_evidence_passes = [True, False] if prefer_without_evidence else [False]

    for without_evidence in without_evidence_passes:
        for node_type in tuple(node_types):
            for preferred_role in tuple(preferred_roles):
                for node in find_active_nodes(
                    graph,
                    node_type,
                    role=preferred_role,
                    exclude_role=exclude_role,
                    without_evidence=without_evidence,
                ):
                    if node.id in seen:
                        continue
                    seen.add(node.id)
                    results.append(node)
                    if len(results) >= limit:
                        return results
            for node in find_active_nodes(
                graph,
                node_type,
                exclude_role=exclude_role,
                without_evidence=without_evidence,
            ):
                if node.id in seen:
                    continue
                seen.add(node.id)
                results.append(node)
                if len(results) >= limit:
                    return results
    return results


def _role_support_preferences(role: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if role == "MechanismProposer":
        return ("Method", "Hypothesis", "NoveltyClaim"), ("Hypothesis", "Problem")
    if role == "EvaluationDesigner":
        return ("EvalPlan", "Method"), ("Hypothesis", "Method", "Problem")
    if role == "ImpactReframer":
        return ("Hypothesis", "NoveltyClaim"), ("Problem", "Hypothesis")
    if role == "NoveltyExaminer":
        return ("NoveltyClaim", "Hypothesis"), ("Problem", "Hypothesis")
    return ("Method", "Hypothesis", "EvalPlan"), ("Problem", "Hypothesis", "Method")


def _role_contradiction_preferences(role: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if role == "NoveltyExaminer":
        return ("NoveltyClaim", "Hypothesis", "Problem"), ("Method", "Hypothesis", "Problem")
    return ("Risk", "Assumption"), ("Method", "EvalPlan", "Hypothesis", "Problem")


def _repair_text_for_role(role: str, target: Node) -> str:
    if role == "MechanismProposer":
        return (
            f"Refine the {target.type.lower()} so it addresses the unresolved failure mode with a more explicit "
            "mechanism and clearer scope."
        )
    if role == "FeasibilityCritic":
        return (
            f"Adjust the {target.type.lower()} so the feasibility concern is answered with tighter assumptions, "
            "constraints, or evaluation details."
        )
    if role == "NoveltyExaminer":
        return (
            f"Narrow the {target.type.lower()} so it is more clearly differentiated from nearby prior work and "
            "less vulnerable to overlap."
        )
    if role == "EvaluationDesigner":
        return (
            f"Revise the {target.type.lower()} so its evaluation path, baselines, and success criteria are explicit."
        )
    return (
        f"Reframe the {target.type.lower()} so it better supports the overall idea while directly addressing the "
        "current weakness."
    )


def _generic_candidate_action_specs(
    graph: IdeaGraph,
    round_name: str,
    role: str,
    branch: Branch,
) -> list[dict[str, object]]:
    phase = resolve_round_phase(round_name)
    allowed_actions = set(phase.allowed_actions)
    candidates: list[dict[str, object]] = []

    contradiction_related = contradiction_related_node_ids(graph)
    contradiction_targets = [
        graph.nodes[node_id]
        for node_id in sorted(contradiction_related)
        if node_id in graph.nodes and graph.nodes[node_id].status == "active"
    ]

    if "request_evidence" in allowed_actions:
        evidence_targets = contradiction_targets or _candidate_nodes(
            graph,
            node_types=("NoveltyClaim", "Hypothesis", "Method", "EvalPlan"),
            preferred_roles=(role, "MechanismProposer", "EvaluationDesigner", "ImpactReframer"),
            prefer_without_evidence=True,
            limit=2,
        )
        for target in evidence_targets[:2]:
            if target.evidence:
                continue
            candidates.append(
                _build_action_spec(
                    kind="request_evidence",
                    target_ids=[target.id],
                    payload={
                        "branch_id": branch.id,
                        "query": f"Find grounding, precedent, or evaluation evidence relevant to this {target.type.lower()}.",
                    },
                    rationale=f"Request explicit grounding for the weak {target.type.lower()} node before it shapes later synthesis.",
                    candidate_source="utility_request_evidence",
                )
            )

    if "attach_evidence" in allowed_actions:
        evidence_targets = contradiction_targets or _candidate_nodes(
            graph,
            node_types=("EvalPlan", "Method", "Hypothesis", "NoveltyClaim"),
            preferred_roles=("EvaluationDesigner", "MechanismProposer", "ImpactReframer", role),
            prefer_without_evidence=True,
            exclude_role=role if phase.key == "stress_test" else None,
            limit=2,
        )
        for index, target in enumerate(evidence_targets[:2]):
            if target.evidence:
                continue
            candidates.append(
                _build_action_spec(
                    kind="attach_evidence",
                    target_ids=[target.id],
                    payload={
                        "branch_id": branch.id,
                        "evidence": literature_item(graph, index),
                    },
                    rationale=f"Attach concrete evidence to the {target.type.lower()} node to improve grounding and support coverage.",
                    candidate_source="utility_attach_evidence",
                )
            )

    if "mark_overlap" in allowed_actions and role == "NoveltyExaminer":
        overlap_targets = _candidate_nodes(
            graph,
            node_types=("NoveltyClaim", "Hypothesis", "Problem"),
            preferred_roles=("NoveltyExaminer", "ImpactReframer", role),
            prefer_without_evidence=True,
            limit=2,
        )
        for index, target in enumerate(overlap_targets[:2]):
            candidates.append(
                _build_action_spec(
                    kind="mark_overlap",
                    target_ids=[target.id],
                    payload={
                        "branch_id": branch.id,
                        "paper_id": f"paper-{index + 1:03d}",
                        "evidence": literature_item(graph, index + 1),
                    },
                    rationale=f"Mark potential prior-work overlap on the {target.type.lower()} node before overcommitting to novelty.",
                    candidate_source="utility_mark_overlap",
                )
            )

    if "add_support_edge" in allowed_actions:
        source_types, target_types = _role_support_preferences(role)
        support_sources = _candidate_nodes(
            graph,
            node_types=source_types,
            preferred_roles=(role, "MechanismProposer", "EvaluationDesigner", "ImpactReframer"),
            limit=2,
        )
        support_targets = _candidate_nodes(
            graph,
            node_types=target_types,
            preferred_roles=("ImpactReframer", "MechanismProposer", "EvaluationDesigner"),
            limit=2,
        )
        for source in support_sources:
            for target in support_targets:
                if source.id == target.id:
                    continue
                if edge_exists(
                    graph,
                    source_id=source.id,
                    relation="supports",
                    target_id=target.id,
                ):
                    continue
                candidates.append(
                    _build_action_spec(
                        kind="add_support_edge",
                        target_ids=[source.id, target.id],
                        payload={"branch_id": branch.id},
                        rationale=f"Add one support edge to strengthen a more coherent problem-to-method path via the {target.type.lower()} node.",
                        candidate_source="utility_add_support",
                    )
                )

    if "add_dependency_edge" in allowed_actions:
        dependency_sources = _candidate_nodes(
            graph,
            node_types=("EvalPlan", "Method"),
            preferred_roles=("EvaluationDesigner", role, "MechanismProposer"),
            limit=2,
        )
        dependency_targets = _candidate_nodes(
            graph,
            node_types=("Hypothesis", "Problem", "Method"),
            preferred_roles=("MechanismProposer", "ImpactReframer"),
            limit=2,
        )
        for source in dependency_sources:
            for target in dependency_targets:
                if source.id == target.id:
                    continue
                if edge_exists(
                    graph,
                    source_id=source.id,
                    relation="depends_on",
                    target_id=target.id,
                ):
                    continue
                candidates.append(
                    _build_action_spec(
                        kind="add_dependency_edge",
                        target_ids=[source.id, target.id],
                        payload={"branch_id": branch.id},
                        rationale=f"Make the dependency structure explicit so the {source.type.lower()} node is easier to evaluate or repair later.",
                        candidate_source="utility_add_dependency",
                    )
                )

    if "add_contradiction_edge" in allowed_actions:
        source_types, target_types = _role_contradiction_preferences(role)
        contradiction_sources = _candidate_nodes(
            graph,
            node_types=source_types,
            preferred_roles=(role, "FeasibilityCritic", "NoveltyExaminer"),
            limit=2,
        )
        contradiction_targets = _candidate_nodes(
            graph,
            node_types=target_types,
            preferred_roles=("MechanismProposer", "EvaluationDesigner", "ImpactReframer"),
            exclude_role=role,
            limit=2,
        )
        for source in contradiction_sources:
            for target in contradiction_targets:
                if source.id == target.id:
                    continue
                if edge_exists(
                    graph,
                    source_id=source.id,
                    relation="contradicts",
                    target_id=target.id,
                ):
                    continue
                candidates.append(
                    _build_action_spec(
                        kind="add_contradiction_edge",
                        target_ids=[source.id, target.id],
                        payload={"branch_id": branch.id},
                        rationale=f"Expose a concrete contradiction so weak claims can be repaired before final synthesis.",
                        candidate_source="utility_add_contradiction",
                    )
                )

    if "propose_repair" in allowed_actions:
        repair_targets: list[Node] = []
        primary_target = choose_repair_target(graph, role)
        if primary_target is not None:
            repair_targets.append(primary_target)
        repair_targets.extend(
            node
            for node in _candidate_nodes(
                graph,
                node_types=("Method", "EvalPlan", "Hypothesis", "NoveltyClaim", "Problem"),
                preferred_roles=("MechanismProposer", "EvaluationDesigner", "ImpactReframer"),
                limit=2,
            )
            if node.id not in {target.id for target in repair_targets}
        )
        for target in repair_targets[:2]:
            candidates.append(
                _build_action_spec(
                    kind="propose_repair",
                    target_ids=[target.id],
                    payload={
                        "branch_id": branch.id,
                        "repair_text": _repair_text_for_role(role, target),
                    },
                    rationale=f"Repair the {target.type.lower()} node to improve coherence and reduce downstream contradiction.",
                    candidate_source="utility_propose_repair",
                )
            )

    if "freeze_branch" in allowed_actions:
        candidates.append(
            _build_action_spec(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": branch.id},
                rationale="Freeze this branch only if preserving it is better than spending another edit on it right now.",
                candidate_source="utility_freeze_branch",
            )
        )

    return _dedupe_action_specs(candidates)


def generic_candidate_action_specs(
    graph: IdeaGraph,
    round_name: str,
    role: str,
    branch: Branch,
) -> list[dict[str, object]]:
    return _generic_candidate_action_specs(graph, round_name, role, branch)


def _reference_subgraph(graph: IdeaGraph) -> dict[str, object]:
    candidate = _best_candidate_subgraph(graph)
    if candidate is not None:
        return candidate
    legacy = _legacy_select_final_subgraph(graph)
    legacy_snapshot = _compute_maturity_snapshot(
        graph,
        update_history=False,
        node_ids=set(legacy["node_ids"]),
    )
    legacy.update(
        {
            "support_coverage": legacy_snapshot.support_coverage,
            "unresolved_contradiction_ratio": legacy_snapshot.unresolved_contradiction_ratio,
            "utility_stable": legacy_snapshot.utility_stable,
            "completeness": legacy_snapshot.completeness,
            "is_mature": legacy_snapshot.is_mature,
            "utility_breakdown": asdict(legacy_snapshot.utility_breakdown),
            "selection_mode": "legacy_reference",
        }
    )
    return legacy


def _score_action_spec(
    graph: IdeaGraph,
    *,
    round_name: str,
    role: str,
    spec: dict[str, object],
    reference_subgraph: dict[str, object],
    reference_snapshot: MaturitySnapshot,
) -> dict[str, object]:
    try:
        simulated_graph = deepcopy(graph)
        simulated_action = make_action(
            simulated_graph,
            round_name=round_name,
            role=role,
            kind=str(spec.get("kind", "")).strip(),
            target_ids=[str(item).strip() for item in spec.get("target_ids", []) if str(item).strip()],
            payload=dict(spec.get("payload", {}) or {}),
            rationale=str(spec.get("rationale", "")).strip(),
        )
        apply_action(simulated_graph, simulated_action)
        after_subgraph = _reference_subgraph(simulated_graph)
        after_snapshot = _compute_maturity_snapshot(
            simulated_graph,
            update_history=False,
            node_ids=set(after_subgraph["node_ids"]),
        )

        utility_gain = after_snapshot.utility - reference_snapshot.utility
        support_gain = after_snapshot.support_coverage - reference_snapshot.support_coverage
        contradiction_gain = (
            reference_snapshot.unresolved_contradiction_ratio - after_snapshot.unresolved_contradiction_ratio
        )
        coherence_gain = (
            after_snapshot.utility_breakdown.coherence - reference_snapshot.utility_breakdown.coherence
        )
        maturity_gain = 1.0 if after_snapshot.is_mature and not reference_snapshot.is_mature else 0.0
        completeness_gain = 1.0 if after_snapshot.completeness and not reference_snapshot.completeness else 0.0
        stability_gain = 1.0 if after_snapshot.utility_stable and not reference_snapshot.utility_stable else 0.0

        predicted_gain = round(
            utility_gain
            + (1.75 * maturity_gain)
            + (1.20 * contradiction_gain)
            + (0.90 * support_gain)
            + (0.80 * coherence_gain)
            + (0.35 * completeness_gain)
            + (0.25 * stability_gain),
            3,
        )
        return {
            "valid": True,
            "predicted_gain": predicted_gain,
            "utility_gain": round(utility_gain, 3),
            "support_gain": round(support_gain, 3),
            "contradiction_gain": round(contradiction_gain, 3),
            "coherence_gain": round(coherence_gain, 3),
            "maturity_gain": maturity_gain,
            "completeness_gain": completeness_gain,
            "stability_gain": stability_gain,
            "after_snapshot": after_snapshot,
            "after_subgraph": {
                "selection_mode": after_subgraph.get("selection_mode"),
                "utility": after_subgraph.get("utility"),
                "support_coverage": after_subgraph.get("support_coverage"),
                "unresolved_contradiction_ratio": after_subgraph.get("unresolved_contradiction_ratio"),
                "is_mature": after_subgraph.get("is_mature"),
            },
        }
    except Exception as exc:
        return {
            "valid": False,
            "predicted_gain": float("-inf"),
            "error": str(exc),
        }


def _record_action_selection_trace(
    graph: IdeaGraph,
    *,
    round_name: str,
    role: str,
    selected_candidate: dict[str, object],
    ranked_candidates: list[dict[str, object]],
) -> None:
    entry = {
        "round": round_name,
        "role": role,
        "selected_kind": selected_candidate.get("kind"),
        "selected_targets": list(selected_candidate.get("target_ids", [])),
        "selected_candidate_source": selected_candidate.get("candidate_source"),
        "selected_predicted_gain": selected_candidate.get("predicted_gain"),
        "ranked_candidates": [
            {
                "kind": candidate.get("kind"),
                "target_ids": list(candidate.get("target_ids", [])),
                "candidate_source": candidate.get("candidate_source"),
                "predicted_gain": candidate.get("predicted_gain"),
                "valid": candidate.get("valid", False),
                "error": candidate.get("error", ""),
            }
            for candidate in ranked_candidates[:6]
        ],
    }
    selection_log = graph.metadata.setdefault("action_selection_log", [])
    if isinstance(selection_log, list):
        selection_log.append(entry)


def _record_runtime_controller_trace(
    graph: IdeaGraph,
    *,
    round_name: str,
    role: str,
    controller_kind: str,
    heuristic_candidate: dict[str, object],
    selected_candidate: dict[str, object],
    controller_decision: dict[str, object],
    scored_candidates: Sequence[Mapping[str, object]],
) -> None:
    def _snapshot(candidate: Mapping[str, object]) -> dict[str, object]:
        target_ids = candidate.get("target_ids", ())
        if isinstance(target_ids, Sequence) and not isinstance(target_ids, (str, bytes)):
            normalized_target_ids = [str(item) for item in target_ids]
        else:
            normalized_target_ids = []
        payload = candidate.get("payload", {})
        normalized_payload = dict(payload) if isinstance(payload, Mapping) else {}
        return {
            "candidate_id": candidate.get("candidate_id"),
            "kind": candidate.get("kind"),
            "target_ids": normalized_target_ids,
            "payload": normalized_payload,
            "rationale": candidate.get("rationale"),
            "candidate_source": candidate.get("candidate_source"),
            "predicted_gain": candidate.get("predicted_gain"),
            "critic_score": candidate.get("critic_score"),
            "controller_fallback_reason": candidate.get("controller_fallback_reason"),
        }

    selected_fallback_ids = selected_candidate.get("controller_fallback_candidate_ids", ())
    if isinstance(selected_fallback_ids, Sequence) and not isinstance(selected_fallback_ids, (str, bytes)):
        normalized_fallback_ids = [str(item) for item in selected_fallback_ids]
    else:
        normalized_fallback_ids = []

    entry = {
        "round": round_name,
        "role": role,
        "controller_kind": controller_kind,
        "heuristic_candidate_id": heuristic_candidate.get("candidate_id"),
        "heuristic_kind": heuristic_candidate.get("kind"),
        "heuristic_predicted_gain": heuristic_candidate.get("predicted_gain"),
        "heuristic_critic_score": heuristic_candidate.get("critic_score"),
        "selected_candidate_id": selected_candidate.get("candidate_id"),
        "selected_kind": selected_candidate.get("kind"),
        "selected_source": controller_decision.get("selected_source"),
        "selected_fallback_reason": selected_candidate.get("controller_fallback_reason"),
        "selected_fallback_candidate_ids": normalized_fallback_ids,
        "override_margin": controller_decision.get("override_margin"),
        "used_heuristic_fallback": controller_decision.get("used_heuristic_fallback"),
        "heuristic_candidate": _snapshot(heuristic_candidate),
        "selected_candidate": _snapshot(selected_candidate),
        "top_scored_candidates": [
            _snapshot(candidate)
            for candidate in sorted(
                (dict(item) for item in scored_candidates),
                key=lambda item: float(item.get("critic_score", float("-inf"))),
                reverse=True,
            )[:6]
        ],
    }
    controller_log = graph.metadata.setdefault("runtime_controller_log", [])
    if isinstance(controller_log, list):
        controller_log.append(entry)


def _select_ranked_action(
    graph: IdeaGraph,
    round_name: str,
    role: str,
    *,
    record_trace: bool,
    runtime_controller: Any | None = None,
    runtime_controller_metadata: dict[str, Any] | None = None,
) -> tuple[GraphAction, dict[str, object]]:
    baseline_action = _legacy_choose_round_action(deepcopy(graph), round_name, role)
    candidates = enumerate_candidate_specs(
        graph,
        round_name=round_name,
        role=role,
        baseline_action=baseline_action,
    )
    candidates = [
        {
            **candidate,
            "candidate_id": f"{round_name}:{role}:{index:03d}",
        }
        for index, candidate in enumerate(candidates)
    ]
    edit_candidates = [
        candidate for candidate in candidates if str(candidate.get("kind", "")).strip() != "commit"
    ]

    reference_subgraph = _reference_subgraph(graph)
    reference_snapshot = _compute_maturity_snapshot(
        graph,
        update_history=False,
        node_ids=set(reference_subgraph["node_ids"]),
    )

    ranked_candidates: list[dict[str, object]] = []
    for candidate in edit_candidates:
        ranked_candidates.append(
            {
                **candidate,
                **_score_action_spec(
                    graph,
                    round_name=round_name,
                    role=role,
                    spec=candidate,
                    reference_subgraph=reference_subgraph,
                    reference_snapshot=reference_snapshot,
                ),
            }
        )

    valid_candidates = [candidate for candidate in ranked_candidates if candidate.get("valid")]
    if valid_candidates:
        phase = resolve_round_phase(round_name)
        contradiction_target_ids = {
            edge.target_id
            for edge in unresolved_contradiction_edges(graph)
            if edge.target_id in graph.nodes
        }
        contradiction_related_ids = contradiction_related_node_ids(graph)

        if phase.key == "repair" and contradiction_target_ids:
            contradiction_repairs = [
                candidate
                for candidate in valid_candidates
                if candidate.get("kind") == "propose_repair"
                and list(candidate.get("target_ids", []))
                and str(list(candidate.get("target_ids", []))[0]) in contradiction_target_ids
            ]
            if contradiction_repairs:
                valid_candidates = contradiction_repairs
            else:
                contradiction_touching = [
                    candidate
                    for candidate in valid_candidates
                    if any(str(target_id) in contradiction_related_ids for target_id in candidate.get("target_ids", []))
                ]
                if contradiction_touching:
                    valid_candidates = contradiction_touching

        if phase.key == "stress_test" and role == "FeasibilityCritic":
            evidence_candidates = [
                candidate for candidate in valid_candidates if candidate.get("kind") == "attach_evidence"
            ]
            if evidence_candidates:
                valid_candidates = evidence_candidates

        if phase.key == "structure" and role in {"MechanismProposer", "ImpactReframer"}:
            structural_support = [
                candidate for candidate in valid_candidates if candidate.get("kind") == "add_support_edge"
            ]
            if structural_support:
                valid_candidates = structural_support

        selected_candidate = sorted(
            valid_candidates,
            key=lambda candidate: (
                float(candidate.get("predicted_gain", float("-inf"))),
                float(candidate.get("utility_gain", float("-inf"))),
                float(candidate.get("support_gain", float("-inf"))),
                float(candidate.get("coherence_gain", float("-inf"))),
            ),
            reverse=True,
        )[0]
    else:
        selected_candidate = ranked_candidates[0]

    heuristic_selected_candidate = dict(selected_candidate)

    if runtime_controller is not None and runtime_controller_metadata is not None and valid_candidates:
        controller_config = runtime_controller_metadata.get("config")
        if controller_config is not None:
            controller_kind = str(runtime_controller_metadata.get("kind", "")).strip() or "text_critic_rerank"
            round_index = (
                int(round_name[5:])
                if str(round_name).startswith("Round") and str(round_name)[5:].isdigit()
                else 0
            )
            controller_state = {
                "round_index": round_index,
                "support_coverage": reference_snapshot.support_coverage,
                "unresolved_contradiction_ratio": reference_snapshot.unresolved_contradiction_ratio,
                "completeness": reference_snapshot.completeness,
                "is_mature": reference_snapshot.is_mature,
            }
            heuristic_candidate_id = str(heuristic_selected_candidate.get("candidate_id", "")).strip()
            if controller_kind == "text_critic_rerank":
                controller_decision = select_text_critic_candidate(
                    graph,
                    round_name=round_name,
                    role=role,
                    state_features=controller_state,
                    candidate_specs=valid_candidates,
                    heuristic_candidate_id=heuristic_candidate_id,
                    model=runtime_controller,
                    config=controller_config,
                )
            elif controller_kind == "relation_graph_critic_rerank":
                controller_decision = select_relation_graph_critic_candidate(
                    graph,
                    round_name=round_name,
                    role=role,
                    state_features=controller_state,
                    candidate_specs=valid_candidates,
                    heuristic_candidate_id=heuristic_candidate_id,
                    runtime_bundle=runtime_controller,
                    config=controller_config,
                )
            else:
                controller_decision = None

            if controller_decision is not None:
                selected_candidate_id = str(controller_decision.policy_decision.selected_candidate_id)
                selected_candidate_from_ranked = next(
                    (
                        candidate
                        for candidate in valid_candidates
                        if str(candidate.get("candidate_id", "")).strip() == selected_candidate_id
                    ),
                    heuristic_selected_candidate,
                )
                selected_candidate = {
                    **selected_candidate_from_ranked,
                    "critic_score": controller_decision.selected_spec.get("critic_score"),
                    "controller_kind": controller_kind,
                    "controller_selected_source": controller_decision.policy_decision.selected_source,
                    "controller_override_margin": controller_decision.policy_decision.override_margin,
                    "controller_used_heuristic_fallback": controller_decision.policy_decision.used_heuristic_fallback,
                }
                fallback_reason = controller_decision.selected_spec.get("controller_fallback_reason")
                if fallback_reason is not None:
                    selected_candidate["controller_fallback_reason"] = str(fallback_reason)
                fallback_ids = controller_decision.selected_spec.get("controller_fallback_candidate_ids")
                if isinstance(fallback_ids, Sequence) and not isinstance(fallback_ids, (str, bytes)):
                    selected_candidate["controller_fallback_candidate_ids"] = tuple(str(item) for item in fallback_ids)

                if record_trace:
                    _record_runtime_controller_trace(
                        graph,
                        round_name=round_name,
                        role=role,
                        controller_kind=controller_kind,
                        heuristic_candidate=heuristic_selected_candidate,
                        selected_candidate=selected_candidate,
                        controller_decision={
                            "selected_source": controller_decision.policy_decision.selected_source,
                            "override_margin": controller_decision.policy_decision.override_margin,
                            "used_heuristic_fallback": controller_decision.policy_decision.used_heuristic_fallback,
                        },
                        scored_candidates=controller_decision.scored_candidates,
                    )

    selected_action = make_action(
        graph,
        round_name=round_name,
        role=role,
        kind=str(selected_candidate.get("kind", "")).strip(),
        target_ids=[str(item).strip() for item in selected_candidate.get("target_ids", []) if str(item).strip()],
        payload=dict(selected_candidate.get("payload", {}) or {}),
        rationale=str(selected_candidate.get("rationale", "")).strip(),
    )
    if record_trace:
        _record_action_selection_trace(
            graph,
            round_name=round_name,
            role=role,
            selected_candidate=selected_candidate,
            ranked_candidates=sorted(
                ranked_candidates,
                key=lambda candidate: (
                    1 if candidate.get("valid") else 0,
                    float(candidate.get("predicted_gain", float("-inf"))),
                ),
                reverse=True,
            ),
        )
    return selected_action, selected_candidate


def choose_round_action(graph: IdeaGraph, round_name: str, role: str) -> GraphAction:
    action, _ = _select_ranked_action(graph, round_name, role, record_trace=True)
    return action


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


def _clamp(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _subgraph_nodes(graph: IdeaGraph, node_ids: set[str] | None = None) -> list[Node]:
    return [
        node
        for node in graph.active_nodes()
        if node_ids is None or node.id in node_ids
    ]


def _subgraph_edges(graph: IdeaGraph, node_ids: set[str]) -> list[Edge]:
    return [
        edge
        for edge in graph.edges
        if edge.source_id in node_ids and edge.target_id in node_ids
    ]


def _node_selection_score(graph: IdeaGraph, node: Node) -> float:
    incoming = graph.incoming_edges(node.id)
    outgoing = graph.outgoing_edges(node.id)
    support_count = sum(1 for edge in incoming if edge.relation == "supports")
    repair_count = sum(1 for edge in incoming if edge.relation == "repairs")
    dependency_count = sum(
        1 for edge in incoming + outgoing if edge.relation == "depends_on"
    )
    unresolved_contradictions = sum(
        1 for edge in incoming + outgoing if edge.relation == "contradicts" and not edge.resolved
    )
    overlap_count = sum(
        1 for edge in incoming + outgoing if edge.relation == "overlaps_prior"
    )
    evidence_bonus = min(2, len(node.evidence))
    return (
        node.confidence
        + (0.20 * support_count)
        + (0.28 * repair_count)
        + (0.10 * dependency_count)
        + (0.12 * evidence_bonus)
        - (0.25 * unresolved_contradictions)
        - (0.12 * overlap_count)
    )


def _claim_chain_score(nodes: list[Node], edges: list[Edge]) -> float:
    if not nodes:
        return 0.0

    nodes_by_id = {node.id: node for node in nodes}
    positive_relations = {"supports", "refines", "depends_on", "repairs"}

    def link(left_types: set[str], right_types: set[str]) -> float:
        for edge in edges:
            if edge.relation not in positive_relations:
                continue
            source = nodes_by_id.get(edge.source_id)
            target = nodes_by_id.get(edge.target_id)
            if source is None or target is None:
                continue
            if source.type in left_types and target.type in right_types:
                return 1.0
            if target.type in left_types and source.type in right_types:
                return 1.0
        return 0.0

    segments = [
        link({"Hypothesis", "NoveltyClaim"}, {"Problem"}),
        link({"Method", "Repair"}, {"Hypothesis", "Problem"}),
        link({"EvalPlan"}, {"Method", "Hypothesis"}),
    ]
    return round(sum(segments) / len(segments), 2)


def _open_risk_penalty(nodes: list[Node], edges: list[Edge]) -> float:
    risk_nodes = [node for node in nodes if node.type == "Risk"]
    if not risk_nodes:
        return 0.0
    repaired_target_ids = {edge.target_id for edge in edges if edge.relation == "repairs"}
    open_risk_count = sum(
        1
        for node in risk_nodes
        if node.id not in repaired_target_ids and not node.evidence
    )
    return round(open_risk_count / len(risk_nodes), 2)


def _normalized_token_set(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if len(token) >= 4
        and token
        not in {
            "with",
            "that",
            "this",
            "from",
            "using",
            "use",
            "into",
            "task",
            "tasks",
            "method",
            "methods",
            "approach",
            "approaches",
            "model",
            "models",
            "report",
            "metric",
            "metrics",
            "benchmark",
            "datasets",
            "dataset",
            "generic",
        }
    }


def _contains_anchor(text: str, anchor: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_anchor = normalize_text(anchor)
    return bool(normalized_anchor and normalized_anchor in normalized_text)


def _reference_snippet_texts(metadata: dict[str, object]) -> list[str]:
    payload = metadata.get("paper_grounding", {})
    if not isinstance(payload, dict):
        return []
    raw_snippets = payload.get("reference_paper_snippets", [])
    if not isinstance(raw_snippets, list):
        return []

    texts: list[str] = []
    for item in raw_snippets[:6]:
        if not isinstance(item, dict):
            continue
        for field_name in ("method", "abstract", "evaluation", "introduction", "conclusion"):
            value = str(item.get(field_name, "")).strip()
            if value:
                texts.append(value)
    return texts


def _benchmark_specificity_score(graph: IdeaGraph, nodes: list[Node]) -> float:
    if not nodes:
        return 0.0
    grounding = build_literature_grounding(
        literature=graph.literature,
        metadata=_utility_grounding_metadata(graph),
    )
    text = " ".join(node.text for node in nodes if node.text)
    anchors = [
        *grounding.dataset_items[:3],
        *grounding.metric_items[:4],
    ]
    if not anchors:
        return 0.0
    hit_count = sum(1 for anchor in anchors if _contains_anchor(text, anchor))
    return round(_clamp(hit_count / min(3, len(anchors))), 2)


def _experiment_method_alignment_score(nodes: list[Node]) -> float:
    methods = [node for node in nodes if node.type == "Method"]
    evaluations = [node for node in nodes if node.type == "EvalPlan"]
    if not methods or not evaluations:
        return 0.0

    method_tokens = set().union(*(_normalized_token_set(node.text) for node in methods))
    evaluation_tokens = set().union(*(_normalized_token_set(node.text) for node in evaluations))
    shared_tokens = method_tokens & evaluation_tokens

    evaluation_text = " ".join(node.text for node in evaluations)
    explicit_eval_signal = 0.0
    if any(marker in normalize_text(evaluation_text) for marker in ("ablate", "stress", "held out", "error rate", "success rate")):
        explicit_eval_signal += 0.4
    if any(marker in normalize_text(evaluation_text) for marker in ("osworld", "rmse", "mae", "crps", "iou", "f1", "accuracy")):
        explicit_eval_signal += 0.4
    if "compare against" in normalize_text(evaluation_text):
        explicit_eval_signal += 0.2

    shared_score = min(1.0, len(shared_tokens) / 3.0) if shared_tokens else 0.0
    return round(_clamp((0.55 * shared_score) + (0.45 * explicit_eval_signal)), 2)


def _role_slot_balance_score(nodes: list[Node]) -> float:
    required_roles = {
        "ImpactReframer",
        "MechanismProposer",
        "NoveltyExaminer",
        "EvaluationDesigner",
    }
    present_roles = {node.role for node in nodes if node.role in required_roles}
    required_types = {"Problem", "Hypothesis", "Method", "EvalPlan", "NoveltyClaim"}
    present_types = {node.type for node in nodes if node.type in required_types}
    role_score = len(present_roles) / len(required_roles)
    type_score = len(present_types) / len(required_types)
    return round((0.45 * role_score) + (0.55 * type_score), 2)


def _reference_copy_penalty(graph: IdeaGraph, nodes: list[Node]) -> float:
    reference_texts = [normalize_text(text) for text in _reference_snippet_texts(_utility_grounding_metadata(graph))]
    reference_texts = [text for text in reference_texts if text]
    if not reference_texts:
        return 0.0

    penalty = 0.0
    for node in nodes:
        if node.type not in {"Method", "Hypothesis", "NoveltyClaim"}:
            continue
        normalized_node = normalize_text(node.text)
        if not normalized_node:
            continue
        node_tokens = _normalized_token_set(node.text)
        for reference_text in reference_texts:
            if normalized_node == reference_text:
                penalty = max(penalty, 1.0)
                continue
            if len(normalized_node) >= 32 and normalized_node in reference_text:
                penalty = max(penalty, 0.9)
                continue
            reference_tokens = {
                token
                for token in reference_text.split()
                if len(token) >= 4
            }
            if node_tokens and reference_tokens:
                overlap = len(node_tokens & reference_tokens) / max(1, len(node_tokens | reference_tokens))
                if overlap >= 0.85:
                    penalty = max(penalty, round(overlap, 2))
    return round(penalty, 2)


def utility_breakdown(graph: IdeaGraph, node_ids: set[str] | None = None) -> UtilityBreakdown:
    active_nodes = _subgraph_nodes(graph, node_ids)
    if not active_nodes:
        return UtilityBreakdown()

    active_node_ids = {node.id for node in active_nodes}
    relevant_edges = _subgraph_edges(graph, active_node_ids)
    active_types = {node.type for node in active_nodes}
    tracked_nodes = [
        node
        for node in active_nodes
        if node.type in {"Problem", "Hypothesis", "Method", "EvalPlan", "NoveltyClaim"}
    ]
    supported_nodes = [
        node
        for node in active_nodes
        if node.type in {"Hypothesis", "Method", "NoveltyClaim", "EvalPlan"}
    ]
    contradictions = [edge for edge in relevant_edges if edge.relation == "contradicts"]
    unresolved = [edge for edge in contradictions if not edge.resolved]
    overlap_edges = [edge for edge in relevant_edges if edge.relation == "overlaps_prior"]

    average_confidence = (
        sum(node.confidence for node in tracked_nodes) / len(tracked_nodes)
        if tracked_nodes
        else sum(node.confidence for node in active_nodes) / len(active_nodes)
    )
    novelty_presence = 1.0 if "NoveltyClaim" in active_types else (0.55 if {"Hypothesis", "Method"}.issubset(active_types) else 0.2)
    overlap_penalty = len(overlap_edges) / max(1, len(relevant_edges)) if relevant_edges else 0.0
    novelty = _clamp((0.65 * novelty_presence) + (0.35 * (1.0 - overlap_penalty)))
    promise = _clamp((0.70 * average_confidence) + (0.30 * novelty))

    supported_count = 0
    for node in supported_nodes:
        incoming = [edge for edge in relevant_edges if edge.target_id == node.id]
        has_support = any(edge.relation == "supports" for edge in incoming)
        has_repair = any(edge.relation == "repairs" for edge in incoming)
        if has_support or has_repair or node.evidence:
            supported_count += 1
    support = supported_count / len(supported_nodes) if supported_nodes else 0.0

    evidence = (
        sum(1 for node in tracked_nodes if node.evidence) / len(tracked_nodes)
        if tracked_nodes
        else 0.0
    )
    completeness = 1.0 if {"Problem", "Hypothesis", "Method", "EvalPlan"}.issubset(active_types) else 0.0
    contradiction_penalty = len(unresolved) / len(contradictions) if contradictions else 0.0
    coherence = _clamp(
        (0.45 * completeness)
        + (0.35 * _claim_chain_score(active_nodes, relevant_edges))
        + (0.20 * (1.0 - contradiction_penalty))
    )
    benchmark_specificity = _benchmark_specificity_score(graph, active_nodes)
    experiment_alignment = _experiment_method_alignment_score(active_nodes)
    role_balance = _role_slot_balance_score(active_nodes)
    reference_copy_penalty = _reference_copy_penalty(graph, active_nodes)
    open_risk_penalty = _open_risk_penalty(active_nodes, relevant_edges)
    size_penalty = _clamp(max(0, len(active_nodes) - 7) / 5.0)

    total = round(
        10.0
        * _clamp(
            (0.25 * promise)
            + (0.21 * support)
            + (0.17 * coherence)
            + (0.15 * evidence)
            + (0.15 * novelty)
            + (0.10 * benchmark_specificity)
            + (0.07 * experiment_alignment)
            + (0.03 * role_balance)
            - (0.15 * contradiction_penalty)
            - (0.10 * reference_copy_penalty)
            - (0.10 * open_risk_penalty)
            - (0.05 * size_penalty)
        ),
        2,
    )
    return UtilityBreakdown(
        promise=round(promise, 2),
        support=round(support, 2),
        coherence=round(coherence, 2),
        evidence=round(evidence, 2),
        novelty=round(novelty, 2),
        benchmark_specificity=round(benchmark_specificity, 2),
        experiment_alignment=round(experiment_alignment, 2),
        role_balance=round(role_balance, 2),
        reference_copy_penalty=round(reference_copy_penalty, 2),
        contradiction_penalty=round(contradiction_penalty, 2),
        open_risk_penalty=round(open_risk_penalty, 2),
        size_penalty=round(size_penalty, 2),
        total=total,
    )


def utility_score(graph: IdeaGraph, node_ids: set[str] | None = None) -> float:
    return utility_breakdown(graph, node_ids).total


def _utility_stability(history: list[float]) -> bool:
    if len(history) < 3:
        return False
    a, b, c = history[-3:]
    return abs(c - b) < 0.45 and abs(b - a) < 0.45


def _compute_maturity_snapshot(
    graph: IdeaGraph,
    *,
    update_history: bool,
    node_ids: set[str] | None = None,
) -> MaturitySnapshot:
    tracked_types = {"Hypothesis", "Method", "NoveltyClaim", "EvalPlan"}
    active_nodes = _subgraph_nodes(graph, node_ids)
    active_node_ids = {node.id for node in active_nodes}
    relevant_edges = _subgraph_edges(graph, active_node_ids)
    tracked_nodes = [node for node in active_nodes if node.type in tracked_types]

    supported_count = 0
    for node in tracked_nodes:
        incoming = [edge for edge in relevant_edges if edge.target_id == node.id]
        has_support = any(edge.relation == "supports" for edge in incoming)
        has_repair = any(edge.relation == "repairs" for edge in incoming)
        if has_support or has_repair or node.evidence:
            supported_count += 1

    support_coverage = 0.0 if not tracked_nodes else round(supported_count / len(tracked_nodes), 2)
    contradictions = [edge for edge in relevant_edges if edge.relation == "contradicts"]
    unresolved = [edge for edge in contradictions if not edge.resolved]
    unresolved_ratio = 0.0 if not contradictions else round(len(unresolved) / len(contradictions), 2)

    active_types = {node.type for node in active_nodes}
    structural_completeness = {"Problem", "Hypothesis", "Method", "EvalPlan"}.issubset(active_types)
    claim_chain = select_claim_chain(graph, node_ids=active_node_ids or None)
    claim_chain_ready = bool(claim_chain and claim_chain.get("coverage", {}).get("is_synthesis_ready"))
    completeness = structural_completeness and claim_chain_ready

    breakdown = utility_breakdown(graph, active_node_ids or None)
    utility = breakdown.total
    if update_history:
        graph.utility_history.append(utility)
        history = list(graph.utility_history)
    else:
        history = [*graph.utility_history, utility]
    utility_stable = _utility_stability(history)
    weak_context_specificity = _keyword_only_specificity_score(graph, active_node_ids or None)
    weak_context_mode = bool(_weak_context_scaffold(graph))
    benchmark_mode = bool(str(graph.metadata.get("benchmark", "")).strip()) and not weak_context_mode
    benchmark_specific_ready = (
        (not benchmark_mode)
        or (
            breakdown.benchmark_specificity >= 0.3
            and breakdown.experiment_alignment >= 0.2
        )
    )
    benchmark_high_confidence_ready = (
        (not benchmark_mode)
        or (
            breakdown.benchmark_specificity >= 0.45
            and breakdown.experiment_alignment >= 0.3
        )
    )
    min_rounds_before_maturity = max(
        2,
        int(graph.metadata.get("idea_graph_min_rounds_before_maturity", 2) or 2),
    )
    high_confidence_mature = (
        len(history) >= max(3, min_rounds_before_maturity)
        and support_coverage >= 0.84
        and unresolved_ratio <= 0.18
        and completeness
        and breakdown.coherence >= 0.7
        and breakdown.evidence >= 0.18
        and utility >= 6.6
        and benchmark_high_confidence_ready
        and (not weak_context_mode or weak_context_specificity >= 0.62)
    )

    return MaturitySnapshot(
        support_coverage=support_coverage,
        unresolved_contradiction_ratio=unresolved_ratio,
        utility=utility,
        utility_stable=utility_stable,
        completeness=completeness,
        is_mature=(
            (
                len(history) >= (max(3, min_rounds_before_maturity) if weak_context_mode else min_rounds_before_maturity)
                and support_coverage >= 0.74
                and unresolved_ratio <= 0.33
                and completeness
                and breakdown.coherence >= 0.6
                and breakdown.evidence >= 0.12
                and benchmark_specific_ready
                and (not weak_context_mode or weak_context_specificity >= 0.55)
                and (
                    utility_stable
                    or (
                        len(history) >= (max(3, min_rounds_before_maturity) if weak_context_mode else min_rounds_before_maturity)
                        and support_coverage >= 0.82
                        and breakdown.evidence >= 0.2
                        and utility >= 7.5
                        and benchmark_high_confidence_ready
                        and (not weak_context_mode or weak_context_specificity >= 0.62)
                    )
                )
            )
            or high_confidence_mature
        ),
        utility_breakdown=breakdown,
    )


def _is_connected_subgraph(node_ids: set[str], edges: list[Edge]) -> bool:
    if not node_ids:
        return False
    if len(node_ids) == 1:
        return True

    adjacency = {node_id: set() for node_id in node_ids}
    for edge in edges:
        if edge.source_id in adjacency and edge.target_id in adjacency:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)

    start = next(iter(node_ids))
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return seen == node_ids


def _top_candidate_nodes_by_type(graph: IdeaGraph, node_type: str, *, limit: int = 3) -> list[Node]:
    candidates = [node for node in graph.active_nodes() if node.type == node_type]
    return sorted(candidates, key=lambda node: _node_selection_score(graph, node), reverse=True)[:limit]


def _expand_candidate_subgraph(graph: IdeaGraph, core_nodes: tuple[Node, Node, Node, Node]) -> dict[str, object]:
    selected_nodes = list(core_nodes)
    selected_node_ids = {node.id for node in selected_nodes}

    neighbor_candidates: list[tuple[float, Node]] = []
    neighbor_type_bonus = {
        "NoveltyClaim": 0.26,
        "Repair": 0.24,
        "Risk": 0.18,
        "Assumption": 0.16,
        "Method": 0.10,
        "EvalPlan": 0.10,
    }
    for node in graph.active_nodes():
        if node.id in selected_node_ids:
            continue
        incident_edges = [
            edge
            for edge in graph.incoming_edges(node.id) + graph.outgoing_edges(node.id)
            if edge.source_id in selected_node_ids or edge.target_id in selected_node_ids
        ]
        if not incident_edges:
            continue
        if any(edge.relation == "contradicts" and not edge.resolved for edge in incident_edges):
            continue
        type_bonus = neighbor_type_bonus.get(node.type, 0.0)
        relation_bonus = 0.04 * sum(
            1 for edge in incident_edges if edge.relation in {"supports", "repairs", "refines", "depends_on"}
        )
        evidence_bonus = 0.08 if node.evidence else 0.0
        neighbor_candidates.append(
            (_node_selection_score(graph, node) + type_bonus + relation_bonus + evidence_bonus, node)
        )

    selected_neighbor_types: set[str] = set()
    for _, node in sorted(neighbor_candidates, key=lambda item: item[0], reverse=True):
        if len(selected_nodes) >= 7:
            break
        if node.type in selected_neighbor_types and node.type not in {"Repair"}:
            continue
        selected_nodes.append(node)
        selected_node_ids.add(node.id)
        selected_neighbor_types.add(node.type)

    selected_edges = [
        edge
        for edge in graph.edges
        if edge.source_id in selected_node_ids and edge.target_id in selected_node_ids
    ]
    snapshot = _compute_maturity_snapshot(
        graph,
        update_history=False,
        node_ids=selected_node_ids,
    )
    return {
        "node_ids": [node.id for node in selected_nodes],
        "edge_ids": [edge.id for edge in selected_edges],
        "core_node_ids": [node.id for node in core_nodes],
        "utility": snapshot.utility,
        "utility_breakdown": asdict(snapshot.utility_breakdown),
        "support_coverage": snapshot.support_coverage,
        "unresolved_contradiction_ratio": snapshot.unresolved_contradiction_ratio,
        "utility_stable": snapshot.utility_stable,
        "completeness": snapshot.completeness,
        "is_mature": snapshot.is_mature,
        "connected": _is_connected_subgraph(selected_node_ids, selected_edges),
    }


def _candidate_subgraphs(graph: IdeaGraph) -> list[dict[str, object]]:
    core_types = ("Problem", "Hypothesis", "Method", "EvalPlan")
    candidate_lists = [_top_candidate_nodes_by_type(graph, node_type) for node_type in core_types]
    if any(not candidates for candidates in candidate_lists):
        return []

    candidates: list[dict[str, object]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for combo in product(*candidate_lists):
        core_signature = tuple(sorted(node.id for node in combo))
        if core_signature in seen_signatures:
            continue
        candidate = _expand_candidate_subgraph(graph, combo)
        if not candidate["connected"]:
            continue
        signature = tuple(sorted(candidate["node_ids"]))
        if signature in seen_signatures:
            continue
        seen_signatures.add(core_signature)
        seen_signatures.add(signature)
        candidates.append(candidate)

    return sorted(
        candidates,
        key=lambda item: (
            1 if item["is_mature"] else 0,
            float(item["utility"]),
            float(item["support_coverage"]),
            1.0 - float(item["unresolved_contradiction_ratio"]),
            1 if item["utility_stable"] else 0,
        ),
        reverse=True,
    )


def _best_candidate_subgraph(graph: IdeaGraph, *, prefer_mature: bool = False) -> dict[str, object] | None:
    candidates = _candidate_subgraphs(graph)
    if not candidates:
        return None
    if prefer_mature:
        mature_candidates = [candidate for candidate in candidates if candidate["is_mature"]]
        if mature_candidates:
            return {**mature_candidates[0], "selection_mode": "mature_candidate"}
    return {**candidates[0], "selection_mode": "best_candidate"}


def maturity_snapshot(graph: IdeaGraph) -> MaturitySnapshot:
    claim_chain = select_claim_chain(graph)
    if claim_chain is not None and claim_chain.get("coverage", {}).get("is_synthesis_ready"):
        return _compute_maturity_snapshot(
            graph,
            update_history=True,
            node_ids=set(claim_chain["subgraph"]["node_ids"]),
        )

    candidate = _best_candidate_subgraph(graph)
    if candidate is not None:
        return _compute_maturity_snapshot(
            graph,
            update_history=True,
            node_ids=set(candidate["node_ids"]),
        )

    legacy = _legacy_select_final_subgraph(graph)
    return _compute_maturity_snapshot(
        graph,
        update_history=True,
        node_ids=set(legacy["node_ids"]),
    )


def _legacy_select_final_subgraph(graph: IdeaGraph) -> dict[str, object]:
    selected_nodes = []
    for node_type in ("Problem", "Hypothesis", "Method", "EvalPlan"):
        candidates = [node for node in graph.active_nodes() if node.type == node_type]
        if candidates:
            selected_nodes.append(max(candidates, key=lambda node: _node_selection_score(graph, node)))

    selected_node_ids = {node.id for node in selected_nodes}

    neighbor_candidates: list[tuple[float, Node]] = []
    neighbor_type_bonus = {
        "NoveltyClaim": 0.26,
        "Repair": 0.24,
        "Risk": 0.18,
        "Assumption": 0.16,
        "Method": 0.10,
        "EvalPlan": 0.10,
    }
    for node in graph.active_nodes():
        if node.id in selected_node_ids:
            continue
        incident_edges = [
            edge
            for edge in graph.incoming_edges(node.id) + graph.outgoing_edges(node.id)
            if edge.source_id in selected_node_ids or edge.target_id in selected_node_ids
        ]
        if not incident_edges:
            continue
        if any(edge.relation == "contradicts" and not edge.resolved for edge in incident_edges):
            continue
        type_bonus = neighbor_type_bonus.get(node.type, 0.0)
        relation_bonus = 0.04 * sum(
            1 for edge in incident_edges if edge.relation in {"supports", "repairs", "refines", "depends_on"}
        )
        evidence_bonus = 0.08 if node.evidence else 0.0
        neighbor_candidates.append(
            (_node_selection_score(graph, node) + type_bonus + relation_bonus + evidence_bonus, node)
        )

    selected_neighbor_types: set[str] = set()
    for _, node in sorted(neighbor_candidates, key=lambda item: item[0], reverse=True):
        if len(selected_nodes) >= 7:
            break
        if node.type in selected_neighbor_types and node.type not in {"Repair"}:
            continue
        selected_nodes.append(node)
        selected_node_ids.add(node.id)
        selected_neighbor_types.add(node.type)

    selected_edges = [
        edge
        for edge in graph.edges
        if edge.source_id in selected_node_ids and edge.target_id in selected_node_ids
    ]
    return {
        "node_ids": [node.id for node in selected_nodes],
        "edge_ids": [edge.id for edge in selected_edges],
        "core_node_ids": [node.id for node in selected_nodes[:4]],
        "utility": utility_score(graph, selected_node_ids),
    }


def select_final_subgraph(graph: IdeaGraph) -> dict[str, object]:
    claim_chain = select_claim_chain(graph)
    if claim_chain is not None and claim_chain.get("coverage", {}).get("is_synthesis_ready"):
        claim_subgraph = dict(claim_chain["subgraph"])
        selected_node_ids = set(str(item) for item in claim_subgraph.get("node_ids", []))
        selected_edge_ids = set(str(item) for item in claim_subgraph.get("edge_ids", []))
        snapshot = _compute_maturity_snapshot(
            graph,
            update_history=False,
            node_ids=selected_node_ids,
        )
        claim_subgraph.update(
            {
                "utility": snapshot.utility,
                "utility_breakdown": asdict(snapshot.utility_breakdown),
                "support_coverage": snapshot.support_coverage,
                "unresolved_contradiction_ratio": snapshot.unresolved_contradiction_ratio,
                "utility_stable": snapshot.utility_stable,
                "completeness": snapshot.completeness,
                "is_mature": snapshot.is_mature,
                "connected": _is_connected_subgraph(
                    selected_node_ids,
                    [edge for edge in graph.edges if edge.id in selected_edge_ids],
                ),
                "selection_mode": "claim_chain",
                "claim_chain": claim_chain,
            }
        )
        return claim_subgraph

    candidate = _best_candidate_subgraph(graph, prefer_mature=True)
    if candidate is not None:
        return candidate

    legacy = _legacy_select_final_subgraph(graph)
    legacy_edge_ids = set(legacy["edge_ids"])
    legacy_snapshot = _compute_maturity_snapshot(
        graph,
        update_history=False,
        node_ids=set(legacy["node_ids"]),
    )
    legacy.update(
        {
            "utility_breakdown": asdict(legacy_snapshot.utility_breakdown),
            "support_coverage": legacy_snapshot.support_coverage,
            "unresolved_contradiction_ratio": legacy_snapshot.unresolved_contradiction_ratio,
            "utility_stable": legacy_snapshot.utility_stable,
            "completeness": legacy_snapshot.completeness,
            "is_mature": legacy_snapshot.is_mature,
            "connected": _is_connected_subgraph(
                set(legacy["node_ids"]),
                [edge for edge in graph.edges if edge.id in legacy_edge_ids],
            ),
            "selection_mode": "legacy_heuristic",
        }
    )
    return legacy


def _proposal_title(graph: IdeaGraph, selected: dict[str, str]) -> str:
    topic = _clean_topic_text(graph.topic)
    if topic:
        return topic[0].upper() + topic[1:] if topic[0].islower() else topic
    method = selected.get("Method", "").strip().rstrip(".")
    if method:
        return method.split(".")[0][:140]
    return "Structured Research Idea"


def _clean_topic_text(text: str) -> str:
    cleaned = str(text).strip().rstrip(".")
    prefix = "The topic of this paper is "
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def _first_sentence(text: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return ""
    for delimiter in (". ", "?\n", "!\n", "?", "!"):
        if delimiter in cleaned:
            segment = cleaned.split(delimiter, 1)[0].strip()
            if segment:
                return segment + ("" if segment.endswith((".", "?", "!")) else ".")
    return cleaned if cleaned.endswith((".", "?", "!")) else f"{cleaned}."


def _clean_evaluation_text(text: str, *, topic_text: str, raw_topic: str) -> str:
    cleaned = str(text).strip()
    raw_topic_clean = str(raw_topic).strip().rstrip(".")
    prefixes = [
        f"Evaluate {raw_topic_clean} on ",
        f"Evaluate {topic_text} on ",
    ]
    for prefix in prefixes:
        if prefix and cleaned.startswith(prefix):
            return "Evaluate the proposed approach on " + cleaned[len(prefix) :]
    return cleaned


def _strip_leading_repetition(prefix: str, text: str) -> str:
    prefix_clean = str(prefix).strip().rstrip(".!?")
    text_clean = str(text).strip()
    if not prefix_clean or not text_clean:
        return text_clean
    if text_clean.casefold().startswith(prefix_clean.casefold()):
        remainder = text_clean[len(prefix_clean) :].lstrip(" .!?")
        return remainder if remainder else text_clean
    return text_clean


def _sentences_to_paragraph(parts: list[str]) -> str:
    normalized: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        if text[-1] not in ".!?":
            text = f"{text}."
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return " ".join(normalized)


def synthesize_proposal(graph: IdeaGraph, subgraph: dict[str, object]) -> FinalProposal:
    selected = {graph.nodes[node_id].type: graph.nodes[node_id].text for node_id in subgraph["node_ids"]}
    topic_text = _clean_topic_text(graph.topic)
    generation_metadata = graph.metadata.get("generation_safe_metadata", graph.metadata)
    if not isinstance(generation_metadata, dict):
        generation_metadata = graph.metadata
    grounding = build_literature_grounding(literature=graph.literature, metadata=generation_metadata)
    benchmark_motivation = str(generation_metadata.get("motivation", "")).strip()
    unresolved_count = len(unresolved_contradiction_edges(graph))

    problem = selected.get(
        "Problem",
        f"{topic_text or graph.topic} remains insufficiently specified for rigorous scientific evaluation.",
    )
    if "..." in problem and benchmark_motivation:
        problem = _first_sentence(benchmark_motivation)
    hypothesis = selected.get(
        "Hypothesis",
        "A structured collaboration process can produce a stronger research idea than early whole-idea voting.",
    )
    method = selected.get(
        "Method",
        "Compose a typed idea graph that preserves partial claims, disagreements, and supporting evidence during ideation.",
    )
    if method and normalize_text(hypothesis) == normalize_text(method):
        hypothesis = (
            f"A more structured method for {topic_text or graph.topic} should improve reliability, "
            "robustness, or scientific usefulness over current baselines."
        )

    raw_evaluation = selected.get(
        "EvalPlan",
        "Evaluate the idea against strong baselines, representative benchmark tasks, and targeted ablations.",
    )
    evaluation = grounding.experiment_plan_summary or _clean_evaluation_text(
        raw_evaluation,
        topic_text=topic_text,
        raw_topic=graph.topic,
    )
    if "..." in evaluation:
        evaluation = _clean_evaluation_text(raw_evaluation, topic_text=topic_text, raw_topic=graph.topic)

    existing_methods = grounding.existing_methods_summary

    motivation = benchmark_motivation or _sentences_to_paragraph(
        [
            "Current approaches leave a gap between promising partial ideas and fully testable research proposals",
            hypothesis,
        ]
    )
    motivation = _strip_leading_repetition(problem, motivation)
    if not motivation:
        motivation = (
            f"This motivates a more explicit and testable research strategy for {topic_text or graph.topic}."
        )
    significance = _sentences_to_paragraph(
        [
            f"If successful, this idea would produce a more testable and clearly motivated research direction for {topic_text or graph.topic}.",
            "It would also make the proposal easier to compare against strong baselines and ablations.",
        ]
    )
    caveats_parts = [
        "The current scaffold still relies on heuristic graph updates and lightweight literature grounding."
    ]
    if unresolved_count:
        caveats_parts.append(
            f"The graph still contains {unresolved_count} unresolved contradiction(s), so the proposal should be stress-tested further."
        )
    caveats = " ".join(caveats_parts)
    return FinalProposal(
        title=_proposal_title(graph, selected),
        abstract="",
        problem=problem,
        existing_methods=existing_methods,
        motivation=motivation,
        hypothesis=hypothesis,
        method=method,
        evaluation=evaluation,
        significance=significance,
        caveats=caveats,
    )


def run_experiment(
    topic: str,
    literature: list[str],
    metadata: dict[str, object] | None = None,
    collaboration_backend: CollaborationBackend | None = None,
    runtime_controller: Any | None = None,
    runtime_controller_metadata: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    max_rounds: int = 3,
    stop_when_mature: bool = True,
) -> IdeaGraph:
    graph = IdeaGraph(topic=topic, literature=literature, metadata=dict(metadata or {}))
    runtime_protocol = (
        str(graph.metadata.get("runtime_protocol", "sequential_v1")).strip()
        or "sequential_v1"
    )
    graph.metadata["runtime_protocol"] = runtime_protocol
    graph.metadata.setdefault(
        "literature_grounding",
        build_literature_grounding(literature=graph.literature, metadata=_generation_metadata(graph)).as_dict(),
    )
    graph.metadata["max_rounds_requested"] = max(1, int(max_rounds))
    graph.metadata["stop_when_mature"] = bool(stop_when_mature)
    if runtime_controller_metadata is not None:
        sanitized_runtime_controller = {
            key: value
            for key, value in runtime_controller_metadata.items()
            if key != "config"
        }
        graph.metadata["runtime_controller"] = sanitized_runtime_controller
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
    if core_node_coverage_enabled(graph):
        ensure_core_node_coverage(graph)
        emit_progress(
            graph,
            progress_callback,
            stage="coverage_safeguard",
            message="Core-node coverage safeguard checked and applied where needed.",
            details={"enabled": True},
        )
    else:
        emit_progress(
            graph,
            progress_callback,
            stage="coverage_safeguard",
            message="Core-node coverage safeguard disabled for this protocol variant.",
            details={"enabled": False},
        )
    emit_progress(
        graph,
        progress_callback,
        stage="merge_complete",
        message=f"Seed merge complete: {len(graph.active_nodes())} active nodes, {len(graph.edges)} edges.",
        details={"active_nodes": len(graph.active_nodes()), "edges": len(graph.edges)},
    )
    if runtime_controller_metadata is not None:
        emit_progress(
            graph,
            progress_callback,
            stage="runtime_controller",
            message=(
                f"Runtime controller active: {runtime_controller_metadata.get('kind', 'unknown')} "
                f"with tau_override={runtime_controller_metadata.get('tau_override', 'n/a')}."
            ),
            details={
                "kind": runtime_controller_metadata.get("kind"),
                "model_path": runtime_controller_metadata.get("model_path"),
                "use_commit": runtime_controller_metadata.get("use_commit"),
                "tau_override": runtime_controller_metadata.get("tau_override"),
            },
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
        if runtime_protocol == "parallel_graph_v2":
            from .parallel_replay import append_parallel_edit_rows, append_parallel_round_trace
            from .parallel_runtime import execute_parallel_role_round

            result = execute_parallel_role_round(
                graph,
                round_name=round_name,
                collaboration_backend=collaboration_backend,
                runtime_controller=runtime_controller,
                runtime_controller_metadata=runtime_controller_metadata,
                progress_callback=progress_callback,
            )
            selected_action_payloads = [
                {
                    "id": action.id,
                    "role": action.role,
                    "kind": action.kind,
                    "target_ids": list(action.target_ids),
                    "payload": dict(action.payload),
                    "source": action.source,
                }
                for action in result.selected_actions
            ]
            append_parallel_round_trace(
                graph.metadata,
                {
                    "round": result.round_name,
                    "active_roles": list(result.active_roles),
                    "inactive_roles": [role for role in ROLE_NAMES if role not in result.active_roles],
                    "selected_actions": selected_action_payloads,
                    "skipped_roles": list(result.skipped_roles),
                    "termination_reason": result.termination_reason,
                    "graph_delta": {
                        "node_count_before": result.node_count_before,
                        "node_count_after": result.node_count_after,
                        "node_delta": result.node_count_after - result.node_count_before,
                        "edge_count_before": result.edge_count_before,
                        "edge_count_after": result.edge_count_after,
                        "edge_delta": result.edge_count_after - result.edge_count_before,
                        "action_count_before": result.action_count_before,
                        "action_count_after": result.action_count_after,
                        "action_delta": result.action_count_after - result.action_count_before,
                    },
                },
            )
            append_parallel_edit_rows(
                graph.metadata,
                result.edit_rows,
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
                    f"utility={snapshot.utility}, coherence={snapshot.utility_breakdown.coherence}, "
                    f"mature={snapshot.is_mature}."
                ),
                details={
                    "round": round_name,
                    "runtime_protocol": runtime_protocol,
                    "active_roles": list(result.active_roles),
                    "skipped_roles": list(result.skipped_roles),
                    "selected_action_count": len(result.selected_actions),
                    "support_coverage": snapshot.support_coverage,
                    "unresolved_contradiction_ratio": snapshot.unresolved_contradiction_ratio,
                    "utility": snapshot.utility,
                    "utility_breakdown": asdict(snapshot.utility_breakdown),
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
            continue
        for role in ROLE_NAMES:
            action_source = "deterministic"
            deterministic_ranked_action: GraphAction | None = None
            deterministic_ranked_meta: dict[str, object] | None = None
            if collaboration_backend is None:
                action, deterministic_ranked_meta = _select_ranked_action(
                    graph,
                    round_name,
                    role,
                    record_trace=True,
                    runtime_controller=runtime_controller,
                    runtime_controller_metadata=runtime_controller_metadata,
                )
            else:
                deterministic_ranked_action, deterministic_ranked_meta = _select_ranked_action(
                    graph,
                    round_name,
                    role,
                    record_trace=bool(runtime_controller is not None),
                    runtime_controller=runtime_controller,
                    runtime_controller_metadata=runtime_controller_metadata,
                )
                try:
                    decision = collaboration_backend.choose_action(graph, round_name, role)
                    append_agent_trace(graph, stage=f"{round_name}_action", role=role, trace=decision.trace)
                    llm_action = action_from_decision(
                        graph,
                        round_name=round_name,
                        role=role,
                        decision=decision,
                    )
                    alignment_error = llm_action_alignment_error(graph, round_name, llm_action)
                    if alignment_error is not None:
                        raise ValueError(alignment_error)
                    reference_subgraph = _reference_subgraph(graph)
                    reference_snapshot = _compute_maturity_snapshot(
                        graph,
                        update_history=False,
                        node_ids=set(reference_subgraph["node_ids"]),
                    )
                    llm_candidate = {
                        **_action_spec_from_action(llm_action, candidate_source="llm"),
                        **_score_action_spec(
                            graph,
                            round_name=round_name,
                            role=role,
                            spec=_action_spec_from_action(llm_action, candidate_source="llm"),
                            reference_subgraph=reference_subgraph,
                            reference_snapshot=reference_snapshot,
                        ),
                    }
                    llm_gain = float(llm_candidate.get("predicted_gain", float("-inf")))
                    deterministic_gain = float(
                        (deterministic_ranked_meta or {}).get("predicted_gain", float("-inf"))
                    )
                    override_margin = float(graph.metadata.get("idea_graph_llm_override_margin", 0.4))
                    if (
                        deterministic_ranked_action is not None
                        and deterministic_ranked_meta is not None
                        and deterministic_gain > llm_gain + override_margin
                    ):
                        action = deterministic_ranked_action
                        action_source = "utility_controller_override"
                        graph.metadata.setdefault("utility_controller_overrides", []).append(
                            {
                                "round": round_name,
                                "role": role,
                                "llm_kind": llm_action.kind,
                                "llm_targets": list(llm_action.target_ids),
                                "llm_predicted_gain": llm_gain,
                                "deterministic_kind": deterministic_ranked_action.kind,
                                "deterministic_targets": list(deterministic_ranked_action.target_ids),
                                "deterministic_predicted_gain": deterministic_gain,
                            }
                        )
                        emit_progress(
                            graph,
                            progress_callback,
                            stage="action_rerank",
                            message=(
                                f"{round_name} {_progress_role_name(role)}: utility controller overrode the LLM proposal "
                                f"({llm_action.kind}) with {deterministic_ranked_action.kind}."
                            ),
                            details={
                                "round": round_name,
                                "role": role,
                                "llm_kind": llm_action.kind,
                                "llm_predicted_gain": llm_gain,
                                "deterministic_kind": deterministic_ranked_action.kind,
                                "deterministic_predicted_gain": deterministic_gain,
                            },
                        )
                    else:
                        action = llm_action
                        action_source = "llm"
                except Exception as exc:
                    graph.metadata.setdefault("action_errors", []).append(
                        {"round": round_name, "role": role, "error": str(exc)}
                    )
                    emit_progress(
                        graph,
                        progress_callback,
                        stage="action_fallback",
                        message=f"{round_name} {_progress_role_name(role)}: invalid LLM action, using deterministic fallback. Error: {exc}",
                        details={"round": round_name, "role": role, "error": str(exc)},
                    )
                    if deterministic_ranked_action is not None:
                        action = deterministic_ranked_action
                    else:
                        action = choose_round_action(graph, round_name, role)
                    action_source = "deterministic_fallback"
            try:
                action.source = action_source
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
                        message=f"{round_name} {_progress_role_name(role)}: LLM action could not be applied, using deterministic fallback. Error: {exc}",
                        details={"round": round_name, "role": role, "error": str(exc)},
                    )
                    if deterministic_ranked_action is not None:
                        action = deterministic_ranked_action
                    else:
                        action = choose_round_action(graph, round_name, role)
                    action_source = "deterministic_fallback"
                    action.source = action_source
                    apply_action(graph, action)
                else:
                    raise RuntimeError(
                        f"Failed to apply {action_source} action for {round_name}/{role}: {exc}"
                    ) from exc
            emit_progress(
                graph,
                progress_callback,
                stage="action_applied",
                message=(
                    f"{round_name} {_progress_role_name(role)}: applied {action.kind} "
                    f"via "
                    f"{'LLM proposal' if action_source == 'llm' else 'utility controller override' if action_source == 'utility_controller_override' else 'deterministic fallback' if action_source == 'deterministic_fallback' else 'utility-ranked deterministic policy'}."
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
                f"utility={snapshot.utility}, coherence={snapshot.utility_breakdown.coherence}, "
                f"mature={snapshot.is_mature}."
            ),
            details={
                "round": round_name,
                "support_coverage": snapshot.support_coverage,
                "unresolved_contradiction_ratio": snapshot.unresolved_contradiction_ratio,
                "utility": snapshot.utility,
                "utility_breakdown": asdict(snapshot.utility_breakdown),
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
