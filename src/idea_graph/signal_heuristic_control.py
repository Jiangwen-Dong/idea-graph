from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .claim_chain import select_claim_chain
from .collaboration_protocol import round_index_from_name
from .controller_protocol import (
    SIGNAL_NAMES,
    allowed_actions_for_role,
    is_active_edit_action,
    primary_signal_for_action,
)
from .models import IdeaGraph


_GROUNDING_TARGET = 0.72
_COMPLETENESS_TARGET = 0.72
_CONTRADICTION_LOAD_TARGET = 0.18
_DEPENDENCY_CLOSURE_TARGET = 0.60
_DIAGNOSTIC_EXPOSURE_TARGET = 0.30


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _active_nodes(graph: IdeaGraph):
    return [node for node in graph.nodes.values() if getattr(node, "status", "active") == "active"]


def _active_edges(graph: IdeaGraph):
    active_ids = {node.id for node in _active_nodes(graph)}
    return [
        edge
        for edge in graph.edges
        if str(edge.source_id) in active_ids and str(edge.target_id) in active_ids
    ]


def _core_nodes(graph: IdeaGraph):
    return [
        node
        for node in _active_nodes(graph)
        if str(node.type) in {"Hypothesis", "Method", "EvalPlan", "NoveltyClaim"}
    ]


def _grounding_focus_node_ids(
    graph: IdeaGraph,
    *,
    core_node_ids: set[str] | None = None,
) -> set[str]:
    core_ids = set(core_node_ids or {str(node.id) for node in _core_nodes(graph)})
    if not core_ids:
        return set()

    _, claim_chain_core_ids = _claim_chain_slot_coverage(graph)
    backbone_core_ids = {node_id for node_id in claim_chain_core_ids if node_id in core_ids}
    if backbone_core_ids:
        return backbone_core_ids
    return core_ids


def _claim_chain_slot_coverage(graph: IdeaGraph) -> tuple[float, list[str]]:
    claim_chain = select_claim_chain(graph)
    if isinstance(claim_chain, Mapping):
        coverage = claim_chain.get("coverage", {})
        if isinstance(coverage, Mapping):
            required_slots = coverage.get("required_slots", [])
            slot_count = int(coverage.get("slot_count", 0) or 0)
            total_slots = len(required_slots) if isinstance(required_slots, Sequence) else 0
            if total_slots > 0:
                core_node_ids = []
                subgraph = claim_chain.get("subgraph", {})
                if isinstance(subgraph, Mapping):
                    raw_core_ids = subgraph.get("core_node_ids", [])
                    if isinstance(raw_core_ids, Sequence) and not isinstance(raw_core_ids, (str, bytes)):
                        core_node_ids = [str(node_id) for node_id in raw_core_ids if str(node_id).strip()]
                return _clamp(slot_count / total_slots), core_node_ids

    essential_types = {"Problem", "Hypothesis", "Method", "EvalPlan", "NoveltyClaim"}
    present_types = {str(node.type) for node in _active_nodes(graph) if str(node.type) in essential_types}
    return _clamp(len(present_types) / len(essential_types)), []


def compute_graph_signals(
    graph: IdeaGraph,
    *,
    round_index: int = 1,
) -> dict[str, float | int | bool]:
    active_nodes = _active_nodes(graph)
    active_edges = _active_edges(graph)
    core_nodes = _core_nodes(graph)
    core_node_ids = {str(node.id) for node in core_nodes}
    grounding_focus_ids = _grounding_focus_node_ids(graph, core_node_ids=core_node_ids)
    grounding_focus_count = len(grounding_focus_ids)

    support_targets = set()
    dependency_touched = set()
    for edge in active_edges:
        relation = str(edge.relation)
        if relation == "supports" and str(edge.target_id) in grounding_focus_ids:
            support_targets.add(str(edge.target_id))
        if relation == "depends_on":
            dependency_touched.add(str(edge.source_id))
            dependency_touched.add(str(edge.target_id))

    support_channel = (
        len(support_targets) / grounding_focus_count
        if grounding_focus_count
        else 1.0
    )
    evidence_channel = (
        sum(1 for node in core_nodes if str(node.id) in grounding_focus_ids and list(node.evidence)) / grounding_focus_count
        if grounding_focus_count
        else 1.0
    )
    grounding = _clamp((0.50 * support_channel) + (0.50 * evidence_channel))

    contradiction_edges = [edge for edge in active_edges if str(edge.relation) == "contradicts"]
    unresolved_contradictions = [
        edge for edge in contradiction_edges if not bool(getattr(edge, "resolved", False))
    ]
    total_contradictions = len(contradiction_edges)
    edge_load = (
        len(unresolved_contradictions) / total_contradictions
        if total_contradictions
        else 0.0
    )

    risk_nodes = [node for node in active_nodes if str(node.type) in {"Risk", "Assumption"}]
    repaired_target_ids = {
        str(edge.target_id)
        for edge in active_edges
        if str(edge.relation) == "repairs"
    }
    contradiction_target_ids = {str(edge.target_id) for edge in contradiction_edges}
    open_repair_target_ids = {
        target_id
        for target_id in contradiction_target_ids
        if target_id not in repaired_target_ids
        and any(
            str(edge.target_id) == target_id and not bool(getattr(edge, "resolved", False))
            for edge in contradiction_edges
        )
    }
    open_repair_target_ratio = (
        len(open_repair_target_ids) / len(contradiction_target_ids)
        if contradiction_target_ids
        else 0.0
    )
    contradiction_load = _clamp((0.65 * edge_load) + (0.35 * open_repair_target_ratio))

    slot_coverage, claim_chain_core_ids = _claim_chain_slot_coverage(graph)
    backbone_ids = set(claim_chain_core_ids)
    if not backbone_ids:
        backbone_ids = {
            str(node.id)
            for node in active_nodes
            if str(node.type) in {"Problem", "Hypothesis", "Method", "EvalPlan", "NoveltyClaim"}
        }
    dependency_candidates = {
        str(node.id)
        for node in active_nodes
        if str(node.type) in {"Hypothesis", "Method", "EvalPlan"}
    }
    dependency_closure = (
        len(dependency_touched & dependency_candidates) / len(dependency_candidates)
        if dependency_candidates
        else 1.0
    )
    dependency_need = _clamp(
        (_DEPENDENCY_CLOSURE_TARGET - dependency_closure) / _DEPENDENCY_CLOSURE_TARGET
    )
    support_touched = {
        str(edge.source_id)
        for edge in active_edges
        if str(edge.relation) in {"supports", "depends_on"}
    } | {
        str(edge.target_id)
        for edge in active_edges
        if str(edge.relation) in {"supports", "depends_on"}
    }
    support_connectivity = (
        len(backbone_ids & support_touched) / len(backbone_ids)
        if backbone_ids
        else 1.0
    )
    completeness = _clamp(
        (0.25 * slot_coverage)
        + (0.45 * dependency_closure)
        + (0.30 * support_connectivity)
    )

    contradiction_touched_core_ids = {
        node_id
        for node_id in (
            {str(edge.source_id) for edge in contradiction_edges}
            | {str(edge.target_id) for edge in contradiction_edges}
        )
        if node_id in core_node_ids
    }
    contradiction_exposure = (
        len(contradiction_touched_core_ids) / len(core_nodes)
        if core_nodes
        else 0.0
    )
    structural_readiness = _clamp((0.25 * grounding) + (0.75 * completeness))
    underdiagnosed_gap = _clamp(
        (_DIAGNOSTIC_EXPOSURE_TARGET - contradiction_exposure) / _DIAGNOSTIC_EXPOSURE_TARGET
    )
    risk_anchor_ratio = 1.0 if risk_nodes else 0.0
    diagnostic_need = _clamp(
        underdiagnosed_gap
        * (0.35 + (0.65 * structural_readiness))
        * (0.35 + (0.65 * risk_anchor_ratio))
        * (1.0 - (0.60 * contradiction_load))
    )
    resolution_need = contradiction_load

    maturity = _clamp(
        (0.40 * grounding)
        + (0.35 * completeness)
        + (0.25 * (1.0 - contradiction_load))
    )

    grounding_need = _clamp((_GROUNDING_TARGET - grounding) / _GROUNDING_TARGET)
    completeness_need = _clamp((_COMPLETENESS_TARGET - completeness) / _COMPLETENESS_TARGET)
    residual_need = _clamp(
        (
            grounding_need
            + completeness_need
            + resolution_need
            + (0.75 * diagnostic_need)
        )
        / 3.75
    )

    return {
        "grounding": round(grounding, 6),
        "contradiction_load": round(contradiction_load, 6),
        "completeness": round(completeness, 6),
        "maturity": round(maturity, 6),
        "support_channel": round(support_channel, 6),
        "evidence_channel": round(evidence_channel, 6),
        "grounding_focus_count": grounding_focus_count,
        "slot_coverage": round(slot_coverage, 6),
        "dependency_closure": round(dependency_closure, 6),
        "dependency_need": round(dependency_need, 6),
        "support_connectivity": round(support_connectivity, 6),
        "diagnostic_need": round(diagnostic_need, 6),
        "contradiction_exposure": round(contradiction_exposure, 6),
        "resolution_need": round(resolution_need, 6),
        "residual_need": round(residual_need, 6),
        "round_index": int(round_index),
        "open_risk_count": len(risk_nodes),
        "total_risk_count": len(risk_nodes),
        "open_repair_target_count": len(open_repair_target_ids),
        "total_repair_target_count": len(contradiction_target_ids),
        "open_repair_ratio": round(open_repair_target_ratio, 6),
        "unresolved_contradiction_count": len(unresolved_contradictions),
        "total_contradiction_count": total_contradictions,
        "is_mature": bool(
            maturity >= 0.64
            and contradiction_load <= 0.25
            and diagnostic_need <= 0.20
            and not open_repair_target_ids
        ),
    }


def compute_signal_deficits(signals: Mapping[str, Any]) -> dict[str, float]:
    grounding = float(signals.get("grounding", 0.0) or 0.0)
    completeness = float(signals.get("completeness", 0.0) or 0.0)
    contradiction_load = float(signals.get("contradiction_load", 0.0) or 0.0)
    return {
        "grounding": round(_clamp((_GROUNDING_TARGET - grounding) / _GROUNDING_TARGET), 6),
        "completeness": round(_clamp((_COMPLETENESS_TARGET - completeness) / _COMPLETENESS_TARGET), 6),
        "contradiction_load": round(
            _clamp((contradiction_load - _CONTRADICTION_LOAD_TARGET) / (1.0 - _CONTRADICTION_LOAD_TARGET)),
            6,
        ),
        "dependency_need": round(_clamp(float(signals.get("dependency_need", 0.0) or 0.0)), 6),
        "diagnostic_need": round(_clamp(float(signals.get("diagnostic_need", 0.0) or 0.0)), 6),
        "repair_backlog": round(_clamp(float(signals.get("open_repair_ratio", 0.0) or 0.0)), 6),
        "maturity": round(1.0 - _clamp(float(signals.get("maturity", 0.0) or 0.0)), 6),
    }


def graph_signal_payload(
    graph: IdeaGraph,
    *,
    round_name: str | None = None,
) -> tuple[dict[str, object], dict[str, float]]:
    round_index = round_index_from_name(round_name or "Round1")
    signals = compute_graph_signals(graph, round_index=round_index)
    return dict(signals), compute_signal_deficits(signals)


def _phase_key(round_index: int) -> str:
    if round_index <= 1:
        return "structure"
    return "repair"


def _candidate_id(spec: Mapping[str, Any], index: int) -> str:
    return str(spec.get("candidate_id", f"heuristic-candidate:{index:03d}")).strip() or f"heuristic-candidate:{index:03d}"


def _candidate_target_features(
    graph: IdeaGraph,
    spec: Mapping[str, Any],
) -> dict[str, float]:
    target_ids = [
        str(target_id).strip()
        for target_id in spec.get("target_ids", ())
        if str(target_id).strip()
    ]
    if not target_ids:
        return {
            "unresolved_target_mass": 0.0,
            "contradiction_related_hit": 0.0,
            "grounding_gap_hit": 0.0,
            "dependency_gap_hit": 0.0,
        }

    active_edges = _active_edges(graph)
    core_nodes = _core_nodes(graph)
    core_node_ids = {str(node.id) for node in core_nodes}
    grounding_focus_ids = _grounding_focus_node_ids(graph, core_node_ids=core_node_ids)

    support_targets = {
        str(edge.target_id)
        for edge in active_edges
        if str(edge.relation) == "supports" and str(edge.target_id) in grounding_focus_ids
    }
    evidence_targets = {
        str(node.id)
        for node in core_nodes
        if str(node.id) in grounding_focus_ids
        if list(node.evidence)
    }
    dependency_touched = {
        node_id
        for edge in active_edges
        if str(edge.relation) == "depends_on"
        for node_id in (str(edge.source_id), str(edge.target_id))
    }
    dependency_candidates = {
        str(node.id)
        for node in _active_nodes(graph)
        if str(node.type) in {"Hypothesis", "Method", "EvalPlan"}
    }

    unresolved_edges = [
        edge
        for edge in active_edges
        if str(edge.relation) == "contradicts" and not bool(getattr(edge, "resolved", False))
    ]
    unresolved_target_counter = Counter(str(edge.target_id) for edge in unresolved_edges)
    contradiction_related_ids = {
        node_id
        for edge in unresolved_edges
        for node_id in (str(edge.source_id), str(edge.target_id))
    }
    max_unresolved_target_mass = max(unresolved_target_counter.values(), default=0)

    unresolved_target_mass = (
        max(
            (unresolved_target_counter.get(target_id, 0) / max_unresolved_target_mass)
            for target_id in target_ids
        )
        if max_unresolved_target_mass > 0
        else 0.0
    )
    contradiction_related_hit = 1.0 if any(target_id in contradiction_related_ids for target_id in target_ids) else 0.0

    grounding_gap_targets = {
        node_id
        for node_id in grounding_focus_ids
        if node_id not in support_targets or node_id not in evidence_targets
    }
    grounding_gap_hit = 1.0 if any(target_id in grounding_gap_targets for target_id in target_ids) else 0.0

    dependency_gap_targets = {
        node_id
        for node_id in dependency_candidates
        if node_id not in dependency_touched
    }
    dependency_gap_hit = 1.0 if any(target_id in dependency_gap_targets for target_id in target_ids) else 0.0

    return {
        "unresolved_target_mass": round(unresolved_target_mass, 6),
        "contradiction_related_hit": round(contradiction_related_hit, 6),
        "grounding_gap_hit": round(grounding_gap_hit, 6),
        "dependency_gap_hit": round(dependency_gap_hit, 6),
    }


def _heuristic_score(
    kind: str,
    *,
    role: str,
    round_index: int,
    signals: Mapping[str, Any],
    deficits: Mapping[str, float],
    candidate_features: Mapping[str, float] | None = None,
) -> tuple[float, dict[str, float | str]]:
    phase = _phase_key(round_index)
    grounding_deficit = float(deficits.get("grounding", 0.0))
    completeness_deficit = float(deficits.get("completeness", 0.0))
    contradiction_pressure = float(deficits.get("contradiction_load", 0.0))
    maturity = float(signals.get("maturity", 0.0) or 0.0)
    diagnostic_need = float(signals.get("diagnostic_need", 0.0) or 0.0)
    dependency_need = float(signals.get("dependency_need", 0.0) or 0.0)
    resolution_need = float(signals.get("resolution_need", 0.0) or 0.0)
    residual_need = float(signals.get("residual_need", 0.0) or 0.0)
    repair_backlog = float(signals.get("open_repair_ratio", 0.0) or 0.0)
    features = dict(candidate_features or {})
    unresolved_target_mass = float(features.get("unresolved_target_mass", 0.0) or 0.0)
    contradiction_related_hit = float(features.get("contradiction_related_hit", 0.0) or 0.0)
    grounding_gap_hit = float(features.get("grounding_gap_hit", 0.0) or 0.0)
    dependency_gap_hit = float(features.get("dependency_gap_hit", 0.0) or 0.0)
    grounding_consolidation_pressure = 0.0
    if phase == "repair" and resolution_need <= 0.05 and repair_backlog <= 0.05:
        grounding_consolidation_pressure = max(0.0, grounding_deficit - completeness_deficit)

    role_bonus = 0.0
    if kind == "add_support_edge" and role in {"ImpactReframer", "MechanismProposer", "EvaluationDesigner"}:
        role_bonus = 0.08
    elif kind == "attach_evidence" and role in {"EvaluationDesigner", "NoveltyExaminer", "FeasibilityCritic"}:
        role_bonus = 0.06
    elif kind == "add_dependency_edge" and role in {"EvaluationDesigner", "MechanismProposer"}:
        role_bonus = 0.16
    elif kind == "add_contradiction_edge" and role in {"FeasibilityCritic", "NoveltyExaminer"}:
        role_bonus = 0.16
    elif kind == "propose_repair" and role == "FeasibilityCritic":
        role_bonus = 0.14

    phase_bonus = 0.0
    if phase == "structure" and kind == "add_support_edge":
        phase_bonus = 0.04
    elif phase == "structure" and kind == "add_dependency_edge":
        phase_bonus = 0.14
    elif phase == "structure" and kind == "add_contradiction_edge":
        phase_bonus = 0.12
    elif phase == "repair" and kind == "propose_repair":
        phase_bonus = 0.08

    if kind == "add_support_edge":
        score = (
            (0.64 * grounding_deficit)
            + (0.10 * completeness_deficit)
            + (0.08 * dependency_need)
            - (0.08 * contradiction_pressure)
            + (0.12 * grounding_gap_hit)
            + ((0.22 if phase == "repair" else 0.08) * contradiction_related_hit)
            - ((0.32 if phase == "repair" else 0.0) * repair_backlog)
            + (0.12 * grounding_consolidation_pressure * (0.5 + (0.5 * grounding_gap_hit)))
        )
    elif kind == "attach_evidence":
        score = (
            (0.78 * grounding_deficit)
            + (0.04 * completeness_deficit)
            - (0.12 * dependency_need)
            + (0.18 * grounding_gap_hit)
            + ((0.18 if phase == "repair" else 0.06) * contradiction_related_hit)
            - ((0.24 if phase == "repair" else 0.0) * repair_backlog)
            + (0.25 * grounding_consolidation_pressure * (0.6 + (0.4 * grounding_gap_hit)))
        )
    elif kind == "add_dependency_edge":
        score = (
            (0.85 * completeness_deficit)
            + (0.90 * dependency_need)
            + (0.10 * diagnostic_need)
            - (0.05 * contradiction_pressure)
            + (0.18 * dependency_gap_hit)
            - ((0.18 if phase == "repair" else 0.0) * repair_backlog)
            - (0.40 * grounding_consolidation_pressure * (0.5 + (0.5 * dependency_gap_hit)))
        )
    elif kind == "add_contradiction_edge":
        score = (
            (0.95 * diagnostic_need)
            - (0.30 * contradiction_pressure)
            - (0.08 * maturity)
            + (0.10 * grounding_gap_hit)
            - (1.20 if round_index >= 2 else 0.0)
        )
    elif kind == "propose_repair":
        score = (
            (0.95 * resolution_need)
            + (0.12 * completeness_deficit)
            - (0.06 * maturity)
            + (0.55 * unresolved_target_mass)
            + (0.20 * contradiction_related_hit)
            + (0.70 * repair_backlog)
            - (0.28 if resolution_need > 0.0 and unresolved_target_mass <= 0.0 else 0.0)
        )
    elif kind == "skip":
        score = (
            (0.85 * maturity)
            - (0.90 * residual_need)
            - (0.35 * diagnostic_need)
            + (0.04 * max(0, round_index - 2))
            - (0.25 * repair_backlog)
        )
    else:
        score = -1.0

    total = score + role_bonus + phase_bonus
    return total, {
        "phase": phase,
        "grounding_deficit": round(grounding_deficit, 6),
        "completeness_deficit": round(completeness_deficit, 6),
        "contradiction_pressure": round(contradiction_pressure, 6),
        "diagnostic_need": round(diagnostic_need, 6),
        "dependency_need": round(dependency_need, 6),
        "resolution_need": round(resolution_need, 6),
        "residual_need": round(residual_need, 6),
        "repair_backlog": round(repair_backlog, 6),
        "unresolved_target_mass": round(unresolved_target_mass, 6),
        "contradiction_related_hit": round(contradiction_related_hit, 6),
        "grounding_gap_hit": round(grounding_gap_hit, 6),
        "dependency_gap_hit": round(dependency_gap_hit, 6),
        "grounding_consolidation_pressure": round(grounding_consolidation_pressure, 6),
        "role_bonus": round(role_bonus, 6),
        "phase_bonus": round(phase_bonus, 6),
    }


class SignalHeuristicController:
    def choose(
        self,
        *,
        round_name: str,
        role: str,
        candidate_specs: Sequence[Mapping[str, Any]] | None = None,
        candidates: Sequence[Mapping[str, Any]] | None = None,
        graph: IdeaGraph,
    ) -> dict[str, object]:
        raw_candidates = list(candidate_specs if candidate_specs is not None else candidates or ())
        legal_actions = set(allowed_actions_for_role(role))
        round_index = round_index_from_name(round_name)
        signals = compute_graph_signals(graph, round_index=round_index)
        deficits = compute_signal_deficits(signals)

        scored_candidates: list[dict[str, object]] = []
        for index, spec in enumerate(raw_candidates):
            kind = str(spec.get("kind", "")).strip()
            if not kind:
                continue
            if kind != "skip" and (not is_active_edit_action(kind) or kind not in legal_actions):
                continue
            if kind == "skip" and "skip" not in legal_actions:
                continue
            candidate_features = _candidate_target_features(graph, spec)
            score, components = _heuristic_score(
                kind,
                role=role,
                round_index=round_index,
                signals=signals,
                deficits=deficits,
                candidate_features=candidate_features,
            )
            scored_candidates.append(
                {
                    **dict(spec),
                    "candidate_id": _candidate_id(spec, index),
                    "critic_score": round(score, 6),
                    "critic_base_score": round(score, 6),
                    "critic_score_calibrated": round(score, 6),
                    "critic_calibration_bias": 0.0,
                    "critic_action_family": primary_signal_for_action(kind) or "heuristic",
                    "critic_aligned_signal": primary_signal_for_action(kind),
                    "critic_calibration_enabled": False,
                    "critic_calibration_feedback": dict(signals),
                    "critic_calibration_deficits": dict(deficits),
                    "heuristic_components": components,
                }
            )

        if not scored_candidates:
            raise ValueError(f"No legal heuristic candidates were available for role '{role}'.")

        selected = max(
            scored_candidates,
            key=lambda row: (float(row.get("critic_score", float("-inf"))), str(row.get("candidate_id", ""))),
        )
        return {
            "candidate_id": str(selected["candidate_id"]),
            "selected_source": "signal_heuristic_control",
            "scored_candidates": tuple(scored_candidates),
            "signals": dict(signals),
            "deficits": dict(deficits),
        }

    def score_commit_graph(self, graph: IdeaGraph, *, snapshot=None) -> float:
        del snapshot
        signals = compute_graph_signals(graph, round_index=1)
        maturity = float(signals["maturity"])
        contradiction_load = float(signals["contradiction_load"])
        diagnostic_need = float(signals.get("diagnostic_need", 0.0) or 0.0)
        dependency_need = float(signals.get("dependency_need", 0.0) or 0.0)
        repair_backlog = float(signals.get("open_repair_ratio", 0.0) or 0.0)
        residual_need = float(signals["residual_need"])
        score = (
            (0.72 * maturity)
            + (0.14 * (1.0 - contradiction_load))
            - (0.18 * diagnostic_need)
            - (0.12 * dependency_need)
            - (0.14 * repair_backlog)
            - (0.14 * residual_need)
        )
        return _clamp(score)


__all__ = [
    "SIGNAL_NAMES",
    "SignalHeuristicController",
    "compute_graph_signals",
    "compute_signal_deficits",
    "graph_signal_payload",
]
