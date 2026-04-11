from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Protocol

from .claim_chain import select_claim_chain
from .benchmark_mode import build_generation_safe_metadata
from .collaboration_protocol import (
    ACTION_PROMPT_HINTS,
    ACTION_REQUIRED_PAYLOAD_FIELDS,
    ACTION_TARGET_COUNTS,
    resolve_round_phase,
)
from .literature_grounding import LiteratureGrounding, build_literature_grounding
from .llm import ChatCompletionResult, OpenAICompatibleChatClient
from .models import FinalProposal, IdeaGraph
from .schema import EDGE_TYPES, NODE_TYPES, ROLE_NAMES
from .settings import OpenAICompatibleSettings

ROLE_DISPLAY_NAMES = {
    "ImpactReframer": "TaskFramer",
    "NoveltyExaminer": "LiteratureGrounder",
    "MechanismProposer": "MethodArchitect",
    "EvaluationDesigner": "ExperimentDesigner",
    "FeasibilityCritic": "SkepticRepairer",
}

ROLE_GUIDANCE = {
    "MechanismProposer": (
        "MethodArchitect: own one core mechanism and the design choices that make it work. "
        "Prefer hypotheses and methods that define a concrete research mechanism."
    ),
    "FeasibilityCritic": (
        "SkepticRepairer: own mismatch detection, risks, assumptions, and concrete repairs that keep the idea realistic. "
        "Prefer risks, assumptions, repair candidates, and evaluation pressure tests that surface practical failure modes."
    ),
    "NoveltyExaminer": (
        "LiteratureGrounder: own the visible reference-based gap, overlap boundaries, novelty boundaries, and novelty positioning relative to the provided literature only. "
        "Prefer literature-grounded gap statements, overlap checks, and evidence requests grounded in nearby prior work."
    ),
    "EvaluationDesigner": (
        "ExperimentDesigner: own the evaluation logic that directly tests the mechanism. "
        "Prefer evaluations, datasets or tasks, metrics, baselines, and ablations that make claims testable."
    ),
    "ImpactReframer": (
        "TaskFramer: own the exact benchmark task, the concrete failure mode or bottleneck, and why the problem matters. "
        "Prefer problem framing, significance, and alternative research directions with interpretable tradeoffs."
    ),
}

ROLE_PREFERRED_ANCHOR_TYPES = {
    "MechanismProposer": ("Hypothesis", "Method", "Assumption"),
    "FeasibilityCritic": ("Risk", "Assumption", "EvalPlan"),
    "NoveltyExaminer": ("NoveltyClaim", "EvidenceNeed", "Problem"),
    "EvaluationDesigner": ("EvalPlan", "Method", "Risk"),
    "ImpactReframer": ("Problem", "Hypothesis", "NoveltyClaim"),
}

SYMBOLIC_REFERENCE_PATTERN = re.compile(
    r"(?:paper_grounding\.)?reference_paper_snippets\[(\d+)\](?:\.(\w+))?",
    re.IGNORECASE,
)


def _role_prompt_name(role: str) -> str:
    return ROLE_DISPLAY_NAMES.get(role, role)


@dataclass(frozen=True)
class SeedSupportDraft:
    type: str
    text: str
    confidence: float
    relation_to_anchor: str = "supports"


@dataclass(frozen=True)
class SeedDraft:
    anchor_type: str
    anchor_text: str
    anchor_confidence: float
    support_nodes: list[SeedSupportDraft]
    rationale: str = ""
    trace: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionDecision:
    kind: str
    target_ids: list[str]
    payload: dict[str, object]
    rationale: str = ""
    trace: dict[str, object] = field(default_factory=dict)


class CollaborationBackend(Protocol):
    name: str

    def generate_seed(self, graph: IdeaGraph, role: str) -> SeedDraft: ...

    def choose_action(self, graph: IdeaGraph, round_name: str, role: str) -> ActionDecision: ...

    def synthesize_final_proposal(self, graph: IdeaGraph, subgraph: dict[str, object]) -> FinalProposal: ...


def append_agent_trace(
    graph: IdeaGraph,
    *,
    stage: str,
    role: str,
    trace: dict[str, object],
) -> None:
    if not trace:
        return
    traces = graph.metadata.setdefault("agent_traces", [])
    if isinstance(traces, list):
        traces.append({"stage": stage, "role": role, **trace})


def focused_view_for_prompt(graph: IdeaGraph, role: str) -> dict[str, object]:
    own_branch = next((branch for branch in graph.branches.values() if branch.role == role), None)
    all_edges = [
        {
            "id": edge.id,
            "source_id": edge.source_id,
            "relation": edge.relation,
            "target_id": edge.target_id,
            "resolved": edge.resolved,
            "role": edge.role,
        }
        for edge in graph.edges
    ]
    active_nodes = [
        {
            "id": node.id,
            "type": node.type,
            "text": _truncate_text(node.text, max_chars=220),
            "role": node.role,
            "branch_id": node.branch_id,
            "confidence": node.confidence,
            "evidence": [_truncate_text(item, max_chars=160) for item in node.evidence[:2]],
        }
        for node in graph.active_nodes()
    ]
    active_edges = [
        edge
        for edge in all_edges[-24:]
    ]
    branches = [
        {
            "id": branch.id,
            "role": branch.role,
            "node_ids": list(branch.node_ids),
            "frozen": branch.frozen,
            "rejected": branch.rejected,
        }
        for branch in graph.branches.values()
    ]

    unsupported_novelty = [
        {
            "id": node["id"],
            "text": node["text"],
            "role": node["role"],
        }
        for node in active_nodes
        if node["type"] == "NoveltyClaim"
        and not any(edge["relation"] == "supports" and edge["target_id"] == node["id"] for edge in all_edges)
    ][:3]
    all_unresolved_contradictions = [
        {
            "edge_id": edge["id"],
            "source_id": edge["source_id"],
            "target_id": edge["target_id"],
            "source_type": next((node["type"] for node in active_nodes if node["id"] == edge["source_id"]), ""),
            "target_type": next((node["type"] for node in active_nodes if node["id"] == edge["target_id"]), ""),
            "source_role": edge["role"],
        }
        for edge in all_edges
        if edge["relation"] == "contradicts" and not edge["resolved"]
    ]
    unresolved_contradictions = all_unresolved_contradictions[:5]
    tracked_nodes = [
        node
        for node in active_nodes
        if node["type"] in {"Hypothesis", "Method", "NoveltyClaim", "EvalPlan"}
    ]
    supported_node_ids = {
        edge["target_id"]
        for edge in all_edges
        if edge["relation"] in {"supports", "repairs"}
    }
    tracked_nodes_needing_support = [
        {
            "id": node["id"],
            "type": node["type"],
            "role": node["role"],
            "text": node["text"],
        }
        for node in tracked_nodes
        if node["id"] not in supported_node_ids and not node["evidence"]
    ][:6]
    tracked_nodes_needing_evidence = [
        {
            "id": node["id"],
            "type": node["type"],
            "role": node["role"],
            "text": node["text"],
        }
        for node in tracked_nodes
        if not node["evidence"]
    ][:5]
    blocked_request_node_ids = [
        edge["target_id"]
        for edge in all_edges
        if edge["relation"] == "requires_evidence" and edge["source_id"] == edge["target_id"]
    ]
    request_evidence_candidates = [
        node
        for node in tracked_nodes_needing_evidence
        if node["id"] not in blocked_request_node_ids
    ][:4]
    attach_evidence_candidates = tracked_nodes_needing_evidence[:4]
    tracked_count = len(tracked_nodes)
    supported_count = sum(
        1
        for node in tracked_nodes
        if node["id"] in supported_node_ids or node["evidence"]
    )
    contradictions = [edge for edge in all_edges if edge["relation"] == "contradicts"]
    unresolved_ratio = 0.0 if not contradictions else round(len(all_unresolved_contradictions) / len(contradictions), 2)
    support_coverage = 0.0 if not tracked_count else round(supported_count / tracked_count, 2)
    active_types = {node["type"] for node in active_nodes}

    return {
        "role": role,
        "role_guidance": ROLE_GUIDANCE.get(role, ""),
        "own_branch_id": own_branch.id if own_branch is not None else "",
        "topic": graph.topic,
        "context_packet": _compact_generation_context(graph),
        "evidence_candidates": _paper_evidence_candidates(graph.metadata, limit=4),
        "maturity_snapshot": {
            "support_coverage": support_coverage,
            "unresolved_contradiction_ratio": unresolved_ratio,
            "completeness": {"Problem", "Hypothesis", "Method", "EvalPlan"}.issubset(active_types),
        },
        "branches": branches[:4],
        "nodes": active_nodes[:18],
        "edges": active_edges,
        "unsupported_novelty_claims": unsupported_novelty,
        "unresolved_contradictions": unresolved_contradictions,
        "tracked_nodes_needing_support": tracked_nodes_needing_support[:5],
        "tracked_nodes_needing_evidence": tracked_nodes_needing_evidence,
        "request_evidence_candidates": request_evidence_candidates,
        "attach_evidence_candidates": attach_evidence_candidates,
        "blocked_request_node_ids": blocked_request_node_ids[:6],
        "recent_actions": _recent_actions_for_prompt(graph, limit=6),
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON must be an object.")
    return payload


def _clamp_confidence(value: Any, default: float = 0.65) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _coerce_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = _coerce_string(item)
        if text:
            normalized.append(text)
    return normalized


def _unique_strings(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def _first_sentence(text: Any, *, max_chars: int = 220) -> str:
    cleaned = _coerce_string(text)
    if not cleaned:
        return ""
    for separator in (". ", "! ", "? "):
        if separator in cleaned:
            cleaned = cleaned.split(separator, 1)[0].strip().rstrip(".!?") + "."
            break
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."
    return cleaned


def _truncate_text(text: Any, *, max_chars: int = 220) -> str:
    cleaned = _coerce_string(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _safe_paper_grounding(metadata: dict[str, Any], *, limit: int = 6) -> dict[str, object]:
    payload = metadata.get("paper_grounding", {})
    if not isinstance(payload, dict):
        return {"reference_paper_snippets": []}

    raw_snippets = payload.get("reference_paper_snippets", [])
    if not isinstance(raw_snippets, list):
        raw_snippets = []

    safe_snippets: list[dict[str, str]] = []
    for index, item in enumerate(raw_snippets[:limit], start=1):
        if not isinstance(item, dict):
            continue
        safe_snippets.append(
            {
                "paper_id": f"paper-{index:03d}",
                "resolved_title": _coerce_string(item.get("resolved_title") or item.get("raw_title")),
                "abstract": _first_sentence(item.get("abstract")),
                "introduction": _first_sentence(item.get("introduction")),
                "method": _first_sentence(item.get("method")),
                "evaluation": _first_sentence(item.get("evaluation")),
                "conclusion": _first_sentence(item.get("conclusion")),
            }
        )
    return {"reference_paper_snippets": safe_snippets}


def _prompt_safe_metadata(metadata: dict[str, Any]) -> dict[str, object]:
    source_metadata = (
        build_generation_safe_metadata(metadata)
        if _coerce_string(metadata.get("benchmark"))
        else dict(metadata)
    )
    safe: dict[str, object] = {}
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
    for key, value in source_metadata.items():
        if key in blocked_keys:
            continue
        if key == "paper_grounding":
            safe[key] = _safe_paper_grounding({"paper_grounding": value})
            continue
        safe[key] = value
    return safe


def _paper_evidence_candidates(metadata: dict[str, Any], *, limit: int = 6) -> list[dict[str, str]]:
    paper_grounding = _safe_paper_grounding(metadata, limit=limit)
    raw_snippets = paper_grounding.get("reference_paper_snippets", [])
    if not isinstance(raw_snippets, list):
        return []

    candidates: list[dict[str, str]] = []
    for item in raw_snippets[:limit]:
        if not isinstance(item, dict):
            continue
        paper_id = _coerce_string(item.get("paper_id"))
        title = _coerce_string(item.get("resolved_title"))
        for field_name in ("method", "abstract", "evaluation", "introduction", "conclusion"):
            evidence = _coerce_string(item.get(field_name))
            if not evidence:
                continue
            candidates.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "section": field_name,
                    "evidence": evidence,
                }
            )
            break
    return candidates[:limit]


def _benchmark_packet_evidence_candidates(metadata: dict[str, Any], *, limit: int = 6) -> list[dict[str, str]]:
    packet = metadata.get("benchmark_input_packet", {})
    if not isinstance(packet, dict):
        return []

    raw_references = packet.get("reference_packet", [])
    if not isinstance(raw_references, list):
        raw_references = []

    candidates: list[dict[str, str]] = []
    for index, item in enumerate(raw_references[:limit], start=1):
        if not isinstance(item, dict):
            continue
        title = _coerce_string(item.get("title"))
        evidence = _first_sentence(item.get("snippet"), max_chars=220)
        if not evidence:
            continue
        candidates.append(
            {
                "paper_id": f"packet-{index:03d}",
                "title": title,
                "section": "reference_packet",
                "evidence": evidence,
            }
        )

    keyword = _coerce_string(packet.get("keyword"))
    topic = _coerce_string(packet.get("topic"))
    if keyword:
        candidates.append(
            {
                "paper_id": "benchmark-keyword",
                "title": "Benchmark keyword",
                "section": "keyword",
                "evidence": f"The benchmark keyword is {keyword}.",
            }
        )
    elif topic:
        candidates.append(
            {
                "paper_id": "benchmark-topic",
                "title": "Benchmark topic",
                "section": "topic",
                "evidence": _first_sentence(topic, max_chars=220),
            }
        )
    return candidates[:limit]


def _grounding_evidence_candidates(graph: IdeaGraph, *, limit: int = 4) -> list[dict[str, str]]:
    safe_metadata = _prompt_safe_metadata(graph.metadata)
    grounding = build_literature_grounding(literature=graph.literature, metadata=safe_metadata)
    candidates: list[dict[str, str]] = []

    for index, item in enumerate(grounding.design_highlights[:limit], start=1):
        evidence = _first_sentence(item, max_chars=220)
        if evidence:
            candidates.append(
                {
                    "paper_id": f"grounding-{index:03d}",
                    "title": "Grounding highlight",
                    "section": "design_highlight",
                    "evidence": evidence,
                }
            )

    experiment_summary = _first_sentence(grounding.experiment_plan_summary, max_chars=220)
    if experiment_summary:
        candidates.append(
            {
                "paper_id": "grounding-experiment",
                "title": "Grounding experiment summary",
                "section": "experiment_plan_summary",
                "evidence": experiment_summary,
            }
        )
    return candidates[:limit]


def _compact_benchmark_packet(graph: IdeaGraph, *, max_references: int = 4, snippet_chars: int = 180) -> dict[str, object]:
    packet = graph.metadata.get("benchmark_input_packet", {})
    if not isinstance(packet, dict):
        packet = {}
    reference_packet = packet.get("reference_packet", [])
    if not isinstance(reference_packet, list):
        reference_packet = []

    compact_references: list[dict[str, str]] = []
    for item in reference_packet[:max_references]:
        if not isinstance(item, dict):
            continue
        title = _coerce_string(item.get("title"))
        snippet = _truncate_text(item.get("snippet"), max_chars=snippet_chars)
        if title or snippet:
            compact_references.append({"title": title, "snippet": snippet})

    return {
        "benchmark": _coerce_string(packet.get("benchmark")),
        "topic": _coerce_string(packet.get("topic") or graph.topic),
        "keyword": _coerce_string(packet.get("keyword") or graph.metadata.get("keyword")),
        "task_instruction": _truncate_text(packet.get("task_instruction"), max_chars=220),
        "reference_packet": compact_references,
        "constraints": _list_of_strings(packet.get("constraints"))[:4],
    }


def _compact_generation_context(graph: IdeaGraph) -> dict[str, object]:
    safe_metadata = _prompt_safe_metadata(graph.metadata)
    grounding = build_literature_grounding(literature=graph.literature, metadata=safe_metadata).as_dict()
    weak_context_scaffold = grounding.get("weak_context_scaffold", {})
    weak_context_mode = bool(weak_context_scaffold)
    return {
        "benchmark_packet": _compact_benchmark_packet(graph),
        "design_highlights": _list_of_strings(grounding.get("design_highlights"))[:3],
        "design_anchor_terms": _design_anchor_terms(_list_of_strings(grounding.get("design_highlights"))[:4]),
        "dataset_items": _list_of_strings(grounding.get("dataset_items"))[:4],
        "metric_items": _list_of_strings(grounding.get("metric_items"))[:6],
        "weak_context_mode": weak_context_mode,
        "weak_context_scaffold": weak_context_scaffold if isinstance(weak_context_scaffold, dict) else {},
    }


def _recent_actions_for_prompt(graph: IdeaGraph, *, limit: int = 8) -> list[dict[str, object]]:
    recent_actions: list[dict[str, object]] = []
    for action in graph.actions[-limit:]:
        recent_actions.append(
            {
                "round": action.round_name,
                "role": action.role,
                "kind": action.kind,
                "target_ids": list(action.target_ids),
            }
        )
    return recent_actions


def _repair_priority_targets(graph: IdeaGraph, *, limit: int = 6) -> list[dict[str, str]]:
    active_nodes = {
        node.id: node
        for node in graph.active_nodes()
    }
    priorities: list[dict[str, str]] = []
    for edge in graph.edges:
        if edge.relation != "contradicts" or edge.resolved:
            continue
        target = active_nodes.get(edge.target_id)
        source = active_nodes.get(edge.source_id)
        if target is None:
            continue
        priorities.append(
            {
                "target_id": target.id,
                "target_type": target.type,
                "target_role": target.role,
                "target_text": target.text,
                "contradicted_by": source.id if source is not None else edge.source_id,
                "contradiction_source_text": source.text if source is not None else "",
            }
        )
    return priorities[:limit]


def _graph_has_edge(
    graph: IdeaGraph,
    *,
    source_id: str,
    relation: str,
    target_id: str,
    evidence_id: str = "",
) -> bool:
    normalized_evidence_id = _coerce_string(evidence_id)
    for edge in graph.edges:
        if edge.source_id != source_id or edge.relation != relation or edge.target_id != target_id:
            continue
        if normalized_evidence_id and edge.evidence_id != normalized_evidence_id:
            continue
        return True
    return False


def _own_branch_id(graph: IdeaGraph, role: str) -> str:
    for branch in graph.branches.values():
        if branch.role == role:
            return branch.id
    return ""


def _looks_symbolic_reference(text: str) -> bool:
    cleaned = _coerce_string(text)
    if not cleaned:
        return False
    return bool(SYMBOLIC_REFERENCE_PATTERN.search(cleaned)) or any(
        marker in cleaned
        for marker in ("raw_record.", "metadata.", "target_paper_snippet.", "reference_paper_snippets[")
    )


def _resolve_symbolic_reference_text(text: str, metadata: dict[str, Any]) -> str:
    cleaned = _coerce_string(text)
    if not cleaned:
        return ""

    payload = metadata.get("paper_grounding", {})
    if not isinstance(payload, dict):
        return cleaned
    snippets = payload.get("reference_paper_snippets", [])
    if not isinstance(snippets, list):
        return cleaned

    resolved_parts: list[str] = []
    for segment in [part.strip() for part in cleaned.split(",") if part.strip()]:
        match = SYMBOLIC_REFERENCE_PATTERN.fullmatch(segment)
        if match is None:
            continue
        index = int(match.group(1))
        field_name = _coerce_string(match.group(2), "abstract") or "abstract"
        if not (0 <= index < len(snippets)):
            continue
        snippet = snippets[index]
        if not isinstance(snippet, dict):
            continue
        resolved_text = _first_sentence(
            snippet.get(field_name)
            or snippet.get("method")
            or snippet.get("abstract")
            or snippet.get("evaluation")
            or snippet.get("text_excerpt")
        )
        resolved_title = _coerce_string(snippet.get("resolved_title") or snippet.get("raw_title"))
        if resolved_text:
            if resolved_title:
                resolved_parts.append(f"{resolved_title}: {resolved_text}")
            else:
                resolved_parts.append(resolved_text)
    if resolved_parts:
        return "; ".join(_unique_strings(resolved_parts))
    return cleaned


def _normalize_paper_id(value: Any, metadata: dict[str, Any]) -> str:
    cleaned = _coerce_string(value)
    candidates = _paper_evidence_candidates(metadata)
    known_ids = {candidate["paper_id"] for candidate in candidates}
    if cleaned in known_ids:
        return cleaned

    match = SYMBOLIC_REFERENCE_PATTERN.fullmatch(cleaned)
    if match is not None:
        index = int(match.group(1))
        resolved = f"paper-{index + 1:03d}"
        if resolved in known_ids:
            return resolved

    if cleaned.isdigit():
        resolved = f"paper-{int(cleaned) + 1:03d}"
        if resolved in known_ids:
            return resolved

    lowered = cleaned.casefold()
    for candidate in candidates:
        if lowered == candidate["title"].casefold():
            return candidate["paper_id"]
    return cleaned


def _design_anchor_terms(design_highlights: list[str]) -> list[str]:
    anchors: list[str] = []
    for item in design_highlights:
        cleaned = _coerce_string(item)
        if not cleaned:
            continue
        if ":" in cleaned:
            label = cleaned.split(":", 1)[0].strip()
        else:
            label = " ".join(cleaned.split()[:6]).strip()
        if label:
            anchors.append(label)
    return _unique_strings(anchors)[:4]


def _design_support_terms(design_highlights: list[str]) -> list[str]:
    terms: list[str] = []
    for item in design_highlights:
        cleaned = _coerce_string(item)
        if not cleaned:
            continue
        label = cleaned
        detail = ""
        if ":" in cleaned:
            label, detail = cleaned.split(":", 1)
            detail = detail.strip().rstrip(".")
        label = label.strip()
        if label:
            terms.append(label)
        if detail:
            terms.append(detail)
            for piece in re.split(r",|;|\band\b|\bor\b", detail):
                normalized = " ".join(piece.split())
                if len(normalized.split()) >= 2:
                    terms.append(normalized)
    return _unique_strings(terms)[:12]


def _naturalize_design_highlight(design_highlight: Any) -> str:
    cleaned = _coerce_string(design_highlight).rstrip(".")
    if not cleaned:
        return ""
    if ":" not in cleaned:
        return _first_sentence(cleaned)
    label, detail = cleaned.split(":", 1)
    label = label.strip()
    detail = detail.strip().rstrip(".")
    if not label:
        return _first_sentence(detail)
    if not detail:
        return f"Use {label.casefold()} as a core component."
    return f"Use {label.casefold()} to {detail}."


def _best_attach_evidence_text(
    graph: IdeaGraph,
    *,
    target_id: str,
    preferred_text: Any,
) -> str:
    node = graph.nodes.get(target_id)
    existing = set(node.evidence) if node is not None else set()

    preferred = _resolve_symbolic_reference_text(_coerce_string(preferred_text), graph.metadata)
    if preferred and preferred not in existing and not _looks_symbolic_reference(preferred):
        return preferred

    for candidate in (
        _paper_evidence_candidates(graph.metadata)
        + _benchmark_packet_evidence_candidates(graph.metadata)
        + _grounding_evidence_candidates(graph)
    ):
        evidence = _coerce_string(candidate.get("evidence"))
        if not evidence:
            continue
        title = _coerce_string(candidate.get("title"))
        rendered = f"{title}: {evidence}" if title else evidence
        if rendered not in existing:
            return rendered
        if evidence not in existing:
            return evidence
    topic_anchor = _coerce_string(graph.metadata.get("keyword") or graph.topic)
    if topic_anchor:
        fallback = f"Topic anchor: {_first_sentence(topic_anchor, max_chars=220)}"
        if fallback not in existing:
            return fallback
    return ""


def _unresolved_contradiction_target_ids(graph: IdeaGraph) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for edge in graph.edges:
        if edge.relation != "contradicts" or edge.resolved:
            continue
        if edge.source_id not in graph.nodes or edge.target_id not in graph.nodes:
            continue
        source = graph.nodes[edge.source_id]
        target = graph.nodes[edge.target_id]
        if source.status != "active" or target.status != "active":
            continue
        if edge.target_id in seen:
            continue
        seen.add(edge.target_id)
        ordered.append(edge.target_id)
    return ordered


def _contradiction_related_node_ids(graph: IdeaGraph) -> set[str]:
    related: set[str] = set()
    for edge in graph.edges:
        if edge.relation != "contradicts" or edge.resolved:
            continue
        if edge.source_id not in graph.nodes or edge.target_id not in graph.nodes:
            continue
        source = graph.nodes[edge.source_id]
        target = graph.nodes[edge.target_id]
        if source.status != "active" or target.status != "active":
            continue
        related.add(edge.source_id)
        related.add(edge.target_id)
    return related


def _attachable_target_ids(
    graph: IdeaGraph,
    *,
    related_only: bool = False,
    limit: int = 8,
) -> list[str]:
    candidates: list[str] = []
    related_ids = _contradiction_related_node_ids(graph)
    preferred_types = ("EvalPlan", "Method", "Hypothesis", "NoveltyClaim", "Problem")
    for node_type in preferred_types:
        for node in graph.active_nodes():
            if node.type != node_type or node.evidence:
                continue
            if related_only and node.id not in related_ids:
                continue
            if not _best_attach_evidence_text(graph, target_id=node.id, preferred_text=""):
                continue
            candidates.append(node.id)
            if len(candidates) >= limit:
                return candidates
    return candidates


def _dynamic_allowed_actions(graph: IdeaGraph, round_name: str) -> tuple[str, ...]:
    phase = resolve_round_phase(round_name)
    allowed = list(phase.allowed_actions)
    unresolved_targets = _unresolved_contradiction_target_ids(graph)

    if phase.key == "repair":
        if unresolved_targets:
            allowed = [kind for kind in allowed if kind != "freeze_branch"]
        else:
            allowed = [kind for kind in allowed if kind != "propose_repair"]
        if "attach_evidence" in allowed and not _attachable_target_ids(
            graph,
            related_only=bool(unresolved_targets),
        ):
            allowed = [kind for kind in allowed if kind != "attach_evidence"]
        if not allowed:
            allowed = ["freeze_branch"]
        return tuple(allowed)

    if "attach_evidence" in allowed and not _attachable_target_ids(graph):
        allowed = [kind for kind in allowed if kind != "attach_evidence"]
    if not allowed:
        return tuple(phase.allowed_actions)
    return tuple(allowed)


def _alternate_request_evidence_target(graph: IdeaGraph, *, exclude_ids: set[str]) -> str:
    blocked = {
        edge.target_id
        for edge in graph.edges
        if edge.relation == "requires_evidence" and edge.source_id == edge.target_id
    }
    preferred_types = ("NoveltyClaim", "Hypothesis", "Method", "EvalPlan")
    for node_type in preferred_types:
        for node in graph.active_nodes():
            if node.type != node_type or node.id in exclude_ids or node.evidence or node.id in blocked:
                continue
            return node.id
    return ""


def _ordered_active_node_ids_by_type(
    graph: IdeaGraph,
    *,
    preferred_types: tuple[str, ...],
    preferred_ids: list[str] | None = None,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    for node_id in preferred_ids or []:
        node = graph.nodes.get(node_id)
        if node is None or node.status != "active" or node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node_id)

    for node_type in preferred_types:
        for node in graph.active_nodes():
            if node.type != node_type or node.id in seen:
                continue
            seen.add(node.id)
            ordered.append(node.id)

    return ordered


def _alternate_edge_pair(
    graph: IdeaGraph,
    *,
    relation: str,
    preferred_source_ids: list[str],
    preferred_target_ids: list[str],
) -> list[str]:
    relation_preferences = {
        "supports": {
            "sources": ("Method", "Hypothesis", "EvalPlan", "Repair", "NoveltyClaim"),
            "targets": ("Problem", "Hypothesis", "Method", "NoveltyClaim", "EvalPlan"),
        },
        "contradicts": {
            "sources": ("Risk", "Assumption", "NoveltyClaim", "Problem", "EvalPlan"),
            "targets": ("Hypothesis", "Method", "NoveltyClaim", "EvalPlan", "Problem"),
        },
        "depends_on": {
            "sources": ("EvalPlan", "Method", "Hypothesis", "NoveltyClaim"),
            "targets": ("Hypothesis", "Method", "Problem", "Assumption"),
        },
    }
    preferences = relation_preferences.get(
        relation,
        {
            "sources": ("Method", "Hypothesis", "Problem", "EvalPlan", "Risk", "NoveltyClaim"),
            "targets": ("Problem", "Hypothesis", "Method", "EvalPlan", "NoveltyClaim", "Assumption"),
        },
    )

    source_candidates = _ordered_active_node_ids_by_type(
        graph,
        preferred_types=preferences["sources"],
        preferred_ids=preferred_source_ids,
    )
    target_candidates = _ordered_active_node_ids_by_type(
        graph,
        preferred_types=preferences["targets"],
        preferred_ids=preferred_target_ids,
    )

    for source_id in source_candidates:
        for target_id in target_candidates:
            if source_id == target_id:
                continue
            if _graph_has_edge(
                graph,
                source_id=source_id,
                relation=relation,
                target_id=target_id,
            ):
                continue
            return [source_id, target_id]
    return []


def _salvage_grounding_target(
    graph: IdeaGraph,
    *,
    preferred_ids: list[str],
    related_only: bool,
) -> str:
    related_ids = _contradiction_related_node_ids(graph)
    for node_id in preferred_ids:
        node = graph.nodes.get(node_id)
        if node is None or node.status != "active":
            continue
        if related_only and node_id not in related_ids:
            continue
        if _best_attach_evidence_text(graph, target_id=node_id, preferred_text=""):
            return node_id

    attachable = _attachable_target_ids(
        graph,
        related_only=related_only,
        limit=1,
    )
    if attachable:
        return attachable[0]
    return ""


def _repair_text_for_target(graph: IdeaGraph, target_id: str) -> str:
    node = graph.nodes.get(target_id)
    if node is None:
        return "Revise the target claim so it directly addresses the current contradiction."
    return (
        f"Revise the {node.type.lower()} so it resolves the current contradiction with clearer constraints, "
        "evidence, or evaluation details."
    )


def _salvage_action_decision(
    graph: IdeaGraph,
    *,
    role: str,
    kind: str,
    target_ids: list[str],
    payload: dict[str, object],
    rationale: str,
) -> tuple[str, list[str], dict[str, object], str]:
    branch_id = _coerce_string(payload.get("branch_id")) or _own_branch_id(graph, role)
    normalized_payload = dict(payload)
    if branch_id:
        normalized_payload["branch_id"] = branch_id

    if kind == "attach_evidence" and target_ids:
        evidence_text = _best_attach_evidence_text(
            graph,
            target_id=target_ids[0],
            preferred_text=normalized_payload.get("evidence"),
        )
        if evidence_text:
            normalized_payload["evidence"] = evidence_text
        elif not _coerce_string(normalized_payload.get("evidence")) and not _unresolved_contradiction_target_ids(graph):
            return (
                "freeze_branch",
                [],
                {"branch_id": branch_id},
                rationale or "No concrete evidence remains to attach, so preserve the branch for final synthesis.",
            )

    edge_relation_by_kind = {
        "add_support_edge": "supports",
        "add_contradiction_edge": "contradicts",
        "add_dependency_edge": "depends_on",
    }
    relation = edge_relation_by_kind.get(kind)
    if relation is not None and len(target_ids) >= 2:
        if _graph_has_edge(
            graph,
            source_id=target_ids[0],
            relation=relation,
            target_id=target_ids[1],
        ):
            alternate_pair = _alternate_edge_pair(
                graph,
                relation=relation,
                preferred_source_ids=target_ids,
                preferred_target_ids=list(reversed(target_ids)),
            )
            if alternate_pair:
                target_ids = alternate_pair
            elif kind in {"add_support_edge", "add_contradiction_edge"}:
                grounding_target = _salvage_grounding_target(
                    graph,
                    preferred_ids=target_ids,
                    related_only=(kind == "add_contradiction_edge"),
                )
                evidence_text = (
                    _best_attach_evidence_text(
                        graph,
                        target_id=grounding_target,
                        preferred_text=normalized_payload.get("evidence"),
                    )
                    if grounding_target
                    else ""
                )
                if grounding_target and evidence_text:
                    return (
                        "attach_evidence",
                        [grounding_target],
                        {
                            "branch_id": branch_id,
                            "evidence": evidence_text,
                        },
                        rationale or "The selected edge already exists, so add new grounding instead of duplicating structure.",
                    )
            elif not _unresolved_contradiction_target_ids(graph):
                return (
                    "freeze_branch",
                    [],
                    {"branch_id": branch_id},
                    rationale or "The selected dependency already exists, so preserve the current branch instead of repeating it.",
                )

    if kind == "request_evidence" and target_ids:
        if _graph_has_edge(
            graph,
            source_id=target_ids[0],
            relation="requires_evidence",
            target_id=target_ids[0],
        ):
            alternate_target = _alternate_request_evidence_target(graph, exclude_ids=set(target_ids))
            if alternate_target:
                target_ids = [alternate_target]
            else:
                evidence_text = _best_attach_evidence_text(
                    graph,
                    target_id=target_ids[0],
                    preferred_text=normalized_payload.get("evidence"),
                )
                if evidence_text:
                    return (
                        "attach_evidence",
                        [target_ids[0]],
                        {
                            "branch_id": branch_id,
                            "evidence": evidence_text,
                        },
                        rationale or "An evidence request already exists for this node, so attach grounding directly instead.",
                    )
                if not _unresolved_contradiction_target_ids(graph):
                    return (
                        "freeze_branch",
                        [],
                        {"branch_id": branch_id},
                        rationale or "The requested evidence is already being tracked, so preserve the branch instead of duplicating the request.",
                    )

    if kind == "propose_repair":
        unresolved_targets = _unresolved_contradiction_target_ids(graph)
        if not unresolved_targets:
            return (
                "freeze_branch",
                [],
                {"branch_id": branch_id},
                rationale or "No unresolved contradictions remain, so preserve the current branch for synthesis.",
            )
        if not target_ids or target_ids[0] not in unresolved_targets:
            target_ids = [unresolved_targets[0]]
        if not _coerce_string(normalized_payload.get("repair_text")) and target_ids:
            normalized_payload["repair_text"] = _repair_text_for_target(graph, target_ids[0])

    return kind, target_ids, normalized_payload, rationale


def _normalize_action_payload(
    graph: IdeaGraph,
    *,
    role: str,
    kind: str,
    target_ids: list[str],
    payload: dict[str, object],
    rationale: str,
) -> dict[str, object]:
    normalized = {str(key): value for key, value in payload.items()}

    own_branch_id = _own_branch_id(graph, role)
    if kind != "freeze_branch" and own_branch_id:
        normalized["branch_id"] = own_branch_id
    elif not _coerce_string(normalized.get("branch_id")) and own_branch_id:
        normalized["branch_id"] = own_branch_id

    if "evidence" in normalized:
        normalized["evidence"] = _resolve_symbolic_reference_text(
            str(normalized.get("evidence", "")),
            graph.metadata,
        )
    if kind == "mark_overlap":
        normalized["paper_id"] = _normalize_paper_id(normalized.get("paper_id"), graph.metadata)
    if kind == "request_evidence" and not _coerce_string(normalized.get("query")) and target_ids:
        node = graph.nodes.get(target_ids[0])
        node_hint = node.text if node is not None else rationale
        normalized["query"] = f"Need concrete literature grounding for: {_first_sentence(node_hint, max_chars=140)}"

    return normalized


def _validate_action_semantics(
    graph: IdeaGraph,
    *,
    kind: str,
    target_ids: list[str],
    payload: dict[str, object],
) -> None:
    if kind == "add_support_edge" and _graph_has_edge(
        graph,
        source_id=target_ids[0],
        relation="supports",
        target_id=target_ids[1],
    ):
        raise ValueError("Support edge already exists for the selected source and target.")

    if kind == "add_contradiction_edge" and _graph_has_edge(
        graph,
        source_id=target_ids[0],
        relation="contradicts",
        target_id=target_ids[1],
    ):
        raise ValueError("Contradiction edge already exists for the selected source and target.")

    if kind == "add_dependency_edge" and _graph_has_edge(
        graph,
        source_id=target_ids[0],
        relation="depends_on",
        target_id=target_ids[1],
    ):
        raise ValueError("Dependency edge already exists for the selected source and target.")

    if kind == "request_evidence" and _graph_has_edge(
        graph,
        source_id=target_ids[0],
        relation="requires_evidence",
        target_id=target_ids[0],
    ):
        raise ValueError("This node already has an outstanding evidence request edge.")

    if kind == "attach_evidence":
        evidence = _coerce_string(payload.get("evidence"))
        if _looks_symbolic_reference(evidence):
            raise ValueError("attach_evidence must contain concrete evidence text, not a symbolic field reference.")
        node = graph.nodes[target_ids[0]]
        if evidence in node.evidence:
            raise ValueError("The selected evidence is already attached to this node.")

    if kind == "mark_overlap":
        paper_id = _coerce_string(payload.get("paper_id"))
        evidence = _coerce_string(payload.get("evidence"))
        known_ids = {candidate["paper_id"] for candidate in _paper_evidence_candidates(graph.metadata)}
        if known_ids and paper_id not in known_ids:
            raise ValueError("mark_overlap must reference one of the provided paper_id values.")
        if evidence and _looks_symbolic_reference(evidence):
            raise ValueError("mark_overlap evidence must be concrete text, not a symbolic field reference.")
        if _graph_has_edge(
            graph,
            source_id=target_ids[0],
            relation="overlaps_prior",
            target_id=target_ids[0],
            evidence_id=paper_id,
        ):
            raise ValueError("This overlap marker already exists for the selected node and paper.")

    if kind == "propose_repair":
        repair_text = _coerce_string(payload.get("repair_text"))
        if _looks_symbolic_reference(repair_text):
            raise ValueError("propose_repair must contain concrete repair text, not a symbolic field reference.")


def _literature_display_items(graph: IdeaGraph, *, limit: int = 8) -> list[str]:
    entries: list[str] = []
    for item in graph.literature:
        text = _coerce_string(item)
        if not text:
            continue
        entries.append(text.split("|", 1)[0].strip())
    reference_titles = _list_of_strings(graph.metadata.get("reference_titles"))
    entries.extend(reference_titles)
    return _unique_strings(entries)[:limit]


def _baseline_prompt_instruction(metadata: dict[str, Any]) -> str:
    baseline_key = _coerce_string(metadata.get("baseline_name"))
    baseline_name = _coerce_string(metadata.get("baseline_display_name") or metadata.get("baseline_name"))
    proxy_target = _coerce_string(metadata.get("baseline_proxy_target"))
    strategy = _coerce_string(metadata.get("baseline_strategy"))
    if not baseline_name:
        return ""
    parts = [f"Current baseline wrapper: {baseline_name}."]
    if strategy:
        parts.append(f"Execution strategy: {strategy}.")
    if proxy_target:
        parts.append(
            f"This is a local proxy wrapper intended to approximate {proxy_target}; do not assume it is an exact reproduction."
        )
    baseline_specific_guidance = {
        "ours-eig": (
            "Optimize for typed-graph rigor: add complementary claims, expose contradictions early, attach evidence when possible, and converge toward one mature high-utility subgraph rather than a loose discussion transcript."
        ),
        "ours-delayed-consensus": (
            "Optimize for typed-graph rigor: add complementary claims, expose contradictions early, attach evidence when possible, and converge toward one mature high-utility subgraph rather than a loose discussion transcript."
        ),
        "virsci-proxy": (
            "Approximate a discussion-oriented virtual-scientist panel: keep multiple viewpoints alive for longer, surface tradeoffs explicitly, and let the graph capture debate before convergence."
        ),
        "direct": (
            "Favor a compact single-pass idea without simulated debate or extra self-critique."
        ),
        "self-refine": (
            "Favor a strong draft followed by explicit internal critique and revision for specificity."
        ),
        "ai-researcher-proxy": (
            "Favor literature-grounded candidate ideation, benchmark-faithful topic anchoring, and deliberate selection among alternatives."
        ),
        "scipip-proxy": (
            "Favor structured decomposition: identify the core bottleneck, relate it to nearby work, and turn it into one coherent method plus evaluation plan."
        ),
    }
    guidance = baseline_specific_guidance.get(baseline_key)
    if guidance:
        parts.append(guidance)
    return " ".join(parts)


def _seed_system_prompt(role: str) -> str:
    preferred_anchor_types = ROLE_PREFERRED_ANCHOR_TYPES.get(role, tuple(NODE_TYPES))
    return (
        f"You are the {_role_prompt_name(role)} agent in a delayed-consensus scientific ideation system. "
        f"(Internal role id: {role}.) "
        f"{ROLE_GUIDANCE.get(role, '')} "
        "Return strict JSON only. Do not use markdown. "
        "Create one partial seed graph, not a complete abstract. "
        f"Allowed node types: {', '.join(NODE_TYPES)}. "
        f"Preferred anchor types for your role: {', '.join(preferred_anchor_types)}. "
        "Choose an anchor type that best fits your role and the topic; do not default to one type unless the context supports it. "
        "Prefer one concrete bottleneck, module, evaluation asset, or scientific tension from the provided context rather than generic claims like 'improve performance'. "
        "Support nodes should add complementary structure such as method, evidence needs, risks, or evaluation signals rather than repeating the anchor. "
        "Output schema: "
        '{"anchor":{"type":"<one allowed node type>","text":"...","confidence":0.7},'
        '"support_nodes":[{"type":"<one allowed node type>","text":"...","confidence":0.6,"relation_to_anchor":"supports|contradicts|depends_on|requires_evidence|overlaps_prior|repairs|refines"}],'
        '"rationale":"One short sentence explaining why this partial graph is useful now."}'
    )


def _seed_user_prompt(graph: IdeaGraph, role: str) -> str:
    context_packet = _compact_generation_context(graph)
    weak_context_mode = bool(context_packet.get("weak_context_mode"))
    payload = {
        "role": role,
        "topic": graph.topic,
        "context_packet": context_packet,
        "baseline_context": _baseline_prompt_instruction(graph.metadata),
        "constraints": {
            "exactly_one_anchor": True,
            "support_node_count_range": [1, 3],
            "no_full_abstract": True,
        },
    }
    if weak_context_mode:
        payload["weak_context_guidance"] = {
            "enabled": True,
            "goal": (
                "Use the weak-context scaffold to open one concrete but distinct branch. "
                "Favor divergent thinking across problem framing, mechanism choice, and evaluation design "
                "instead of converging immediately to a generic solution."
            ),
            "divergence_axes": _list_of_strings(
                context_packet.get("weak_context_scaffold", {}).get("divergence_axes")
                if isinstance(context_packet.get("weak_context_scaffold"), dict)
                else []
            )[:3],
            "design_highlights": _list_of_strings(context_packet.get("design_highlights"))[:3],
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _action_system_prompt(
    graph: IdeaGraph,
    role: str,
    round_name: str,
    allowed_actions: tuple[str, ...],
) -> str:
    phase = resolve_round_phase(round_name)
    weak_context_mode = bool(_compact_generation_context(graph).get("weak_context_mode"))
    action_lines = []
    for action_kind in allowed_actions:
        hint = ACTION_PROMPT_HINTS[action_kind]
        target_shape = hint["target_shape"]
        payload_fields = hint["payload_fields"]
        target_text = "[]" if not target_shape else "[" + ", ".join(target_shape) + "]"
        payload_text = ", ".join(payload_fields)
        action_lines.append(
            f"- {action_kind}: target_ids={target_text}; payload fields={payload_text}; {hint['when_to_use']}"
        )
    decision_focus = " ".join(phase.decision_focus)
    phase_specific_instruction = ""
    if phase.key == "repair" and _unresolved_contradiction_target_ids(graph):
        phase_specific_instruction = (
            " If you choose propose_repair or attach_evidence during repair, target one of the provided "
            "repair_priority_targets whenever possible."
        )
    elif phase.key == "repair":
        phase_specific_instruction = (
            " No unresolved contradictions remain, so this repair-phase turn is now consolidation. "
            "Do not invent a new repair. Prefer attaching grounding to a weak node, adding one missing support edge, "
            "or freezing a coherent branch."
        )
    weak_context_instruction = ""
    if weak_context_mode:
        weak_context_instruction = (
            " This is a keyword-only weak-context benchmark. Preserve useful divergence for longer: "
            "prefer edits that make the graph more keyword-specific, add one concrete mechanism or evaluation anchor, "
            "or keep distinct branches alive when they reflect different scaffold directions. "
            "Do not freeze a generic branch just because it is locally coherent."
        )
    return (
        f"You are the {_role_prompt_name(role)} agent in round {round_name} of delayed-consensus scientific collaboration. "
        f"(Internal role id: {role}.) "
        f"Current phase: {phase.title}. "
        f"Round objective: {phase.objective} "
        f"{ROLE_GUIDANCE.get(role, '')} "
        "Choose exactly one action that best improves the graph at this moment. "
        "Do not default to a specific action kind just because it appears first in a schema. "
        "Prefer benchmark-grounded edits over generic ones: when the context packet exposes concrete reference titles, "
        "design-anchor terms, datasets, or metrics, use them to make the graph more specific rather than asking for "
        "another vague evidence request. "
        "During structure turns, a concrete support or dependency edge grounded in the benchmark context is usually "
        "higher value than a generic request_evidence action. "
        "Use request_evidence mainly when the current prompt does not already provide one concrete supporting clue. "
        f"{decision_focus} "
        f"{phase_specific_instruction}"
        f"{weak_context_instruction}"
        "Return strict JSON only. Do not use markdown. "
        f"Allowed action kinds for this phase: {', '.join(allowed_actions)}. "
        "Use existing node IDs and branch IDs from the provided graph snapshot. "
        "For non-freeze actions, use your own branch_id from the prompt even when targeting another branch's node. "
        "For attach_evidence or mark_overlap, payload.evidence must be concrete evidence text or a concise paraphrase "
        "from one provided evidence candidate. Never return field references such as "
        "reference_paper_snippets[0].abstract. "
        "Prefer the focused-view candidate lists for request_evidence and attach_evidence, and avoid blocked_request_node_ids. "
        "If the benchmark_focus block contains anchor terms such as module names or representations, favor edits that "
        "make those terms explicit in the graph. "
        "Avoid duplicating an edge, evidence request, or evidence attachment that already appears in recent action history. "
        "Your rationale should explain why this action is higher leverage than the other allowed actions right now. "
        "Action schemas:\n"
        + "\n".join(action_lines)
        + "\nOutput schema: "
        '{"kind":"<one allowed action kind>","target_ids":["..."],"payload":{"branch_id":"..."},'
        '"rationale":"One short sentence grounded in the current graph state."}'
    )


def _action_user_prompt(graph: IdeaGraph, round_name: str, role: str) -> str:
    phase = resolve_round_phase(round_name)
    allowed_actions = _dynamic_allowed_actions(graph, round_name)
    context_packet = _compact_generation_context(graph)
    action_requirements = {
        action_kind: {
            "target_count": ACTION_TARGET_COUNTS[action_kind],
            "required_payload_fields": list(ACTION_REQUIRED_PAYLOAD_FIELDS.get(action_kind, ())),
            "prompt_hint": ACTION_PROMPT_HINTS[action_kind]["when_to_use"],
        }
        for action_kind in allowed_actions
    }
    unresolved_targets = _unresolved_contradiction_target_ids(graph)
    if phase.key == "repair" and unresolved_targets:
        decision_instruction = (
            "Pick one action that most improves contradiction handling, maturity, or grounding. "
            "Prefer repair or evidence actions that directly touch contradiction-related nodes, and avoid duplicating an existing support edge."
        )
    elif phase.key == "repair":
        decision_instruction = (
            "No unresolved contradictions remain. Consolidate the graph instead of inventing another repair: "
            "add missing grounding, reinforce one coherent path, or freeze a branch that is already strong enough."
        )
    else:
        decision_instruction = (
            "Pick one action that most improves maturity, grounding, contradiction handling, or branch quality. "
            "If support edges already exist for the same pair, avoid duplicating them. "
            "When attaching evidence, use actual text from the provided evidence candidates instead of symbolic references."
        )
    evidence_candidates = (
        _paper_evidence_candidates(graph.metadata, limit=4)
        + _benchmark_packet_evidence_candidates(graph.metadata, limit=4)
        + _grounding_evidence_candidates(graph, limit=4)
    )[:8]
    payload = {
        "round": round_name,
        "phase": phase.key,
        "phase_title": phase.title,
        "allowed_actions": list(allowed_actions),
        "context_packet": context_packet,
        "baseline_context": _baseline_prompt_instruction(graph.metadata),
        "action_requirements": action_requirements,
        "benchmark_focus": {
            "topic": context_packet.get("benchmark_packet", {}).get("topic"),
            "keyword": context_packet.get("benchmark_packet", {}).get("keyword"),
            "reference_titles": [
                item.get("title", "")
                for item in context_packet.get("benchmark_packet", {}).get("reference_packet", [])
                if isinstance(item, dict)
            ],
            "design_anchor_terms": list(context_packet.get("design_anchor_terms", [])),
            "dataset_items": list(context_packet.get("dataset_items", [])),
            "metric_items": list(context_packet.get("metric_items", [])),
        },
        "evidence_candidates": evidence_candidates,
        "focused_view": focused_view_for_prompt(graph, role),
        "repair_priority_targets": _repair_priority_targets(graph),
        "decision_instruction": decision_instruction,
    }
    weak_context_scaffold = context_packet.get("weak_context_scaffold", {})
    if isinstance(weak_context_scaffold, dict) and weak_context_scaffold:
        payload["weak_context_guidance"] = {
            "enabled": True,
            "goal": (
                "Increase keyword-specific structure before convergence. Prefer edits that make the graph mention the "
                "benchmark keyword together with one concrete mechanism family and one concrete evaluation anchor."
            ),
            "divergence_axes": _list_of_strings(weak_context_scaffold.get("divergence_axes"))[:3],
            "mechanism_terms": _list_of_strings(weak_context_scaffold.get("mechanism_terms"))[:5],
            "risk_items": _list_of_strings(weak_context_scaffold.get("risk_items"))[:3],
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _synthesis_system_prompt() -> str:
    return (
        "You are the synthesis module for a multi-agent scientific ideation graph. "
        "Return strict JSON only. Do not use markdown. "
        "Write a compact structured research idea, not a full paper. "
        "The output must be coherent, specific, and grounded in the provided graph context and literature metadata. "
        "Do not invent detailed claims about prior work that are not supported by the input. "
        "If the literature context is sparse, say so explicitly and keep the existing-methods summary cautious. "
        "Each field should contribute distinct information instead of repeating the same sentences across sections. "
        "Do not output an abstract field; use the concrete sections below instead. "
        "Avoid generic titles such as 'Language-Driven X' or 'Improving Y' when a more specific mechanism is available. "
        "Avoid generic method phrases such as 'hybrid model', 'improve performance', or 'strong baselines' when the "
        "input already provides concrete modules, representations, or reference systems. "
        "When design highlights or design_anchor_terms are provided, explicitly reuse at least one concrete module, bottleneck, or mechanism name rather than writing only a high-level combination. "
        "When selected nodes contain Repair, Risk, Assumption, or NoveltyClaim content, use them to sharpen method details, caveats, or novelty positioning instead of dropping them. "
        "When dataset_items or metric_items are provided, mention at least one dataset and one metric explicitly in the evaluation field rather than using vague placeholders like 'benchmark datasets' or generic metrics. "
        "Favor clean natural prose over copied snippet fragments. If a literature fragment looks truncated, noisy, or copied from OCR/PDF extraction, omit it instead of reproducing it. "
        "Existing Methods should name 2-3 concrete reference directions when the input provides them, and should state one concrete limitation instead of only listing titles. "
        "The Method section should describe 2-4 concrete design choices and how they interact. "
        "The Evaluation section should name datasets, metrics, baselines, and at least one ablation or stress test when the input supports them. "
        "Use slot_summary as the canonical scaffold: keep the method centered on the mechanism slot, keep the evaluation centered on the evaluation slot, "
        "and use the caveat slot to sharpen limitations rather than dropping it. "
        "If the context is keyword-only and does not support named datasets, stay cautious and do not invent benchmark names. "
        "If a weak_context_scaffold is provided, use it to make the proposal keyword-specific and to keep one divergent mechanism family explicit. "
        "Prefer one coherent method story rather than stitching together several loosely related mechanisms. "
        "Make the method concrete by naming 2-4 distinct design choices or stages when the graph supports them. "
        "Make the evaluation concrete by naming candidate datasets, baselines, metrics, and at least one ablation or stress test when the context supports them. "
        "Use the fields to produce a proposal that is richer than isolated graph nodes but shorter than a full manuscript. "
        'JSON schema: {"title":"8-18 word title",'
        '"problem":"2-3 sentences","existing_methods":"2-4 sentences grounded in provided literature",'
        '"motivation":"1-3 sentences on why the idea is needed","hypothesis":"1-2 sentences",'
        '"method":"3-5 sentences describing the proposed method","evaluation":"3-5 sentences describing the experiment plan",'
        '"significance":"1-2 sentences on expected contribution","caveats":"1-2 sentences on assumptions or risks"}'
    )


def _slot_summary_from_claim_chain(
    graph: IdeaGraph,
    selected_claim_chain: dict[str, object] | None,
) -> dict[str, object]:
    if not isinstance(selected_claim_chain, dict):
        return {}
    slots = selected_claim_chain.get("slots", {})
    if not isinstance(slots, dict):
        return {}

    summary: dict[str, object] = {}
    for slot_name, node_id in slots.items():
        if not isinstance(node_id, str) or node_id not in graph.nodes:
            continue
        node = graph.nodes[node_id]
        summary[slot_name] = {
            "id": node.id,
            "type": node.type,
            "role": node.role,
            "text": _truncate_text(node.text, max_chars=260),
            "evidence": [_truncate_text(item, max_chars=180) for item in node.evidence[:2]],
        }
    if summary:
        summary["synthesis_contract"] = {
            "problem_from_slot": "problem",
            "method_from_slot": "mechanism",
            "evaluation_from_slot": "evaluation",
            "caveats_from_slot": "caveat",
            "existing_methods_from_grounding": True,
        }
    return summary


def _synthesis_user_prompt(graph: IdeaGraph, subgraph: dict[str, object]) -> str:
    node_ids = [str(item) for item in subgraph.get("node_ids", [])]
    edge_ids = [str(item) for item in subgraph.get("edge_ids", [])]
    selected_node_ids = {node_id for node_id in node_ids if node_id in graph.nodes}
    local_edges = [
        edge
        for edge in graph.edges
        if edge.source_id in selected_node_ids or edge.target_id in selected_node_ids
    ]
    local_node_ids = set(selected_node_ids)
    for edge in local_edges:
        if edge.source_id in graph.nodes:
            local_node_ids.add(edge.source_id)
        if edge.target_id in graph.nodes:
            local_node_ids.add(edge.target_id)
    safe_metadata = _prompt_safe_metadata(graph.metadata)
    grounding = build_literature_grounding(literature=graph.literature, metadata=safe_metadata)
    selected_claim_chain = select_claim_chain(graph, node_ids=local_node_ids or None)
    slot_summary = _slot_summary_from_claim_chain(graph, selected_claim_chain)
    latest_round = graph.round_summaries[-1][1] if graph.round_summaries else None
    payload = {
        "topic": graph.topic,
        "context_packet": _compact_generation_context(graph),
        "baseline_context": _baseline_prompt_instruction(graph.metadata),
        "literature_titles": _literature_display_items(graph, limit=6),
        "evidence_candidates": (
            _paper_evidence_candidates(graph.metadata, limit=4)
            + _benchmark_packet_evidence_candidates(graph.metadata, limit=4)
            + _grounding_evidence_candidates(graph, limit=4)
        )[:8],
        "literature_grounding": {
            "source": grounding.source,
            "reference_titles": grounding.reference_titles[:5],
            "design_highlights": grounding.design_highlights[:3],
            "design_anchor_terms": _design_anchor_terms(grounding.design_highlights[:4]),
            "dataset_items": grounding.dataset_items[:4],
            "metric_items": grounding.metric_items[:6],
            "existing_methods_summary": _truncate_text(grounding.existing_methods_summary, max_chars=320),
            "experiment_plan_summary": _truncate_text(grounding.experiment_plan_summary, max_chars=260),
            "weak_context_scaffold": grounding.weak_context_scaffold if isinstance(grounding.weak_context_scaffold, dict) else {},
        },
        "specificity_requirements": {
            "reuse_design_anchor_terms": _design_anchor_terms(grounding.design_highlights[:4]),
            "mention_dataset_items": grounding.dataset_items[:2],
            "mention_metric_items": grounding.metric_items[:3],
            "name_reference_titles": grounding.reference_titles[:3],
            "include_ablation": True,
            "keep_keyword_explicit": _coerce_string(graph.metadata.get("keyword")),
        },
        "writing_target": {
            "style": "compact structured research idea",
            "length": "sectioned proposal, shorter than a paper",
            "must_follow_claim_chain": True,
            "must_not_do": [
                "do not merely copy node text verbatim into multiple sections",
                "do not fabricate detailed prior-work claims without support",
                "do not write a full paper introduction or related-work section",
                "do not repeat the same sentence across problem, motivation, method, and evaluation",
                "do not use placeholders like strong baselines or task-specific metrics when concrete names are available",
            ],
        },
        "quality_checklist": {
            "title_should_name_a_concrete_mechanism_when_available": True,
            "existing_methods_should_include_concrete_reference_directions": True,
            "method_should_include_2_to_4_concrete_design_choices": True,
            "evaluation_should_include_dataset_metric_baseline_and_ablation_when_supported": True,
            "problem_and_motivation_should_not_repeat_each_other": True,
        },
        "selected_claim_chain": selected_claim_chain or {},
        "slot_summary": slot_summary,
        "latest_round_snapshot": (
            {
                "support_coverage": latest_round.support_coverage,
                "unresolved_contradiction_ratio": latest_round.unresolved_contradiction_ratio,
                "utility": latest_round.utility,
                "utility_breakdown": {
                    "promise": latest_round.utility_breakdown.promise,
                    "support": latest_round.utility_breakdown.support,
                    "coherence": latest_round.utility_breakdown.coherence,
                    "evidence": latest_round.utility_breakdown.evidence,
                    "novelty": latest_round.utility_breakdown.novelty,
                },
            }
            if latest_round is not None
            else {}
        ),
        "selected_nodes": [
            {
                "id": node_id,
                "type": graph.nodes[node_id].type,
                "text": _truncate_text(graph.nodes[node_id].text, max_chars=240),
                "evidence": [_truncate_text(item, max_chars=180) for item in graph.nodes[node_id].evidence[:2]],
            }
            for node_id in node_ids
            if node_id in graph.nodes
        ],
        "local_supporting_nodes": [
            {
                "id": node_id,
                "type": graph.nodes[node_id].type,
                "text": _truncate_text(graph.nodes[node_id].text, max_chars=220),
                "role": graph.nodes[node_id].role,
                "confidence": graph.nodes[node_id].confidence,
                "evidence": [_truncate_text(item, max_chars=160) for item in graph.nodes[node_id].evidence[:2]],
            }
            for node_id in sorted(local_node_ids)
            if node_id in graph.nodes and node_id not in selected_node_ids
        ][:10],
        "selected_edges": [
            {
                "id": edge.id,
                "source_id": edge.source_id,
                "relation": edge.relation,
                "target_id": edge.target_id,
            }
            for edge in graph.edges
            if edge.id in edge_ids
        ],
        "local_graph_edges": [
            {
                "id": edge.id,
                "source_id": edge.source_id,
                "relation": edge.relation,
                "target_id": edge.target_id,
                "resolved": edge.resolved,
            }
            for edge in local_edges[:24]
        ],
        "unresolved_local_contradictions": [
            {
                "id": edge.id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
            }
            for edge in local_edges
            if edge.relation == "contradicts" and not edge.resolved
        ][:8],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_match_text(value: Any) -> str:
    return " ".join(_coerce_string(value).casefold().split())


def _contains_any_term(text: str, terms: list[str]) -> bool:
    haystack = _normalize_match_text(text)
    if not haystack:
        return False
    for term in terms:
        needle = _normalize_match_text(term)
        if needle and needle in haystack:
            return True
    return False


def _matching_terms(text: str, terms: list[str]) -> list[str]:
    haystack = _normalize_match_text(text)
    if not haystack:
        return []
    matches: list[str] = []
    for term in terms:
        needle = _normalize_match_text(term)
        if needle and needle in haystack:
            matches.append(_coerce_string(term))
    return _unique_strings(matches)


def _content_terms(text: Any, *, limit: int = 8) -> list[str]:
    generic_tokens = {
        "with",
        "that",
        "this",
        "from",
        "using",
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
        "better",
        "baseline",
        "baselines",
        "current",
        "integrate",
    }
    ordered: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9\-]+", _coerce_string(text).lower()):
        if len(token) < 5 or token in generic_tokens or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
        if len(ordered) >= limit:
            break
    return ordered


def _contains_key_terms(text: Any, reference_text: Any, *, min_hits: int = 2) -> bool:
    haystack = _normalize_match_text(text)
    terms = _content_terms(reference_text)
    if not haystack or not terms:
        return False
    hit_count = sum(1 for term in terms if term in haystack)
    return hit_count >= min(min_hits, len(terms))


def _join_natural_strings(items: list[str]) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _append_unique_sentence(base_text: str, sentence: str) -> str:
    base = _coerce_string(base_text)
    addition = _coerce_string(sentence)
    if not addition:
        return base
    if not base:
        return addition
    if _normalize_match_text(addition) in _normalize_match_text(base):
        return base
    separator = "" if base.endswith((".", "!", "?")) else "."
    return f"{base}{separator} {addition}".strip()


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = _normalize_match_text(text)
    return any(_normalize_match_text(phrase) in normalized for phrase in phrases if phrase)


def _split_sentences(text: Any) -> list[str]:
    cleaned = _coerce_string(text)
    if not cleaned:
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]


def _is_noisy_proposal_sentence(sentence: str) -> bool:
    normalized = _normalize_match_text(sentence)
    if not normalized:
        return True
    noisy_markers = (
        "experiments were conducted on paper introduces",
        "paper introduces",
        "captured in diverse real scenarios",
        "project website",
        "see the project website",
        "task instruction see examples above",
        "a11y tree",
        "keyboardmouse",
        "action observation input predict",
        "package name",
        "view id resource name",
        "bounds in screen",
        "class name",
        "content description",
        "et al",
    )
    if any(marker in normalized for marker in noisy_markers):
        return True
    return bool(
        re.search(r"\bfigure\s*\d+\b", normalized)
        or re.search(r"\btable\s*\d+\b", normalized)
        or normalized.startswith("figure ")
        or normalized.startswith("table ")
    )


def _replace_weak_context_placeholder_phrases(text: Any) -> str:
    cleaned = _coerce_string(text)
    if not cleaned:
        return ""
    replacements = {
        "keyword-specific case studies": "domain-grounded case studies",
        "generic problem framing": "overly generic task formulations",
        "insufficient mechanism grounding": "weak linkage between the mechanism and the domain structure",
    }
    for source, target in replacements.items():
        cleaned = re.sub(source, target, cleaned, flags=re.IGNORECASE)
    return cleaned


def _remove_sentences_with_phrases(text: Any, phrases: tuple[str, ...]) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return _coerce_string(text)
    normalized_phrases = [_normalize_match_text(item) for item in phrases if item]
    kept = [
        sentence
        for sentence in sentences
        if not any(phrase in _normalize_match_text(sentence) for phrase in normalized_phrases)
    ]
    return " ".join(kept) if kept else ""


def _compose_weak_context_existing_methods(
    keyword: str,
    directions: list[str],
    risk_items: list[str],
) -> str:
    direction_text = _join_natural_strings(directions[:3]) or "strong task-specific predictive models"
    summary = f"Common approaches for tasks centered on {keyword} include {direction_text}."
    if risk_items:
        summary = _append_unique_sentence(
            summary,
            f"These families often struggle with {_join_natural_strings(risk_items[:2])}.",
        )
    return summary


def _compose_weak_context_focus_claim(keyword: str, method_terms: list[str]) -> str:
    focus_terms = _unique_strings(method_terms[:2])
    if len(focus_terms) >= 2:
        return (
            f"The key claim is that combining {focus_terms[0]} with {focus_terms[1]} "
            f"will generalize better than generic predictors for tasks centered on {keyword}."
        )
    if focus_terms:
        return (
            f"The key claim is that explicitly modeling {focus_terms[0]} "
            f"will generalize better than generic predictors for tasks centered on {keyword}."
        )
    return (
        f"The key claim is that an explicit mechanism tailored to {keyword} "
        "will generalize better than a generic predictor."
    )


def _compose_weak_context_method(keyword: str, method_terms: list[str]) -> str:
    focus_terms = _unique_strings(method_terms[:3])
    if focus_terms:
        return (
            f"The method centers on {_join_natural_strings(focus_terms)}, "
            f"keeping the model tied to {keyword} rather than relying on a generic predictor."
        )
    return (
        f"The method introduces an explicit inductive bias tailored to {keyword}, "
        "together with robustness checks for distribution shift."
    )


def _method_has_concrete_detail(text: Any) -> bool:
    markers = (
        "encoder",
        "decoder",
        "attention",
        "message passing",
        "graph",
        "loss",
        "objective",
        "constraint",
        "module",
        "backbone",
        "network",
        "message-passing",
        "electron configuration",
        "group and period",
    )
    return _contains_any_term(_coerce_string(text), list(markers))


def _compose_weak_context_evaluation(
    keyword: str,
    evaluation_assets: list[str],
    metric_items: list[str],
    compare_directions: list[str],
    method_terms: list[str],
    risk_items: list[str],
) -> str:
    asset_text = _join_natural_strings(evaluation_assets[:3]) or f"{keyword}-relevant benchmark tasks"
    metric_text = _join_natural_strings(metric_items[:4]) or "task-relevant accuracy and robustness metrics"
    compare_text = _join_natural_strings(compare_directions[:3]) or "strong task-specific baselines"
    ablation_text = _join_natural_strings(_unique_strings(method_terms[:2])) or "the main proposed components"
    evaluation = (
        f"Evaluate on {asset_text}. "
        f"Report {metric_text}. "
        f"Compare against {compare_text}. "
        f"Include ablations on {ablation_text}."
    )
    if risk_items:
        evaluation = _append_unique_sentence(
            evaluation,
            f"Stress test {_join_natural_strings(risk_items[:2])}.",
        )
    return evaluation


def _rewrite_imperative_method_sentences(text: Any) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return _coerce_string(text)
    rewritten: list[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if not re.match(r"^Use\s+", stripped, flags=re.IGNORECASE):
            rewritten.append(stripped)
            continue
        match = re.match(r"^Use\s+.+?\s+approach\s+to\s+(.+?)\.?$", stripped, flags=re.IGNORECASE)
        if match:
            rewritten.append(f"The method is designed to {match.group(1).strip().rstrip('.')}.")
            continue
        rewritten.append(f"The method uses {re.sub(r'^Use\s+', '', stripped, flags=re.IGNORECASE).rstrip('.')}.")
    return " ".join(rewritten)


def _has_fragmentary_benchmark_plan(text: Any) -> bool:
    for sentence in _split_sentences(text):
        lowered = _normalize_match_text(sentence)
        word_count = len(sentence.split())
        if lowered.startswith("evaluate on ") and word_count <= 8:
            return True
        if lowered.startswith("report ") and word_count <= 8:
            return True
    return False


def _compose_grounded_benchmark_evaluation(
    dataset_items: list[str],
    metric_items: list[str],
    reference_titles: list[str],
    design_anchor_terms: list[str],
) -> str:
    dataset_text = _join_natural_strings(dataset_items[:3]) or "the benchmark datasets"
    metric_text = _join_natural_strings(metric_items[:4]) or "task-relevant quantitative metrics"
    compare_text = _join_natural_strings(reference_titles[:2]) or "strong benchmark baselines"
    ablation_text = _join_natural_strings(design_anchor_terms[:2]) or "the main proposed components"
    return (
        f"Evaluate on {dataset_text}. "
        f"Report {metric_text}. "
        f"Compare against {compare_text}. "
        f"Include ablations on {ablation_text}."
    )


def _compose_claim_chain_grounded_evaluation(
    evaluation_text: str,
    dataset_items: list[str],
    metric_items: list[str],
    reference_titles: list[str],
    design_anchor_terms: list[str],
) -> str:
    evaluation = _clean_proposal_section_text(evaluation_text)
    if not evaluation or _has_fragmentary_benchmark_plan(evaluation):
        return _compose_grounded_benchmark_evaluation(
            dataset_items,
            metric_items,
            reference_titles,
            design_anchor_terms,
        )
    if dataset_items and not _contains_any_term(evaluation, dataset_items):
        evaluation = _append_unique_sentence(
            evaluation,
            f"Evaluate on {', '.join(dataset_items[:2])}.",
        )
    if metric_items and not _contains_any_term(evaluation, metric_items):
        evaluation = _append_unique_sentence(
            evaluation,
            f"Report {', '.join(metric_items[:3])}.",
        )
    if reference_titles and not _contains_any_phrase(evaluation, ("compare against", "baseline")):
        evaluation = _append_unique_sentence(
            evaluation,
            f"Compare against {', '.join(reference_titles[:2])}.",
        )
    if design_anchor_terms and not _contains_any_phrase(evaluation, ("ablation", "stress test")):
        evaluation = _append_unique_sentence(
            evaluation,
            f"Include ablations on {', '.join(design_anchor_terms[:2])}.",
        )
    return evaluation


def _clean_proposal_section_text(text: Any) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return _coerce_string(text)
    kept: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = _normalize_match_text(sentence)
        if not normalized or normalized in seen:
            continue
        if _is_noisy_proposal_sentence(sentence):
            continue
        kept.append(sentence)
        seen.add(normalized)
    return " ".join(kept) if kept else _coerce_string(text)


def _remove_exact_scaffold_method_sentences(method_text: str, design_highlights: list[str]) -> str:
    sentences = _split_sentences(method_text)
    if len(sentences) <= 1:
        return _coerce_string(method_text)
    scaffold_sentences = {
        _normalize_match_text(_naturalize_design_highlight(item))
        for item in design_highlights
        if _naturalize_design_highlight(item)
    }
    if not scaffold_sentences:
        return _coerce_string(method_text)
    kept_sentences = [
        sentence
        for sentence in sentences
        if _normalize_match_text(sentence) not in scaffold_sentences
    ]
    if not kept_sentences:
        return _coerce_string(method_text)
    return " ".join(kept_sentences)


def _postprocess_grounding(graph: IdeaGraph) -> LiteratureGrounding:
    if graph.metadata.get("benchmark_mode"):
        return build_literature_grounding(
            literature=graph.literature,
            metadata=_prompt_safe_metadata(graph.metadata),
        )
    stored = graph.metadata.get("literature_grounding", {})
    if isinstance(stored, dict) and stored:
        return LiteratureGrounding(
            source=_coerce_string(stored.get("source")),
            target_paper=_coerce_string(stored.get("target_paper")),
            reference_titles=_list_of_strings(stored.get("reference_titles")),
            design_highlights=_list_of_strings(stored.get("design_highlights")),
            dataset_items=_list_of_strings(stored.get("dataset_items")),
            metric_items=_list_of_strings(stored.get("metric_items")),
            existing_methods_summary=_coerce_string(stored.get("existing_methods_summary")),
            experiment_plan_summary=_coerce_string(stored.get("experiment_plan_summary")),
            weak_context_scaffold=(
                dict(stored.get("weak_context_scaffold", {}))
                if isinstance(stored.get("weak_context_scaffold"), dict)
                else {}
            ),
        )
    return build_literature_grounding(literature=graph.literature, metadata=graph.metadata)


def _postprocess_final_proposal(graph: IdeaGraph, proposal: FinalProposal) -> FinalProposal:
    grounding = _postprocess_grounding(graph)
    selected_claim_chain = select_claim_chain(graph)
    slot_summary = _slot_summary_from_claim_chain(graph, selected_claim_chain)
    mechanism_slot = slot_summary.get("mechanism", {}) if isinstance(slot_summary, dict) else {}
    evaluation_slot = slot_summary.get("evaluation", {}) if isinstance(slot_summary, dict) else {}
    design_anchor_terms = _design_anchor_terms(grounding.design_highlights[:4])
    design_support_terms = _design_support_terms(grounding.design_highlights[:4])
    weak_context_scaffold = grounding.weak_context_scaffold if isinstance(grounding.weak_context_scaffold, dict) else {}
    scaffold_directions = _list_of_strings(weak_context_scaffold.get("existing_method_directions"))[:3]
    scaffold_risk_items = _list_of_strings(weak_context_scaffold.get("risk_items"))[:3]
    scaffold_method_terms = _list_of_strings(weak_context_scaffold.get("mechanism_terms"))[:4]
    scaffold_design_highlights = _list_of_strings(weak_context_scaffold.get("design_highlights"))[:3]
    scaffold_evaluation_assets = _list_of_strings(weak_context_scaffold.get("evaluation_assets"))[:3]
    scaffold_metric_items = _list_of_strings(weak_context_scaffold.get("metric_items"))[:5]
    scaffold_method_instantiation = _coerce_string(weak_context_scaffold.get("method_instantiation"))
    packet = graph.metadata.get("benchmark_input_packet", {})
    reference_packet = packet.get("reference_packet", []) if isinstance(packet, dict) else []
    keyword = _coerce_string(
        graph.metadata.get("keyword")
        or (packet.get("keyword") if isinstance(packet, dict) else "")
        or graph.topic
    )
    keyword_only_mode = (
        _coerce_string(graph.metadata.get("benchmark")) == "liveideabench"
        or (grounding.source == "titles_only" and not reference_packet)
    )
    generic_evaluation_markers = (
        "strong baselines",
        "task-specific quantitative metrics",
        "task-relevant datasets",
        "realistic benchmark tasks",
        "representative benchmark datasets",
    )
    generic_method_markers = (
        "hybrid model",
        "improve performance",
        "guide rendering",
        "stronger representation",
    )
    weak_context_existing_markers = (
        "safe starting directions",
        "benchmark keyword",
        "held out metadata",
        "row provides a keyword prompt",
        "generic problem framing",
        "insufficient mechanism grounding",
    )
    weak_context_instruction_markers = (
        "The central mechanism should stay explicitly targeted",
        "Focus the method on",
    )
    weak_context_residue_markers = (
        "keyword-specific case studies",
        "generic problem framing",
        "insufficient mechanism grounding",
        "overly generic task formulations",
    )

    proposal.method = _rewrite_imperative_method_sentences(proposal.method)
    mechanism_text = _coerce_string(mechanism_slot.get("text")) if isinstance(mechanism_slot, dict) else ""
    if mechanism_text and (
        _contains_any_phrase(proposal.method, generic_method_markers)
        or not _contains_key_terms(proposal.method, mechanism_text)
    ):
        proposal.method = _rewrite_imperative_method_sentences(mechanism_text)

    if grounding.design_highlights and not _contains_any_term(proposal.method, design_support_terms):
        proposal.method = _append_unique_sentence(
            proposal.method,
            _naturalize_design_highlight(grounding.design_highlights[0]),
        )

    if design_anchor_terms and not _contains_any_phrase(proposal.evaluation, ("ablation", "stress test")):
        proposal.evaluation = _append_unique_sentence(
            proposal.evaluation,
            f"Include ablations on {', '.join(design_anchor_terms[:2])}.",
        )

    if grounding.dataset_items and not _contains_any_term(proposal.evaluation, grounding.dataset_items):
        proposal.evaluation = _append_unique_sentence(
            proposal.evaluation,
            f"Evaluate on {', '.join(grounding.dataset_items[:2])}.",
        )

    if grounding.metric_items and not _contains_any_term(proposal.evaluation, grounding.metric_items):
        proposal.evaluation = _append_unique_sentence(
            proposal.evaluation,
            f"Report {', '.join(grounding.metric_items[:3])}.",
        )

    if grounding.reference_titles and not _contains_any_term(proposal.existing_methods, grounding.reference_titles[:3]):
        summary_sentence = _first_sentence(grounding.existing_methods_summary, max_chars=260)
        proposal.existing_methods = _append_unique_sentence(proposal.existing_methods, summary_sentence)

    if grounding.reference_titles and not _contains_any_phrase(proposal.evaluation, ("compare against", "baseline")):
        compare_titles = ", ".join(grounding.reference_titles[:2])
        proposal.evaluation = _append_unique_sentence(
            proposal.evaluation,
            f"Compare against {compare_titles}-style baselines.",
        )

    if design_anchor_terms and _contains_any_phrase(proposal.method, generic_method_markers):
        proposal.method = _append_unique_sentence(
            proposal.method,
            f"Make {design_anchor_terms[0]} the main differentiating component rather than a generic combination baseline.",
        )

    if grounding.reference_titles and _contains_any_phrase(proposal.existing_methods, ("plausible existing directions", "common directions include")):
        proposal.existing_methods = _append_unique_sentence(
            proposal.existing_methods,
            f"Concrete nearby references include {', '.join(grounding.reference_titles[:3])}.",
        )

    evaluation_text = _coerce_string(evaluation_slot.get("text")) if isinstance(evaluation_slot, dict) else ""
    if grounding.dataset_items and grounding.metric_items and (
        _contains_any_phrase(proposal.evaluation, generic_evaluation_markers)
        or (evaluation_text and not _contains_key_terms(proposal.evaluation, evaluation_text, min_hits=1))
    ):
        proposal.evaluation = _compose_claim_chain_grounded_evaluation(
            evaluation_text or proposal.evaluation,
            grounding.dataset_items,
            grounding.metric_items,
            grounding.reference_titles,
            design_anchor_terms,
        )

    if keyword_only_mode:
        keyword_sentence = f"This matters directly for {keyword}."
        if keyword and keyword.casefold() not in _normalize_match_text(proposal.problem):
            proposal.problem = _append_unique_sentence(proposal.problem, keyword_sentence)
        proposal.hypothesis = _remove_sentences_with_phrases(
            proposal.hypothesis,
            (weak_context_instruction_markers[0],),
        )
        if keyword and keyword.casefold() not in _normalize_match_text(proposal.hypothesis):
            proposal.hypothesis = _append_unique_sentence(
                proposal.hypothesis,
                _compose_weak_context_focus_claim(keyword, scaffold_method_terms),
            )
        proposal.method = _remove_sentences_with_phrases(
            proposal.method,
            (weak_context_instruction_markers[1],),
        )
        existing_lower = _normalize_match_text(proposal.existing_methods)
        if (
            "benchmark keyword" in existing_lower
            or "held out metadata" in existing_lower
            or "row provides a keyword prompt" in existing_lower
        ):
            proposal.existing_methods = (
                f"For {keyword}, plausible existing directions include spatiotemporal forecasting models, "
                "physics-aware simulation, and multi-source data fusion methods."
            )
        if weak_context_scaffold:
            if scaffold_directions and (
                not proposal.existing_methods
                or _contains_any_phrase(proposal.existing_methods, weak_context_existing_markers)
                or _normalize_match_text(proposal.existing_methods).startswith("however")
            ):
                proposal.existing_methods = _compose_weak_context_existing_methods(
                    keyword,
                    scaffold_directions,
                    scaffold_risk_items,
                )
            elif scaffold_risk_items and "however" not in _normalize_match_text(proposal.existing_methods):
                proposal.existing_methods = _append_unique_sentence(
                    proposal.existing_methods,
                    f"However, these directions often struggle with {_join_natural_strings(scaffold_risk_items)}.",
                )
            if (
                not proposal.method
                or _contains_any_phrase(proposal.method, generic_method_markers)
                or _contains_any_phrase(proposal.method, (weak_context_instruction_markers[1],))
            ) and scaffold_method_terms:
                proposal.method = _compose_weak_context_method(keyword, scaffold_method_terms)
            if scaffold_method_instantiation and not _method_has_concrete_detail(proposal.method):
                proposal.method = _append_unique_sentence(
                    proposal.method,
                    _rewrite_imperative_method_sentences(scaffold_method_instantiation),
                )
            design_highlights = scaffold_design_highlights
            covered_highlight_count = sum(
                1
                for item in design_highlights
                if _contains_any_term(proposal.method, _design_support_terms([item]))
            )
            if design_highlights and covered_highlight_count < min(2, len(design_highlights)):
                for item in design_highlights:
                    if _contains_any_term(proposal.method, _design_support_terms([item])):
                        continue
                    proposal.method = _append_unique_sentence(
                        proposal.method,
                        _naturalize_design_highlight(item),
                    )
                    covered_highlight_count += 1
                    if covered_highlight_count >= min(2, len(design_highlights)):
                        break
        evaluation_lower = _normalize_match_text(proposal.evaluation)
        unsupported_markers = (
            "synthetic urban dataset",
            "lerf dataset",
            "3d ovs",
            "polycam",
            "scannet",
        )
        if any(marker in evaluation_lower for marker in unsupported_markers):
            proposal.evaluation = _compose_weak_context_evaluation(
                keyword,
                scaffold_evaluation_assets,
                scaffold_metric_items,
                scaffold_directions,
                scaffold_method_terms,
                scaffold_risk_items,
            )
        if weak_context_scaffold and (
            _contains_any_phrase(proposal.evaluation, generic_evaluation_markers)
            or "realistic benchmark tasks" in evaluation_lower
            or "task-specific quantitative metrics" in evaluation_lower
            or _contains_any_phrase(proposal.evaluation, weak_context_residue_markers)
        ):
            proposal.evaluation = _compose_weak_context_evaluation(
                keyword,
                scaffold_evaluation_assets,
                scaffold_metric_items,
                scaffold_directions,
                scaffold_method_terms,
                scaffold_risk_items,
            )
        if weak_context_scaffold:
            evaluation_assets = scaffold_evaluation_assets
            metric_items = scaffold_metric_items
            risk_items = scaffold_risk_items[:2]
            compare_directions = scaffold_directions
            matched_assets = _matching_terms(proposal.evaluation, evaluation_assets)
            matched_metrics = _matching_terms(proposal.evaluation, metric_items)
            if evaluation_assets and len(matched_assets) < len(evaluation_assets):
                missing_assets = [item for item in evaluation_assets if item not in matched_assets]
                if missing_assets:
                    proposal.evaluation = _append_unique_sentence(
                        proposal.evaluation,
                        f"Also validate on {_join_natural_strings(missing_assets)}.",
                    )
            if metric_items and len(matched_metrics) < len(metric_items):
                missing_metrics = [item for item in metric_items if item not in matched_metrics]
                if missing_metrics:
                    proposal.evaluation = _append_unique_sentence(
                        proposal.evaluation,
                        f"Also report {_join_natural_strings(missing_metrics)}.",
                    )
            if risk_items and not _contains_any_term(proposal.evaluation, risk_items):
                proposal.evaluation = _append_unique_sentence(
                    proposal.evaluation,
                    f"Stress test {_join_natural_strings(risk_items)}.",
                )
            if compare_directions and not _contains_any_phrase(proposal.evaluation, ("compare against", "baseline")):
                proposal.evaluation = _append_unique_sentence(
                    proposal.evaluation,
                    f"Compare against {_join_natural_strings(compare_directions)}.",
                )

    if (
        not keyword_only_mode
        and grounding.dataset_items
        and grounding.metric_items
        and _has_fragmentary_benchmark_plan(proposal.evaluation)
    ):
        proposal.evaluation = _compose_grounded_benchmark_evaluation(
            grounding.dataset_items,
            grounding.metric_items,
            grounding.reference_titles,
            design_anchor_terms,
        )

    proposal.method = _remove_exact_scaffold_method_sentences(
        proposal.method,
        grounding.design_highlights[:4],
    )
    proposal.hypothesis = _replace_weak_context_placeholder_phrases(proposal.hypothesis)
    proposal.existing_methods = _replace_weak_context_placeholder_phrases(proposal.existing_methods)
    proposal.method = _replace_weak_context_placeholder_phrases(proposal.method)
    proposal.evaluation = _replace_weak_context_placeholder_phrases(proposal.evaluation)
    proposal.caveats = _replace_weak_context_placeholder_phrases(proposal.caveats)
    proposal.problem = _clean_proposal_section_text(proposal.problem)
    proposal.hypothesis = _clean_proposal_section_text(proposal.hypothesis)
    proposal.existing_methods = _clean_proposal_section_text(proposal.existing_methods)
    proposal.method = _clean_proposal_section_text(proposal.method)
    proposal.evaluation = _clean_proposal_section_text(proposal.evaluation)
    proposal.caveats = _clean_proposal_section_text(proposal.caveats)
    return proposal


def _trace_payload(result: ChatCompletionResult, *, messages: list[dict[str, str]]) -> dict[str, object]:
    return {
        "request_messages": messages,
        "response_model": result.model,
        "raw_response": result.raw_response,
        "response_text": result.content,
    }


class OpenAICompatibleCollaborationBackend:
    name = "openai-compatible"

    def __init__(self, settings: OpenAICompatibleSettings) -> None:
        self.settings = settings
        self.client = OpenAICompatibleChatClient(settings)

    def _chat(self, *, role: str, messages: list[dict[str, str]]) -> ChatCompletionResult:
        return self.client.create_chat_completion(
            messages=messages,
            model=self.settings.model_for_role(role),
        )

    def _chat_json_object(self, *, role: str, messages: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, object]]:
        attempt_messages = list(messages)
        last_error = "Unknown JSON decoding failure."
        for attempt in range(self.settings.max_retries + 1):
            result = self._chat(role=role, messages=attempt_messages)
            trace = _trace_payload(result, messages=attempt_messages)
            try:
                payload = _extract_json_object(result.content)
                trace["attempt"] = attempt + 1
                return payload, trace
            except Exception as exc:
                last_error = str(exc)
                if attempt >= self.settings.max_retries:
                    raise
                attempt_messages = attempt_messages + [
                    {"role": "assistant", "content": result.content},
                    {
                        "role": "user",
                        "content": (
                            "Your last message was not valid for the required JSON protocol. "
                            "Return a single strict JSON object only, with no markdown, commentary, or code fences."
                        ),
                    },
                ]
        raise ValueError(last_error)

    def generate_seed(self, graph: IdeaGraph, role: str) -> SeedDraft:
        messages = [
            {"role": "system", "content": _seed_system_prompt(role)},
            {"role": "user", "content": _seed_user_prompt(graph, role)},
        ]
        payload, trace = self._chat_json_object(role=role, messages=messages)

        anchor = payload.get("anchor", {})
        if not isinstance(anchor, dict):
            raise ValueError("Seed payload must contain an object field 'anchor'.")
        anchor_type = _coerce_string(anchor.get("type"))
        anchor_text = _coerce_string(anchor.get("text"))
        if anchor_type not in NODE_TYPES:
            raise ValueError(f"Invalid anchor node type: {anchor_type}")
        if not anchor_text:
            raise ValueError("Seed anchor text may not be empty.")

        support_nodes_raw = payload.get("support_nodes", [])
        if not isinstance(support_nodes_raw, list) or not support_nodes_raw:
            raise ValueError("Seed payload must contain a non-empty list field 'support_nodes'.")

        support_nodes: list[SeedSupportDraft] = []
        for item in support_nodes_raw[:3]:
            if not isinstance(item, dict):
                continue
            node_type = _coerce_string(item.get("type"))
            relation = _coerce_string(item.get("relation_to_anchor"), "supports")
            text = _coerce_string(item.get("text"))
            if node_type not in NODE_TYPES or not text:
                continue
            if relation not in EDGE_TYPES:
                relation = "supports"
            support_nodes.append(
                SeedSupportDraft(
                    type=node_type,
                    text=text,
                    confidence=_clamp_confidence(item.get("confidence"), 0.64),
                    relation_to_anchor=relation,
                )
            )
        if not support_nodes:
            raise ValueError("Seed payload did not contain any valid support nodes.")

        return SeedDraft(
            anchor_type=anchor_type,
            anchor_text=anchor_text,
            anchor_confidence=_clamp_confidence(anchor.get("confidence"), 0.72),
            support_nodes=support_nodes,
            rationale=_coerce_string(payload.get("rationale")),
            trace=trace,
        )

    def choose_action(self, graph: IdeaGraph, round_name: str, role: str) -> ActionDecision:
        allowed_actions = _dynamic_allowed_actions(graph, round_name)
        messages = [
            {"role": "system", "content": _action_system_prompt(graph, role, round_name, allowed_actions)},
            {"role": "user", "content": _action_user_prompt(graph, round_name, role)},
        ]
        payload, trace = self._chat_json_object(role=role, messages=messages)

        kind = _coerce_string(payload.get("kind"))
        target_ids = payload.get("target_ids", [])
        if not isinstance(target_ids, list):
            raise ValueError("Action payload must contain a list field 'target_ids'.")
        normalized_target_ids = [_coerce_string(item) for item in target_ids if _coerce_string(item)]

        action_payload_raw = payload.get("payload", {})
        if not isinstance(action_payload_raw, dict):
            action_payload_raw = {}
        action_payload = _normalize_action_payload(
            graph,
            role=role,
            kind=kind,
            target_ids=normalized_target_ids,
            payload=action_payload_raw,
            rationale=_coerce_string(payload.get("rationale")),
        )
        kind, normalized_target_ids, action_payload, rationale = _salvage_action_decision(
            graph,
            role=role,
            kind=kind,
            target_ids=normalized_target_ids,
            payload=action_payload,
            rationale=_coerce_string(payload.get("rationale")),
        )
        if kind not in allowed_actions:
            raise ValueError(f"Invalid action kind '{kind}' for round {round_name}.")
        expected_target_count = ACTION_TARGET_COUNTS[kind]
        if len(normalized_target_ids) != expected_target_count:
            raise ValueError(
                f"Action kind '{kind}' for round {round_name} expects {expected_target_count} target ids, "
                f"but received {len(normalized_target_ids)}."
            )
        branch_id = _coerce_string(action_payload.get("branch_id"))
        if not branch_id:
            raise ValueError("Action payload must contain 'branch_id'.")
        for field_name in ACTION_REQUIRED_PAYLOAD_FIELDS.get(kind, ()):
            value = action_payload.get(field_name)
            if value is None or (isinstance(value, str) and not str(value).strip()):
                raise ValueError(f"Action payload must contain '{field_name}' for action kind '{kind}'.")

        if kind != "freeze_branch":
            for node_id in normalized_target_ids:
                if node_id not in graph.nodes:
                    raise ValueError(f"Action referenced unknown node id '{node_id}'.")
        else:
            if branch_id not in graph.branches:
                raise ValueError(f"Freeze action referenced unknown branch id '{branch_id}'.")

        _validate_action_semantics(
            graph,
            kind=kind,
            target_ids=normalized_target_ids,
            payload=action_payload,
        )

        return ActionDecision(
            kind=kind,
            target_ids=normalized_target_ids,
            payload={str(key): value for key, value in action_payload.items()},
            rationale=_coerce_string(rationale),
            trace=trace,
        )

    def synthesize_final_proposal(self, graph: IdeaGraph, subgraph: dict[str, object]) -> FinalProposal:
        messages = [
            {"role": "system", "content": _synthesis_system_prompt()},
            {"role": "user", "content": _synthesis_user_prompt(graph, subgraph)},
        ]
        payload, trace = self._chat_json_object(role=ROLE_NAMES[0], messages=messages)
        proposal = FinalProposal(
            title=_coerce_string(payload.get("title")),
            problem=_coerce_string(payload.get("problem")),
            existing_methods=_coerce_string(payload.get("existing_methods") or payload.get("existing_method")),
            motivation=_coerce_string(payload.get("motivation")),
            hypothesis=_coerce_string(payload.get("hypothesis")),
            method=_coerce_string(payload.get("method") or payload.get("proposed_method")),
            evaluation=_coerce_string(payload.get("evaluation") or payload.get("experiment_plan")),
            significance=_coerce_string(payload.get("significance") or payload.get("expected_contribution")),
            caveats=_coerce_string(payload.get("caveats") or payload.get("limitations")),
        )
        graph.metadata["final_synthesis_trace"] = trace
        return _postprocess_final_proposal(graph, proposal)
