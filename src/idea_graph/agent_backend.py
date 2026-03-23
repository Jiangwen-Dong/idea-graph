from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Protocol

from .collaboration_protocol import (
    ACTION_PROMPT_HINTS,
    ACTION_REQUIRED_PAYLOAD_FIELDS,
    ACTION_TARGET_COUNTS,
    resolve_round_phase,
)
from .llm import ChatCompletionResult, OpenAICompatibleChatClient
from .models import FinalProposal, IdeaGraph
from .schema import EDGE_TYPES, NODE_TYPES, ROLE_NAMES
from .settings import OpenAICompatibleSettings

ROLE_GUIDANCE = {
    "MechanismProposer": "Prefer hypotheses and methods that define a concrete research mechanism.",
    "FeasibilityCritic": "Prefer risks, assumptions, and evaluation pressure tests that surface practical failure modes.",
    "NoveltyExaminer": "Prefer novelty claims, overlap checks, and evidence requests grounded in nearby prior work.",
    "EvaluationDesigner": "Prefer evaluations, datasets, metrics, and dependency structure that make claims testable.",
    "ImpactReframer": "Prefer problem framing, significance, and alternative research directions with interpretable tradeoffs.",
}

ROLE_PREFERRED_ANCHOR_TYPES = {
    "MechanismProposer": ("Hypothesis", "Method", "Assumption"),
    "FeasibilityCritic": ("Risk", "Assumption", "EvalPlan"),
    "NoveltyExaminer": ("NoveltyClaim", "EvidenceNeed", "Problem"),
    "EvaluationDesigner": ("EvalPlan", "Method", "Risk"),
    "ImpactReframer": ("Problem", "Hypothesis", "NoveltyClaim"),
}


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
            "text": node.text,
            "role": node.role,
            "branch_id": node.branch_id,
            "confidence": node.confidence,
            "evidence": list(node.evidence),
        }
        for node in graph.active_nodes()
    ]
    active_edges = [
        edge
        for edge in all_edges[-40:]
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
    ][:6]
    candidate_node_ids_by_type = {
        node_type: [node["id"] for node in active_nodes if node["type"] == node_type][:6]
        for node_type in NODE_TYPES
    }
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
        "literature": graph.literature[:8],
        "metadata": {key: value for key, value in graph.metadata.items() if key != "agent_traces"},
        "maturity_snapshot": {
            "support_coverage": support_coverage,
            "unresolved_contradiction_ratio": unresolved_ratio,
            "completeness": {"Problem", "Hypothesis", "Method", "EvalPlan"}.issubset(active_types),
        },
        "branches": branches[:5],
        "nodes": active_nodes[:30],
        "edges": active_edges,
        "unsupported_novelty_claims": unsupported_novelty,
        "unresolved_contradictions": unresolved_contradictions,
        "tracked_nodes_needing_support": tracked_nodes_needing_support,
        "tracked_nodes_needing_evidence": tracked_nodes_needing_evidence,
        "candidate_node_ids_by_type": candidate_node_ids_by_type,
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


def _seed_system_prompt(role: str) -> str:
    preferred_anchor_types = ROLE_PREFERRED_ANCHOR_TYPES.get(role, tuple(NODE_TYPES))
    return (
        f"You are the {role} agent in a delayed-consensus scientific ideation system. "
        f"{ROLE_GUIDANCE.get(role, '')} "
        "Return strict JSON only. Do not use markdown. "
        "Create one partial seed graph, not a complete abstract. "
        f"Allowed node types: {', '.join(NODE_TYPES)}. "
        f"Preferred anchor types for your role: {', '.join(preferred_anchor_types)}. "
        "Choose an anchor type that best fits your role and the topic; do not default to one type unless the context supports it. "
        "Support nodes should add complementary structure such as method, evidence needs, risks, or evaluation signals rather than repeating the anchor. "
        "Output schema: "
        '{"anchor":{"type":"<one allowed node type>","text":"...","confidence":0.7},'
        '"support_nodes":[{"type":"<one allowed node type>","text":"...","confidence":0.6,"relation_to_anchor":"supports|contradicts|depends_on|requires_evidence|overlaps_prior|repairs|refines"}],'
        '"rationale":"One short sentence explaining why this partial graph is useful now."}'
    )


def _seed_user_prompt(graph: IdeaGraph, role: str) -> str:
    payload = {
        "role": role,
        "topic": graph.topic,
        "literature": graph.literature[:8],
        "metadata": {key: value for key, value in graph.metadata.items() if key != "agent_traces"},
        "constraints": {
            "exactly_one_anchor": True,
            "support_node_count_range": [1, 3],
            "no_full_abstract": True,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _action_system_prompt(role: str, round_name: str) -> str:
    phase = resolve_round_phase(round_name)
    action_lines = []
    for action_kind in phase.allowed_actions:
        hint = ACTION_PROMPT_HINTS[action_kind]
        target_shape = hint["target_shape"]
        payload_fields = hint["payload_fields"]
        target_text = "[]" if not target_shape else "[" + ", ".join(target_shape) + "]"
        payload_text = ", ".join(payload_fields)
        action_lines.append(
            f"- {action_kind}: target_ids={target_text}; payload fields={payload_text}; {hint['when_to_use']}"
        )
    decision_focus = " ".join(phase.decision_focus)
    return (
        f"You are the {role} agent in round {round_name} of delayed-consensus scientific collaboration. "
        f"Current phase: {phase.title}. "
        f"Round objective: {phase.objective} "
        f"{ROLE_GUIDANCE.get(role, '')} "
        "Choose exactly one action that best improves the graph at this moment. "
        "Do not default to a specific action kind just because it appears first in a schema. "
        f"{decision_focus} "
        "Return strict JSON only. Do not use markdown. "
        f"Allowed action kinds for this phase: {', '.join(phase.allowed_actions)}. "
        "Use existing node IDs and branch IDs from the provided graph snapshot. "
        "Your rationale should explain why this action is higher leverage than the other allowed actions right now. "
        "Action schemas:\n"
        + "\n".join(action_lines)
        + "\nOutput schema: "
        '{"kind":"<one allowed action kind>","target_ids":["..."],"payload":{"branch_id":"..."},'
        '"rationale":"One short sentence grounded in the current graph state."}'
    )


def _action_user_prompt(graph: IdeaGraph, round_name: str, role: str) -> str:
    phase = resolve_round_phase(round_name)
    action_requirements = {
        action_kind: {
            "target_count": ACTION_TARGET_COUNTS[action_kind],
            "required_payload_fields": list(ACTION_REQUIRED_PAYLOAD_FIELDS.get(action_kind, ())),
            "prompt_hint": ACTION_PROMPT_HINTS[action_kind]["when_to_use"],
        }
        for action_kind in phase.allowed_actions
    }
    payload = {
        "round": round_name,
        "phase": phase.key,
        "phase_title": phase.title,
        "allowed_actions": list(phase.allowed_actions),
        "action_requirements": action_requirements,
        "focused_view": focused_view_for_prompt(graph, role),
        "decision_instruction": (
            "Pick one action that most improves maturity, grounding, contradiction handling, or branch quality. "
            "If unresolved contradictions are present, consider whether a repair or evidence action is more useful than adding another support edge. "
            "If support edges already exist for the same pair, avoid duplicating them."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _synthesis_system_prompt() -> str:
    return (
        "You are the synthesis module for a multi-agent scientific ideation graph. "
        "Return strict JSON only. Do not use markdown. "
        "Write a concise structured proposal from the selected subgraph. "
        'JSON schema: {"problem":"...","hypothesis":"...","method":"...","evaluation":"...",'
        '"significance":"...","caveats":"..."}'
    )


def _synthesis_user_prompt(graph: IdeaGraph, subgraph: dict[str, object]) -> str:
    node_ids = [str(item) for item in subgraph.get("node_ids", [])]
    edge_ids = [str(item) for item in subgraph.get("edge_ids", [])]
    payload = {
        "topic": graph.topic,
        "literature": graph.literature[:8],
        "metadata": {key: value for key, value in graph.metadata.items() if key != "agent_traces"},
        "selected_nodes": [
            {
                "id": node_id,
                "type": graph.nodes[node_id].type,
                "text": graph.nodes[node_id].text,
                "evidence": list(graph.nodes[node_id].evidence),
            }
            for node_id in node_ids
            if node_id in graph.nodes
        ],
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
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


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
        messages = [
            {"role": "system", "content": _action_system_prompt(role, round_name)},
            {"role": "user", "content": _action_user_prompt(graph, round_name, role)},
        ]
        payload, trace = self._chat_json_object(role=role, messages=messages)

        phase = resolve_round_phase(round_name)
        kind = _coerce_string(payload.get("kind"))
        if kind not in phase.allowed_actions:
            raise ValueError(f"Invalid action kind '{kind}' for round {round_name}.")

        target_ids = payload.get("target_ids", [])
        if not isinstance(target_ids, list):
            raise ValueError("Action payload must contain a list field 'target_ids'.")
        normalized_target_ids = [_coerce_string(item) for item in target_ids if _coerce_string(item)]
        expected_target_count = ACTION_TARGET_COUNTS[kind]
        if len(normalized_target_ids) != expected_target_count:
            raise ValueError(
                f"Action kind '{kind}' for round {round_name} expects {expected_target_count} target ids, "
                f"but received {len(normalized_target_ids)}."
            )

        action_payload = payload.get("payload", {})
        if not isinstance(action_payload, dict):
            action_payload = {}
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

        return ActionDecision(
            kind=kind,
            target_ids=normalized_target_ids,
            payload={str(key): value for key, value in action_payload.items()},
            rationale=_coerce_string(payload.get("rationale")),
            trace=trace,
        )

    def synthesize_final_proposal(self, graph: IdeaGraph, subgraph: dict[str, object]) -> FinalProposal:
        messages = [
            {"role": "system", "content": _synthesis_system_prompt()},
            {"role": "user", "content": _synthesis_user_prompt(graph, subgraph)},
        ]
        payload, trace = self._chat_json_object(role=ROLE_NAMES[0], messages=messages)
        proposal = FinalProposal(
            problem=_coerce_string(payload.get("problem")),
            hypothesis=_coerce_string(payload.get("hypothesis")),
            method=_coerce_string(payload.get("method")),
            evaluation=_coerce_string(payload.get("evaluation")),
            significance=_coerce_string(payload.get("significance")),
            caveats=_coerce_string(payload.get("caveats")),
        )
        graph.metadata["final_synthesis_trace"] = trace
        return proposal
