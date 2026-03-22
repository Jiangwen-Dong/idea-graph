from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Protocol

from .llm import ChatCompletionResult, OpenAICompatibleChatClient
from .models import FinalProposal, IdeaGraph
from .schema import EDGE_TYPES, NODE_TYPES, ROLE_NAMES
from .settings import OpenAICompatibleSettings


ROUND_ALLOWED_ACTIONS = {
    "Round1": (
        "add_support_edge",
        "add_contradiction_edge",
        "add_dependency_edge",
        "mark_overlap",
        "request_evidence",
    ),
    "Round2": (
        "attach_evidence",
        "mark_overlap",
        "request_evidence",
        "add_support_edge",
        "add_contradiction_edge",
    ),
    "Round3": (
        "propose_repair",
        "add_support_edge",
        "freeze_branch",
        "attach_evidence",
    ),
}

ROUND_OBJECTIVES = {
    "Round1": "Expose the graph structure by adding support, contradiction, dependency, overlap, or evidence-request signals.",
    "Round2": "Stress-test branches created by other roles using evidence attachment, overlap marking, or critique.",
    "Round3": "Repair or recombine the graph into stronger branches without collapsing into early whole-idea voting.",
}

ROLE_GUIDANCE = {
    "MechanismProposer": "Prefer hypotheses and methods that define a concrete research mechanism.",
    "FeasibilityCritic": "Prefer risks, assumptions, and evaluation pressure tests that surface practical failure modes.",
    "NoveltyExaminer": "Prefer novelty claims, overlap checks, and evidence requests grounded in nearby prior work.",
    "EvaluationDesigner": "Prefer evaluations, datasets, metrics, and dependency structure that make claims testable.",
    "ImpactReframer": "Prefer problem framing, significance, and alternative research directions with interpretable tradeoffs.",
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
        {
            "id": edge.id,
            "source_id": edge.source_id,
            "relation": edge.relation,
            "target_id": edge.target_id,
            "resolved": edge.resolved,
            "role": edge.role,
        }
        for edge in graph.edges[-40:]
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
        node["id"]
        for node in active_nodes
        if node["type"] == "NoveltyClaim"
        and not any(edge["relation"] == "supports" and edge["target_id"] == node["id"] for edge in active_edges)
    ][:3]
    unresolved_contradictions = [
        edge["id"] for edge in active_edges if edge["relation"] == "contradicts" and not edge["resolved"]
    ][:5]
    candidate_node_ids_by_type = {
        node_type: [node["id"] for node in active_nodes if node["type"] == node_type][:6]
        for node_type in NODE_TYPES
    }

    return {
        "role": role,
        "role_guidance": ROLE_GUIDANCE.get(role, ""),
        "topic": graph.topic,
        "literature": graph.literature[:8],
        "metadata": {key: value for key, value in graph.metadata.items() if key != "agent_traces"},
        "branches": branches[:5],
        "nodes": active_nodes[:30],
        "edges": active_edges,
        "unsupported_novelty_claims": unsupported_novelty,
        "unresolved_contradictions": unresolved_contradictions,
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
    return (
        f"You are the {role} agent in a delayed-consensus scientific ideation system. "
        f"{ROLE_GUIDANCE.get(role, '')} "
        "Return strict JSON only. Do not use markdown. "
        "Create one partial seed graph, not a complete abstract. "
        f"Allowed node types: {', '.join(NODE_TYPES)}. "
        "JSON schema: "
        '{"anchor":{"type":"Hypothesis","text":"...","confidence":0.7},'
        '"support_nodes":[{"type":"Method","text":"...","confidence":0.6,"relation_to_anchor":"supports"}],'
        '"rationale":"..."}'
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
    allowed = ROUND_ALLOWED_ACTIONS[round_name]
    return (
        f"You are the {role} agent in round {round_name} of delayed-consensus scientific collaboration. "
        f"Round objective: {ROUND_OBJECTIVES[round_name]} "
        f"{ROLE_GUIDANCE.get(role, '')} "
        "Return strict JSON only. Do not use markdown. "
        f"Allowed action kinds for this round: {', '.join(allowed)}. "
        "Use existing node IDs and branch IDs from the provided graph snapshot. "
        "JSON schema: "
        '{"kind":"add_support_edge","target_ids":["N001","N002"],"payload":{"branch_id":"B001"},'
        '"rationale":"..."}'
    )


def _action_user_prompt(graph: IdeaGraph, round_name: str, role: str) -> str:
    payload = {
        "round": round_name,
        "allowed_actions": list(ROUND_ALLOWED_ACTIONS[round_name]),
        "focused_view": focused_view_for_prompt(graph, role),
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

        kind = _coerce_string(payload.get("kind"))
        if kind not in ROUND_ALLOWED_ACTIONS[round_name]:
            raise ValueError(f"Invalid action kind '{kind}' for round {round_name}.")

        target_ids = payload.get("target_ids", [])
        if not isinstance(target_ids, list):
            raise ValueError("Action payload must contain a list field 'target_ids'.")
        normalized_target_ids = [_coerce_string(item) for item in target_ids if _coerce_string(item)]

        action_payload = payload.get("payload", {})
        if not isinstance(action_payload, dict):
            action_payload = {}
        branch_id = _coerce_string(action_payload.get("branch_id"))
        if not branch_id:
            raise ValueError("Action payload must contain 'branch_id'.")

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
