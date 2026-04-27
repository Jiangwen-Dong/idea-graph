from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_candidates import flatten_candidate_text
from .critic_policy import (
    CriticPolicyDecision,
    SafeCriticPolicyConfig,
    ScoredCandidate,
    choose_critic_action,
)
from .models import Edge, IdeaGraph, Node
from .text_critic import _strip_leaky_segments


@dataclass(frozen=True)
class TextCriticRuntimeConfig:
    tau_override: float = 0.05
    tau_override_by_round: Mapping[int | str, float] | None = None
    tau_commit: float = 0.08
    gamma_commit: float = 0.50
    gamma_commit_by_round: Mapping[int | str, float] | None = None
    min_commit_round: int = 3
    use_commit: bool = False
    guard_support_threshold: float = 0.66
    guard_support_gain_floor: float = 0.10
    guard_requires_contradiction_progress: bool = False
    guard_commit_support_threshold: float = 0.0
    guard_commit_utility_floor: float = 0.0


@dataclass(frozen=True)
class TextCriticRuntimeDecision:
    selected_spec: dict[str, object]
    policy_decision: CriticPolicyDecision
    scored_candidates: tuple[dict[str, object], ...]


def _active_nodes(graph: IdeaGraph) -> list[Node]:
    return sorted(graph.active_nodes(), key=lambda node: node.id)


def _active_edges(graph: IdeaGraph, active_node_ids: set[str]) -> list[Edge]:
    edges = [
        edge
        for edge in graph.edges
        if edge.source_id in active_node_ids and edge.target_id in active_node_ids
    ]
    return sorted(edges, key=lambda edge: (edge.source_id, edge.relation, edge.target_id, edge.id))


def _unresolved_contradiction_count(graph: IdeaGraph, active_node_ids: set[str]) -> int:
    return sum(
        1
        for edge in graph.edges
        if edge.relation == "contradicts"
        and not edge.resolved
        and edge.source_id in active_node_ids
        and edge.target_id in active_node_ids
    )


def flatten_graph_state_text(graph: IdeaGraph) -> str:
    nodes = _active_nodes(graph)
    active_node_ids = {node.id for node in nodes}
    edges = _active_edges(graph, active_node_ids)

    node_fragments: list[str] = []
    for node in nodes:
        node_fragments.append(f"{node.id}:{node.type}:{node.text}")

    edge_fragments: list[str] = []
    for edge in edges:
        edge_fragments.append(f"{edge.source_id}-{edge.relation}->{edge.target_id}")

    return (
        f"nodes={len(nodes)};"
        f"edges={len(edges)};"
        f"contradictions={_unresolved_contradiction_count(graph, active_node_ids)};"
        f"node_details=[{' || '.join(node_fragments)}];"
        f"edge_details=[{' || '.join(edge_fragments)}]"
    )


def render_text_critic_input(state_text: str, candidate_text: str) -> str:
    return f"{state_text} [SEP] {_strip_leaky_segments(candidate_text)}"


def _parse_round_index(round_name: str) -> int:
    text = str(round_name).strip()
    if not text.lower().startswith("round"):
        return 0
    suffix = text[5:]
    try:
        return int(suffix)
    except ValueError:
        return 0


def _normalized_candidate_specs(
    candidate_specs: Sequence[Mapping[str, Any]],
    *,
    use_commit: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(candidate_specs):
        kind = str(spec.get("kind", "")).strip()
        if not use_commit and kind == "commit":
            continue
        row = dict(spec)
        row["candidate_id"] = str(spec.get("candidate_id", f"candidate:{index:03d}")).strip() or f"candidate:{index:03d}"
        rows.append(row)
    return rows


def _candidate_float(spec: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(spec.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _candidate_after_is_mature(spec: Mapping[str, Any]) -> bool:
    after_subgraph = spec.get("after_subgraph")
    if not isinstance(after_subgraph, Mapping):
        return False
    return bool(after_subgraph.get("is_mature", False))


def _policy_candidate_from_scored_spec(spec: Mapping[str, Any]) -> ScoredCandidate:
    return ScoredCandidate(
        candidate_id=str(spec["candidate_id"]),
        score=float(spec["critic_score"]),
        is_commit=str(spec.get("kind", "")).strip() == "commit",
        confidence=float(spec["critic_score"]),
        predicted_gain=_candidate_float(spec, "predicted_gain"),
        support_gain=_candidate_float(spec, "support_gain"),
        contradiction_gain=_candidate_float(spec, "contradiction_gain"),
        maturity_gain=_candidate_float(spec, "maturity_gain"),
        after_is_mature=_candidate_after_is_mature(spec),
    )


def select_text_critic_candidate(
    graph: IdeaGraph,
    *,
    round_name: str,
    role: str,
    state_features: Mapping[str, Any] | None = None,
    candidate_specs: Sequence[Mapping[str, Any]],
    heuristic_candidate_id: str,
    model: Any,
    config: TextCriticRuntimeConfig,
) -> TextCriticRuntimeDecision:
    normalized_specs = _normalized_candidate_specs(candidate_specs, use_commit=bool(config.use_commit))
    if not normalized_specs:
        raise ValueError("candidate_specs must not be empty after runtime-critic filtering.")

    state_text = flatten_graph_state_text(graph)
    candidate_texts = [flatten_candidate_text(graph, spec) for spec in normalized_specs]
    model_inputs = [render_text_critic_input(state_text, text) for text in candidate_texts]
    scores = model.score(model_inputs)
    if len(scores) != len(normalized_specs):
        raise ValueError(
            f"Runtime critic returned {len(scores)} scores for {len(normalized_specs)} candidate specs."
        )

    scored_candidates: list[dict[str, object]] = []
    scored_policy_candidates: list[ScoredCandidate] = []
    for spec, score in zip(normalized_specs, scores):
        candidate_id = str(spec["candidate_id"])
        score_value = float(score)
        scored_candidates.append(
            {
                **spec,
                "critic_score": score_value,
                "candidate_text": flatten_candidate_text(graph, spec),
                "after_is_mature": _candidate_after_is_mature(spec),
            }
        )
        scored_policy_candidates.append(
            _policy_candidate_from_scored_spec(scored_candidates[-1])
        )

    scored_lookup = {str(row["candidate_id"]): row for row in scored_candidates}
    heuristic_candidate = scored_lookup.get(str(heuristic_candidate_id).strip())
    if heuristic_candidate is None:
        raise ValueError(f"heuristic_candidate_id '{heuristic_candidate_id}' is not present in candidate_specs.")

    policy_decision = choose_critic_action(
        state={
            "round_index": _parse_round_index(round_name),
            **dict(state_features or {}),
        },
        critic_candidates=scored_policy_candidates,
        heuristic_candidate=_policy_candidate_from_scored_spec(heuristic_candidate),
        config=SafeCriticPolicyConfig(
            min_commit_round=int(config.min_commit_round),
            tau_override=float(config.tau_override),
            tau_override_by_round=dict(config.tau_override_by_round or {}),
            tau_commit=float(config.tau_commit),
            gamma_commit=float(config.gamma_commit),
            gamma_commit_by_round=dict(config.gamma_commit_by_round or {}),
            guard_support_threshold=float(config.guard_support_threshold),
            guard_support_gain_floor=float(config.guard_support_gain_floor),
            guard_requires_contradiction_progress=bool(
                config.guard_requires_contradiction_progress
            ),
            guard_commit_support_threshold=float(config.guard_commit_support_threshold),
            guard_commit_utility_floor=float(config.guard_commit_utility_floor),
        ),
    )
    selected_spec = dict(scored_lookup[policy_decision.selected_candidate_id])
    if policy_decision.fallback_reason and "controller_fallback_reason" not in selected_spec:
        selected_spec["controller_fallback_reason"] = policy_decision.fallback_reason
    return TextCriticRuntimeDecision(
        selected_spec=selected_spec,
        policy_decision=policy_decision,
        scored_candidates=tuple(scored_candidates),
    )


@lru_cache(maxsize=8)
def load_pickled_text_critic_model(model_path: str) -> Any:
    resolved = Path(model_path).resolve()
    with resolved.open("rb") as handle:
        return pickle.load(handle)
