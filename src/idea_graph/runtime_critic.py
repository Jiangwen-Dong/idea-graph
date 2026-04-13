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
    tau_commit: float = 0.08
    gamma_commit: float = 0.60
    min_commit_round: int = 2
    use_commit: bool = False


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


def select_text_critic_candidate(
    graph: IdeaGraph,
    *,
    round_name: str,
    role: str,
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
            }
        )
        scored_policy_candidates.append(
            ScoredCandidate(
                candidate_id=candidate_id,
                score=score_value,
                is_commit=str(spec.get("kind", "")).strip() == "commit",
                confidence=score_value,
            )
        )

    scored_lookup = {str(row["candidate_id"]): row for row in scored_candidates}
    heuristic_candidate = scored_lookup.get(str(heuristic_candidate_id).strip())
    if heuristic_candidate is None:
        raise ValueError(f"heuristic_candidate_id '{heuristic_candidate_id}' is not present in candidate_specs.")

    policy_decision = choose_critic_action(
        state={"round_index": _parse_round_index(round_name)},
        critic_candidates=scored_policy_candidates,
        heuristic_candidate=ScoredCandidate(
            candidate_id=str(heuristic_candidate["candidate_id"]),
            score=float(heuristic_candidate["critic_score"]),
            is_commit=False,
            confidence=float(heuristic_candidate["critic_score"]),
        ),
        config=SafeCriticPolicyConfig(
            min_commit_round=int(config.min_commit_round),
            tau_override=float(config.tau_override),
            tau_commit=float(config.tau_commit),
            gamma_commit=float(config.gamma_commit),
        ),
    )
    selected_spec = scored_lookup[policy_decision.selected_candidate_id]
    return TextCriticRuntimeDecision(
        selected_spec=dict(selected_spec),
        policy_decision=policy_decision,
        scored_candidates=tuple(scored_candidates),
    )


@lru_cache(maxsize=8)
def load_pickled_text_critic_model(model_path: str) -> Any:
    resolved = Path(model_path).resolve()
    with resolved.open("rb") as handle:
        return pickle.load(handle)
