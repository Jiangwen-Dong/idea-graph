from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Callable

from .agent_backend import OpenAICompatibleCollaborationBackend
from .benchmark_mode import apply_io_mode
from .engine import emit_progress, run_experiment
from .external_baselines import run_external_baseline
from .literature_grounding import build_literature_grounding
from .models import FinalProposal, IdeaGraph


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    display_name: str
    strategy: str
    description: str
    is_proxy: bool = False
    proxy_target: str = ""
    prompt_style: str = ""
    candidate_count: int = 1


BASELINE_SPECS: dict[str, BaselineSpec] = {
    "ours-delayed-consensus": BaselineSpec(
        name="ours-delayed-consensus",
        display_name="Ours Delayed Consensus",
        strategy="delayed_consensus",
        description="Typed-graph multi-agent collaboration with delayed consensus.",
        prompt_style="ours",
    ),
    "direct": BaselineSpec(
        name="direct",
        display_name="Direct",
        strategy="direct",
        description="One-pass single-agent structured idea generation.",
        prompt_style="direct",
    ),
    "self-refine": BaselineSpec(
        name="self-refine",
        display_name="Self Refine",
        strategy="self_refine",
        description="Single-agent draft, critique, and revision baseline.",
        prompt_style="self_refine",
    ),
    "ai-researcher": BaselineSpec(
        name="ai-researcher",
        display_name="AI-Researcher",
        strategy="external",
        description="External wrapper that runs the official AI-Researcher ideation pipeline from its upstream repository.",
    ),
    "scipip": BaselineSpec(
        name="scipip",
        display_name="SciPIP",
        strategy="external",
        description="External wrapper that runs the official SciPIP pipeline from its upstream repository.",
    ),
    "virsci": BaselineSpec(
        name="virsci",
        display_name="VirSci",
        strategy="external",
        description="External wrapper that runs the official Virtual-Scientists pipeline from its upstream repository when the task setting is supported.",
    ),
    "ai-researcher-proxy": BaselineSpec(
        name="ai-researcher-proxy",
        display_name="AI-Researcher Proxy",
        strategy="candidate_rank",
        description="Local proxy wrapper for the AI-Researcher ideation pipeline with literature-grounded candidate generation and selection.",
        is_proxy=True,
        proxy_target="AI-Researcher",
        prompt_style="ai_researcher_proxy",
        candidate_count=4,
    ),
    "scipip-proxy": BaselineSpec(
        name="scipip-proxy",
        display_name="SciPIP Proxy",
        strategy="direct",
        description="Local proxy wrapper emphasizing structured motivation and experiment decomposition.",
        is_proxy=True,
        proxy_target="SciPIP",
        prompt_style="scipip_proxy",
    ),
    "virsci-proxy": BaselineSpec(
        name="virsci-proxy",
        display_name="VirSci Proxy",
        strategy="delayed_consensus",
        description="Local proxy wrapper for a discussion-oriented multi-agent baseline.",
        is_proxy=True,
        proxy_target="VirSci",
        prompt_style="virsci_proxy",
    ),
}


PROMPT_STYLE_GUIDANCE = {
    "ours": (
        "Preserve delayed consensus, typed intermediate claims, disagreement tracking, and section-level rigor."
    ),
    "direct": (
        "Produce one concise, strong idea directly from the provided packet without extra self-critique."
    ),
    "self_refine": (
        "Produce a strong first draft, then use explicit critique to revise weak sections."
    ),
    "ai_researcher_proxy": (
        "Emphasize literature-grounded candidate generation, proposal elaboration, diversity across ideas, and selective ranking."
    ),
    "scipip_proxy": (
        "Emphasize structured decomposition from topic and inspiration context into motivation and experiment plan."
    ),
    "virsci_proxy": (
        "Emphasize diverse agent perspectives, discussion-style synthesis, and explicit tradeoffs across alternatives."
    ),
}


def get_baseline_spec(name: str) -> BaselineSpec:
    try:
        return BASELINE_SPECS[name]
    except KeyError as exc:
        options = ", ".join(sorted(BASELINE_SPECS))
        raise KeyError(f"Unknown baseline '{name}'. Available baselines: {options}") from exc


def baseline_choices() -> list[str]:
    return sorted(BASELINE_SPECS)


def attach_baseline_metadata(
    instance,
    *,
    baseline_name: str,
    io_mode: str = "auto",
):
    baseline = get_baseline_spec(baseline_name)
    instance = apply_io_mode(instance, io_mode=io_mode)
    metadata = dict(instance.metadata)
    metadata["baseline_name"] = baseline.name
    metadata["baseline_display_name"] = baseline.display_name
    metadata["baseline_strategy"] = baseline.strategy
    metadata["baseline_prompt_style"] = baseline.prompt_style
    metadata["baseline_description"] = baseline.description
    metadata["baseline_proxy"] = baseline.is_proxy
    metadata["baseline_proxy_target"] = baseline.proxy_target
    return instance.__class__(
        name=instance.name,
        topic=instance.topic,
        literature=list(instance.literature),
        source_path=instance.source_path,
        metadata=metadata,
    )


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


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


def _coerce_string(value: Any) -> str:
    return _clean_text(value)


def _first_sentence(text: Any) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    for separator in (". ", "! ", "? "):
        if separator in cleaned:
            return cleaned.split(separator, 1)[0].strip().rstrip(".!?") + "."
    return cleaned.rstrip(".!?") + "."


def _proposal_from_payload(payload: dict[str, Any]) -> FinalProposal:
    return FinalProposal(
        title=_coerce_string(payload.get("title")),
        abstract="",
        problem=_coerce_string(payload.get("problem")),
        existing_methods=_coerce_string(payload.get("existing_methods") or payload.get("existing_method")),
        motivation=_coerce_string(payload.get("motivation")),
        hypothesis=_coerce_string(payload.get("hypothesis") or payload.get("core_idea")),
        method=_coerce_string(payload.get("method") or payload.get("method_sketch")),
        evaluation=_coerce_string(payload.get("evaluation") or payload.get("experiment_plan")),
        significance=_coerce_string(payload.get("significance") or payload.get("expected_contribution")),
        caveats=_coerce_string(payload.get("caveats") or payload.get("risk") or payload.get("risks")),
    )


def _proposal_as_prompt_payload(proposal: FinalProposal) -> dict[str, str]:
    return {
        "title": proposal.title,
        "problem": proposal.problem,
        "existing_methods": proposal.existing_methods,
        "motivation": proposal.motivation,
        "hypothesis": proposal.hypothesis,
        "method": proposal.method,
        "evaluation": proposal.evaluation,
        "significance": proposal.significance,
        "caveats": proposal.caveats,
    }


def _topic_text(graph: IdeaGraph) -> str:
    cleaned = _clean_text(graph.topic).rstrip(".")
    prefix = "The topic of this paper is "
    if cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix) :].strip()
    return cleaned or _clean_text(graph.topic)


def _deterministic_direct_proposal(graph: IdeaGraph, baseline: BaselineSpec) -> FinalProposal:
    generation_metadata = graph.metadata.get("generation_safe_metadata", graph.metadata)
    if not isinstance(generation_metadata, dict):
        generation_metadata = graph.metadata
    grounding = build_literature_grounding(literature=graph.literature, metadata=generation_metadata)
    topic_text = _topic_text(graph)
    benchmark_packet = graph.metadata.get("benchmark_input_packet", {})
    task_instruction = _coerce_string(
        benchmark_packet.get("task_instruction") if isinstance(benchmark_packet, dict) else ""
    )
    references = []
    if isinstance(benchmark_packet, dict):
        references = benchmark_packet.get("reference_packet", [])
    reference_titles = []
    if isinstance(references, list):
        for item in references:
            if isinstance(item, dict):
                title = _coerce_string(item.get("title"))
                if title:
                    reference_titles.append(title)

    existing_methods = grounding.existing_methods_summary
    if not existing_methods and reference_titles:
        existing_methods = (
            "Relevant nearby work includes "
            + ", ".join(reference_titles[:3])
            + ", but the current benchmark packet does not expose their full target-paper labels."
        )
    if not existing_methods:
        existing_methods = (
            f"Existing methods for {topic_text} remain only partially grounded in the current context packet."
        )

    motivation = _coerce_string(generation_metadata.get("motivation"))
    if not motivation:
        motivation = (
            f"The benchmark task asks for a concrete and testable idea for {topic_text}. "
            f"{task_instruction or 'The output should remain grounded in the provided context.'}"
        )

    method = (
        f"Design a concise method for {topic_text} that combines a clear mechanism, explicit evaluation hooks, "
        "and literature-aware comparison points instead of only high-level brainstorming."
    )
    if baseline.prompt_style == "scipip_proxy":
        method = (
            f"Decompose {topic_text} into a structured motivation, method sketch, and experiment plan derived "
            "from the benchmark topic and reference packet."
        )

    evaluation = grounding.experiment_plan_summary or (
        f"Evaluate the idea for {topic_text} with strong baselines, benchmark-relevant datasets or tasks, "
        "and targeted ablations that isolate the main proposed mechanism."
    )

    return FinalProposal(
        title=topic_text[:1].upper() + topic_text[1:] if topic_text else baseline.display_name,
        abstract="",
        problem=f"{topic_text} still lacks a compact, testable research formulation in the current benchmark setting.",
        existing_methods=existing_methods,
        motivation=motivation,
        hypothesis=f"A more explicitly structured idea for {topic_text} can improve testability and scientific usefulness.",
        method=method,
        evaluation=evaluation,
        significance=(
            f"If successful, the idea would provide a clearer and more benchmark-faithful research direction for {topic_text}."
        ),
        caveats="The current baseline uses only the benchmark packet and may miss broader external literature context.",
    )


def _deterministic_refine_proposal(graph: IdeaGraph, draft: FinalProposal, baseline: BaselineSpec) -> FinalProposal:
    grounding = build_literature_grounding(literature=graph.literature, metadata=graph.metadata)
    existing_methods = draft.existing_methods
    if grounding.existing_methods_summary and grounding.existing_methods_summary not in existing_methods:
        existing_methods = grounding.existing_methods_summary

    evaluation = draft.evaluation
    if grounding.experiment_plan_summary and grounding.experiment_plan_summary not in evaluation:
        evaluation = grounding.experiment_plan_summary

    caveats = draft.caveats
    if "test" not in caveats.casefold():
        caveats = (
            caveats.rstrip(".")
            + ". Further stress tests should check whether the idea still holds under stronger baselines and ablations."
        )

    significance = draft.significance
    if baseline.prompt_style == "ai_researcher_proxy" and "literature" not in significance.casefold():
        significance = significance.rstrip(".") + ". It should also improve literature-grounded ideation quality."

    return FinalProposal(
        title=draft.title,
        abstract="",
        problem=draft.problem,
        existing_methods=existing_methods,
        motivation=draft.motivation,
        hypothesis=draft.hypothesis,
        method=draft.method,
        evaluation=evaluation,
        significance=significance,
        caveats=caveats,
    )


def _direct_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are a scientific idea generation baseline. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        "Generate exactly one structured research idea using the provided benchmark packet and output schema. "
        "Do not assume access to hidden target-paper labels. "
        'JSON schema: {"title":"...","problem":"...","existing_methods":"...","motivation":"...",'
        '"hypothesis":"...","method":"...","evaluation":"...","significance":"...","caveats":"..."}'
    )


def _direct_user_prompt(graph: IdeaGraph, baseline: BaselineSpec) -> str:
    packet = graph.metadata.get("benchmark_input_packet", {})
    payload = {
        "baseline": asdict(baseline),
        "topic": graph.topic,
        "task_instruction": graph.metadata.get("task_instruction", ""),
        "input_packet": packet,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _critique_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are a scientific idea critic. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        "Critique the current draft using only the benchmark packet and return concise revision guidance. "
        'JSON schema: {"strengths":["..."],"weaknesses":["..."],"revision_focus":["..."]}'
    )


def _critique_user_prompt(graph: IdeaGraph, baseline: BaselineSpec, draft: FinalProposal) -> str:
    payload = {
        "baseline": asdict(baseline),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "draft": _proposal_as_prompt_payload(draft),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _refine_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are revising a scientific research idea after critique. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        "Revise the draft to improve grounding, coherence, and testability while keeping the output concise. "
        'JSON schema: {"title":"...","problem":"...","existing_methods":"...","motivation":"...",'
        '"hypothesis":"...","method":"...","evaluation":"...","significance":"...","caveats":"..."}'
    )


def _refine_user_prompt(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    draft: FinalProposal,
    critique_payload: dict[str, Any],
) -> str:
    payload = {
        "baseline": asdict(baseline),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "draft": _proposal_as_prompt_payload(draft),
        "critique": critique_payload,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _candidate_generation_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    candidate_count = max(2, baseline.candidate_count)
    return (
        "You are generating diverse scientific research ideas for a baseline wrapper. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        f"Generate exactly {candidate_count} different structured idea candidates using only the provided benchmark packet. "
        "Make the candidates meaningfully different in mechanism, framing, or evaluation strategy rather than rephrasing the same idea. "
        "Do not assume access to hidden target-paper labels. "
        'JSON schema: {"candidates":[{"title":"...","problem":"...","existing_methods":"...","motivation":"...",'
        '"hypothesis":"...","method":"...","evaluation":"...","significance":"...","caveats":"..."}]}'
    )


def _candidate_generation_user_prompt(graph: IdeaGraph, baseline: BaselineSpec) -> str:
    payload = {
        "baseline": asdict(baseline),
        "topic": graph.topic,
        "task_instruction": graph.metadata.get("task_instruction", ""),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "candidate_count": max(2, baseline.candidate_count),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _candidate_selection_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are ranking candidate scientific research ideas for a baseline wrapper. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        "Select the single best candidate using novelty, significance, feasibility, clarity, and topic fit. "
        "Prefer the candidate with the strongest overall research promise rather than the longest text. "
        'JSON schema: {"selected_index":0,"reason":"...","scores":[{"index":0,"novelty":1,"significance":1,"feasibility":1,"clarity":1,"topic_fit":1,"overall":1}]}'
    )


def _candidate_selection_user_prompt(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    candidates: list[FinalProposal],
) -> str:
    payload = {
        "baseline": asdict(baseline),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "candidates": [
            {"index": index, **_proposal_as_prompt_payload(candidate)}
            for index, candidate in enumerate(candidates)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _llm_direct_proposal(graph: IdeaGraph, baseline: BaselineSpec, backend: OpenAICompatibleCollaborationBackend) -> FinalProposal:
    result = backend.client.create_chat_completion(
        messages=[
            {"role": "system", "content": _direct_system_prompt(baseline)},
            {"role": "user", "content": _direct_user_prompt(graph, baseline)},
        ],
        model=backend.settings.model,
    )
    payload = _extract_json_object(result.content)
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "direct_generation", "baseline": baseline.name, "raw_response": result.raw_response}
    )
    return _proposal_from_payload(payload)


def _llm_candidate_rank_proposal(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    backend: OpenAICompatibleCollaborationBackend,
) -> FinalProposal:
    generation_result = backend.client.create_chat_completion(
        messages=[
            {"role": "system", "content": _candidate_generation_system_prompt(baseline)},
            {"role": "user", "content": _candidate_generation_user_prompt(graph, baseline)},
        ],
        model=backend.settings.model,
    )
    generation_payload = _extract_json_object(generation_result.content)
    raw_candidates = generation_payload.get("candidates", [])
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("Candidate-generation response did not contain a non-empty 'candidates' list.")

    candidates = [
        _proposal_from_payload(item)
        for item in raw_candidates
        if isinstance(item, dict)
    ]
    if not candidates:
        raise ValueError("Candidate-generation response did not contain any valid proposal objects.")

    graph.metadata.setdefault("baseline_traces", []).append(
        {
            "stage": "candidate_generation",
            "baseline": baseline.name,
            "raw_response": generation_result.raw_response,
        }
    )

    selection_result = backend.client.create_chat_completion(
        messages=[
            {"role": "system", "content": _candidate_selection_system_prompt(baseline)},
            {"role": "user", "content": _candidate_selection_user_prompt(graph, baseline, candidates)},
        ],
        model=backend.settings.model,
    )
    selection_payload = _extract_json_object(selection_result.content)
    graph.metadata.setdefault("baseline_traces", []).append(
        {
            "stage": "candidate_selection",
            "baseline": baseline.name,
            "raw_response": selection_result.raw_response,
        }
    )
    try:
        selected_index = int(selection_payload.get("selected_index", 0))
    except (TypeError, ValueError):
        selected_index = 0
    if not (0 <= selected_index < len(candidates)):
        selected_index = 0

    selected = candidates[selected_index]
    selection_reason = _coerce_string(selection_payload.get("reason"))
    if selection_reason:
        graph.metadata["baseline_selection_reason"] = selection_reason
    return selected


def _llm_self_refine_proposal(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    backend: OpenAICompatibleCollaborationBackend,
) -> FinalProposal:
    draft = _llm_direct_proposal(graph, baseline, backend)
    critique_result = backend.client.create_chat_completion(
        messages=[
            {"role": "system", "content": _critique_system_prompt(baseline)},
            {"role": "user", "content": _critique_user_prompt(graph, baseline, draft)},
        ],
        model=backend.settings.model,
    )
    critique_payload = _extract_json_object(critique_result.content)
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "self_refine_critique", "baseline": baseline.name, "raw_response": critique_result.raw_response}
    )
    refine_result = backend.client.create_chat_completion(
        messages=[
            {"role": "system", "content": _refine_system_prompt(baseline)},
            {"role": "user", "content": _refine_user_prompt(graph, baseline, draft, critique_payload)},
        ],
        model=backend.settings.model,
    )
    payload = _extract_json_object(refine_result.content)
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "self_refine_revision", "baseline": baseline.name, "raw_response": refine_result.raw_response}
    )
    return _proposal_from_payload(payload)


def _build_baseline_graph(
    instance,
    *,
    baseline: BaselineSpec,
) -> IdeaGraph:
    graph = IdeaGraph(topic=instance.topic, literature=list(instance.literature), metadata=dict(instance.metadata))
    graph.metadata.setdefault(
        "literature_grounding",
        build_literature_grounding(literature=graph.literature, metadata=graph.metadata).as_dict(),
    )
    graph.metadata["baseline_name"] = baseline.name
    graph.metadata["baseline_display_name"] = baseline.display_name
    graph.metadata["baseline_strategy"] = baseline.strategy
    graph.metadata["baseline_prompt_style"] = baseline.prompt_style
    graph.metadata["baseline_proxy"] = baseline.is_proxy
    graph.metadata["baseline_proxy_target"] = baseline.proxy_target
    graph.metadata["baseline_description"] = baseline.description
    graph.metadata.setdefault("instance_name", instance.name)
    return graph


def run_baseline_experiment(
    instance,
    *,
    baseline_name: str,
    collaboration_backend=None,
    progress_callback: Callable[[str], None] | None = None,
    max_rounds: int = 3,
    stop_when_mature: bool = True,
    external_baseline_config: dict[str, dict[str, Any]] | None = None,
):
    baseline = get_baseline_spec(baseline_name)
    if (
        _coerce_string(instance.metadata.get("baseline_name")) != baseline_name
        or "benchmark_input_packet" not in instance.metadata
    ):
        instance = attach_baseline_metadata(instance, baseline_name=baseline_name, io_mode="auto")

    if baseline.strategy == "delayed_consensus":
        return run_experiment(
            topic=instance.topic,
            literature=list(instance.literature),
            metadata=dict(instance.metadata),
            collaboration_backend=collaboration_backend,
            progress_callback=progress_callback,
            max_rounds=max_rounds,
            stop_when_mature=stop_when_mature,
        )

    graph = _build_baseline_graph(instance, baseline=baseline)
    emit_progress(
        graph,
        progress_callback,
        stage="start",
        message=(
            f"Initialized baseline '{baseline.name}' using strategy '{baseline.strategy}' for topic '{graph.topic}'."
        ),
        details={"baseline": baseline.name, "strategy": baseline.strategy},
    )

    if baseline.strategy == "external":
        emit_progress(
            graph,
            progress_callback,
            stage="baseline_generation",
            message=f"Running external baseline '{baseline.name}' through its upstream repository adapter.",
        )
        proposal = run_external_baseline(
            graph,
            baseline_name=baseline.name,
            external_config=external_baseline_config,
            progress_callback=progress_callback,
        )
    elif baseline.strategy == "direct":
        emit_progress(
            graph,
            progress_callback,
            stage="baseline_generation",
            message=f"Generating a one-pass structured idea with baseline '{baseline.name}'.",
        )
        if isinstance(collaboration_backend, OpenAICompatibleCollaborationBackend):
            proposal = _llm_direct_proposal(graph, baseline, collaboration_backend)
        else:
            proposal = _deterministic_direct_proposal(graph, baseline)
    elif baseline.strategy == "candidate_rank":
        emit_progress(
            graph,
            progress_callback,
            stage="baseline_generation",
            message=(
                f"Generating and selecting diverse structured idea candidates with baseline '{baseline.name}'."
            ),
        )
        if isinstance(collaboration_backend, OpenAICompatibleCollaborationBackend):
            proposal = _llm_candidate_rank_proposal(graph, baseline, collaboration_backend)
        else:
            draft = _deterministic_direct_proposal(graph, baseline)
            proposal = _deterministic_refine_proposal(graph, draft, baseline)
    elif baseline.strategy == "self_refine":
        emit_progress(
            graph,
            progress_callback,
            stage="baseline_generation",
            message=f"Generating and refining a structured idea with baseline '{baseline.name}'.",
        )
        if isinstance(collaboration_backend, OpenAICompatibleCollaborationBackend):
            proposal = _llm_self_refine_proposal(graph, baseline, collaboration_backend)
        else:
            draft = _deterministic_direct_proposal(graph, baseline)
            proposal = _deterministic_refine_proposal(graph, draft, baseline)
    else:
        raise ValueError(f"Unsupported baseline strategy '{baseline.strategy}'.")

    graph.final_subgraph = {"node_ids": [], "edge_ids": [], "utility": 0.0}
    graph.final_proposal = proposal
    graph.metadata["max_rounds_requested"] = 0
    graph.metadata["stop_when_mature"] = False
    graph.metadata["executed_round_count"] = 0
    graph.metadata["stopped_early"] = False
    if baseline.strategy == "external":
        graph.metadata["stop_reason"] = f"baseline_{baseline.name}_complete"
    else:
        graph.metadata["stop_reason"] = f"baseline_{baseline.strategy}_complete"

    emit_progress(
        graph,
        progress_callback,
        stage="complete",
        message=(
            f"Baseline run complete for '{baseline.name}': {len(graph.nodes)} nodes, "
            f"{len(graph.edges)} edges, {len(graph.actions)} actions."
        ),
        details={"baseline": baseline.name, "nodes": len(graph.nodes), "edges": len(graph.edges)},
    )
    return graph
