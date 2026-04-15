from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable

from .agent_backend import OpenAICompatibleCollaborationBackend
from .benchmark_mode import apply_io_mode
from .engine import emit_progress, run_experiment
from .literature_grounding import build_literature_grounding
from .models import FinalProposal, IdeaGraph
from .relation_graph_runtime_critic import (
    RelationGraphRuntimeConfig,
    load_relation_graph_runtime_bundle,
)
from .runtime_critic import TextCriticRuntimeConfig, load_pickled_text_critic_model


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
    runtime_controller: str = ""


ROOT = Path(__file__).resolve().parents[2]
TEXT_CRITIC_MODEL_RELATIVE_PATH = (
    Path("outputs")
    / "graph_critic_models"
    / "current_benchmarked_ours_eig_full_g46_text_online_real_train_v1"
    / "model.pkl"
)
RELATION_GRAPH_CRITIC_MODEL_RELATIVE_DIR = (
    Path("outputs")
    / "graph_critic_models"
    / "development_pool_v3_relation_graph_sanitized_v1"
)


RUNTIME_CONTROLLER_METADATA_KEYS = (
    "runtime_controller_enabled",
    "runtime_controller_kind",
    "runtime_controller_use_commit",
    "runtime_controller_tau_override",
    "runtime_controller_tau_commit",
    "runtime_controller_gamma_commit",
    "runtime_controller_min_commit_round",
    "runtime_controller_guard_support_threshold",
    "runtime_controller_guard_support_gain_floor",
    "runtime_controller_guard_requires_contradiction_progress",
    "runtime_controller_model_path",
    "runtime_controller_model_dir",
    "runtime_controller_error",
    "runtime_controller_loaded",
)


def _shared_repo_root_from_worktree(root: Path) -> Path | None:
    if root.parent.name != ".worktrees":
        return None
    return root.parent.parent


def _default_graph_critic_models_root() -> Path:
    shared_root = _shared_repo_root_from_worktree(ROOT)
    if shared_root is not None:
        return shared_root
    return ROOT


def _default_text_critic_model_path() -> Path:
    return (_default_graph_critic_models_root() / TEXT_CRITIC_MODEL_RELATIVE_PATH).resolve()


def _default_relation_graph_runtime_model_dir() -> Path:
    return (_default_graph_critic_models_root() / RELATION_GRAPH_CRITIC_MODEL_RELATIVE_DIR).resolve()


DEFAULT_TEXT_CRITIC_MODEL_PATH = _default_text_critic_model_path()


BASELINE_ALIASES: dict[str, str] = {
    "ours-delayed-consensus": "ours-eig",
}


def canonical_baseline_name(name: str) -> str:
    cleaned = str(name).strip()
    return BASELINE_ALIASES.get(cleaned, cleaned)


def is_eig_baseline_name(name: str) -> bool:
    return canonical_baseline_name(name) == "ours-eig"


BASELINE_SPECS: dict[str, BaselineSpec] = {
    "ours-eig": BaselineSpec(
        name="ours-eig",
        display_name="Ours (EIG)",
        strategy="evolving_graph",
        description="Evolving Idea Graph multi-agent collaboration with maturity-based commitment.",
        prompt_style="ours",
    ),
    "ours-eig-critic-text": BaselineSpec(
        name="ours-eig-critic-text",
        display_name="Ours (EIG + Text Critic)",
        strategy="evolving_graph",
        description="Evolving Idea Graph multi-agent collaboration with the G4.8 adapted text critic as a conservative edit reranker.",
        prompt_style="ours",
        runtime_controller="text_critic_rerank",
    ),
    "ours-eig-critic-graph": BaselineSpec(
        name="ours-eig-critic-graph",
        display_name="Ours (EIG + Relation-Graph Critic)",
        strategy="evolving_graph",
        description="Evolving Idea Graph multi-agent collaboration with a relation-graph runtime critic as a conservative edit reranker.",
        prompt_style="ours",
        runtime_controller="relation_graph_critic_rerank",
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
        strategy="self_refine",
        description="Local proxy wrapper emphasizing structured motivation and experiment decomposition.",
        is_proxy=True,
        proxy_target="SciPIP",
        prompt_style="scipip_proxy",
    ),
    "virsci-proxy": BaselineSpec(
        name="virsci-proxy",
        display_name="VirSci Proxy",
        strategy="evolving_graph",
        description="Local proxy wrapper for a discussion-oriented multi-agent baseline.",
        is_proxy=True,
        proxy_target="VirSci",
        prompt_style="virsci_proxy",
    ),
}


PROMPT_STYLE_GUIDANCE = {
    "ours": (
        "Preserve typed intermediate claims, disagreement tracking, maturity-based commitment, and section-level rigor."
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
    name = canonical_baseline_name(name)
    try:
        return BASELINE_SPECS[name]
    except KeyError as exc:
        options = ", ".join(sorted(BASELINE_SPECS))
        raise KeyError(f"Unknown baseline '{name}'. Available baselines: {options}") from exc


def baseline_choices(*, include_aliases: bool = True) -> list[str]:
    names = list(BASELINE_SPECS)
    if include_aliases:
        names.extend(BASELINE_ALIASES)
    return sorted(set(names))


def attach_baseline_metadata(
    instance,
    *,
    baseline_name: str,
    io_mode: str = "auto",
):
    requested_baseline_name = str(baseline_name).strip()
    baseline = get_baseline_spec(requested_baseline_name)
    instance = apply_io_mode(instance, io_mode=io_mode)
    metadata = dict(instance.metadata)
    if requested_baseline_name != baseline.name:
        metadata["baseline_requested_name"] = requested_baseline_name
    metadata["baseline_name"] = baseline.name
    metadata["baseline_display_name"] = baseline.display_name
    metadata["baseline_strategy"] = baseline.strategy
    metadata["baseline_prompt_style"] = baseline.prompt_style
    metadata["baseline_description"] = baseline.description
    metadata["baseline_proxy"] = baseline.is_proxy
    metadata["baseline_proxy_target"] = baseline.proxy_target
    metadata["baseline_runtime_controller"] = baseline.runtime_controller
    for key in RUNTIME_CONTROLLER_METADATA_KEYS:
        metadata.pop(key, None)

    if baseline.runtime_controller == "text_critic_rerank":
        metadata["runtime_controller_enabled"] = True
        metadata["runtime_controller_kind"] = "text_critic_rerank"
        metadata["runtime_controller_use_commit"] = False
        metadata["runtime_controller_tau_override"] = 0.05
        metadata["runtime_controller_model_path"] = str(_default_text_critic_model_path())
    elif baseline.runtime_controller == "relation_graph_critic_rerank":
        metadata["runtime_controller_enabled"] = True
        metadata["runtime_controller_kind"] = "relation_graph_critic_rerank"
        metadata["runtime_controller_use_commit"] = False
        metadata["runtime_controller_tau_override"] = 0.05
        metadata["runtime_controller_model_dir"] = str(_default_relation_graph_runtime_model_dir())
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
    prefixes = (
        "The topic of this paper is ",
        "Ideation topic keyword: ",
    )
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned or _clean_text(graph.topic)


def _reference_packet(graph: IdeaGraph) -> list[dict[str, str]]:
    packet = graph.metadata.get("benchmark_input_packet", {})
    if not isinstance(packet, dict):
        return []
    references = packet.get("reference_packet", [])
    if not isinstance(references, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in references:
        if not isinstance(item, dict):
            continue
        title = _coerce_string(item.get("title"))
        snippet = _coerce_string(item.get("snippet"))
        if title or snippet:
            normalized.append({"title": title, "snippet": snippet})
    return normalized


def _generation_metadata(graph: IdeaGraph) -> dict[str, Any]:
    payload = graph.metadata.get("generation_safe_metadata", graph.metadata)
    return payload if isinstance(payload, dict) else graph.metadata


def _baseline_anchor_terms(graph: IdeaGraph) -> list[str]:
    topic = _topic_text(graph)
    corpus = " ".join(
        [
            topic,
            *[
                f"{item.get('title', '')} {item.get('snippet', '')}"
                for item in _reference_packet(graph)
            ],
        ]
    ).lower()
    anchors = [topic] if topic else []
    candidate_phrases = [
        "language field",
        "radiance field",
        "gaussian splatting",
        "open-vocabulary",
        "clip embedding",
        "hierarchical semantics",
        "segmentation",
        "query",
        "localization",
        "panoramic",
        "relative pose",
        "geometric",
        "alignment",
        "uncertainty",
        "compression",
        "retrieval",
    ]
    for phrase in candidate_phrases:
        if phrase in corpus and phrase not in anchors:
            anchors.append(phrase)
    return anchors[:8]


def _baseline_focus_constraints(graph: IdeaGraph, baseline: BaselineSpec) -> list[str]:
    topic = _topic_text(graph)
    anchors = _baseline_anchor_terms(graph)
    anchor_hint = ", ".join(anchors[1:6]) if len(anchors) > 1 else topic
    constraints = [
        f"Keep the main task tightly centered on '{topic}', not an adjacent task family.",
        "Use only the benchmark topic and visible reference packet; do not rely on hidden target-paper fields.",
        "Do not treat cited method papers as datasets. Use method papers as baselines or inspiration; mention datasets only when the packet clearly signals benchmark datasets or evaluation assets.",
        "Avoid generic high-level proposals that could fit many topics; tie the mechanism and evaluation to the given benchmark packet.",
        f"Use packet anchors when helpful: {anchor_hint}.",
    ]
    if baseline.prompt_style == "direct":
        constraints.extend(
            [
                "Choose one strong central mechanism instead of listing several possible directions.",
                "Prefer the simplest benchmark-faithful idea that is still non-trivial and testable.",
            ]
        )
    elif baseline.prompt_style == "self_refine":
        constraints.extend(
            [
                "Use critique to remove generic wording, unsupported claims, and vague evaluation language.",
                "The final revision should improve specificity rather than simply making the text longer.",
            ]
        )
    elif baseline.prompt_style == "scipip_proxy":
        constraints.extend(
            [
                "Emphasize structured decomposition: identify the bottleneck, explain why current methods fail, then propose one coherent pipeline.",
                "Make the experiment plan concrete with datasets, metrics, baselines, and ablations whenever the packet supports them.",
            ]
        )
    elif baseline.prompt_style == "ai_researcher_proxy":
        constraints.extend(_ai_researcher_focus_constraints(graph))
    elif baseline.prompt_style == "virsci_proxy":
        constraints.extend(
            [
                "Preserve multiple viewpoints and tradeoffs before settling on one final idea.",
                "Explicitly surface why the chosen idea wins over nearby alternatives.",
            ]
        )
    return _unique_strings(constraints)


def _grounding_brief(graph: IdeaGraph) -> dict[str, str]:
    grounding = build_literature_grounding(
        literature=graph.literature,
        metadata=_generation_metadata(graph),
    )
    return {
        "existing_methods_summary": grounding.existing_methods_summary,
        "experiment_plan_summary": grounding.experiment_plan_summary,
    }


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = _coerce_string(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def _ai_researcher_anchor_terms(graph: IdeaGraph) -> list[str]:
    return _baseline_anchor_terms(graph)[:6]


def _ai_researcher_context_corpus(graph: IdeaGraph) -> str:
    parts: list[str] = [_topic_text(graph)]
    keyword = _coerce_string(
        _generation_metadata(graph).get("keyword")
        or graph.metadata.get("keyword")
    )
    if keyword:
        parts.append(keyword)
    parts.extend(
        [
            f"{item.get('title', '')} {item.get('snippet', '')}"
            for item in _reference_packet(graph)
        ]
    )
    parts.extend(_reference_support_texts(graph))
    return " ".join(parts).lower()


def _is_ai_researcher_language_field_context(graph: IdeaGraph) -> bool:
    corpus = _ai_researcher_context_corpus(graph)
    return any(
        term in corpus
        for term in (
            "language field",
            "radiance field",
            "language embedded radiance",
            "lerf",
            "open-vocabulary scene understanding",
        )
    )


def _ai_researcher_focus_constraints(graph: IdeaGraph) -> list[str]:
    topic = _topic_text(graph)
    anchors = _ai_researcher_anchor_terms(graph)
    anchor_hint = ", ".join(anchors[1:]) if len(anchors) > 1 else topic
    if _is_ai_researcher_language_field_context(graph):
        contribution_hint = (
            "Prefer contributions about language or radiance field representation, "
            "querying, grounding, efficiency, or semantic structure."
        )
        drift_hint = (
            "Penalize drift into generic text-to-3D generation, generic 3D reconstruction, "
            "generic scene synthesis, or temporal/video extensions unless the packet directly "
            "supports that shift."
        )
    else:
        contribution_hint = (
            "Prefer one concrete technical idea whose mechanism and evaluation stay close to the "
            "benchmark topic and visible references."
        )
        drift_hint = (
            "Penalize drift into adjacent task families, unsupported modality shifts, or broad "
            "generic proposals that are not justified by the packet."
        )
    return [
        f"Keep the main task tightly centered on '{topic}', not an adjacent task family.",
        contribution_hint,
        drift_hint,
        f"Use the packet anchors when helpful: {anchor_hint}.",
        "Do not treat cited method papers as datasets. Use method papers as baselines or inspiration; mention datasets only when the packet clearly indicates a benchmark dataset.",
        "Favor ideas whose evaluation can be justified directly from the benchmark topic and visible references.",
    ]


def _ai_researcher_topic_fidelity_score(graph: IdeaGraph, proposal: FinalProposal) -> float:
    title_text = proposal.title.lower()
    problem_text = proposal.problem.lower()
    hypothesis_text = proposal.hypothesis.lower()
    method_text = proposal.method.lower()
    text = " ".join(
        [
            proposal.title,
            proposal.problem,
            proposal.existing_methods,
            proposal.motivation,
            proposal.hypothesis,
            proposal.method,
            proposal.evaluation,
            proposal.significance,
        ]
    ).lower()
    anchors = _ai_researcher_anchor_terms(graph)
    if not anchors:
        return 0.0

    anchor_hits = sum(1 for term in anchors if term and term.lower() in text)
    score = 0.45 * (anchor_hits / len(anchors))

    topic = _topic_text(graph).lower()
    context_corpus = _ai_researcher_context_corpus(graph)
    title_problem_method = " ".join([proposal.title, proposal.problem, proposal.method]).lower()
    evaluation_text = proposal.evaluation.lower()
    if topic and topic in text:
        score += 0.25
    keyword = _coerce_string(
        _generation_metadata(graph).get("keyword")
        or graph.metadata.get("keyword")
    ).lower()
    if keyword and keyword in text:
        score += 0.25

    topical_terms = [topic, *anchors[1:]]
    topical_hits = sum(1 for term in topical_terms if term and term.lower() in title_problem_method)
    score += min(0.18, 0.06 * topical_hits)

    if any(
        signal in evaluation_text
        for signal in ("benchmark", "dataset", "metric", "accuracy", "calibration", "ablation", "precision", "recall")
    ):
        score += 0.06

    unsupported_drift_terms = {
        "language field": ("language field", "3d language field"),
        "radiance field": ("radiance field", "nerf"),
        "gaussian splatting": ("gaussian splatting",),
        "text-to-3d": ("text-to-3d",),
        "scene generation": ("scene generation", "scene synthesis"),
        "3d reconstruction": ("3d reconstruction",),
        "temporal": ("temporal", "video-based"),
    }
    for context_term, proposal_terms in unsupported_drift_terms.items():
        if context_term not in context_corpus and any(term in text for term in proposal_terms):
            score -= 0.12

    if _is_ai_researcher_language_field_context(graph):
        if "3d language field" in title_text:
            score += 0.25
        elif "language field" in title_text:
            score += 0.18
        else:
            score -= 0.18
        if "3d language field" in problem_text or "language field modeling" in problem_text:
            score += 0.18
        if "language field" in hypothesis_text or "language field" in method_text:
            score += 0.12
        if "modeling" in title_text or "modeling" in problem_text or "modeling" in method_text:
            score += 0.08

        if any(term in text for term in ("language field", "radiance field", "language-embedded radiance")):
            score += 0.25
        else:
            score -= 0.35

        for positive_term in ("open-vocabulary", "gaussian splatting", "localization"):
            if positive_term in text:
                score += 0.08

        for drift_term in ("text-to-3d", "scene generation", "3d reconstruction", "temporal", "video-based"):
            if drift_term in text:
                score -= 0.12
    else:
        topic_tokens = [
            token
            for token in re.findall(r"[a-z0-9][a-z0-9-]+", topic)
            if len(token) >= 5
        ]
        distinct_hits = sum(1 for token in topic_tokens if token in title_problem_method)
        score += min(0.16, 0.08 * distinct_hits)
        if keyword and keyword not in title_problem_method and keyword not in evaluation_text:
            score -= 0.10

    return max(0.0, min(1.0, score))


def _reference_support_texts(graph: IdeaGraph) -> list[str]:
    texts: list[str] = []
    raw_grounding = graph.metadata.get("paper_grounding", {})
    if isinstance(raw_grounding, dict):
        raw_references = raw_grounding.get("reference_paper_snippets", [])
        if isinstance(raw_references, list):
            for item in raw_references:
                if not isinstance(item, dict):
                    continue
                for field_name in ("snippet", "abstract", "introduction", "method", "evaluation", "text_excerpt"):
                    value = _coerce_string(item.get(field_name))
                    if value:
                        texts.append(value)
    for item in _reference_packet(graph):
        for field_name in ("title", "snippet"):
            value = _coerce_string(item.get(field_name))
            if value:
                texts.append(value)
    return texts


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _ai_researcher_proxy_postprocess_proposal(graph: IdeaGraph, proposal: FinalProposal) -> FinalProposal:
    references = _reference_packet(graph)
    reference_titles = [item.get("title", "") for item in references if item.get("title")]
    support_text = " ".join(_reference_support_texts(graph))

    title = proposal.title
    if "language field" not in title.lower():
        title = f"{title.rstrip()} for 3D Language Field Modeling".strip()
    if "3d language field" not in title.lower() and "language field" in title.lower():
        title = title.replace("Language Field", "3D Language Field", 1)
    if len(title.split()) > 18:
        title = "Efficient Open-Vocabulary Querying in 3D Language Fields"

    problem = proposal.problem
    if "3d language field" not in problem.lower():
        problem = problem.rstrip(".") + " In particular, scalable 3D language field modeling remains difficult when queries must stay open-vocabulary and mask-free."

    existing_methods = proposal.existing_methods
    if "lerf" not in existing_methods.lower() and any("lerf" in title_item.lower() for title_item in reference_titles):
        existing_methods = existing_methods.rstrip(".") + " LERF demonstrates language-embedded radiance fields for open-vocabulary querying."
    if "gaussian splatting" not in existing_methods.lower() and any("gaussian splatting" in title_item.lower() for title_item in reference_titles):
        existing_methods = existing_methods.rstrip(".") + " 3D Gaussian Splatting offers efficient rendering but does not by itself solve language-field querying."

    motivation = proposal.motivation
    if "3d language field modeling" not in motivation.lower():
        motivation = motivation.rstrip(".") + " This matters because 3D language field modeling should support efficient open-vocabulary interaction in real scenes."

    hypothesis = proposal.hypothesis
    if "language field" not in hypothesis.lower():
        hypothesis = hypothesis.rstrip(".") + " within a 3D language field."

    method = proposal.method
    if "3d language field" not in method.lower():
        method = method.rstrip(".") + " The resulting representation is optimized specifically as a 3D language field rather than a generic text-guided scene generator."

    evaluation = proposal.evaluation
    if "lerf dataset" not in evaluation.lower() and "lerf dataset" in support_text.lower():
        evaluation = evaluation.rstrip(".") + " Use the LERF dataset for open-vocabulary querying analysis."
    if "3d-ovs" not in evaluation.lower() and "3d-ovs" in support_text.lower():
        evaluation = evaluation.rstrip(".") + " Also test on the 3D-OVS dataset for open-vocabulary semantic behavior."
    if "localization accuracy" not in evaluation.lower() and "localization accuracy" in support_text.lower():
        evaluation = evaluation.rstrip(".") + " Report localization accuracy in addition to query precision and recall."

    significance = proposal.significance
    if "3d language field modeling" not in significance.lower():
        significance = significance.rstrip(".") + " This would strengthen efficient 3D language field modeling for open-vocabulary interaction."

    return FinalProposal(
        title=title,
        abstract="",
        problem=problem,
        existing_methods=existing_methods,
        motivation=motivation,
        hypothesis=hypothesis,
        method=method,
        evaluation=evaluation,
        significance=significance,
        caveats=proposal.caveats,
    )


def _split_sentences(text: Any) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]


def _is_noisy_sentence(text: str) -> bool:
    lowered = _clean_text(text).casefold()
    if not lowered:
        return True
    noisy_markers = (
        "experiments were conducted on paper introduces",
        "paper introduces",
        "captured in diverse real scenarios",
        "project website",
        "see the project website",
        "figure",
        "table",
        "et al",
    )
    return any(marker in lowered for marker in noisy_markers)


def _clean_section_text(text: Any) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return _clean_text(text)
    kept: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = _clean_text(sentence).casefold()
        if not normalized or normalized in seen:
            continue
        if _is_noisy_sentence(sentence):
            continue
        kept.append(sentence)
        seen.add(normalized)
    return " ".join(kept) if kept else _clean_text(text)


def _baseline_postprocess_proposal(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    proposal: FinalProposal,
) -> FinalProposal:
    if baseline.prompt_style == "ai_researcher_proxy" and _is_ai_researcher_language_field_context(graph):
        return _ai_researcher_proxy_postprocess_proposal(graph, proposal)

    topic = _topic_text(graph)
    anchors = _baseline_anchor_terms(graph)
    anchor_phrase = anchors[1] if len(anchors) > 1 else topic
    grounding = _grounding_brief(graph)
    benchmark_packet = graph.metadata.get("benchmark_input_packet", {})
    reference_packet = benchmark_packet.get("reference_packet", []) if isinstance(benchmark_packet, dict) else []
    keyword = _coerce_string(
        graph.metadata.get("keyword")
        or (benchmark_packet.get("keyword") if isinstance(benchmark_packet, dict) else "")
        or topic
    )
    keyword_only_mode = (
        _coerce_string(graph.metadata.get("benchmark")) == "liveideabench"
        or (not reference_packet and "benchmark keyword:" in grounding["existing_methods_summary"].lower())
    )

    title = proposal.title or topic.title()
    if baseline.prompt_style == "scipip_proxy" and topic and topic.lower() not in title.lower():
        title = f"{title.rstrip()} for {topic}".strip()
    if len(title.split()) > 18:
        title = title[:96].rsplit(" ", 1)[0].strip()

    problem = proposal.problem or f"{topic} remains insufficiently solved under the current benchmark setting."
    if topic and topic.lower() not in problem.lower():
        problem = problem.rstrip(".") + f" This matters directly for {topic}."

    existing_methods = proposal.existing_methods or grounding["existing_methods_summary"]
    if baseline.prompt_style == "scipip_proxy" and grounding["existing_methods_summary"]:
        summary = grounding["existing_methods_summary"]
        should_append_summary = bool(summary) and summary not in existing_methods
        if should_append_summary and len(existing_methods.split()) >= 38:
            should_append_summary = False
        if should_append_summary:
            existing_methods = existing_methods.rstrip(".") + " " + summary
    if keyword_only_mode and (
        "benchmark keyword:" in existing_methods.lower()
        or "held-out metadata" in existing_methods.lower()
        or "row provides a keyword prompt" in existing_methods.lower()
    ):
        existing_methods = (
            f"For {keyword}, common directions include spatiotemporal forecasting models, "
            "physics-aware simulation, and multi-source data fusion."
        )

    motivation = proposal.motivation or f"A more precise and testable idea for {topic} is needed."
    if baseline.prompt_style == "scipip_proxy" and "why" not in motivation.lower():
        motivation = motivation.rstrip(".") + " The key motivation is that current methods do not adequately address the core bottleneck exposed by the benchmark packet."

    hypothesis = proposal.hypothesis or f"A more explicit mechanism around {anchor_phrase} can improve results for {topic}."
    method = proposal.method
    if baseline.prompt_style == "scipip_proxy" and method:
        if "first" not in method.lower():
            method = (
                "First, identify the core bottleneck from the benchmark packet. "
                + method.rstrip(".")
                + ". Then connect that bottleneck to one coherent pipeline and targeted ablations."
            )
    if not method:
        method = f"Design one concrete mechanism around {anchor_phrase} for {topic}, with explicit implementation steps and ablations."

    evaluation = proposal.evaluation or grounding["experiment_plan_summary"]
    experiment_summary = grounding["experiment_plan_summary"]
    should_append_experiment_summary = bool(experiment_summary) and experiment_summary not in evaluation
    if should_append_experiment_summary and baseline.prompt_style in {"self_refine", "scipip_proxy"}:
        if (
            experiment_summary.casefold().startswith("compare against strong baselines")
            and _contains_any(evaluation, ["compare against", "ablation", "quantitative metric"])
        ):
            should_append_experiment_summary = False
        if baseline.prompt_style == "scipip_proxy" and len(evaluation.split()) >= 28:
            should_append_experiment_summary = False
    if should_append_experiment_summary:
        if baseline.prompt_style in {"self_refine", "scipip_proxy"}:
            evaluation = evaluation.rstrip(".") + " " + grounding["experiment_plan_summary"]
    if "metric" not in evaluation.lower() and "accuracy" not in evaluation.lower():
        evaluation = evaluation.rstrip(".") + " Report quantitative metrics and compare against strong baselines."
    if keyword_only_mode and any(
        marker in evaluation.lower()
        for marker in ("synthetic urban dataset", "lerf dataset", "3d-ovs", "polycam", "scannet")
    ):
        evaluation = (
            f"Evaluate on realistic benchmark tasks for {keyword}, compare against strong data-driven and hybrid baselines, "
            "report task-specific quantitative metrics, and include ablations over the main components."
        )

    significance = proposal.significance or f"If successful, the idea would make {topic} more accurate, robust, or testable."
    if topic and topic.lower() not in significance.lower():
        significance = significance.rstrip(".") + f" This could make progress on {topic} more reliable and testable."

    caveats = proposal.caveats or "The idea may still depend on incomplete literature context and should be validated with targeted ablations."

    existing_methods = _clean_section_text(existing_methods)
    method = _clean_section_text(method)
    evaluation = _clean_section_text(evaluation)
    significance = _clean_section_text(significance)

    return FinalProposal(
        title=title,
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


def _deterministic_direct_proposal(graph: IdeaGraph, baseline: BaselineSpec) -> FinalProposal:
    generation_metadata = _generation_metadata(graph)
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

    proposal = FinalProposal(
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
    return _baseline_postprocess_proposal(graph, baseline, proposal)


def _deterministic_refine_proposal(graph: IdeaGraph, draft: FinalProposal, baseline: BaselineSpec) -> FinalProposal:
    grounding = build_literature_grounding(
        literature=graph.literature,
        metadata=_generation_metadata(graph),
    )
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

    proposal = FinalProposal(
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
    return _baseline_postprocess_proposal(graph, baseline, proposal)


def _direct_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are a scientific idea generation baseline. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        "Generate exactly one structured research idea using the provided benchmark packet and output schema. "
        "Do not assume access to hidden target-paper labels. "
        "Benchmark fidelity matters more than writing style. "
        "Each section must add distinct information instead of repeating the same sentence in different fields. "
        "Keep the idea specific, testable, and grounded in the visible references. "
        "Do not copy raw extraction fragments from literature snippets; if a dataset or method fragment looks noisy or truncated, omit it. "
        "Prefer one coherent mechanism and one coherent evaluation story rather than several loosely connected ideas. "
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
        "focus_constraints": _baseline_focus_constraints(graph, baseline),
        "anchor_terms": _baseline_anchor_terms(graph),
        "literature_grounding": _grounding_brief(graph),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _critique_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are a scientific idea critic. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        "Critique the current draft using only the benchmark packet and return concise revision guidance. "
        "Focus on benchmark fidelity, literature grounding, unsupported claims, vague evaluation design, repetition across sections, noisy copied snippet fragments, and overcomplicated mechanism drift. "
        'JSON schema: {"strengths":["..."],"weaknesses":["..."],"revision_focus":["..."]}'
    )


def _critique_user_prompt(graph: IdeaGraph, baseline: BaselineSpec, draft: FinalProposal) -> str:
    payload = {
        "baseline": asdict(baseline),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _baseline_focus_constraints(graph, baseline),
        "anchor_terms": _baseline_anchor_terms(graph),
        "literature_grounding": _grounding_brief(graph),
        "draft": _proposal_as_prompt_payload(draft),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _refine_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are revising a scientific research idea after critique. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        "Revise the draft to improve grounding, coherence, benchmark fidelity, and testability while keeping the output concise. "
        "Do not add generic filler; prefer sharper, more benchmark-faithful content. "
        "If the draft contains noisy copied snippet fragments, remove them rather than elaborating on them. "
        "If the draft contains multiple loosely connected mechanisms, simplify to one coherent method story. "
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
        "focus_constraints": _baseline_focus_constraints(graph, baseline),
        "anchor_terms": _baseline_anchor_terms(graph),
        "literature_grounding": _grounding_brief(graph),
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
        "All candidates must remain benchmark-faithful and avoid drifting into adjacent tasks. "
        'JSON schema: {"candidates":[{"title":"...","problem":"...","existing_methods":"...","motivation":"...",'
        '"hypothesis":"...","method":"...","evaluation":"...","significance":"...","caveats":"..."}]}'
    )


def _candidate_generation_user_prompt(graph: IdeaGraph, baseline: BaselineSpec) -> str:
    payload = {
        "baseline": asdict(baseline),
        "topic": graph.topic,
        "task_instruction": graph.metadata.get("task_instruction", ""),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _baseline_focus_constraints(graph, baseline),
        "anchor_terms": _baseline_anchor_terms(graph),
        "literature_grounding": _grounding_brief(graph),
        "candidate_count": max(2, baseline.candidate_count),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _ai_researcher_seed_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    candidate_count = max(2, baseline.candidate_count)
    return (
        "You are implementing a lightweight AI-Researcher-style baseline for scientific idea generation. "
        f"{guidance} "
        "Benchmark faithfulness is the first requirement. "
        "Stage 1 is seed ideation only. Use the topic and reference packet as inspiration, but do not copy or restate the references. "
        f"Generate exactly {candidate_count} diverse seed ideas with clearly different mechanisms or problem framings. "
        "The ideas must stay inside the benchmark task family rather than drifting to a nearby but different task. "
        "Return strict JSON only, with no markdown or commentary. "
        'JSON schema: {"seed_ideas":[{"idea_name":"...","problem_focus":"...","existing_gap":"...","core_mechanism":"...","evaluation_hint":"..."}]}'
    )


def _ai_researcher_seed_user_prompt(graph: IdeaGraph, baseline: BaselineSpec) -> str:
    payload = {
        "baseline": asdict(baseline),
        "topic": graph.topic,
        "task_instruction": graph.metadata.get("task_instruction", ""),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _ai_researcher_focus_constraints(graph),
        "anchor_terms": _ai_researcher_anchor_terms(graph),
        "candidate_count": max(2, baseline.candidate_count),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_seed_idea_payload(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "idea_name": _coerce_string(payload.get("idea_name") or payload.get("title") or payload.get("name")),
        "problem_focus": _coerce_string(payload.get("problem_focus") or payload.get("problem")),
        "existing_gap": _coerce_string(payload.get("existing_gap") or payload.get("gap") or payload.get("existing_methods")),
        "core_mechanism": _coerce_string(payload.get("core_mechanism") or payload.get("mechanism") or payload.get("hypothesis")),
        "evaluation_hint": _coerce_string(payload.get("evaluation_hint") or payload.get("evaluation")),
    }


def _ai_researcher_expansion_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are implementing a lightweight AI-Researcher-style baseline for scientific idea generation. "
        f"{guidance} "
        "Stage 2 expands one seed idea into one full structured proposal. "
        "Preserve benchmark faithfulness: if the seed drifts away from the benchmark task, pull it back toward the benchmark topic instead of amplifying the drift. "
        "Keep one coherent proposal with one main mechanism, grounded in the provided packet, and non-repetitive across sections. "
        "Do not copy raw extraction fragments from literature snippets; if a detail looks noisy or truncated, omit it. "
        "Return strict JSON only, with no markdown or commentary. "
        'JSON schema: {"title":"...","problem":"...","existing_methods":"...","motivation":"...",'
        '"hypothesis":"...","method":"...","evaluation":"...","significance":"...","caveats":"..."}'
    )


def _ai_researcher_expansion_user_prompt(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    seed_idea: dict[str, str],
) -> str:
    payload = {
        "baseline": asdict(baseline),
        "topic": graph.topic,
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _ai_researcher_focus_constraints(graph),
        "anchor_terms": _ai_researcher_anchor_terms(graph),
        "seed_idea": seed_idea,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _ai_researcher_ranking_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are implementing a lightweight AI-Researcher-style ranking stage. "
        f"{guidance} "
        "Rank the expanded candidates and choose the single best one. "
        "Benchmark fidelity is the first gate: a candidate that sounds exciting but drifts away from the benchmark task should lose. "
        "Use topic fidelity, literature grounding, novelty, significance, feasibility, clarity, and experiment quality. "
        "Penalize noisy copied snippet fragments, unsupported dataset mentions, and multi-mechanism proposals that lack one clear core idea. "
        "Do not reward longer text unless it improves substance. "
        "Return strict JSON only, with no markdown or commentary. "
        'JSON schema: {"selected_index":0,"reason":"...","scores":[{"index":0,"topic_fidelity":1,"novelty":1,"significance":1,"feasibility":1,"clarity":1,"literature_grounding":1,"experiment_quality":1,"overall":1}]}'
    )


def _ai_researcher_ranking_user_prompt(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    candidates: list[FinalProposal],
) -> str:
    payload = {
        "baseline": asdict(baseline),
        "input_packet": graph.metadata.get("benchmark_input_packet", {}),
        "focus_constraints": _ai_researcher_focus_constraints(graph),
        "anchor_terms": _ai_researcher_anchor_terms(graph),
        "candidates": [
            {"index": index, **_proposal_as_prompt_payload(candidate)}
            for index, candidate in enumerate(candidates)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _llm_json_object(
    backend: OpenAICompatibleCollaborationBackend,
    *,
    role: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[dict[str, Any], dict[str, object]]:
    attempt_messages = list(messages)
    last_error = "Unknown JSON decoding failure."
    for attempt in range(backend.settings.max_retries + 1):
        result = backend.client.create_chat_completion(
            messages=attempt_messages,
            model=backend.settings.model_for_role(role),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        trace: dict[str, object] = {
            "role": role,
            "attempt": attempt + 1,
            "messages": attempt_messages,
            "raw_response": result.raw_response,
        }
        try:
            payload = _extract_json_object(result.content)
            return payload, trace
        except Exception as exc:
            last_error = str(exc)
            trace["error"] = last_error
            if attempt >= backend.settings.max_retries:
                raise
            attempt_messages = attempt_messages + [
                {"role": "assistant", "content": result.content},
                {
                    "role": "user",
                    "content": (
                        "Your last message was not valid strict JSON for the required schema. "
                        "Return one JSON object only, with no markdown, no code fences, and no extra commentary."
                    ),
                },
            ]
    raise ValueError(last_error)


def _candidate_selection_system_prompt(baseline: BaselineSpec) -> str:
    guidance = PROMPT_STYLE_GUIDANCE.get(baseline.prompt_style, "")
    return (
        "You are ranking candidate scientific research ideas for a baseline wrapper. "
        f"{guidance} "
        "Return strict JSON only. Do not use markdown. "
        "Select the single best candidate using novelty, significance, feasibility, clarity, and topic fit. "
        "Benchmark fidelity and literature grounding should dominate tie-breaking. "
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
        "focus_constraints": _baseline_focus_constraints(graph, baseline),
        "anchor_terms": _baseline_anchor_terms(graph),
        "candidates": [
            {"index": index, **_proposal_as_prompt_payload(candidate)}
            for index, candidate in enumerate(candidates)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _llm_direct_proposal(graph: IdeaGraph, baseline: BaselineSpec, backend: OpenAICompatibleCollaborationBackend) -> FinalProposal:
    payload, trace = _llm_json_object(
        backend,
        role="BaselineDirect",
        messages=[
            {"role": "system", "content": _direct_system_prompt(baseline)},
            {"role": "user", "content": _direct_user_prompt(graph, baseline)},
        ],
        temperature=0.35,
        max_tokens=1400,
    )
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "direct_generation", "baseline": baseline.name, **trace}
    )
    proposal = _proposal_from_payload(payload)
    return _baseline_postprocess_proposal(graph, baseline, proposal)


def _llm_candidate_rank_proposal(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    backend: OpenAICompatibleCollaborationBackend,
    progress_callback: Callable[[str], None] | None = None,
) -> FinalProposal:
    if baseline.prompt_style == "ai_researcher_proxy":
        return _llm_ai_researcher_proxy_proposal(
            graph,
            baseline,
            backend,
            progress_callback=progress_callback,
        )

    generation_payload, generation_trace = _llm_json_object(
        backend,
        role="BaselineCandidateGeneration",
        messages=[
            {"role": "system", "content": _candidate_generation_system_prompt(baseline)},
            {"role": "user", "content": _candidate_generation_user_prompt(graph, baseline)},
        ],
        temperature=0.7,
        max_tokens=2200,
    )
    raw_candidates = generation_payload.get("candidates", [])
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("Candidate-generation response did not contain a non-empty 'candidates' list.")

    candidates = [
        _baseline_postprocess_proposal(graph, baseline, _proposal_from_payload(item))
        for item in raw_candidates
        if isinstance(item, dict)
    ]
    if not candidates:
        raise ValueError("Candidate-generation response did not contain any valid proposal objects.")

    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "candidate_generation", "baseline": baseline.name, **generation_trace}
    )

    selection_payload, selection_trace = _llm_json_object(
        backend,
        role="BaselineCandidateSelection",
        messages=[
            {"role": "system", "content": _candidate_selection_system_prompt(baseline)},
            {"role": "user", "content": _candidate_selection_user_prompt(graph, baseline, candidates)},
        ],
        temperature=0.0,
        max_tokens=1400,
    )
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "candidate_selection", "baseline": baseline.name, **selection_trace}
    )
    try:
        selected_index = int(selection_payload.get("selected_index", 0))
    except (TypeError, ValueError):
        selected_index = 0
    if not (0 <= selected_index < len(candidates)):
        selected_index = 0

    selected = _baseline_postprocess_proposal(graph, baseline, candidates[selected_index])
    selection_reason = _coerce_string(selection_payload.get("reason"))
    if selection_reason:
        graph.metadata["baseline_selection_reason"] = selection_reason
    return selected


def _llm_ai_researcher_proxy_proposal(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    backend: OpenAICompatibleCollaborationBackend,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> FinalProposal:
    emit_progress(
        graph,
        progress_callback,
        stage="baseline_seed_generation",
        message=f"Baseline '{baseline.name}': generating literature-grounded seed ideas.",
    )
    seed_payload, seed_trace = _llm_json_object(
        backend,
        role="BaselineSeedGeneration",
        messages=[
            {"role": "system", "content": _ai_researcher_seed_system_prompt(baseline)},
            {"role": "user", "content": _ai_researcher_seed_user_prompt(graph, baseline)},
        ],
        temperature=0.8,
        max_tokens=1800,
    )
    raw_seed_ideas = seed_payload.get("seed_ideas", [])
    if not isinstance(raw_seed_ideas, list) or not raw_seed_ideas:
        raise ValueError("AI-Researcher-style seed generation did not return a non-empty 'seed_ideas' list.")
    seed_ideas = [
        _normalize_seed_idea_payload(item)
        for item in raw_seed_ideas
        if isinstance(item, dict)
    ]
    seed_ideas = [item for item in seed_ideas if any(item.values())]
    if not seed_ideas:
        raise ValueError("AI-Researcher-style seed generation did not return any usable seed ideas.")
    graph.metadata["ai_researcher_proxy_seed_ideas"] = seed_ideas
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "seed_idea_generation", "baseline": baseline.name, **seed_trace}
    )

    candidates: list[FinalProposal] = []
    expansion_errors: list[dict[str, str]] = []
    for index, seed_idea in enumerate(seed_ideas):
        emit_progress(
            graph,
            progress_callback,
            stage="baseline_candidate_expansion",
            message=(
                f"Baseline '{baseline.name}': expanding seed candidate {index + 1}/{len(seed_ideas)}"
            ),
        )
        try:
            expansion_payload, expansion_trace = _llm_json_object(
                backend,
                role="BaselineCandidateExpansion",
                messages=[
                    {"role": "system", "content": _ai_researcher_expansion_system_prompt(baseline)},
                    {"role": "user", "content": _ai_researcher_expansion_user_prompt(graph, baseline, seed_idea)},
                ],
                temperature=0.35,
                max_tokens=1800,
            )
            proposal = _proposal_from_payload(expansion_payload)
            if not proposal.title:
                proposal.title = seed_idea.get("idea_name", "")
            if not proposal.problem:
                proposal.problem = seed_idea.get("problem_focus", "")
            if not proposal.hypothesis:
                proposal.hypothesis = seed_idea.get("core_mechanism", "")
            if not proposal.evaluation:
                proposal.evaluation = seed_idea.get("evaluation_hint", "")
            proposal = _baseline_postprocess_proposal(graph, baseline, proposal)
            candidates.append(proposal)
            graph.metadata.setdefault("baseline_traces", []).append(
                {
                    "stage": "candidate_expansion",
                    "baseline": baseline.name,
                    "seed_index": index,
                    "seed_idea": seed_idea,
                    **expansion_trace,
                }
            )
        except Exception as exc:
            expansion_errors.append(
                {
                    "seed_index": str(index),
                    "idea_name": seed_idea.get("idea_name", ""),
                    "error": str(exc),
                }
            )
    if expansion_errors:
        graph.metadata["ai_researcher_proxy_expansion_errors"] = expansion_errors
    if not candidates:
        raise ValueError("AI-Researcher-style expansion did not produce any valid candidate proposals.")

    graph.metadata["ai_researcher_proxy_candidate_count"] = len(candidates)
    graph.metadata["ai_researcher_proxy_candidates"] = [
        _proposal_as_prompt_payload(candidate)
        for candidate in candidates
    ]

    emit_progress(
        graph,
        progress_callback,
        stage="baseline_candidate_selection",
        message=f"Baseline '{baseline.name}': ranking expanded candidates.",
    )
    ranking_error = ""
    try:
        selection_payload, selection_trace = _llm_json_object(
            backend,
            role="BaselineCandidateSelection",
            messages=[
                {"role": "system", "content": _ai_researcher_ranking_system_prompt(baseline)},
                {"role": "user", "content": _ai_researcher_ranking_user_prompt(graph, baseline, candidates)},
            ],
            temperature=0.0,
            max_tokens=1600,
        )
        graph.metadata.setdefault("baseline_traces", []).append(
            {"stage": "candidate_selection", "baseline": baseline.name, **selection_trace}
        )
        selection_reason = _coerce_string(selection_payload.get("reason"))
        scores = selection_payload.get("scores", [])
        score_rows = scores if isinstance(scores, list) else []
        if isinstance(scores, list):
            graph.metadata["ai_researcher_proxy_scores"] = scores
        llm_overall_by_index: dict[int, float] = {}
        for row in score_rows:
            if not isinstance(row, dict):
                continue
            try:
                row_index = int(row.get("index", -1))
            except (TypeError, ValueError):
                continue
            try:
                llm_overall_by_index[row_index] = float(row.get("overall", 0.0))
            except (TypeError, ValueError):
                llm_overall_by_index[row_index] = 0.0

        combined_rows: list[dict[str, float | int]] = []
        for index, candidate in enumerate(candidates):
            candidate_for_scoring = _baseline_postprocess_proposal(graph, baseline, candidate)
            topic_fidelity = _ai_researcher_topic_fidelity_score(graph, candidate_for_scoring)
            llm_overall = llm_overall_by_index.get(index, 0.0)
            combined_score = llm_overall + 3.0 * topic_fidelity
            combined_rows.append(
                {
                    "index": index,
                    "llm_overall": round(llm_overall, 3),
                    "topic_fidelity": round(topic_fidelity, 3),
                    "combined_score": round(combined_score, 3),
                }
            )
        graph.metadata["ai_researcher_proxy_combined_scores"] = combined_rows
        best_row = max(combined_rows, key=lambda item: float(item["combined_score"]))
        selected_index = int(best_row["index"])
        selected = _baseline_postprocess_proposal(graph, baseline, candidates[selected_index])
        if selection_reason:
            try:
                llm_selected_index = int(selection_payload.get("selected_index", selected_index))
            except (TypeError, ValueError):
                llm_selected_index = selected_index
            if selected_index != llm_selected_index:
                graph.metadata["baseline_selection_reason"] = (
                    selection_reason
                    + " Final selection was adjusted by a deterministic topic-fidelity reranker to reduce benchmark drift."
                )
            else:
                graph.metadata["baseline_selection_reason"] = selection_reason
        return selected
    except Exception as exc:
        ranking_error = str(exc)

    graph.metadata["baseline_selection_error"] = ranking_error
    return _baseline_postprocess_proposal(graph, baseline, candidates[0])


def _llm_self_refine_proposal(
    graph: IdeaGraph,
    baseline: BaselineSpec,
    backend: OpenAICompatibleCollaborationBackend,
) -> FinalProposal:
    draft = _llm_direct_proposal(graph, baseline, backend)
    critique_payload, critique_trace = _llm_json_object(
        backend,
        role="BaselineCritique",
        messages=[
            {"role": "system", "content": _critique_system_prompt(baseline)},
            {"role": "user", "content": _critique_user_prompt(graph, baseline, draft)},
        ],
        temperature=0.0,
        max_tokens=1200,
    )
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "self_refine_critique", "baseline": baseline.name, **critique_trace}
    )
    payload, refine_trace = _llm_json_object(
        backend,
        role="BaselineRevision",
        messages=[
            {"role": "system", "content": _refine_system_prompt(baseline)},
            {"role": "user", "content": _refine_user_prompt(graph, baseline, draft, critique_payload)},
        ],
        temperature=0.25,
        max_tokens=1500,
    )
    graph.metadata.setdefault("baseline_traces", []).append(
        {"stage": "self_refine_revision", "baseline": baseline.name, **refine_trace}
    )
    proposal = _proposal_from_payload(payload)
    return _baseline_postprocess_proposal(graph, baseline, proposal)


def _build_baseline_graph(
    instance,
    *,
    baseline: BaselineSpec,
) -> IdeaGraph:
    graph = IdeaGraph(topic=instance.topic, literature=list(instance.literature), metadata=dict(instance.metadata))
    graph.metadata.setdefault(
        "literature_grounding",
        build_literature_grounding(
            literature=graph.literature,
            metadata=_generation_metadata(graph),
        ).as_dict(),
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


def _maybe_build_runtime_controller(graph: IdeaGraph, baseline: BaselineSpec) -> tuple[Any | None, dict[str, Any] | None]:
    runtime_kind = str(graph.metadata.get("runtime_controller_kind", "")).strip() or baseline.runtime_controller
    if not runtime_kind:
        return None, None

    if runtime_kind == "text_critic_rerank":
        model_path = Path(
            str(
                graph.metadata.get("runtime_controller_model_path")
                or _default_text_critic_model_path()
            )
        )
        if not model_path.exists():
            graph.metadata["runtime_controller_error"] = f"Missing runtime controller model at {model_path}."
            return None, None

        model = load_pickled_text_critic_model(str(model_path))
        config = TextCriticRuntimeConfig(
            tau_override=float(graph.metadata.get("runtime_controller_tau_override", 0.05)),
            tau_commit=float(graph.metadata.get("runtime_controller_tau_commit", 0.08)),
            gamma_commit=float(graph.metadata.get("runtime_controller_gamma_commit", 0.60)),
            min_commit_round=int(graph.metadata.get("runtime_controller_min_commit_round", 2)),
            use_commit=bool(graph.metadata.get("runtime_controller_use_commit", False)),
            guard_support_threshold=float(
                graph.metadata.get("runtime_controller_guard_support_threshold", 0.66)
            ),
            guard_support_gain_floor=float(
                graph.metadata.get("runtime_controller_guard_support_gain_floor", 0.10)
            ),
            guard_requires_contradiction_progress=bool(
                graph.metadata.get("runtime_controller_guard_requires_contradiction_progress", False)
            ),
        )
        controller_metadata = {
            "kind": "text_critic_rerank",
            "model_path": str(model_path.resolve()),
            "use_commit": bool(config.use_commit),
            "tau_override": float(config.tau_override),
        }
        graph.metadata["runtime_controller_loaded"] = controller_metadata
        return model, {"config": config, **controller_metadata}

    if runtime_kind == "relation_graph_critic_rerank":
        model_dir = Path(
            str(
                graph.metadata.get("runtime_controller_model_dir")
                or _default_relation_graph_runtime_model_dir()
            )
        )
        if not model_dir.exists():
            graph.metadata["runtime_controller_error"] = f"Missing runtime controller model directory at {model_dir}."
            return None, None

        runtime_bundle = load_relation_graph_runtime_bundle(model_dir)
        config = RelationGraphRuntimeConfig(
            tau_override=float(graph.metadata.get("runtime_controller_tau_override", 0.05)),
            tau_commit=float(graph.metadata.get("runtime_controller_tau_commit", 0.08)),
            gamma_commit=float(graph.metadata.get("runtime_controller_gamma_commit", 0.60)),
            min_commit_round=int(graph.metadata.get("runtime_controller_min_commit_round", 2)),
            use_commit=bool(graph.metadata.get("runtime_controller_use_commit", False)),
            guard_support_threshold=float(
                graph.metadata.get("runtime_controller_guard_support_threshold", 0.66)
            ),
            guard_support_gain_floor=float(
                graph.metadata.get("runtime_controller_guard_support_gain_floor", 0.10)
            ),
            guard_requires_contradiction_progress=bool(
                graph.metadata.get("runtime_controller_guard_requires_contradiction_progress", False)
            ),
        )
        controller_metadata = {
            "kind": "relation_graph_critic_rerank",
            "model_dir": str(model_dir.resolve()),
            "model_path": str(model_dir.resolve()),
            "use_commit": bool(config.use_commit),
            "tau_override": float(config.tau_override),
        }
        graph.metadata["runtime_controller_loaded"] = controller_metadata
        return runtime_bundle, {"config": config, **controller_metadata}

    graph.metadata["runtime_controller_error"] = f"Unsupported runtime controller kind '{runtime_kind}'."
    return None, None


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
        _coerce_string(instance.metadata.get("baseline_name")) != baseline.name
        or "benchmark_input_packet" not in instance.metadata
    ):
        instance = attach_baseline_metadata(instance, baseline_name=baseline_name, io_mode="auto")

    if baseline.strategy == "evolving_graph":
        baseline_graph = _build_baseline_graph(instance, baseline=baseline)
        runtime_controller = None
        runtime_controller_metadata = None
        if bool(baseline_graph.metadata.get("runtime_controller_enabled", False)):
            runtime_controller_kind = str(
                baseline_graph.metadata.get("runtime_controller_kind", baseline.runtime_controller)
            ).strip()
            try:
                runtime_controller, runtime_controller_metadata = _maybe_build_runtime_controller(
                    baseline_graph,
                    baseline,
                )
            except Exception as exc:
                baseline_graph.metadata["runtime_controller_error"] = str(exc)
                raise RuntimeError(
                    f"Runtime controller '{runtime_controller_kind or baseline.runtime_controller}' failed to load "
                    f"for baseline '{baseline.name}': {exc}"
                ) from exc
            if runtime_controller is None or runtime_controller_metadata is None:
                error_detail = str(
                    baseline_graph.metadata.get("runtime_controller_error", "unknown runtime controller load failure")
                ).strip()
                raise RuntimeError(
                    f"Runtime controller '{runtime_controller_kind or baseline.runtime_controller}' failed "
                    f"for baseline '{baseline.name}': {error_detail}"
                )
        return run_experiment(
            topic=instance.topic,
            literature=list(instance.literature),
            metadata=dict(baseline_graph.metadata),
            collaboration_backend=collaboration_backend,
            progress_callback=progress_callback,
            max_rounds=max_rounds,
            stop_when_mature=stop_when_mature,
            runtime_controller=runtime_controller,
            runtime_controller_metadata=runtime_controller_metadata,
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
        from .external_baselines import run_external_baseline

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
            try:
                proposal = _llm_direct_proposal(graph, baseline, collaboration_backend)
            except Exception as exc:
                graph.metadata["baseline_generation_error"] = str(exc)
                emit_progress(
                    graph,
                    progress_callback,
                    stage="baseline_fallback",
                    message=(
                        f"Baseline '{baseline.name}' returned an invalid LLM response. "
                        "Falling back to the deterministic baseline implementation."
                    ),
                )
                proposal = _deterministic_direct_proposal(graph, baseline)
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
            try:
                proposal = _llm_candidate_rank_proposal(
                    graph,
                    baseline,
                    collaboration_backend,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                graph.metadata["baseline_generation_error"] = str(exc)
                emit_progress(
                    graph,
                    progress_callback,
                    stage="baseline_fallback",
                    message=(
                        f"Baseline '{baseline.name}' returned an invalid LLM response. "
                        "Falling back to the deterministic baseline implementation."
                    ),
                )
                draft = _deterministic_direct_proposal(graph, baseline)
                proposal = _deterministic_refine_proposal(graph, draft, baseline)
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
            try:
                proposal = _llm_self_refine_proposal(graph, baseline, collaboration_backend)
            except Exception as exc:
                graph.metadata["baseline_generation_error"] = str(exc)
                emit_progress(
                    graph,
                    progress_callback,
                    stage="baseline_fallback",
                    message=(
                        f"Baseline '{baseline.name}' returned an invalid LLM response. "
                        "Falling back to the deterministic baseline implementation."
                    ),
                )
                draft = _deterministic_direct_proposal(graph, baseline)
                proposal = _deterministic_refine_proposal(graph, draft, baseline)
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
