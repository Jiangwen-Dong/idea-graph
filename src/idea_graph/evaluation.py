from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from .literature_grounding import build_literature_grounding
from .models import FinalProposal, IdeaGraph

RUBRIC_NAME = "idea_graph_research_idea_eval_v1"

CATEGORY_WEIGHTS = {
    "expert_style_quality": 0.55,
    "benchmark_alignment": 0.30,
    "graph_process": 0.15,
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "use",
    "with",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\+\.]*")


@dataclass(frozen=True)
class IdeaMetricScore:
    key: str
    display_name: str
    category: str
    score: float
    max_score: float
    rationale: str
    signals: dict[str, float] = field(default_factory=dict)
    available: bool = True


@dataclass(frozen=True)
class IdeaEvaluation:
    rubric_name: str
    benchmark: str
    category_scores: dict[str, float]
    overall_score: float
    metrics: list[IdeaMetricScore]
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _round_score(value: float) -> float:
    return round(value, 2)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _normalize_text(value: Any) -> str:
    cleaned = _clean_text(value).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", cleaned).split())


def _tokens(text: Any) -> set[str]:
    values: set[str] = set()
    for token in TOKEN_PATTERN.findall(_clean_text(text).lower()):
        if len(token) <= 1 or token in STOPWORDS:
            continue
        values.add(token)
    return values


def _token_f1(text_a: Any, text_b: Any) -> float:
    tokens_a = _tokens(text_a)
    tokens_b = _tokens(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    return (2.0 * overlap) / (len(tokens_a) + len(tokens_b))


def _mean(values: list[float]) -> float:
    filtered = [float(value) for value in values]
    if not filtered:
        return 0.0
    return sum(filtered) / len(filtered)


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        cleaned = _clean_text(item)
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _first_sentence(text: Any) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    for separator in (". ", "! ", "? "):
        if separator in cleaned:
            return cleaned.split(separator, 1)[0].strip().rstrip(".!?") + "."
    return cleaned.rstrip(".!?") + "."


def _section_word_count(text: str) -> int:
    return len(_clean_text(text).split())


def _section_length_quality(text: str) -> float:
    count = _section_word_count(text)
    if count == 0:
        return 0.0
    if 12 <= count <= 140:
        return 1.0
    if count < 12:
        return _clamp(count / 12.0)
    if count <= 220:
        return _clamp(1.0 - ((count - 140) / 160.0))
    return _clamp(0.5 - ((count - 220) / 260.0))


def _keyword_coverage(text: str, phrases: list[str]) -> float:
    normalized_text = _normalize_text(text)
    cleaned_phrases = [_normalize_text(item) for item in phrases if _normalize_text(item)]
    if not normalized_text or not cleaned_phrases:
        return 0.0
    hits = 0
    for phrase in cleaned_phrases:
        if phrase in normalized_text:
            hits += 1
            continue
        phrase_tokens = set(phrase.split())
        if phrase_tokens and len(phrase_tokens & set(normalized_text.split())) >= max(1, len(phrase_tokens) // 2):
            hits += 1
    return hits / len(cleaned_phrases)


def _keyword_score(text: str, keywords: list[str]) -> float:
    text_tokens = _tokens(text)
    keyword_tokens = {token for item in keywords for token in _tokens(item)}
    if not text_tokens or not keyword_tokens:
        return 0.0
    return len(text_tokens & keyword_tokens) / len(keyword_tokens)


def _proposal_sections(proposal: FinalProposal | None) -> dict[str, str]:
    proposal = proposal or FinalProposal()
    return {
        "title": _clean_text(proposal.title),
        "problem": _clean_text(proposal.problem),
        "existing_methods": _clean_text(proposal.existing_methods),
        "motivation": _clean_text(proposal.motivation),
        "hypothesis": _clean_text(proposal.hypothesis),
        "method": _clean_text(proposal.method),
        "evaluation": _clean_text(proposal.evaluation),
        "significance": _clean_text(proposal.significance),
        "caveats": _clean_text(proposal.caveats),
    }


def _reference_texts(metadata: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    texts.extend(_list_of_strings(metadata.get("reference_titles")))
    paper_grounding = metadata.get("paper_grounding", {})
    if isinstance(paper_grounding, dict):
        snippets = paper_grounding.get("reference_paper_snippets", [])
        if isinstance(snippets, list):
            for snippet in snippets[:6]:
                if not isinstance(snippet, dict):
                    continue
                title = _clean_text(snippet.get("resolved_title") or snippet.get("raw_title"))
                summary = _first_sentence(
                    snippet.get("method")
                    or snippet.get("abstract")
                    or snippet.get("evaluation")
                    or snippet.get("introduction")
                )
                texts.append(" ".join(item for item in (title, summary) if item))
    return [item for item in texts if item]


def _tracked_nodes(graph: IdeaGraph) -> list[object]:
    return [
        node
        for node in graph.active_nodes()
        if node.type in {"Hypothesis", "Method", "NoveltyClaim", "EvalPlan"}
    ]


def _support_and_evidence_ratios(graph: IdeaGraph) -> tuple[float, float]:
    tracked_nodes = _tracked_nodes(graph)
    if not tracked_nodes:
        return 0.0, 0.0
    supported_ids = {
        edge.target_id
        for edge in graph.edges
        if edge.relation in {"supports", "repairs"}
    }
    supported = 0
    evidenced = 0
    for node in tracked_nodes:
        if node.id in supported_ids or node.evidence:
            supported += 1
        if node.evidence:
            evidenced += 1
    return supported / len(tracked_nodes), evidenced / len(tracked_nodes)


def _risk_and_repair_balance(graph: IdeaGraph) -> float:
    active_nodes = graph.active_nodes()
    risk_count = sum(1 for node in active_nodes if node.type == "Risk")
    repair_count = sum(1 for node in active_nodes if node.type == "Repair")
    if risk_count == 0 and repair_count == 0:
        return 0.5
    return repair_count / max(1, risk_count + repair_count)


def _latest_snapshot_values(graph: IdeaGraph) -> tuple[float, float, float, float]:
    if not graph.round_summaries:
        return 0.0, 1.0, 0.0, 0.0
    snapshot = graph.round_summaries[-1][1]
    completeness = 1.0 if snapshot.completeness else 0.0
    utility_norm = _clamp(snapshot.utility / 12.0)
    return (
        snapshot.support_coverage,
        snapshot.unresolved_contradiction_ratio,
        completeness,
        utility_norm,
    )


def _graph_action_diversity(graph: IdeaGraph) -> float:
    if not graph.actions:
        return 0.0
    unique_kinds = {action.kind for action in graph.actions}
    return len(unique_kinds) / 8.0


def _grounding_payload(graph: IdeaGraph) -> dict[str, Any]:
    payload = graph.metadata.get("literature_grounding", {})
    if isinstance(payload, dict) and payload:
        return payload
    return build_literature_grounding(literature=graph.literature, metadata=graph.metadata).as_dict()


def _available_metric(
    *,
    key: str,
    display_name: str,
    category: str,
    score: float,
    rationale: str,
    signals: dict[str, float],
    available: bool = True,
) -> IdeaMetricScore:
    return IdeaMetricScore(
        key=key,
        display_name=display_name,
        category=category,
        score=_round_score(score),
        max_score=10.0,
        rationale=rationale,
        signals={name: _round_score(value) for name, value in signals.items()},
        available=available,
    )


def _score_clarity_and_coherence(sections: dict[str, str]) -> IdeaMetricScore:
    proposal_sections = [
        sections["problem"],
        sections["existing_methods"],
        sections["motivation"],
        sections["hypothesis"],
        sections["method"],
        sections["evaluation"],
        sections["significance"],
        sections["caveats"],
    ]
    filled = [section for section in proposal_sections if section]
    presence_ratio = len(filled) / len(proposal_sections)
    length_quality = _mean([_section_length_quality(section) for section in filled]) if filled else 0.0
    pairwise = []
    for idx, left in enumerate(filled):
        for right in filled[idx + 1 :]:
            pairwise.append(_token_f1(left, right))
    distinctness = 1.0 - _clamp(_mean(pairwise))
    normalized = _clamp((0.45 * presence_ratio) + (0.30 * length_quality) + (0.25 * distinctness))
    rationale = (
        "Rewards complete, reasonably detailed, and non-redundant section writing."
    )
    return _available_metric(
        key="clarity_coherence",
        display_name="Clarity And Coherence",
        category="expert_style_quality",
        score=normalized * 10.0,
        rationale=rationale,
        signals={
            "presence_ratio": presence_ratio,
            "length_quality": length_quality,
            "distinctness": distinctness,
        },
    )


def _score_novelty(graph: IdeaGraph, sections: dict[str, str], metadata: dict[str, Any]) -> IdeaMetricScore:
    proposal_core = " ".join(
        item
        for item in (
            sections["title"],
            sections["hypothesis"],
            sections["method"],
            sections["significance"],
        )
        if item
    )
    reference_texts = _reference_texts(metadata)
    max_reference_similarity = max((_token_f1(proposal_core, reference) for reference in reference_texts), default=0.0)
    overlap_count = sum(1 for edge in graph.edges if edge.relation == "overlaps_prior")
    novelty_claim_count = sum(1 for node in graph.active_nodes() if node.type == "NoveltyClaim")
    overlap_penalty = overlap_count / max(1, novelty_claim_count + overlap_count)
    novelty_claim_signal = _clamp(novelty_claim_count / 2.0)
    normalized = _clamp(
        (0.45 * (1.0 - max_reference_similarity))
        + (0.35 * (1.0 - overlap_penalty))
        + (0.20 * novelty_claim_signal)
    )
    rationale = (
        "Inspired by the novelty axis in recent ICLR-style evaluations, while penalizing high similarity "
        "to nearby references and explicit prior-overlap markers."
    )
    return _available_metric(
        key="novelty",
        display_name="Novelty",
        category="expert_style_quality",
        score=normalized * 10.0,
        rationale=rationale,
        signals={
            "reference_similarity_penalty": max_reference_similarity,
            "overlap_penalty": overlap_penalty,
            "novelty_claim_signal": novelty_claim_signal,
        },
    )


def _score_significance(sections: dict[str, str], topic: str) -> IdeaMetricScore:
    problem_quality = _section_length_quality(sections["problem"])
    significance_quality = _section_length_quality(sections["significance"])
    topic_alignment = _token_f1(" ".join([sections["problem"], sections["significance"]]), topic)
    normalized = _clamp(
        (0.40 * problem_quality)
        + (0.35 * significance_quality)
        + (0.25 * topic_alignment)
    )
    rationale = (
        "Tracks the excitement/significance style criterion by rewarding a clear problem statement, an explicit "
        "expected contribution, and alignment with the research topic."
    )
    return _available_metric(
        key="significance",
        display_name="Significance",
        category="expert_style_quality",
        score=normalized * 10.0,
        rationale=rationale,
        signals={
            "problem_quality": problem_quality,
            "significance_quality": significance_quality,
            "topic_alignment": topic_alignment,
        },
    )


def _score_experiment_specificity(sections: dict[str, str], dataset_items: list[str], metric_items: list[str]) -> tuple[float, dict[str, float]]:
    evaluation_text = sections["evaluation"]
    dataset_coverage = _keyword_coverage(evaluation_text, dataset_items)
    metric_coverage = _keyword_coverage(evaluation_text, metric_items)
    experiment_terms = [
        "baseline",
        "ablation",
        "benchmark",
        "dataset",
        "metric",
        "evaluate",
        "compare",
        "report",
        "analysis",
        "robustness",
    ]
    experiment_keyword_score = _keyword_score(evaluation_text, experiment_terms)
    specificity = _clamp(
        (0.45 * dataset_coverage)
        + (0.35 * metric_coverage)
        + (0.20 * experiment_keyword_score)
    )
    return specificity, {
        "dataset_coverage": dataset_coverage,
        "metric_coverage": metric_coverage,
        "experiment_keyword_score": experiment_keyword_score,
    }


def _score_feasibility(graph: IdeaGraph, sections: dict[str, str], dataset_items: list[str], metric_items: list[str]) -> IdeaMetricScore:
    support_ratio, _ = _support_and_evidence_ratios(graph)
    _, unresolved_ratio, _, _ = _latest_snapshot_values(graph)
    repair_balance = _risk_and_repair_balance(graph)
    experiment_specificity, specificity_signals = _score_experiment_specificity(sections, dataset_items, metric_items)
    normalized = _clamp(
        (0.35 * (1.0 - unresolved_ratio))
        + (0.25 * support_ratio)
        + (0.20 * repair_balance)
        + (0.20 * experiment_specificity)
    )
    rationale = (
        "Follows the feasibility dimension from prior idea-evaluation papers by combining contradiction resolution, "
        "support coverage, repairs, and the practicality of the evaluation plan."
    )
    return _available_metric(
        key="feasibility",
        display_name="Feasibility",
        category="expert_style_quality",
        score=normalized * 10.0,
        rationale=rationale,
        signals={
            "contradiction_resolution": 1.0 - unresolved_ratio,
            "support_ratio": support_ratio,
            "repair_balance": repair_balance,
            **specificity_signals,
        },
    )


def _score_effectiveness(graph: IdeaGraph, sections: dict[str, str], dataset_items: list[str], metric_items: list[str]) -> IdeaMetricScore:
    completeness_signal = 0.0
    active_types = {node.type for node in graph.active_nodes()}
    if {"Problem", "Hypothesis", "Method", "EvalPlan"}.issubset(active_types):
        completeness_signal = 1.0
    proposal_chain_signal = _mean(
        [
            1.0 if sections["hypothesis"] else 0.0,
            1.0 if sections["method"] else 0.0,
            1.0 if sections["evaluation"] else 0.0,
        ]
    )
    logic_edge_signal = _clamp(
        sum(
            1
            for edge in graph.edges
            if edge.relation in {"supports", "depends_on"}
            and edge.source_id in graph.nodes
            and edge.target_id in graph.nodes
            and graph.nodes[edge.source_id].type in {"Method", "Hypothesis", "EvalPlan"}
            and graph.nodes[edge.target_id].type in {"Problem", "Hypothesis", "Method", "EvalPlan"}
        )
        / 4.0
    )
    experiment_specificity, specificity_signals = _score_experiment_specificity(sections, dataset_items, metric_items)
    normalized = _clamp(
        (0.30 * completeness_signal)
        + (0.20 * proposal_chain_signal)
        + (0.20 * logic_edge_signal)
        + (0.30 * experiment_specificity)
    )
    rationale = (
        "Approximates effectiveness/testability by checking whether the idea forms a coherent problem-hypothesis-method-"
        "evaluation chain with concrete experimental hooks."
    )
    return _available_metric(
        key="effectiveness",
        display_name="Effectiveness And Testability",
        category="expert_style_quality",
        score=normalized * 10.0,
        rationale=rationale,
        signals={
            "graph_completeness": completeness_signal,
            "proposal_chain_signal": proposal_chain_signal,
            "logic_edge_signal": logic_edge_signal,
            **specificity_signals,
        },
    )


def _score_grounding(graph: IdeaGraph, sections: dict[str, str], metadata: dict[str, Any], grounding: dict[str, Any]) -> IdeaMetricScore:
    _, evidence_ratio = _support_and_evidence_ratios(graph)
    existing_methods_alignment = _token_f1(
        sections["existing_methods"],
        grounding.get("existing_methods_summary", ""),
    )
    dataset_items = _list_of_strings(grounding.get("dataset_items"))
    metric_items = _list_of_strings(grounding.get("metric_items"))
    experiment_specificity, specificity_signals = _score_experiment_specificity(sections, dataset_items, metric_items)
    normalized = _clamp(
        (0.45 * evidence_ratio)
        + (0.30 * existing_methods_alignment)
        + (0.25 * experiment_specificity)
    )
    rationale = (
        "Captures the AI Idea Bench style of judging ideas against general reference material by combining graph evidence, "
        "literature-aware existing-methods discussion, and experiment grounding."
    )
    return _available_metric(
        key="grounding",
        display_name="Literature Grounding",
        category="benchmark_alignment",
        score=normalized * 10.0,
        rationale=rationale,
        signals={
            "evidence_ratio": evidence_ratio,
            "existing_methods_alignment": existing_methods_alignment,
            **specificity_signals,
        },
    )


def _topic_keywords(metadata: dict[str, Any], topic: str) -> list[str]:
    raw_record = metadata.get("raw_record", {})
    keywords: list[str] = []
    if isinstance(raw_record, dict):
        summary = raw_record.get("summary", {})
        if isinstance(summary, dict):
            split_topic = summary.get("split_topic", [])
            if isinstance(split_topic, list):
                for item in split_topic[:6]:
                    if not isinstance(item, dict):
                        continue
                    keyword = _clean_text(item.get("keyword"))
                    if keyword:
                        keywords.append(keyword)
    if not keywords:
        keywords.append(topic)
    return keywords


def _score_topic_alignment(sections: dict[str, str], metadata: dict[str, Any], topic: str) -> IdeaMetricScore:
    proposal_focus = " ".join(
        item for item in (sections["problem"], sections["hypothesis"], sections["method"]) if item
    )
    similarity = _token_f1(proposal_focus, topic)
    keyword_signal = _keyword_score(proposal_focus, _topic_keywords(metadata, topic))
    normalized = _clamp((0.65 * similarity) + (0.35 * keyword_signal))
    rationale = (
        "Measures whether the generated idea stays aligned with the benchmark topic and its decomposed keywords."
    )
    return _available_metric(
        key="topic_alignment",
        display_name="Topic Alignment",
        category="benchmark_alignment",
        score=normalized * 10.0,
        rationale=rationale,
        signals={
            "topic_similarity": similarity,
            "keyword_signal": keyword_signal,
        },
    )


def _score_ground_truth_concordance(sections: dict[str, str], metadata: dict[str, Any]) -> IdeaMetricScore:
    gold_motivation = _clean_text(metadata.get("motivation"))
    gold_method = _clean_text(metadata.get("method_summary"))
    paper_grounding = metadata.get("paper_grounding", {})
    if not gold_method and isinstance(paper_grounding, dict):
        target_snippet = paper_grounding.get("target_paper_snippet", {})
        if isinstance(target_snippet, dict):
            gold_method = _first_sentence(
                target_snippet.get("method")
                or target_snippet.get("abstract")
                or target_snippet.get("introduction")
            )
    if not gold_motivation and not gold_method:
        return _available_metric(
            key="ground_truth_concordance",
            display_name="Ground-Truth Concordance",
            category="benchmark_alignment",
            score=0.0,
            rationale="Not available because this instance does not expose held-out target-paper motivation or method metadata.",
            signals={},
            available=False,
        )
    motivation_similarity = _token_f1(
        " ".join([sections["problem"], sections["motivation"]]),
        gold_motivation,
    ) if gold_motivation else 0.0
    method_similarity = _token_f1(
        " ".join([sections["hypothesis"], sections["method"]]),
        gold_method,
    ) if gold_method else 0.0
    weights = []
    if gold_motivation:
        weights.append((0.4, motivation_similarity))
    if gold_method:
        weights.append((0.6, method_similarity))
    total_weight = sum(weight for weight, _ in weights) or 1.0
    normalized = _clamp(sum(weight * value for weight, value in weights) / total_weight)
    rationale = (
        "Implements the AI Idea Bench ground-truth-content dimension by comparing the proposal against held-out "
        "motivation and method summaries from the target paper."
    )
    return _available_metric(
        key="ground_truth_concordance",
        display_name="Ground-Truth Concordance",
        category="benchmark_alignment",
        score=normalized * 10.0,
        rationale=rationale,
        signals={
            "motivation_similarity": motivation_similarity,
            "method_similarity": method_similarity,
        },
    )


def _score_experiment_alignment(sections: dict[str, str], grounding: dict[str, Any]) -> IdeaMetricScore:
    dataset_items = _list_of_strings(grounding.get("dataset_items"))
    metric_items = _list_of_strings(grounding.get("metric_items"))
    if not dataset_items and not metric_items:
        return _available_metric(
            key="experiment_alignment",
            display_name="Experiment Alignment",
            category="benchmark_alignment",
            score=0.0,
            rationale="Not available because benchmark dataset/metric metadata is missing.",
            signals={},
            available=False,
        )
    dataset_coverage = _keyword_coverage(sections["evaluation"], dataset_items)
    metric_coverage = _keyword_coverage(sections["evaluation"], metric_items)
    available_values = []
    if dataset_items:
        available_values.append(dataset_coverage)
    if metric_items:
        available_values.append(metric_coverage)
    normalized = _clamp(_mean(available_values))
    rationale = (
        "Checks whether the evaluation section matches benchmark datasets and metrics, which is a core part of "
        "AI Idea Bench style alignment."
    )
    return _available_metric(
        key="experiment_alignment",
        display_name="Experiment Alignment",
        category="benchmark_alignment",
        score=normalized * 10.0,
        rationale=rationale,
        signals={
            "dataset_coverage": dataset_coverage,
            "metric_coverage": metric_coverage,
        },
    )


def _score_graph_maturity(graph: IdeaGraph) -> IdeaMetricScore:
    support_coverage, unresolved_ratio, completeness, utility_norm = _latest_snapshot_values(graph)
    action_diversity = _clamp(_graph_action_diversity(graph))
    normalized = _clamp(
        (0.30 * support_coverage)
        + (0.25 * (1.0 - unresolved_ratio))
        + (0.20 * completeness)
        + (0.15 * utility_norm)
        + (0.10 * action_diversity)
    )
    rationale = (
        "Uses the collaboration trajectory itself as a process metric: mature graphs should show coverage, repaired "
        "contradictions, a complete claim chain, and some action diversity."
    )
    return _available_metric(
        key="graph_maturity",
        display_name="Graph Maturity",
        category="graph_process",
        score=normalized * 10.0,
        rationale=rationale,
        signals={
            "support_coverage": support_coverage,
            "contradiction_resolution": 1.0 - unresolved_ratio,
            "completeness": completeness,
            "utility_norm": utility_norm,
            "action_diversity": action_diversity,
        },
    )


def _category_scores(metrics: list[IdeaMetricScore]) -> dict[str, float]:
    by_category: dict[str, list[float]] = {}
    for metric in metrics:
        if not metric.available:
            continue
        by_category.setdefault(metric.category, []).append(metric.score)
    return {
        category: _round_score(_mean(scores))
        for category, scores in by_category.items()
    }


def _overall_score(category_scores: dict[str, float]) -> float:
    active_weights = {
        category: weight
        for category, weight in CATEGORY_WEIGHTS.items()
        if category in category_scores
    }
    if not active_weights:
        return 0.0
    weight_sum = sum(active_weights.values())
    return _round_score(
        sum(category_scores[category] * weight for category, weight in active_weights.items()) / weight_sum
    )


def evaluate_graph(graph: IdeaGraph) -> IdeaEvaluation:
    sections = _proposal_sections(graph.final_proposal)
    metadata = graph.metadata
    grounding = _grounding_payload(graph)
    dataset_items = _list_of_strings(grounding.get("dataset_items"))
    metric_items = _list_of_strings(grounding.get("metric_items"))

    metrics = [
        _score_novelty(graph, sections, metadata),
        _score_significance(sections, graph.topic),
        _score_feasibility(graph, sections, dataset_items, metric_items),
        _score_effectiveness(graph, sections, dataset_items, metric_items),
        _score_clarity_and_coherence(sections),
        _score_grounding(graph, sections, metadata, grounding),
        _score_topic_alignment(sections, metadata, graph.topic),
        _score_ground_truth_concordance(sections, metadata),
        _score_experiment_alignment(sections, grounding),
        _score_graph_maturity(graph),
    ]

    category_scores = _category_scores(metrics)
    notes = [
        "Expert-style quality metrics are inspired by recent ICLR-style idea evaluation rubrics, especially novelty, feasibility, effectiveness, and overall scientific promise.",
        "Benchmark-alignment metrics are inspired by AI Idea Bench 2025, which evaluates both concordance with held-out ground-truth papers and judgment against general reference material.",
        "These scores are deterministic local surrogates rather than human-review or LLM-judge scores, so they are best used for rapid iteration and ablation comparisons.",
    ]
    return IdeaEvaluation(
        rubric_name=RUBRIC_NAME,
        benchmark=_clean_text(metadata.get("benchmark", "")),
        category_scores=category_scores,
        overall_score=_overall_score(category_scores),
        metrics=metrics,
        notes=notes,
    )


def format_evaluation_markdown(evaluation: IdeaEvaluation) -> str:
    lines = [
        "# Idea Evaluation",
        "",
        f"- Rubric: `{evaluation.rubric_name}`",
        f"- Benchmark: `{evaluation.benchmark or 'none'}`",
        f"- Overall score: `{evaluation.overall_score:.2f}/10`",
        "",
        "## Category Scores",
        "",
    ]
    for category, score in evaluation.category_scores.items():
        lines.append(f"- `{category}`: `{score:.2f}/10`")
    lines.extend(["", "## Metric Breakdown", ""])
    for metric in evaluation.metrics:
        availability = "" if metric.available else " (not available for this instance)"
        lines.append(f"### {metric.display_name}")
        lines.append("")
        lines.append(f"- Score: `{metric.score:.2f}/{metric.max_score:.0f}`{availability}")
        lines.append(f"- Category: `{metric.category}`")
        lines.append(f"- Rationale: {metric.rationale}")
        if metric.signals:
            lines.append("- Signals:")
            for key, value in metric.signals.items():
                lines.append(f"  - `{key}`: `{value:.2f}`")
        lines.append("")
    if evaluation.notes:
        lines.extend(["## Notes", ""])
        for note in evaluation.notes:
            lines.append(f"- {note}")
    return "\n".join(lines).strip() + "\n"
