from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ROLE_NAMES = (
    "MechanismProposer",
    "FeasibilityCritic",
    "NoveltyExaminer",
    "EvaluationDesigner",
    "ImpactReframer",
)

NODE_TYPES = (
    "Problem",
    "Hypothesis",
    "Method",
    "Assumption",
    "Risk",
    "EvidenceNeed",
    "EvalPlan",
    "NoveltyClaim",
    "Repair",
)

EDGE_TYPES = (
    "supports",
    "contradicts",
    "refines",
    "depends_on",
    "requires_evidence",
    "overlaps_prior",
    "repairs",
)


@dataclass(frozen=True)
class SeedTemplate:
    anchor_type: str
    anchor_text: str
    support_nodes: tuple[tuple[str, str], ...]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate(text: str, limit: int = 180) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _first_sentence(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    for separator in (". ", "! ", "? "):
        if separator in cleaned:
            return cleaned.split(separator, 1)[0].strip().rstrip(".!?") + "."
    return cleaned.rstrip(".!?") + "."


def _topic_fragment(topic: str) -> str:
    cleaned = _clean_text(topic).rstrip(".")
    return cleaned or "the target research area"


def _coalesce(*values: str) -> str:
    for value in values:
        if _clean_text(value):
            return _clean_text(value)
    return ""


def _reference_titles(literature: list[str], metadata: dict[str, Any]) -> list[str]:
    titles = metadata.get("reference_titles", [])
    if isinstance(titles, list):
        normalized = [_clean_text(item) for item in titles if _clean_text(item)]
        if normalized:
            return normalized
    return [_clean_text(item) for item in literature if _clean_text(item)]


def _method_summary(metadata: dict[str, Any]) -> str:
    raw_record = metadata.get("raw_record", {})
    method_summary = _clean_text(metadata.get("method_summary", ""))
    if method_summary:
        return method_summary
    if isinstance(raw_record, dict):
        summary = raw_record.get("summary", {})
        if isinstance(summary, dict):
            method = summary.get("method", {})
            if isinstance(method, dict):
                return _clean_text(method.get("targeted_designs_summary", ""))
    return ""


def _motivation(metadata: dict[str, Any]) -> str:
    raw_record = metadata.get("raw_record", {})
    motivation = _clean_text(metadata.get("motivation", ""))
    if motivation:
        return motivation
    if isinstance(raw_record, dict):
        summary = raw_record.get("summary", {})
        if isinstance(summary, dict):
            return _clean_text(summary.get("motivation", ""))
    return ""


def _datasets_and_metrics(metadata: dict[str, Any]) -> tuple[str, str]:
    raw_record = metadata.get("raw_record", {})
    if isinstance(raw_record, dict):
        summary = raw_record.get("summary", {})
        if isinstance(summary, dict):
            method = summary.get("method", {})
            if isinstance(method, dict):
                return (
                    _clean_text(method.get("datasets", "")),
                    _clean_text(method.get("metrics", "")),
                )
    return "", ""


def _keyword(metadata: dict[str, Any]) -> str:
    return _clean_text(metadata.get("keyword", ""))


def _target_paper(metadata: dict[str, Any]) -> str:
    return _clean_text(metadata.get("target_paper", ""))


def build_seed_template(
    role: str,
    topic: str,
    *,
    literature: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> SeedTemplate:
    literature = literature or []
    metadata = metadata or {}
    topic_fragment = _topic_fragment(topic)
    references = _reference_titles(literature, metadata)
    reference_hint = _coalesce(references[0] if references else "", "the provided benchmark context")
    motivation = _motivation(metadata)
    method_summary = _method_summary(metadata)
    datasets, metrics = _datasets_and_metrics(metadata)
    keyword = _keyword(metadata)
    target_paper = _target_paper(metadata)
    benchmark_idea = _clean_text(metadata.get("idea", ""))
    novelty_baseline = _coalesce(target_paper, reference_hint if reference_hint != "the provided benchmark context" else "")
    if datasets and metrics:
        evaluation_anchor = f"Evaluate {topic_fragment} on {datasets} using {metrics}."
    elif datasets:
        evaluation_anchor = f"Evaluate {topic_fragment} on {datasets} with benchmark-aware novelty and feasibility checks."
    elif metrics:
        evaluation_anchor = f"Evaluate {topic_fragment} using {metrics} together with novelty and significance checks."
    else:
        evaluation_anchor = ""

    if role == "MechanismProposer":
        anchor_text = _truncate(
            _coalesce(
                _first_sentence(method_summary).replace("LangSplat", target_paper or "This approach"),
                f"Use claim-level graph collaboration to design a concrete research mechanism for {topic_fragment}.",
            )
        )
        method_text = _truncate(
            _coalesce(
                _first_sentence(method_summary),
                f"Build the method around {reference_hint} while keeping the focus on {topic_fragment}.",
            )
        )
        assumption_text = _truncate(
            f"Useful mechanism components for {topic_fragment} can be decomposed into partial claims grounded by {reference_hint}."
        )
        return SeedTemplate(
            anchor_type="Hypothesis",
            anchor_text=anchor_text,
            support_nodes=(
                ("Method", method_text),
                ("Assumption", assumption_text),
            ),
        )

    if role == "FeasibilityCritic":
        risk_text = _truncate(
            _coalesce(
                _first_sentence(motivation),
                f"Research on {topic_fragment} may fail if the design lacks concrete constraints, data, or evaluation.",
            )
        )
        assumption_text = _truncate(
            f"{topic_fragment} needs a compact representation so collaboration does not become noisier than the research problem itself."
        )
        eval_text = _truncate(
            _coalesce(
                f"Monitor support coverage, contradiction repair, and benchmark-specific feasibility signals for {topic_fragment}.",
                "Monitor support coverage and contradiction repair after each round.",
            )
        )
        return SeedTemplate(
            anchor_type="Risk",
            anchor_text=risk_text,
            support_nodes=(
                ("Assumption", assumption_text),
                ("EvalPlan", eval_text),
            ),
        )

    if role == "NoveltyExaminer":
        novelty_text = _truncate(
            _coalesce(
                f"A new angle on {topic_fragment} should be differentiated from {novelty_baseline}."
                if novelty_baseline
                else "",
                f"The key novelty should be a concrete, benchmark-aware idea for {topic_fragment}.",
            )
        )
        evidence_need = _truncate(
            f"Check whether claims about {topic_fragment} overlap with {reference_hint} and other nearby work before asserting novelty."
        )
        return SeedTemplate(
            anchor_type="NoveltyClaim",
            anchor_text=novelty_text,
            support_nodes=(
                ("EvidenceNeed", evidence_need),
            ),
        )

    if role == "EvaluationDesigner":
        eval_anchor = _truncate(
            _coalesce(
                evaluation_anchor,
                f"Evaluate ideas for {topic_fragment} with benchmark-aware novelty, feasibility, coherence, and significance checks.",
            )
        )
        method_text = _truncate(
            _coalesce(
                f"Use the provided references, especially {reference_hint}, as the initial grounding context for {topic_fragment}.",
                f"Use three collaboration rounds after merging seed graphs for {topic_fragment}.",
            )
        )
        risk_text = _truncate(
            f"Without task-specific evaluation for {topic_fragment}, gains may reflect generic fluency rather than scientific quality."
        )
        return SeedTemplate(
            anchor_type="EvalPlan",
            anchor_text=eval_anchor,
            support_nodes=(
                ("Method", method_text),
                ("Risk", risk_text),
            ),
        )

    if role == "ImpactReframer":
        problem_text = _truncate(
            _coalesce(
                _first_sentence(motivation),
                f"Current work on {topic_fragment} still lacks a clear path from partial claims to robust scientific proposals.",
                f"Current work on {keyword or topic_fragment} still needs stronger problem framing and clearer impact.",
            )
        )
        hypothesis_text = _truncate(
            _coalesce(
                _first_sentence(benchmark_idea) if keyword else "",
                f"Preserving disagreement may uncover stronger research directions for {topic_fragment}.",
            )
        )
        novelty_text = _truncate(
            f"Traceable alternatives could make ideation for {topic_fragment} more interpretable than single-draft generation."
        )
        return SeedTemplate(
            anchor_type="Problem",
            anchor_text=problem_text,
            support_nodes=(
                ("Hypothesis", hypothesis_text),
                ("NoveltyClaim", novelty_text),
            ),
        )

    raise ValueError(f"Unknown role: {role}")
