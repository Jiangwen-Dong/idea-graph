from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


@dataclass(frozen=True)
class LiteratureGrounding:
    source: str
    target_paper: str
    reference_titles: list[str]
    design_highlights: list[str]
    dataset_items: list[str]
    metric_items: list[str]
    existing_methods_summary: str
    experiment_plan_summary: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _first_sentence(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    for separator in (". ", "! ", "? "):
        if separator in cleaned:
            return cleaned.split(separator, 1)[0].strip().rstrip(".!?") + "."
    return cleaned.rstrip(".!?") + "."


def _unique_strings(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _split_outside_parentheses(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def _join_natural(items: list[str]) -> str:
    cleaned = [item for item in (_clean_text(item) for item in items) if item]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _reference_titles(literature: list[str], metadata: dict[str, Any]) -> list[str]:
    titles = metadata.get("reference_titles", [])
    extracted: list[str] = []
    if isinstance(titles, list):
        extracted.extend(_clean_text(item) for item in titles if _clean_text(item))
    for item in literature:
        title = _clean_text(str(item).split("|", 1)[0])
        if title:
            extracted.append(title)
    return _unique_strings(extracted)


def _raw_record(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = metadata.get("raw_record", {})
    return payload if isinstance(payload, dict) else {}


def _summary(metadata: dict[str, Any]) -> dict[str, Any]:
    raw_record = _raw_record(metadata)
    payload = raw_record.get("summary", {})
    return payload if isinstance(payload, dict) else {}


def _method_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    summary = _summary(metadata)
    payload = summary.get("method", {})
    return payload if isinstance(payload, dict) else {}


def _target_paper(metadata: dict[str, Any]) -> str:
    return _clean_text(metadata.get("target_paper", ""))


def _paper_grounding(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = metadata.get("paper_grounding", {})
    return payload if isinstance(payload, dict) else {}


def _target_paper_snippet(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = _paper_grounding(metadata).get("target_paper_snippet", {})
    return payload if isinstance(payload, dict) else {}


def _reference_paper_snippets(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _paper_grounding(metadata).get("reference_paper_snippets", [])
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _method_summary(metadata: dict[str, Any]) -> str:
    target_snippet = _target_paper_snippet(metadata)
    for key in ("method", "abstract", "introduction", "text_excerpt"):
        value = _clean_text(target_snippet.get(key, ""))
        if value:
            return value
    value = _clean_text(metadata.get("method_summary", ""))
    if value:
        return value
    return _clean_text(_method_payload(metadata).get("targeted_designs_summary", ""))


def _datasets_text(metadata: dict[str, Any]) -> str:
    return _clean_text(_method_payload(metadata).get("datasets", ""))


def _metrics_text(metadata: dict[str, Any]) -> str:
    return _clean_text(_method_payload(metadata).get("metrics", ""))


def _design_highlights(metadata: dict[str, Any], *, limit: int = 3) -> list[str]:
    payload = _method_payload(metadata)
    details = payload.get("targeted_designs_details", [])
    if isinstance(details, list) and details:
        highlights: list[str] = []
        for item in details[:limit]:
            if not isinstance(item, dict):
                continue
            name = _clean_text(item.get("design_name", ""))
            description = _first_sentence(_clean_text(item.get("description", "")))
            if name and description:
                highlights.append(f"{name}: {description}")
            elif name:
                highlights.append(name)
            elif description:
                highlights.append(description)
        return _unique_strings(highlights)[:limit]

    snippet_highlights: list[str] = []
    for snippet in _reference_paper_snippets(metadata)[:limit]:
        resolved_title = _clean_text(snippet.get("resolved_title", ""))
        method = _first_sentence(
            _clean_text(snippet.get("method", "")) or _clean_text(snippet.get("abstract", ""))
        )
        if (
            resolved_title
            and method
            and len(resolved_title) <= 140
            and 40 <= len(method) <= 220
            and "figure" not in method.casefold()
            and "abstract" not in resolved_title.casefold()
        ):
            snippet_highlights.append(f"{resolved_title}: {method}")
    return _unique_strings(snippet_highlights)[:limit]


def _dataset_items(metadata: dict[str, Any]) -> list[str]:
    sentence = _first_sentence(_datasets_text(metadata))
    if not sentence:
        return []
    prefix_candidates = (
        "Experiments were conducted on several datasets, including ",
        "The datasets include ",
        "Datasets include ",
    )
    clause = sentence
    for prefix in prefix_candidates:
        if clause.startswith(prefix):
            clause = clause[len(prefix) :]
            break
    clause = clause.rstrip(".")
    clause = clause.replace(", and ", ", ")
    clause = re.sub(r"^and\s+", "", clause, flags=re.IGNORECASE)

    items: list[str] = []
    for item in _split_outside_parentheses(clause):
        cleaned = _clean_text(item)
        cleaned = re.sub(r"^and\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^custom datasets?\s+", "", cleaned, flags=re.IGNORECASE)
        if cleaned:
            items.append(cleaned)
    return _unique_strings(items)


def _metric_items(metadata: dict[str, Any]) -> list[str]:
    metrics_text = _metrics_text(metadata)
    if not metrics_text:
        return []
    matches = re.findall(r"([A-Z][A-Za-z\- ]+?) \(([A-Z]{2,8})\)", metrics_text)
    if matches:
        cleaned_items = []
        for name, abbr in matches:
            cleaned_name = re.sub(r"^Evaluation metrics include\s+", "", name.strip(), flags=re.IGNORECASE)
            cleaned_items.append(f"{cleaned_name} ({abbr})")
        return _unique_strings(cleaned_items)

    sentence = _first_sentence(metrics_text)
    if not sentence:
        return []
    prefix_candidates = (
        "Evaluation metrics include ",
        "Metrics include ",
    )
    clause = sentence
    for prefix in prefix_candidates:
        if clause.startswith(prefix):
            clause = clause[len(prefix) :]
            break
    clause = clause.rstrip(".").replace(", and ", ", ")
    items = []
    for item in _split_outside_parentheses(clause):
        cleaned = re.sub(r"\s+for .*$", "", _clean_text(item), flags=re.IGNORECASE)
        if cleaned:
            items.append(cleaned)
    return _unique_strings(items)


def build_literature_grounding(
    *,
    literature: list[str],
    metadata: dict[str, Any],
) -> LiteratureGrounding:
    reference_titles = _reference_titles(literature, metadata)[:6]
    target_paper = _target_paper(metadata)
    method_summary = _method_summary(metadata)
    design_highlights = _design_highlights(metadata)
    dataset_items = _dataset_items(metadata)
    metric_items = _metric_items(metadata)
    paper_grounding = _paper_grounding(metadata)
    paper_grounding_source = "paper_snippets" if paper_grounding.get("reference_paper_snippets") or paper_grounding.get("target_paper_snippet") else ""

    existing_parts: list[str] = []
    if reference_titles:
        existing_parts.append(
            "The provided literature context includes "
            + _join_natural(reference_titles[:4])
            + "."
        )
    if method_summary:
        method_sentence = _first_sentence(method_summary)
        if target_paper and method_sentence.casefold().startswith(target_paper.casefold()):
            existing_parts.append(method_sentence)
        elif target_paper:
            existing_parts.append(
                f"The benchmark target paper {target_paper} can be summarized as follows: {method_sentence}"
            )
        else:
            existing_parts.append(f"A representative target method can be summarized as follows: {method_sentence}")
    if design_highlights:
        existing_parts.append(
            "Notable design elements in this context include "
            + _join_natural([item.rstrip(".") for item in design_highlights[:3]])
            + "."
        )
    if not existing_parts:
        existing_parts.append(
            "Only limited literature context is available, so the existing-method summary is provisional."
        )

    experiment_parts: list[str] = []
    if dataset_items:
        experiment_parts.append("Evaluate on " + _join_natural(dataset_items) + ".")
    if metric_items:
        experiment_parts.append("Report " + _join_natural(metric_items) + ".")
    if not experiment_parts:
        experiment_parts.append(
            "Compare against strong baselines using task-relevant datasets, ablations, and quantitative metrics."
        )

    if paper_grounding_source:
        source = paper_grounding_source
    elif any([method_summary, dataset_items, metric_items, design_highlights]):
        source = "metadata_structured"
    else:
        source = "titles_only"
    return LiteratureGrounding(
        source=source,
        target_paper=target_paper,
        reference_titles=reference_titles,
        design_highlights=design_highlights,
        dataset_items=dataset_items,
        metric_items=metric_items,
        existing_methods_summary=" ".join(existing_parts),
        experiment_plan_summary=" ".join(experiment_parts),
    )
