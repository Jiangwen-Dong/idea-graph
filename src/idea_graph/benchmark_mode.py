from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .instances import ExperimentInstance


CANONICAL_OUTPUT_FIELDS = (
    "title",
    "problem",
    "existing_methods",
    "motivation",
    "hypothesis",
    "method",
    "evaluation",
    "significance",
    "caveats",
)


@dataclass(frozen=True)
class OutputSectionSpec:
    field: str
    title: str
    guidance: str


OUTPUT_SECTION_SPECS = (
    OutputSectionSpec("title", "Title", "8-18 word research-idea title."),
    OutputSectionSpec("problem", "Problem", "1-3 sentences describing the core gap or problem."),
    OutputSectionSpec(
        "existing_methods",
        "Existing Methods and Limitation",
        "2-4 sentences grounded in the provided benchmark context.",
    ),
    OutputSectionSpec(
        "motivation",
        "Motivation",
        "1-2 sentences explaining why the idea matters now.",
    ),
    OutputSectionSpec(
        "hypothesis",
        "Core Idea / Hypothesis",
        "1-2 sentences stating the main scientific claim.",
    ),
    OutputSectionSpec(
        "method",
        "Method Sketch",
        "3-5 sentences describing the proposed method at a high level.",
    ),
    OutputSectionSpec(
        "evaluation",
        "Experiment Plan",
        "2-5 sentences on datasets, metrics, baselines, or validation strategy.",
    ),
    OutputSectionSpec(
        "significance",
        "Expected Contribution",
        "1-2 sentences on expected value or significance.",
    ),
    OutputSectionSpec(
        "caveats",
        "Risk / Caveat",
        "1-2 sentences on assumptions, failure modes, or risks.",
    ),
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _first_sentence(text: Any, *, max_chars: int = 220) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    for separator in (". ", "! ", "? "):
        if separator in cleaned:
            cleaned = cleaned.split(separator, 1)[0].strip().rstrip(".!?") + "."
            break
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."
    return cleaned


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text:
            normalized.append(text)
    return normalized


def _reference_snippet_entries(metadata: dict[str, Any], *, limit: int = 6) -> list[dict[str, str]]:
    paper_grounding = metadata.get("paper_grounding", {})
    if not isinstance(paper_grounding, dict):
        return []
    raw_snippets = paper_grounding.get("reference_paper_snippets", [])
    if not isinstance(raw_snippets, list):
        return []

    entries: list[dict[str, str]] = []
    for item in raw_snippets[:limit]:
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("resolved_title") or item.get("raw_title"))
        if not title:
            continue
        snippet = _first_sentence(
            item.get("method")
            or item.get("abstract")
            or item.get("evaluation")
            or item.get("introduction")
        )
        entry = {"title": title}
        if snippet:
            entry["snippet"] = snippet
        entries.append(entry)
    return entries


def _fallback_reference_entries(instance: ExperimentInstance, *, limit: int = 6) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    titles = _list_of_strings(instance.metadata.get("reference_titles"))
    for raw in list(instance.literature) + titles:
        title = _clean_text(raw.split("|", 1)[0] if "|" in str(raw) else raw)
        if not title:
            continue
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        entries.append({"title": title})
        if len(entries) >= limit:
            break
    return entries


def _canonical_output_schema() -> dict[str, object]:
    return {
        "fields": list(CANONICAL_OUTPUT_FIELDS),
        "sections": [asdict(spec) for spec in OUTPUT_SECTION_SPECS],
        "notes": [
            "Return exactly one structured research idea.",
            "Do not output an abstract field.",
            "Avoid repeating the same content across multiple sections.",
        ],
    }


def build_benchmark_input_packet(instance: ExperimentInstance) -> dict[str, object]:
    benchmark = _clean_text(instance.metadata.get("benchmark"))
    topic = _clean_text(instance.topic)

    if benchmark == "AI_Idea_Bench_2025":
        references = _reference_snippet_entries(instance.metadata) or _fallback_reference_entries(instance)
        return {
            "mode": "benchmark",
            "benchmark": benchmark,
            "task_instruction": (
                "Generate one structured research idea using only the benchmark topic and the provided "
                "inspiration/reference packet. Hidden target-paper fields are reserved for evaluation."
            ),
            "topic": topic,
            "reference_packet": references,
            "constraints": [
                "Use only the topic and reference packet.",
                "Do not assume access to the hidden target paper.",
                "Keep the output concise and sectioned.",
            ],
            "output_schema": _canonical_output_schema(),
        }

    if benchmark == "liveideabench":
        keyword = _clean_text(instance.metadata.get("keyword")) or topic
        return {
            "mode": "benchmark",
            "benchmark": benchmark,
            "task_instruction": (
                "Generate one structured research idea from the benchmark keyword only. "
                "Treat scored benchmark idea text as held-out metadata rather than generation input."
            ),
            "topic": topic,
            "keyword": keyword,
            "reference_packet": [],
            "constraints": [
                "Do not use held-out scored benchmark idea text.",
                "Do not add external retrieval in benchmark mode.",
                "Keep the output concise and sectioned.",
            ],
            "output_schema": _canonical_output_schema(),
        }

    return {
        "mode": "assistant",
        "benchmark": benchmark or "",
        "task_instruction": (
            "Generate one structured research idea from the provided topic and context."
        ),
        "topic": topic,
        "reference_packet": _fallback_reference_entries(instance),
        "constraints": [
            "Use the provided topic and context packet.",
            "Keep the output concise and sectioned.",
        ],
        "output_schema": _canonical_output_schema(),
    }


def build_generation_safe_metadata(metadata: dict[str, Any]) -> dict[str, object]:
    benchmark = _clean_text(metadata.get("benchmark"))
    if not benchmark:
        return dict(metadata)

    blocked_keys = {
        "target_paper",
        "target_paper_path",
        "motivation",
        "method_summary",
        "raw_record",
        "idea",
        "full_response",
        "raw_critique",
        "parsed_score",
        "originality",
        "feasibility",
        "fluency",
        "average_score",
        "literature_grounding",
    }
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        if key in blocked_keys:
            continue
        if key == "paper_grounding":
            if isinstance(value, dict):
                safe[key] = {
                    "reference_paper_snippets": _reference_snippet_entries({"paper_grounding": value})
                }
            continue
        safe[key] = value
    return safe


def resolve_io_mode(instance: ExperimentInstance, requested_mode: str = "auto") -> str:
    normalized = _clean_text(requested_mode).lower() or "auto"
    if normalized not in {"auto", "benchmark", "assistant"}:
        raise ValueError(f"Unsupported io_mode '{requested_mode}'.")
    if normalized == "auto":
        return "benchmark" if _clean_text(instance.metadata.get("benchmark")) else "assistant"
    return normalized


def apply_io_mode(instance: ExperimentInstance, *, io_mode: str = "auto") -> ExperimentInstance:
    resolved_mode = resolve_io_mode(instance, requested_mode=io_mode)
    metadata = dict(instance.metadata)

    if resolved_mode == "benchmark":
        packet = build_benchmark_input_packet(instance)
    else:
        packet = build_benchmark_input_packet(
            ExperimentInstance(
                name=instance.name,
                topic=instance.topic,
                literature=list(instance.literature),
                source_path=instance.source_path,
                metadata={},
            )
        )

    metadata["io_mode"] = resolved_mode
    metadata["benchmark_mode"] = resolved_mode == "benchmark"
    metadata["benchmark_input_packet"] = packet
    metadata["output_schema"] = packet["output_schema"]
    metadata["output_sections"] = [asdict(spec) for spec in OUTPUT_SECTION_SPECS]
    metadata["task_instruction"] = packet["task_instruction"]
    metadata["generation_safe_metadata"] = build_generation_safe_metadata(metadata)

    return ExperimentInstance(
        name=instance.name,
        topic=instance.topic,
        literature=list(instance.literature),
        source_path=instance.source_path,
        metadata=metadata,
    )
