from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from .benchmarks.ai_idea_bench_2025 import load_ai_idea_bench_2025_records
from .llm import OpenAICompatibleChatClient
from .models import FinalProposal, IdeaGraph
from .settings import OpenAICompatibleSettings


@dataclass(frozen=True)
class BenchmarkNativeMetric:
    key: str
    display_name: str
    score: float
    max_score: float
    rationale: str
    available: bool = True
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkNativeEvaluation:
    protocol_name: str
    benchmark: str
    metrics: list[BenchmarkNativeMetric]
    summary: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _round_score(value: float) -> float:
    return round(float(value), 2)


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    candidates = [cleaned]
    if cleaned.startswith("```"):
        fence_stripped = cleaned.strip("`")
        if fence_stripped.startswith("json"):
            fence_stripped = fence_stripped[4:].strip()
        candidates.append(fence_stripped)

    decoder = json.JSONDecoder()
    errors: list[Exception] = []

    def _try_parse(candidate: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError as exc:
            errors.append(exc)
        try:
            payload, _ = decoder.raw_decode(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError as exc:
            errors.append(exc)

        repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
        if repaired != candidate:
            try:
                payload, _ = decoder.raw_decode(repaired)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError as exc:
                errors.append(exc)
        return None

    for candidate in candidates:
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed

    brace_positions = [idx for idx, char in enumerate(cleaned) if char == "{"]
    for start in brace_positions:
        parsed = _try_parse(cleaned[start:])
        if parsed is not None:
            return parsed

    if "{" not in cleaned or "}" not in cleaned:
        raise ValueError("Model response did not contain a JSON object.")
    if errors:
        raise errors[-1]
    raise ValueError("Model response JSON must be an object.")


def _normalize_text(value: Any) -> str:
    return " ".join(
        "".join(char.lower() if char.isalnum() else " " for char in _clean_text(value)).split()
    )


def _tokens(value: Any) -> set[str]:
    return {token for token in _normalize_text(value).split() if len(token) > 2}


def _token_f1(left: Any, right: Any) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return (2.0 * overlap) / (len(left_tokens) + len(right_tokens))


def _first_sentence(text: Any) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    for separator in (". ", "! ", "? "):
        if separator in cleaned:
            return cleaned.split(separator, 1)[0].strip().rstrip(".!?") + "."
    return cleaned.rstrip(".!?") + "."


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


def _metric(
    *,
    key: str,
    display_name: str,
    score: float,
    max_score: float,
    rationale: str,
    available: bool = True,
    details: dict[str, object] | None = None,
) -> BenchmarkNativeMetric:
    return BenchmarkNativeMetric(
        key=key,
        display_name=display_name,
        score=_round_score(score),
        max_score=max_score,
        rationale=rationale,
        available=available,
        details=details or {},
    )


def _unavailable_metric(
    *,
    key: str,
    display_name: str,
    rationale: str,
) -> BenchmarkNativeMetric:
    return _metric(
        key=key,
        display_name=display_name,
        score=0.0,
        max_score=1.0,
        rationale=rationale,
        available=False,
    )


class _NativeJudge:
    def __init__(self, settings: OpenAICompatibleSettings | None) -> None:
        self.settings = settings
        self.client = OpenAICompatibleChatClient(settings) if settings is not None else None

    def available(self) -> bool:
        return self.client is not None

    def score_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 900,
    ) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Benchmark-native scoring needs an OpenAI-compatible judge configuration.")
        result = self.client.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.settings.model if self.settings is not None else None,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        payload = _extract_json_object(result.content)
        payload["_raw_response"] = result.raw_response
        return payload


def _summary_from_metrics(metrics: list[BenchmarkNativeMetric]) -> dict[str, float]:
    available = [metric for metric in metrics if metric.available and metric.max_score > 0]
    if not available:
        return {
            "available_metric_count": 0.0,
            "available_average_normalized_10": 0.0,
        }
    normalized = [(metric.score / metric.max_score) * 10.0 for metric in available]
    return {
        "available_metric_count": float(len(available)),
        "available_average_normalized_10": _round_score(sum(normalized) / len(normalized)),
    }


def _format_multiple_choice_options(options: list[tuple[str, str]]) -> str:
    return "\n".join(f"- {label}: {text}" for label, text in options)


def _ai_idea_bench_generated_texts(graph: IdeaGraph) -> tuple[str, str]:
    sections = _proposal_sections(graph.final_proposal)
    motivation_text = " ".join(
        item
        for item in (
            sections["problem"],
            sections["existing_methods"],
            sections["motivation"],
            sections["significance"],
        )
        if item
    )
    experiment_text = " ".join(
        item
        for item in (
            sections["hypothesis"],
            sections["method"],
            sections["evaluation"],
            sections["caveats"],
        )
        if item
    )
    return motivation_text, experiment_text


def _benchmark_root(graph: IdeaGraph) -> Path | None:
    raw_root = graph.metadata.get("benchmark_root")
    if raw_root:
        return Path(str(raw_root))
    return None


def _ai_idea_bench_distractor_records(graph: IdeaGraph, *, limit: int = 3) -> list[dict[str, Any]]:
    root = _benchmark_root(graph)
    benchmark_index = graph.metadata.get("benchmark_index")
    if root is None or benchmark_index is None:
        return []
    try:
        records = load_ai_idea_bench_2025_records(root)
    except Exception:
        return []

    target_topic = _clean_text(graph.topic)
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        if record.benchmark_index == benchmark_index:
            continue
        topic = record.revised_topic or record.topic
        motivation = _clean_text(record.motivation)
        method_summary = _clean_text(record.method_summary)
        if not motivation and not method_summary:
            continue
        score = _token_f1(target_topic, topic)
        scored.append(
            (
                score,
                {
                    "benchmark_index": record.benchmark_index,
                    "topic": topic,
                    "motivation": motivation,
                    "method_summary": method_summary,
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in scored[:limit]]


def _ai_idea_bench_mcq_options(
    graph: IdeaGraph,
    *,
    target_text: str,
    distractor_field: str,
) -> tuple[list[tuple[str, str]], str]:
    distractors = [
        _clean_text(record.get(distractor_field))
        for record in _ai_idea_bench_distractor_records(graph)
        if _clean_text(record.get(distractor_field))
    ][:3]
    if len(distractors) < 3 or not target_text:
        return [], ""

    option_payloads = [target_text, *distractors]
    rotation = int(graph.metadata.get("benchmark_index", 0) or 0) % len(option_payloads)
    option_payloads = option_payloads[rotation:] + option_payloads[:rotation]
    labels = ["A", "B", "C", "D"]
    options = list(zip(labels, option_payloads))
    correct_label = labels[option_payloads.index(target_text)]
    return options, correct_label


def _mcq_motivation_prompts(input_motivation: str, options: list[tuple[str, str]]) -> tuple[str, str]:
    system_prompt = (
        "You are an AI motivation analyzer. Compare the input research motivation against four options and "
        "identify the closest match based on underlying problem, theme, and intent rather than superficial keywords. "
        "Return strict JSON only."
    )
    user_prompt = (
        "Return JSON in the form "
        '{"closest_option":"A","explanation":"..."}\n\n'
        f"Input motivation:\n{input_motivation}\n\n"
        f"Options:\n{_format_multiple_choice_options(options)}"
    )
    return system_prompt, user_prompt


def _mcq_experiment_prompts(input_plan: str, options: list[tuple[str, str]]) -> tuple[str, str]:
    system_prompt = (
        "You are comparing scientific experiment plans. Match the input plan to the closest of four options by "
        "considering structural alignment, theoretical basis, and problem focus. Ignore wording and ordering differences. "
        "Return strict JSON only."
    )
    user_prompt = (
        "Return JSON in the form "
        '{"closest_plan":"A","explanation":"..."}\n\n'
        f"Input experiment plan:\n{input_plan}\n\n"
        f"Options:\n{_format_multiple_choice_options(options)}"
    )
    return system_prompt, user_prompt


def _i2i_motivation_prompts(generated: str, gold: str) -> tuple[str, str]:
    system_prompt = (
        "You are evaluating similarity between two research motivations. Focus on structural alignment, theoretical "
        "foundations, and problem focus. Return strict JSON only."
    )
    user_prompt = (
        "Compare the two motivations and rate their similarity on a 1-5 scale, where 1 means no similarity and 5 means "
        "complete similarity. Return JSON in the form "
        '{"motivation_similarity":{"rating":4,"explanation":"..."}}.\n\n'
        f"Motivation 1:\n{generated}\n\nMotivation 2:\n{gold}"
    )
    return system_prompt, user_prompt


def _i2i_experiment_prompts(generated: str, gold: str) -> tuple[str, str]:
    system_prompt = (
        "You are evaluating similarity between two experiment plans based on structure, theoretical alignment, and "
        "problem focus. Return strict JSON only."
    )
    user_prompt = (
        "Compare the two experiment plans and rate their similarity on a 1-5 scale, where 1 means no similarity and 5 "
        "means complete similarity. Return JSON in the form "
        '{"experiment_plan_similarity":{"rating":4,"explanation":"..."}}.\n\n'
        f"Experiment Plan 1:\n{generated}\n\nExperiment Plan 2:\n{gold}"
    )
    return system_prompt, user_prompt


def _i2t_prompts(topic: str, motivation: str, experiment_plan: str) -> tuple[str, str]:
    system_prompt = (
        "You are evaluating whether a research idea's motivation and experiment plan align with the target topic. "
        "Return strict JSON only."
    )
    user_prompt = (
        "Score topic alignment from 1-5 for both the motivation and experiment plan. Return JSON in the form "
        '{"motivation":{"alignment":4,"comments":"..."},"experiment_plan":{"alignment":5,"comments":"..."}}.\n\n'
        f"Topic:\n{topic}\n\nMotivation:\n{motivation}\n\nExperiment Plan:\n{experiment_plan}"
    )
    return system_prompt, user_prompt


def _fps_prompts(motivation: str, experiment_plan: str) -> tuple[str, str]:
    system_prompt = (
        "You are evaluating how well an experiment plan addresses the motivation of a research idea. "
        "Return strict JSON only."
    )
    user_prompt = (
        "Provide a single overall score from 1-5 for how well the experiment plan addresses the motivation. Return JSON "
        'in the form {"overall_score":4,"overall_rationale":"..."}.'
        f"\n\nMotivation:\n{motivation}\n\nExperiment Plan:\n{experiment_plan}"
    )
    return system_prompt, user_prompt


def _extract_nested_score(payload: dict[str, Any], *path: str) -> float:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    try:
        return float(current)
    except (TypeError, ValueError):
        return 0.0


def _ai_idea_bench_native_evaluation(
    graph: IdeaGraph,
    judge: _NativeJudge,
) -> BenchmarkNativeEvaluation:
    metrics: list[BenchmarkNativeMetric] = []
    notes = [
        "This scorer follows the public AI Idea Bench 2025 metric structure and official prompt style where that protocol is directly reproducible from the released repository.",
        "Some official benchmark metrics need extra assets or cross-system pools and are marked unavailable until those prerequisites are supplied.",
    ]

    motivation_text, experiment_text = _ai_idea_bench_generated_texts(graph)
    gold_motivation = _clean_text(graph.metadata.get("motivation"))
    gold_method = _clean_text(graph.metadata.get("method_summary"))
    topic = _clean_text(graph.topic)

    if judge.available() and gold_motivation and gold_method:
        system_prompt, user_prompt = _i2i_motivation_prompts(motivation_text, gold_motivation)
        payload = judge.score_json(system_prompt=system_prompt, user_prompt=user_prompt)
        metrics.append(
            _metric(
                key="i2i_motivation",
                display_name="I2I Motivation",
                score=_extract_nested_score(payload, "motivation_similarity", "rating"),
                max_score=5.0,
                rationale="Official-style idea-to-idea motivation similarity against the held-out target-paper motivation.",
                details={"explanation": _clean_text(payload.get("motivation_similarity", {}).get("explanation")) if isinstance(payload.get("motivation_similarity"), dict) else ""},
            )
        )

        system_prompt, user_prompt = _i2i_experiment_prompts(experiment_text, gold_method)
        payload = judge.score_json(system_prompt=system_prompt, user_prompt=user_prompt)
        metrics.append(
            _metric(
                key="i2i_experiment",
                display_name="I2I Experiment",
                score=_extract_nested_score(payload, "experiment_plan_similarity", "rating"),
                max_score=5.0,
                rationale="Official-style idea-to-idea experiment-plan similarity against the held-out target-paper method summary.",
                details={"explanation": _clean_text(payload.get("experiment_plan_similarity", {}).get("explanation")) if isinstance(payload.get("experiment_plan_similarity"), dict) else ""},
            )
        )
    else:
        unavailable_reason = (
            "Needs both a judge model and held-out target-paper motivation/method metadata."
        )
        metrics.append(_unavailable_metric(key="i2i_motivation", display_name="I2I Motivation", rationale=unavailable_reason))
        metrics.append(_unavailable_metric(key="i2i_experiment", display_name="I2I Experiment", rationale=unavailable_reason))

    if judge.available():
        system_prompt, user_prompt = _i2t_prompts(topic, motivation_text, experiment_text)
        payload = judge.score_json(system_prompt=system_prompt, user_prompt=user_prompt)
        metrics.append(
            _metric(
                key="i2t_motivation",
                display_name="I2T Motivation",
                score=_extract_nested_score(payload, "motivation", "alignment"),
                max_score=5.0,
                rationale="Official-style idea-to-topic alignment for the motivation component.",
                details={"comments": _clean_text(payload.get("motivation", {}).get("comments")) if isinstance(payload.get("motivation"), dict) else ""},
            )
        )
        metrics.append(
            _metric(
                key="i2t_experiment",
                display_name="I2T Experiment",
                score=_extract_nested_score(payload, "experiment_plan", "alignment"),
                max_score=5.0,
                rationale="Official-style idea-to-topic alignment for the experiment-plan component.",
                details={"comments": _clean_text(payload.get("experiment_plan", {}).get("comments")) if isinstance(payload.get("experiment_plan"), dict) else ""},
            )
        )

        system_prompt, user_prompt = _fps_prompts(motivation_text, experiment_text)
        payload = judge.score_json(system_prompt=system_prompt, user_prompt=user_prompt)
        metrics.append(
            _metric(
                key="fps",
                display_name="FPS",
                score=_extract_nested_score(payload, "overall_score"),
                max_score=5.0,
                rationale="Official-style motivation-to-experiment consistency scoring.",
                details={"overall_rationale": _clean_text(payload.get("overall_rationale"))},
            )
        )
    else:
        unavailable_reason = "Needs a judge model to run the released AI Idea Bench prompt-based evaluators."
        metrics.append(_unavailable_metric(key="i2t_motivation", display_name="I2T Motivation", rationale=unavailable_reason))
        metrics.append(_unavailable_metric(key="i2t_experiment", display_name="I2T Experiment", rationale=unavailable_reason))
        metrics.append(_unavailable_metric(key="fps", display_name="FPS", rationale=unavailable_reason))

    motivation_options, motivation_answer = _ai_idea_bench_mcq_options(
        graph,
        target_text=gold_motivation,
        distractor_field="motivation",
    )
    experiment_options, experiment_answer = _ai_idea_bench_mcq_options(
        graph,
        target_text=gold_method,
        distractor_field="method_summary",
    )
    if judge.available() and motivation_options and motivation_answer:
        system_prompt, user_prompt = _mcq_motivation_prompts(motivation_text, motivation_options)
        payload = judge.score_json(system_prompt=system_prompt, user_prompt=user_prompt)
        prediction = _clean_text(payload.get("closest_option")).upper()
        metrics.append(
            _metric(
                key="imcq_motivation",
                display_name="IMCQ Motivation",
                score=1.0 if prediction == motivation_answer else 0.0,
                max_score=1.0,
                rationale="Official-style motivation multiple-choice matching against the hidden target-paper option.",
                details={"prediction": prediction, "answer": motivation_answer},
            )
        )
    else:
        metrics.append(
            _unavailable_metric(
                key="imcq_motivation",
                display_name="IMCQ Motivation",
                rationale="Needs a judge model plus enough benchmark metadata to assemble multiple-choice distractors.",
            )
        )

    if judge.available() and experiment_options and experiment_answer:
        system_prompt, user_prompt = _mcq_experiment_prompts(experiment_text, experiment_options)
        payload = judge.score_json(system_prompt=system_prompt, user_prompt=user_prompt)
        prediction = _clean_text(payload.get("closest_plan")).upper()
        metrics.append(
            _metric(
                key="imcq_experiment",
                display_name="IMCQ Experiment",
                score=1.0 if prediction == experiment_answer else 0.0,
                max_score=1.0,
                rationale="Official-style experiment-plan multiple-choice matching against the hidden target-paper option.",
                details={"prediction": prediction, "answer": experiment_answer},
            )
        )
    else:
        metrics.append(
            _unavailable_metric(
                key="imcq_experiment",
                display_name="IMCQ Experiment",
                rationale="Needs a judge model plus enough benchmark metadata to assemble multiple-choice distractors.",
            )
        )

    metrics.append(
        _unavailable_metric(
            key="na",
            display_name="NA",
            rationale="The released novelty-assessment pipeline requires additional current/historical-paper assets beyond the base metadata currently cached in this repo.",
        )
    )
    metrics.append(
        _unavailable_metric(
            key="fa",
            display_name="FA",
            rationale="The released feasibility-assessment pipeline requires experiment-plan keyword extraction plus Semantic Scholar search credentials, which are not guaranteed in the base run path.",
        )
    )
    metrics.append(
        _unavailable_metric(
            key="ic",
            display_name="IC",
            rationale="Ideas Competition is a cross-system comparison metric and should be computed over matched batches of runs rather than a single artifact.",
        )
    )

    return BenchmarkNativeEvaluation(
        protocol_name="ai_idea_bench_2025_public_protocol_v1",
        benchmark=_clean_text(graph.metadata.get("benchmark")),
        metrics=metrics,
        summary=_summary_from_metrics(metrics),
        notes=notes,
    )


def _liveideabench_dimensions(graph: IdeaGraph) -> list[tuple[str, str]]:
    raw_record = graph.metadata.get("raw_record", {})
    dimensions = [
        ("originality", "Originality"),
        ("feasibility", "Feasibility"),
        ("fluency", "Fluency"),
    ]
    if isinstance(raw_record, dict):
        if any(key in raw_record for key in ("clarity", "clar")):
            dimensions.append(("clarity", "Clarity"))
        if any(key in raw_record for key in ("flexibility", "flex")):
            dimensions.append(("flexibility", "Flexibility"))
    return dimensions


def _liveideabench_prompt(graph: IdeaGraph, dimensions: list[tuple[str, str]]) -> tuple[str, str]:
    keyword = _clean_text(graph.metadata.get("keyword")) or _clean_text(graph.topic)
    sections = _proposal_sections(graph.final_proposal)
    proposal_text = "\n".join(
        f"{label}: {value}"
        for label, value in (
            ("Title", sections["title"]),
            ("Problem", sections["problem"]),
            ("Existing Methods", sections["existing_methods"]),
            ("Motivation", sections["motivation"]),
            ("Hypothesis", sections["hypothesis"]),
            ("Method", sections["method"]),
            ("Evaluation", sections["evaluation"]),
            ("Significance", sections["significance"]),
            ("Caveats", sections["caveats"]),
        )
        if value
    )
    dimension_block = "\n".join(
        f"- {display_name}: score from 1 to 10 with a concise rationale."
        for _, display_name in dimensions
    )
    json_block = ", ".join(f'"{key}": {{"score": 7, "rationale": "..."}}' for key, _ in dimensions)
    system_prompt = (
        "You are evaluating a scientific idea under the LiveIdeaBench minimal-context creativity setting. "
        "Use the provided keyword as the only benchmark context and score the idea dimension by dimension. "
        "Return strict JSON only."
    )
    user_prompt = (
        "Score the idea using the following dimensions:\n"
        f"{dimension_block}\n\n"
        "Return JSON in the form "
        f'{{"scores": {{{json_block}}}, "overall_average": 7.2}}.\n\n'
        f"Keyword:\n{keyword}\n\nStructured idea:\n{proposal_text}"
    )
    return system_prompt, user_prompt


def _liveideabench_native_evaluation(
    graph: IdeaGraph,
    judge: _NativeJudge,
) -> BenchmarkNativeEvaluation:
    dimensions = _liveideabench_dimensions(graph)
    metrics: list[BenchmarkNativeMetric] = []
    notes = [
        "The current repo loader targets the public Hugging Face LiveIdeaBench release and scores generated ideas with the benchmark's native minimal-context creativity dimensions.",
        "Because the public dataset release stores judged rows rather than a standalone scorer, this implementation uses a rubric-aligned LLM judge instead of the original judge ensemble.",
    ]

    if not judge.available():
        for key, display_name in dimensions:
            metrics.append(
                _unavailable_metric(
                    key=key,
                    display_name=display_name,
                    rationale="Needs a judge model to run the LiveIdeaBench native rubric.",
                )
            )
        return BenchmarkNativeEvaluation(
            protocol_name="liveideabench_public_rubric_v1",
            benchmark=_clean_text(graph.metadata.get("benchmark")),
            metrics=metrics,
            summary=_summary_from_metrics(metrics),
            notes=notes,
        )

    system_prompt, user_prompt = _liveideabench_prompt(graph, dimensions)
    payload = judge.score_json(system_prompt=system_prompt, user_prompt=user_prompt)
    score_payload = payload.get("scores", {})
    if not isinstance(score_payload, dict):
        score_payload = {}
    for key, display_name in dimensions:
        metric_payload = score_payload.get(key, {})
        score = 0.0
        rationale = "LiveIdeaBench native rubric score."
        details: dict[str, object] = {}
        if isinstance(metric_payload, dict):
            try:
                score = float(metric_payload.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            details["rationale"] = _clean_text(metric_payload.get("rationale"))
        metrics.append(
            _metric(
                key=key,
                display_name=display_name,
                score=score,
                max_score=10.0,
                rationale=rationale,
                details=details,
            )
        )
    overall_average = 0.0
    try:
        overall_average = float(payload.get("overall_average", 0.0))
    except (TypeError, ValueError):
        overall_average = 0.0
    metrics.append(
        _metric(
            key="average",
            display_name="Average",
            score=overall_average,
            max_score=10.0,
            rationale="Average score across the benchmark-native LiveIdeaBench dimensions returned by the judge.",
        )
    )
    return BenchmarkNativeEvaluation(
        protocol_name="liveideabench_public_rubric_v1",
        benchmark=_clean_text(graph.metadata.get("benchmark")),
        metrics=metrics,
        summary=_summary_from_metrics(metrics),
        notes=notes,
    )


def evaluate_benchmark_native(
    graph: IdeaGraph,
    *,
    settings: OpenAICompatibleSettings | None = None,
) -> BenchmarkNativeEvaluation:
    benchmark = _clean_text(graph.metadata.get("benchmark"))
    judge = _NativeJudge(settings)

    if benchmark == "AI_Idea_Bench_2025":
        return _ai_idea_bench_native_evaluation(graph, judge)
    if benchmark == "liveideabench":
        return _liveideabench_native_evaluation(graph, judge)

    metric = _unavailable_metric(
        key="native_scoring",
        display_name="Native Scoring",
        rationale="No benchmark-native scorer is registered for this run because the graph is not tied to a supported benchmark.",
    )
    return BenchmarkNativeEvaluation(
        protocol_name="none",
        benchmark=benchmark,
        metrics=[metric],
        summary=_summary_from_metrics([metric]),
        notes=["Native benchmark scoring is only defined for explicit benchmark runs."],
    )


def format_benchmark_native_markdown(evaluation: BenchmarkNativeEvaluation) -> str:
    lines = [
        "# Benchmark-Native Evaluation",
        "",
        f"- Protocol: `{evaluation.protocol_name}`",
        f"- Benchmark: `{evaluation.benchmark or 'none'}`",
    ]
    if evaluation.summary:
        for key, value in evaluation.summary.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Metric Breakdown", ""])
    for metric in evaluation.metrics:
        availability = "" if metric.available else " (not available for this run)"
        lines.append(f"### {metric.display_name}")
        lines.append("")
        lines.append(f"- Score: `{metric.score:.2f}/{metric.max_score:.2f}`{availability}")
        lines.append(f"- Rationale: {metric.rationale}")
        if metric.details:
            lines.append("- Details:")
            for key, value in metric.details.items():
                lines.append(f"  - `{key}`: `{_clean_text(value)}`")
        lines.append("")
    if evaluation.notes:
        lines.extend(["## Notes", ""])
        for note in evaluation.notes:
            lines.append(f"- {note}")
    return "\n".join(lines).strip() + "\n"
