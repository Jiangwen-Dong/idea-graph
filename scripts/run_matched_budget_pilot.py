from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.agent_backend import OpenAICompatibleCollaborationBackend
from idea_graph.baselines import attach_baseline_metadata, run_baseline_experiment
from idea_graph.benchmarks import (
    ai_idea_bench_2025_instance_from_record,
    get_ai_idea_bench_2025_record,
)
from idea_graph.evaluation import evaluate_graph
from idea_graph.io import write_run_artifacts
from idea_graph.instances import ExperimentInstance
from idea_graph.settings import OpenAICompatibleSettings


@dataclass(frozen=True)
class PilotMethodPlan:
    baseline_name: str
    restarts: int
    max_rounds: int
    stop_when_mature: bool
    rationale: str


PILOT_METHOD_PLANS: dict[str, PilotMethodPlan] = {
    "direct": PilotMethodPlan(
        baseline_name="direct",
        restarts=4,
        max_rounds=1,
        stop_when_mature=True,
        rationale="One-pass lower bound, with multiple iid restarts for a more stable low-cost reference.",
    ),
    "self-refine": PilotMethodPlan(
        baseline_name="self-refine",
        restarts=2,
        max_rounds=1,
        stop_when_mature=True,
        rationale="Draft-critique-revise control baseline with two independent trajectories.",
    ),
    "scipip-proxy": PilotMethodPlan(
        baseline_name="scipip-proxy",
        restarts=2,
        max_rounds=1,
        stop_when_mature=True,
        rationale="Structured decomposition proxy with self-refinement and two independent trajectories.",
    ),
    "ai-researcher-proxy": PilotMethodPlan(
        baseline_name="ai-researcher-proxy",
        restarts=1,
        max_rounds=1,
        stop_when_mature=True,
        rationale="Literature-grounded proxy with internal seed generation, expansion, and ranking.",
    ),
    "ours-delayed-consensus": PilotMethodPlan(
        baseline_name="ours-delayed-consensus",
        restarts=1,
        max_rounds=2,
        stop_when_mature=True,
        rationale="Main graph-mediated method with delayed consensus and maturity-based early stopping.",
    ),
}


def print_progress(message: str) -> None:
    print(f"[pilot] {message}", flush=True)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _usage_from_raw_response(raw_response: Any) -> dict[str, int]:
    if not isinstance(raw_response, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage = raw_response.get("usage", {})
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = _coerce_int(usage.get("prompt_tokens"))
    completion_tokens = _coerce_int(usage.get("completion_tokens"))
    total_tokens = _coerce_int(usage.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def summarize_graph_usage(graph) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    llm_call_count = 0

    metadata = graph.metadata if isinstance(graph.metadata, dict) else {}
    for trace_key in ("baseline_traces", "agent_traces"):
        trace_items = metadata.get(trace_key, [])
        if not isinstance(trace_items, list):
            continue
        for item in trace_items:
            if not isinstance(item, dict):
                continue
            usage = _usage_from_raw_response(item.get("raw_response"))
            if usage["total_tokens"] <= 0:
                continue
            prompt_tokens += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]
            total_tokens += usage["total_tokens"]
            llm_call_count += 1

    final_trace = metadata.get("final_synthesis_trace", {})
    if isinstance(final_trace, dict):
        usage = _usage_from_raw_response(final_trace.get("raw_response"))
        if usage["total_tokens"] > 0:
            prompt_tokens += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]
            total_tokens += usage["total_tokens"]
            llm_call_count += 1

    return {
        "llm_call_count": llm_call_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def selection_score(evaluation_payload: dict[str, Any]) -> float:
    overall = float(evaluation_payload.get("overall_score", 0.0) or 0.0)
    category_scores = evaluation_payload.get("category_scores", {})
    if not isinstance(category_scores, dict):
        category_scores = {}
    benchmark_alignment = float(category_scores.get("benchmark_alignment", 0.0) or 0.0)
    expert_style_quality = float(category_scores.get("expert_style_quality", 0.0) or 0.0)
    return (0.65 * overall) + (0.25 * benchmark_alignment) + (0.10 * expert_style_quality)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small cost-aware ideation pilot (legacy filename retained for compatibility)."
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "data" / "benchmarks" / "ai_idea_bench_2025",
        help="Root directory for AI Idea Bench 2025 benchmark files.",
    )
    parser.add_argument(
        "--benchmark-indices",
        type=int,
        nargs="+",
        default=[13, 15],
        help="Benchmark indices to include in the pilot.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=[
            "direct",
            "self-refine",
            "scipip-proxy",
            "ai-researcher-proxy",
            "ours-delayed-consensus",
        ],
        help="Baselines to include in the pilot.",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        required=True,
        help="Path to the OpenAI-compatible backend config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "pilots",
        help="Directory for pilot artifacts.",
    )
    parser.add_argument(
        "--pilot-name",
        default="cost-aware-pilot",
        help="Short suffix for the pilot directory name.",
    )
    return parser


def load_instance(benchmark_root: Path, benchmark_index: int) -> ExperimentInstance:
    record = get_ai_idea_bench_2025_record(benchmark_root, benchmark_index)
    return ai_idea_bench_2025_instance_from_record(record, benchmark_root=benchmark_root)


def mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(float(statistics.mean(values)), 3)


def format_markdown_summary(pilot_payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Cost-Aware Baseline Pilot")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Benchmark: `{pilot_payload['benchmark']}`")
    lines.append(f"- Indices: `{', '.join(str(item) for item in pilot_payload['benchmark_indices'])}`")
    lines.append(f"- Model: `{pilot_payload['model']}`")
    lines.append(f"- Generated at: `{pilot_payload['generated_at']}`")
    lines.append("")
    lines.append("### Comparison Policy")
    lines.append("")
    for item in pilot_payload.get("method_plans", []):
        lines.append(
            f"- `{item['baseline_name']}`: restarts={item['restarts']}, "
            f"max_rounds={item['max_rounds']}, stop_when_mature={item['stop_when_mature']}. "
            f"{item['rationale']}"
        )
    lines.append("")
    lines.append("## Raw Data Table")
    lines.append("")
    lines.append("| Baseline | Index | Restart | Selection Score | Overall | Benchmark Align | Expert Quality | Graph Process | Calls | Total Tokens | Run Dir |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in pilot_payload.get("raw_rows", []):
        lines.append(
            f"| `{row['baseline_name']}` | {row['benchmark_index']} | {row['restart']} | "
            f"{row['selection_score']:.3f} | {row['overall_score']:.2f} | "
            f"{row['benchmark_alignment']:.2f} | {row['expert_style_quality']:.2f} | "
            f"{row['graph_process']:.2f} | {row['llm_call_count']} | {row['total_tokens']} | "
            f"`{row['run_dir_name']}` |"
        )
    lines.append("")
    lines.append("## Selected Runs")
    lines.append("")
    lines.append("| Baseline | Mean Overall | Mean Benchmark Align | Mean Expert Quality | Mean Graph Process | Mean Calls | Mean Tokens | Score / 10k Tokens |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in pilot_payload.get("aggregate_rows", []):
        lines.append(
            f"| `{row['baseline_name']}` | {row['mean_overall_score']:.2f} | "
            f"{row['mean_benchmark_alignment']:.2f} | {row['mean_expert_style_quality']:.2f} | "
            f"{row['mean_graph_process']:.2f} | {row['mean_llm_call_count']:.2f} | "
            f"{row['mean_total_tokens']:.0f} | {row['score_per_10k_tokens']:.3f} |"
        )
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    for item in pilot_payload.get("findings", []):
        lines.append(f"1. {item}")
    lines.append("")
    lines.append("## Suggested Next Experiments")
    lines.append("")
    for item in pilot_payload.get("next_steps", []):
        lines.append(f"1. {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    settings = OpenAICompatibleSettings.from_json_file(args.llm_config)
    backend = OpenAICompatibleCollaborationBackend(settings)

    missing = [name for name in args.baselines if name not in PILOT_METHOD_PLANS]
    if missing:
        raise SystemExit(f"Unsupported pilot baseline(s): {', '.join(missing)}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pilot_dir = args.output_dir / f"{timestamp}-{args.pilot_name}"
    runs_root = pilot_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    print_progress(
        f"Running pilot on benchmark indices {args.benchmark_indices} with baselines {args.baselines}."
    )

    for benchmark_index in args.benchmark_indices:
        instance = load_instance(args.benchmark_root, benchmark_index)
        print_progress(
            f"Loaded benchmark index {benchmark_index}: {_clean_text(instance.topic)[:120]}"
        )
        for baseline_name in args.baselines:
            plan = PILOT_METHOD_PLANS[baseline_name]
            print_progress(
                f"Running baseline '{baseline_name}' on index {benchmark_index} with {plan.restarts} restart(s)."
            )
            best_row: dict[str, Any] | None = None
            successful_runs = 0
            for restart in range(plan.restarts):
                prepared_instance = attach_baseline_metadata(
                    instance,
                    baseline_name=baseline_name,
                    io_mode="auto",
                )
                metadata = dict(prepared_instance.metadata)
                metadata["pilot_name"] = args.pilot_name
                metadata["pilot_restart"] = restart
                metadata["pilot_budget_plan"] = asdict(plan)
                prepared_instance = ExperimentInstance(
                    name=prepared_instance.name,
                    topic=prepared_instance.topic,
                    literature=list(prepared_instance.literature),
                    source_path=prepared_instance.source_path,
                    metadata=metadata,
                )

                try:
                    graph = run_baseline_experiment(
                        prepared_instance,
                        baseline_name=baseline_name,
                        collaboration_backend=backend,
                        progress_callback=lambda message, bn=baseline_name, bi=benchmark_index, rr=restart: print_progress(
                            f"[{bn}][{bi}][r{rr}] {message}"
                        ),
                        max_rounds=plan.max_rounds,
                        stop_when_mature=plan.stop_when_mature,
                    )
                except Exception as exc:
                    raw_rows.append(
                        {
                            "baseline_name": baseline_name,
                            "benchmark_index": benchmark_index,
                            "restart": restart,
                            "error": str(exc),
                        }
                    )
                    continue

                run_dir = write_run_artifacts(
                    graph,
                    output_root=runs_root,
                    instance=prepared_instance,
                )
                evaluation = evaluate_graph(graph).as_dict()
                usage = summarize_graph_usage(graph)
                category_scores = evaluation.get("category_scores", {})
                if not isinstance(category_scores, dict):
                    category_scores = {}

                row = {
                    "baseline_name": baseline_name,
                    "benchmark_index": benchmark_index,
                    "restart": restart,
                    "selection_score": selection_score(evaluation),
                    "overall_score": float(evaluation.get("overall_score", 0.0) or 0.0),
                    "benchmark_alignment": float(category_scores.get("benchmark_alignment", 0.0) or 0.0),
                    "expert_style_quality": float(category_scores.get("expert_style_quality", 0.0) or 0.0),
                    "graph_process": float(category_scores.get("graph_process", 0.0) or 0.0),
                    "llm_call_count": usage["llm_call_count"],
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "run_dir": str(run_dir),
                    "run_dir_name": run_dir.name,
                    "stop_reason": graph.metadata.get("stop_reason", ""),
                    "executed_round_count": graph.metadata.get("executed_round_count", 0),
                    "title": graph.final_proposal.title if graph.final_proposal is not None else "",
                }
                raw_rows.append(row)
                successful_runs += 1
                if best_row is None or row["selection_score"] > best_row["selection_score"]:
                    best_row = row

            if best_row is None:
                print_progress(
                    f"Baseline '{baseline_name}' failed on benchmark index {benchmark_index}."
                )
                continue
            best_row = dict(best_row)
            best_row["successful_runs"] = successful_runs
            selected_rows.append(best_row)

    for baseline_name in args.baselines:
        rows = [row for row in selected_rows if row["baseline_name"] == baseline_name]
        if not rows:
            continue
        mean_overall = mean_or_zero([row["overall_score"] for row in rows])
        mean_alignment = mean_or_zero([row["benchmark_alignment"] for row in rows])
        mean_expert = mean_or_zero([row["expert_style_quality"] for row in rows])
        mean_graph = mean_or_zero([row["graph_process"] for row in rows])
        mean_calls = mean_or_zero([float(row["llm_call_count"]) for row in rows])
        mean_tokens = mean_or_zero([float(row["total_tokens"]) for row in rows])
        score_per_10k_tokens = 0.0 if mean_tokens <= 0 else round((10000.0 * mean_overall) / mean_tokens, 3)
        aggregate_rows.append(
            {
                "baseline_name": baseline_name,
                "mean_overall_score": mean_overall,
                "mean_benchmark_alignment": mean_alignment,
                "mean_expert_style_quality": mean_expert,
                "mean_graph_process": mean_graph,
                "mean_llm_call_count": mean_calls,
                "mean_total_tokens": mean_tokens,
                "score_per_10k_tokens": score_per_10k_tokens,
                "selected_run_dirs": [row["run_dir"] for row in rows],
            }
        )

    ranked_by_overall = sorted(
        aggregate_rows,
        key=lambda item: (item["mean_overall_score"], item["mean_benchmark_alignment"]),
        reverse=True,
    )
    ranked_by_efficiency = sorted(
        aggregate_rows,
        key=lambda item: item["score_per_10k_tokens"],
        reverse=True,
    )
    findings: list[str] = []
    if ranked_by_overall:
        top = ranked_by_overall[0]
        findings.append(
            f"`{top['baseline_name']}` achieved the strongest mean pilot score "
            f"({top['mean_overall_score']:.2f}/10 overall; {top['mean_benchmark_alignment']:.2f}/10 benchmark alignment)."
        )
    if ranked_by_efficiency:
        efficient = ranked_by_efficiency[0]
        findings.append(
            f"`{efficient['baseline_name']}` delivered the best score efficiency in this pilot "
            f"({efficient['score_per_10k_tokens']:.3f} score per 10k tokens)."
        )
    if aggregate_rows:
        ours_row = next((row for row in aggregate_rows if row["baseline_name"] == "ours-delayed-consensus"), None)
        direct_row = next((row for row in aggregate_rows if row["baseline_name"] == "direct"), None)
        if ours_row is not None and direct_row is not None:
            delta = round(ours_row["mean_overall_score"] - direct_row["mean_overall_score"], 2)
            findings.append(
                f"The main method's mean overall score differed from the direct baseline by {delta:+.2f} points in this pilot."
            )

    next_steps = [
        "Scale the same pilot script from two representative indices to a broader benchmark slice once the prompt and stopping protocol are stable.",
        "Add benchmark-native judge metrics on the selected pilot runs only after the local deterministic pilot table is stable.",
        "If a baseline is still qualitatively weak, improve its prompt or restart policy before launching a large sweep rather than after the fact.",
    ]

    pilot_payload = {
        "pilot_name": args.pilot_name,
        "benchmark": "AI_Idea_Bench_2025",
        "benchmark_indices": args.benchmark_indices,
        "model": settings.model,
        "generated_at": datetime.now().isoformat(),
        "method_plans": [asdict(PILOT_METHOD_PLANS[name]) for name in args.baselines],
        "raw_rows": raw_rows,
        "selected_rows": selected_rows,
        "aggregate_rows": aggregate_rows,
        "findings": findings,
        "next_steps": next_steps,
    }

    (pilot_dir / "pilot_summary.json").write_text(
        json.dumps(pilot_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (pilot_dir / "pilot_summary.md").write_text(
        format_markdown_summary(pilot_payload),
        encoding="utf-8",
    )

    print("== Pilot Directory ==")
    print(pilot_dir)
    print()
    print("== Aggregate ==")
    for row in aggregate_rows:
        print(
            f"{row['baseline_name']}: overall={row['mean_overall_score']:.2f}, "
            f"alignment={row['mean_benchmark_alignment']:.2f}, "
            f"calls={row['mean_llm_call_count']:.2f}, tokens={row['mean_total_tokens']:.0f}"
        )


if __name__ == "__main__":
    main()
