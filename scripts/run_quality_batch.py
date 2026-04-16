from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
from idea_graph.benchmark_scoring import evaluate_benchmark_native
from idea_graph.benchmarks import (
    ai_idea_bench_2025_instance_from_record,
    get_ai_idea_bench_2025_record,
    get_liveideabench_record,
    liveideabench_instance_from_record,
)
from idea_graph.evaluation import evaluate_graph
from idea_graph.experiment_plans import (
    ExperimentMethodPlan,
    canonical_method_plan_name,
    get_method_plan_catalog,
    prepare_instance_for_method_plan,
)
from idea_graph.external_baselines import load_external_baseline_config
from idea_graph.io import write_run_artifacts
from idea_graph.instances import ExperimentInstance
from idea_graph.repo_paths import default_ai_benchmark_root, default_live_benchmark_root
from idea_graph.settings import OpenAICompatibleSettings


@dataclass(frozen=True)
class BenchmarkTarget:
    benchmark: str
    selector: int
    display_selector: str
    instance_name: str
    topic_preview: str
    instance: ExperimentInstance


def print_progress(message: str) -> None:
    print(f"[batch] {message}", flush=True)


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
    return float(evaluation_payload.get("overall_score", 0.0) or 0.0)


def mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(float(statistics.mean(values)), 3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small cross-benchmark quality batch over AI Idea Bench 2025 and LiveIdeaBench."
    )
    parser.add_argument(
        "--ai-benchmark-root",
        type=Path,
        default=default_ai_benchmark_root(ROOT),
        help="Root directory for AI Idea Bench 2025 files.",
    )
    parser.add_argument(
        "--live-benchmark-root",
        type=Path,
        default=default_live_benchmark_root(ROOT),
        help="Root directory for LiveIdeaBench files.",
    )
    parser.add_argument(
        "--plan-preset",
        choices=("main", "ablation"),
        default="main",
        help="Method-plan preset to use. 'main' reproduces the paper comparison, while 'ablation' exposes protocol variants.",
    )
    parser.add_argument(
        "--ai-indices",
        type=int,
        nargs="+",
        default=[13, 15, 18, 21, 33],
        help="AI Idea Bench 2025 benchmark indices to include.",
    )
    parser.add_argument(
        "--live-row-indices",
        type=int,
        nargs="+",
        default=[0, 23, 47, 70, 96],
        help="LiveIdeaBench row indices to include.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=None,
        help="Method-plan names to include in the batch. Defaults depend on --plan-preset.",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        required=True,
        help="Path to the OpenAI-compatible backend config JSON.",
    )
    parser.add_argument(
        "--external-baseline-config",
        type=Path,
        default=None,
        help="Optional JSON config for exact external baselines such as ai-researcher.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "quality_batches",
        help="Directory for batch artifacts.",
    )
    parser.add_argument(
        "--batch-name",
        default="cross-benchmark-quality-batch",
        help="Short suffix for the batch directory name.",
    )
    parser.add_argument(
        "--native-eval",
        action="store_true",
        help="Run benchmark-native evaluation on selected runs. This adds extra judge calls.",
    )
    parser.add_argument(
        "--native-eval-baselines",
        nargs="+",
        default=[],
        help="Optional subset of baselines for benchmark-native evaluation. Empty means all baselines when --native-eval is enabled.",
    )
    parser.add_argument(
        "--native-eval-max-runs",
        type=int,
        default=0,
        help="Optional cap on the number of runs receiving benchmark-native evaluation. 0 means no cap.",
    )
    return parser


def load_ai_target(benchmark_root: Path, benchmark_index: int) -> BenchmarkTarget:
    record = get_ai_idea_bench_2025_record(benchmark_root, benchmark_index)
    instance = ai_idea_bench_2025_instance_from_record(record, benchmark_root=benchmark_root)
    return BenchmarkTarget(
        benchmark="AI_Idea_Bench_2025",
        selector=benchmark_index,
        display_selector=str(benchmark_index),
        instance_name=instance.name,
        topic_preview=_clean_text(instance.topic)[:140],
        instance=instance,
    )


def load_live_target(benchmark_root: Path, row_index: int) -> BenchmarkTarget:
    record = get_liveideabench_record(benchmark_root, row_index)
    instance = liveideabench_instance_from_record(record, benchmark_root=benchmark_root)
    keyword = _clean_text(record.keyword)
    display_selector = f"{row_index}:{keyword}" if keyword else str(row_index)
    return BenchmarkTarget(
        benchmark="liveideabench",
        selector=row_index,
        display_selector=display_selector,
        instance_name=instance.name,
        topic_preview=_clean_text(instance.topic)[:140],
        instance=instance,
    )


def load_targets(args: argparse.Namespace) -> list[BenchmarkTarget]:
    targets: list[BenchmarkTarget] = []
    for benchmark_index in args.ai_indices:
        targets.append(load_ai_target(args.ai_benchmark_root, benchmark_index))
    for row_index in args.live_row_indices:
        targets.append(load_live_target(args.live_benchmark_root, row_index))
    return targets


def should_run_native_eval(
    *,
    args: argparse.Namespace,
    method_name: str,
    native_eval_count: int,
) -> bool:
    if not args.native_eval:
        return False
    allowed_baselines = set(args.native_eval_baselines)
    if allowed_baselines and method_name not in allowed_baselines:
        return False
    if args.native_eval_max_runs > 0 and native_eval_count >= args.native_eval_max_runs:
        return False
    return True


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["benchmark"]), str(row["baseline_name"]))
        groups.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (benchmark, baseline_name), items in sorted(groups.items()):
        mean_overall = mean_or_zero([float(item["overall_score"]) for item in items])
        mean_alignment = mean_or_zero([float(item["benchmark_alignment"]) for item in items])
        mean_expert = mean_or_zero([float(item["expert_style_quality"]) for item in items])
        mean_graph = mean_or_zero([float(item["graph_process"]) for item in items])
        mean_calls = mean_or_zero([float(item["llm_call_count"]) for item in items])
        mean_tokens = mean_or_zero([float(item["total_tokens"]) for item in items])
        mean_rounds = mean_or_zero([float(item.get("executed_round_count", 0.0) or 0.0) for item in items])
        mean_actions = mean_or_zero([float(item.get("action_count", 0.0) or 0.0) for item in items])
        runtime_protocols = sorted(
            {
                str(item.get("runtime_protocol", "")).strip() or "sequential_v1"
                for item in items
            }
        )
        native_rows = [
            float(item["native_average_normalized_10"])
            for item in items
            if item.get("native_average_normalized_10") is not None
        ]
        aggregates.append(
            {
                "benchmark": benchmark,
                "baseline_name": baseline_name,
                "mean_overall_score": mean_overall,
                "mean_benchmark_alignment": mean_alignment,
                "mean_expert_style_quality": mean_expert,
                "mean_graph_process": mean_graph,
                "mean_llm_call_count": mean_calls,
                "mean_total_tokens": mean_tokens,
                "mean_executed_round_count": mean_rounds,
                "mean_action_count": mean_actions,
                "runtime_protocols": runtime_protocols,
                "mean_native_average_normalized_10": mean_or_zero(native_rows) if native_rows else None,
                "run_count": len(items),
                "selected_run_dirs": [item["run_dir"] for item in items],
            }
        )
    return aggregates


def overall_aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["baseline_name"]), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for baseline_name, items in sorted(groups.items()):
        mean_overall = mean_or_zero([float(item["overall_score"]) for item in items])
        mean_alignment = mean_or_zero([float(item["benchmark_alignment"]) for item in items])
        mean_expert = mean_or_zero([float(item["expert_style_quality"]) for item in items])
        mean_graph = mean_or_zero([float(item["graph_process"]) for item in items])
        mean_calls = mean_or_zero([float(item["llm_call_count"]) for item in items])
        mean_tokens = mean_or_zero([float(item["total_tokens"]) for item in items])
        mean_rounds = mean_or_zero([float(item.get("executed_round_count", 0.0) or 0.0) for item in items])
        mean_actions = mean_or_zero([float(item.get("action_count", 0.0) or 0.0) for item in items])
        runtime_protocols = sorted(
            {
                str(item.get("runtime_protocol", "")).strip() or "sequential_v1"
                for item in items
            }
        )
        native_rows = [
            float(item["native_average_normalized_10"])
            for item in items
            if item.get("native_average_normalized_10") is not None
        ]
        aggregates.append(
            {
                "baseline_name": baseline_name,
                "mean_overall_score": mean_overall,
                "mean_benchmark_alignment": mean_alignment,
                "mean_expert_style_quality": mean_expert,
                "mean_graph_process": mean_graph,
                "mean_llm_call_count": mean_calls,
                "mean_total_tokens": mean_tokens,
                "mean_executed_round_count": mean_rounds,
                "mean_action_count": mean_actions,
                "runtime_protocols": runtime_protocols,
                "mean_native_average_normalized_10": mean_or_zero(native_rows) if native_rows else None,
                "run_count": len(items),
            }
        )
    return aggregates


def _token_multiplier_lookup(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    direct_row = next((row for row in rows if row["baseline_name"] == "direct"), None)
    direct_tokens = float(direct_row["mean_total_tokens"]) if direct_row is not None else 0.0
    lookup: dict[str, float | None] = {}
    for row in rows:
        lookup[str(row["baseline_name"])] = (
            None if direct_tokens <= 0 else round(float(row["mean_total_tokens"]) / direct_tokens, 2)
        )
    return lookup


def format_markdown_summary(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Cross-Benchmark Quality Batch")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Model: `{payload['model']}`")
    lines.append(f"- Generated at: `{payload['generated_at']}`")
    lines.append(
        f"- AI Idea Bench 2025 indices: `{', '.join(str(item) for item in payload['ai_indices'])}`"
    )
    lines.append(
        f"- LiveIdeaBench row indices: `{', '.join(str(item) for item in payload['live_row_indices'])}`"
    )
    lines.append("")
    lines.append("## Comparison Policy")
    lines.append("")
    for item in payload.get("method_plans", []):
        method_name = str(item.get("name", item.get("baseline_name", "")))
        runner_name = str(item.get("baseline_name", method_name))
        runner_text = "" if runner_name == method_name else f", runner=`{runner_name}`"
        runtime_protocol = str(item.get("runtime_protocol", "")).strip() or "sequential_v1"
        lines.append(
            f"- `{method_name}`{runner_text}: restarts={item['restarts']}, max_rounds={item['max_rounds']}, "
            f"stop_when_mature={item['stop_when_mature']}, runtime_protocol=`{runtime_protocol}`. {item['rationale']}"
        )
    lines.append("")
    lines.append("## Raw Data Table")
    lines.append("")
    lines.append("| Benchmark | Selector | Baseline | Protocol | Overall | Align. | Expert | Graph | Calls | Tokens | Rounds | Actions | Native | Stop | Run Dir |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in payload.get("raw_rows", []):
        if "error" in row:
            continue
        native_value = row.get("native_average_normalized_10")
        native_text = "--" if native_value is None else f"{float(native_value):.2f}"
        runtime_protocol = str(row.get("runtime_protocol", "")).strip() or "sequential_v1"
        lines.append(
            f"| `{row['benchmark']}` | `{row['display_selector']}` | `{row['baseline_name']}` | `{runtime_protocol}` | "
            f"{row['overall_score']:.2f} | {row['benchmark_alignment']:.2f} | "
            f"{row['expert_style_quality']:.2f} | {row['graph_process']:.2f} | "
            f"{row['llm_call_count']} | {row['total_tokens']} | {row.get('executed_round_count', 0)} | {row.get('action_count', 0)} | {native_text} | "
            f"`{row['stop_reason']}` | `{row['run_dir_name']}` |"
        )
    lines.append("")
    error_rows = [row for row in payload.get("raw_rows", []) if "error" in row]
    if error_rows:
        lines.append("## Failed Runs")
        lines.append("")
        lines.append("| Benchmark | Selector | Baseline | Error |")
        lines.append("| --- | --- | --- | --- |")
        for row in error_rows:
            lines.append(
                f"| `{row['benchmark']}` | `{row['display_selector']}` | `{row['baseline_name']}` | "
                f"{_clean_text(row['error'])} |"
            )
    lines.append("")
    lines.append("## Per-Benchmark Aggregate")
    lines.append("")
    lines.append("| Benchmark | Baseline | Protocols | Overall | Align. | Expert | Graph | Calls | Tokens | Rounds | Actions | x Direct | Native |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    benchmark_rows = list(payload.get("aggregate_rows", []))
    benchmark_groups: dict[str, list[dict[str, Any]]] = {}
    for row in benchmark_rows:
        benchmark_groups.setdefault(str(row["benchmark"]), []).append(row)
    for benchmark_name, rows in benchmark_groups.items():
        multiplier_lookup = _token_multiplier_lookup(rows)
        for row in rows:
            native_value = row.get("mean_native_average_normalized_10")
            native_text = "--" if native_value is None else f"{float(native_value):.2f}"
            multiplier_value = multiplier_lookup[row["baseline_name"]]
            multiplier_text = "--" if multiplier_value is None else f"{float(multiplier_value):.2f}x"
            protocols_text = ", ".join(str(item) for item in row.get("runtime_protocols", [])) or "sequential_v1"
            lines.append(
                f"| `{benchmark_name}` | `{row['baseline_name']}` | `{protocols_text}` | {row['mean_overall_score']:.2f} | "
                f"{row['mean_benchmark_alignment']:.2f} | {row['mean_expert_style_quality']:.2f} | "
                f"{row['mean_graph_process']:.2f} | {row['mean_llm_call_count']:.2f} | "
                f"{row['mean_total_tokens']:.0f} | {row['mean_executed_round_count']:.2f} | {row['mean_action_count']:.2f} | {multiplier_text} | {native_text} |"
            )
    lines.append("")
    lines.append("## Overall Aggregate")
    lines.append("")
    lines.append("| Baseline | Protocols | Overall | Align. | Expert | Graph | Calls | Tokens | Rounds | Actions | x Direct | Native |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    overall_rows = list(payload.get("overall_aggregate_rows", []))
    multiplier_lookup = _token_multiplier_lookup(overall_rows)
    for row in overall_rows:
        native_value = row.get("mean_native_average_normalized_10")
        native_text = "--" if native_value is None else f"{float(native_value):.2f}"
        multiplier_value = multiplier_lookup[row["baseline_name"]]
        multiplier_text = "--" if multiplier_value is None else f"{float(multiplier_value):.2f}x"
        protocols_text = ", ".join(str(item) for item in row.get("runtime_protocols", [])) or "sequential_v1"
        lines.append(
            f"| `{row['baseline_name']}` | `{protocols_text}` | {row['mean_overall_score']:.2f} | "
            f"{row['mean_benchmark_alignment']:.2f} | {row['mean_expert_style_quality']:.2f} | "
            f"{row['mean_graph_process']:.2f} | {row['mean_llm_call_count']:.2f} | "
            f"{row['mean_total_tokens']:.0f} | {row['mean_executed_round_count']:.2f} | {row['mean_action_count']:.2f} | {multiplier_text} | {native_text} |"
        )
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    for item in payload.get("findings", []):
        lines.append(f"1. {item}")
    lines.append("")
    lines.append("## Suggested Next Experiments")
    lines.append("")
    for item in payload.get("next_steps", []):
        lines.append(f"1. {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    settings = OpenAICompatibleSettings.from_json_file(args.llm_config)
    backend = OpenAICompatibleCollaborationBackend(settings)
    external_baseline_config = load_external_baseline_config(args.external_baseline_config)

    method_catalog = get_method_plan_catalog(args.plan_preset)
    requested_methods = (
        [canonical_method_plan_name(name) for name in args.baselines]
        if args.baselines
        else list(method_catalog)
    )
    requested_methods = list(dict.fromkeys(requested_methods))
    missing = [name for name in requested_methods if name not in method_catalog]
    if missing:
        raise SystemExit(f"Unsupported method-plan name(s): {', '.join(missing)}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_dir = args.output_dir / f"{timestamp}-{args.batch_name}"
    runs_root = batch_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    targets = load_targets(args)
    raw_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    native_eval_count = 0

    print_progress(
        "Loaded targets: "
        + ", ".join(f"{target.benchmark}:{target.display_selector}" for target in targets)
    )

    for target in targets:
        print_progress(
            f"Target {target.benchmark}/{target.display_selector}: {_clean_text(target.topic_preview)}"
        )
        for method_name in requested_methods:
            plan: ExperimentMethodPlan = method_catalog[method_name]
            print_progress(
                f"Running method '{method_name}' on {target.benchmark}/{target.display_selector}."
            )
            best_row: dict[str, Any] | None = None
            successful_runs = 0
            for restart in range(plan.restarts):
                prepared_instance = prepare_instance_for_method_plan(target.instance, plan=plan)
                metadata = dict(prepared_instance.metadata)
                metadata["batch_name"] = args.batch_name
                metadata["batch_restart"] = restart
                metadata["batch_method_plan"] = plan.as_dict()
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
                        baseline_name=plan.baseline_name,
                        collaboration_backend=backend,
                        external_baseline_config=external_baseline_config,
                        progress_callback=lambda message, mn=method_name, bb=target.benchmark, ss=target.display_selector, rr=restart: print_progress(
                            f"[{bb}][{ss}][{mn}][r{rr}] {message}"
                        ),
                        max_rounds=plan.max_rounds,
                        stop_when_mature=plan.stop_when_mature,
                    )
                except Exception as exc:
                    raw_rows.append(
                        {
                            "benchmark": target.benchmark,
                            "selector": target.selector,
                            "display_selector": target.display_selector,
                            "baseline_name": method_name,
                            "runner_baseline_name": plan.baseline_name,
                            "restart": restart,
                            "error": str(exc),
                        }
                    )
                    continue

                native_evaluation = None
                if should_run_native_eval(
                    args=args,
                    method_name=method_name,
                    native_eval_count=native_eval_count,
                ):
                    try:
                        native_evaluation = evaluate_benchmark_native(graph, settings=settings)
                        native_eval_count += 1
                    except Exception as exc:
                        graph.metadata["batch_native_eval_error"] = str(exc)

                run_dir = write_run_artifacts(
                    graph,
                    output_root=runs_root,
                    instance=prepared_instance,
                    native_evaluation_payload=(
                        native_evaluation.as_dict() if native_evaluation is not None else None
                    ),
                )
                evaluation = evaluate_graph(graph).as_dict()
                usage = summarize_graph_usage(graph)
                category_scores = evaluation.get("category_scores", {})
                if not isinstance(category_scores, dict):
                    category_scores = {}

                native_average = None
                if native_evaluation is not None:
                    summary = native_evaluation.summary if isinstance(native_evaluation.summary, dict) else {}
                    value = summary.get("available_average_normalized_10")
                    if value is not None:
                        native_average = float(value)

                row = {
                    "benchmark": target.benchmark,
                    "selector": target.selector,
                    "display_selector": target.display_selector,
                    "instance_name": prepared_instance.name,
                    "baseline_name": method_name,
                    "runner_baseline_name": plan.baseline_name,
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
                    "native_average_normalized_10": native_average,
                    "run_dir": str(run_dir),
                    "run_dir_name": run_dir.name,
                    "runtime_protocol": str(graph.metadata.get("runtime_protocol", "sequential_v1") or "sequential_v1"),
                    "stop_reason": str(graph.metadata.get("stop_reason", "")),
                    "executed_round_count": int(graph.metadata.get("executed_round_count", 0) or 0),
                    "action_count": len(graph.actions),
                    "title": graph.final_proposal.title if graph.final_proposal is not None else "",
                }
                raw_rows.append(row)
                successful_runs += 1
                if best_row is None or row["selection_score"] > best_row["selection_score"]:
                    best_row = row

            if best_row is None:
                print_progress(
                    f"Method '{method_name}' failed on {target.benchmark}/{target.display_selector}."
                )
                continue
            best_row = dict(best_row)
            best_row["successful_runs"] = successful_runs
            selected_rows.append(best_row)

    clean_selected_rows = [row for row in selected_rows if "error" not in row]
    aggregate_payload = aggregate_rows(clean_selected_rows)
    overall_payload = overall_aggregate_rows(clean_selected_rows)

    findings: list[str] = []
    for benchmark_name in ("AI_Idea_Bench_2025", "liveideabench"):
        rows = [row for row in aggregate_payload if row["benchmark"] == benchmark_name]
        if not rows:
            continue
        best_row = max(
            rows,
            key=lambda item: (item["mean_overall_score"], item["mean_benchmark_alignment"]),
        )
        findings.append(
            f"On `{benchmark_name}`, `{best_row['baseline_name']}` achieved the strongest mean local score "
            f"({best_row['mean_overall_score']:.2f}/10 overall; {best_row['mean_benchmark_alignment']:.2f}/10 alignment)."
        )
    if overall_payload:
        best_overall = max(
            overall_payload,
            key=lambda item: (item["mean_overall_score"], item["mean_benchmark_alignment"]),
        )
        findings.append(
            f"Across the combined small batch, `{best_overall['baseline_name']}` ranked highest by mean overall score "
            f"({best_overall['mean_overall_score']:.2f}/10)."
        )
        ours_row = next((row for row in overall_payload if row["baseline_name"] == "ours-eig"), None)
        direct_row = next((row for row in overall_payload if row["baseline_name"] == "direct"), None)
        if ours_row is not None and direct_row is not None:
            delta = round(float(ours_row["mean_overall_score"]) - float(direct_row["mean_overall_score"]), 2)
            findings.append(
                f"The EIG method differed from the direct baseline by {delta:+.2f} overall points, "
                "so quality and cost should be reported on separate axes."
            )

    next_steps = [
        "Run benchmark-native evaluation on only the strongest baseline and our method if the local cross-benchmark trends remain stable.",
        "Generate per-benchmark paper tables next, keeping process and cost in supplementary analyses rather than the main comparison table.",
        "Ablate the idea-graph method on maturity stop and repair/consolidation prompting now that the action space is state-aware.",
    ]

    payload = {
        "batch_name": args.batch_name,
        "plan_preset": args.plan_preset,
        "generated_at": datetime.now().isoformat(),
        "model": settings.model,
        "ai_indices": args.ai_indices,
        "live_row_indices": args.live_row_indices,
        "method_plans": [method_catalog[name].as_dict() for name in requested_methods],
        "raw_rows": raw_rows,
        "selected_rows": clean_selected_rows,
        "aggregate_rows": aggregate_payload,
        "overall_aggregate_rows": overall_payload,
        "findings": findings,
        "next_steps": next_steps,
    }

    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "batch_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (batch_dir / "batch_summary.md").write_text(
        format_markdown_summary(payload),
        encoding="utf-8",
    )
    write_csv(batch_dir / "raw_rows.csv", raw_rows)
    write_csv(batch_dir / "selected_rows.csv", clean_selected_rows)
    write_csv(batch_dir / "aggregate_rows.csv", aggregate_payload)
    write_csv(batch_dir / "overall_aggregate_rows.csv", overall_payload)

    print("== Batch Directory ==")
    print(batch_dir)
    print()
    print("== Overall Aggregate ==")
    for row in overall_payload:
        native_value = row.get("mean_native_average_normalized_10")
        native_text = "--" if native_value is None else f"{float(native_value):.2f}"
        print(
            f"{row['baseline_name']}: overall={row['mean_overall_score']:.2f}, "
            f"alignment={row['mean_benchmark_alignment']:.2f}, "
            f"calls={row['mean_llm_call_count']:.2f}, tokens={row['mean_total_tokens']:.0f}, "
            f"native={native_text}"
        )


if __name__ == "__main__":
    main()
