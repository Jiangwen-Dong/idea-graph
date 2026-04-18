from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent_backend import OpenAICompatibleCollaborationBackend
from .baselines import (
    attach_baseline_metadata,
    canonical_baseline_name,
    reset_two_head_runtime_controller_defaults,
    run_baseline_experiment,
)
from .benchmark_scoring import evaluate_benchmark_native
from .benchmarks import (
    ai_idea_bench_2025_instance_from_record,
    get_ai_idea_bench_2025_record,
    get_liveideabench_record,
    liveideabench_instance_from_record,
)
from .fs_utils import read_text_file, write_text_file
from .instances import ExperimentInstance
from .io import write_run_artifacts
from .settings import OpenAICompatibleSettings


def _normalize_str(value: object, *, default: str = "") -> str:
    normalized = str(value if value is not None else "").strip()
    return normalized or default


def _normalize_int(value: object, *, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer-like value, got {value!r}.") from exc


def _paper_baseline_name(runtime_baseline_name: str) -> str:
    canonical_name = canonical_baseline_name(runtime_baseline_name)
    if canonical_name == "ours-eig-critic-graph":
        return "ours-eig-graph-critic"
    return canonical_name


def load_packet_rows(
    manifest_path: str | Path,
    *,
    partition_role_filter: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifest = Path(manifest_path)
    seen_group_ids: set[str] = set()
    for line_index, raw_line in enumerate(read_text_file(manifest).splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{manifest} line {line_index} must contain a JSON object.")
        row = dict(payload)
        group_id = _normalize_str(row.get("group_id"))
        if not group_id:
            raise ValueError(f"{manifest} line {line_index} is missing group_id.")
        if group_id in seen_group_ids:
            raise ValueError(f"Duplicate group_id in packet manifest: {group_id}")
        seen_group_ids.add(group_id)

        partition_role = _normalize_str(row.get("partition_role"))
        if partition_role_filter and partition_role != partition_role_filter:
            continue

        row["group_id"] = group_id
        row["benchmark"] = _normalize_str(row.get("benchmark"))
        row["instance_name"] = _normalize_str(row.get("instance_name"))
        row["partition_role"] = partition_role
        row["source_split"] = _normalize_str(row.get("source_split"))
        if "benchmark_index" in row:
            row["benchmark_index"] = _normalize_int(row.get("benchmark_index"))
        if "row_index" in row:
            row["row_index"] = _normalize_int(row.get("row_index"))
        row["benchmark_keyword"] = _normalize_str(row.get("benchmark_keyword"))
        rows.append(row)
    return rows


def packet_row_to_benchmark_args(row: Mapping[str, Any]) -> dict[str, Any]:
    benchmark = _normalize_str(row.get("benchmark"))
    if benchmark == "AI_Idea_Bench_2025":
        benchmark_index = _normalize_int(row.get("benchmark_index"))
        if benchmark_index is None:
            raise ValueError(f"AIIB row is missing benchmark_index: {row!r}")
        return {
            "benchmark": "ai_idea_bench_2025",
            "benchmark_index": benchmark_index,
        }
    if benchmark == "liveideabench":
        row_index = _normalize_int(row.get("row_index"))
        if row_index is None:
            raise ValueError(f"LiveIdeaBench row is missing row_index: {row!r}")
        return {
            "benchmark": "liveideabench",
            "benchmark_index": row_index,
            "benchmark_keyword": _normalize_str(row.get("benchmark_keyword")),
        }
    raise ValueError(f"Unsupported benchmark in packet row: {benchmark!r}")


def resolve_benchmark_root(base_root: str | Path, benchmark_name: str) -> Path:
    base = Path(base_root)
    if base.name == benchmark_name:
        return base
    return base / benchmark_name


def load_benchmark_instance(
    row: Mapping[str, Any],
    *,
    benchmark_root_base: str | Path,
) -> ExperimentInstance:
    benchmark_args = packet_row_to_benchmark_args(row)
    benchmark_name = str(benchmark_args["benchmark"])
    benchmark_root = resolve_benchmark_root(benchmark_root_base, benchmark_name)
    if benchmark_name == "ai_idea_bench_2025":
        record = get_ai_idea_bench_2025_record(benchmark_root, int(benchmark_args["benchmark_index"]))
        return ai_idea_bench_2025_instance_from_record(record, benchmark_root=benchmark_root)
    record = get_liveideabench_record(
        benchmark_root,
        int(benchmark_args["benchmark_index"]),
        keyword=_normalize_str(benchmark_args.get("benchmark_keyword")),
    )
    return liveideabench_instance_from_record(record, benchmark_root=benchmark_root)


def build_openai_backend(
    llm_config_path: str | Path | None,
) -> tuple[OpenAICompatibleSettings | None, OpenAICompatibleCollaborationBackend | None]:
    if llm_config_path is None:
        return None, None
    settings = OpenAICompatibleSettings.from_json_file(llm_config_path)
    return settings, OpenAICompatibleCollaborationBackend(settings)


def apply_runtime_controller_overrides(
    instance: ExperimentInstance,
    *,
    runtime_controller_calibration_path: str | Path | None,
    disable_runtime_calibration: bool,
) -> ExperimentInstance:
    metadata = dict(instance.metadata)
    if disable_runtime_calibration:
        if str(metadata.get("runtime_controller_kind", "")).strip() == "relation_graph_two_head_critic":
            metadata = reset_two_head_runtime_controller_defaults(metadata)
        else:
            metadata.pop("runtime_controller_calibration_path", None)
            metadata.pop("runtime_controller_calibration_source", None)
            metadata.pop("runtime_controller_calibration_version", None)
        metadata["runtime_controller_disable_calibration"] = True
    elif runtime_controller_calibration_path is not None:
        metadata["runtime_controller_calibration_path"] = str(
            Path(runtime_controller_calibration_path).resolve()
        )
        metadata.pop("runtime_controller_disable_calibration", None)
    return ExperimentInstance(
        name=instance.name,
        topic=instance.topic,
        literature=list(instance.literature),
        source_path=instance.source_path,
        metadata=metadata,
    )


def execute_packet_run(
    row: Mapping[str, Any],
    *,
    baseline_name: str,
    output_root: str | Path,
    benchmark_root_base: str | Path,
    max_rounds: int,
    native_eval: bool,
    llm_config_path: str | Path | None,
    runtime_controller_calibration_path: str | Path | None = None,
    disable_runtime_calibration: bool = False,
) -> dict[str, Any]:
    settings, backend = build_openai_backend(llm_config_path)
    if native_eval and settings is None:
        raise ValueError("native_eval requires llm_config_path so the judge backend can be built.")

    instance = load_benchmark_instance(row, benchmark_root_base=benchmark_root_base)
    instance = attach_baseline_metadata(instance, baseline_name=baseline_name, io_mode="auto")
    instance = apply_runtime_controller_overrides(
        instance,
        runtime_controller_calibration_path=runtime_controller_calibration_path,
        disable_runtime_calibration=disable_runtime_calibration,
    )

    experiment_metadata = dict(instance.metadata)
    experiment_metadata["agent_backend"] = "openai-compatible" if backend is not None else "deterministic"
    experiment_metadata["max_rounds_requested"] = max(1, max_rounds)
    experiment_metadata["stop_when_mature"] = True
    experiment_metadata["packet_group_id"] = _normalize_str(row.get("group_id"))
    experiment_metadata["packet_partition_role"] = _normalize_str(row.get("partition_role"))
    experiment_metadata["packet_source_split"] = _normalize_str(row.get("source_split"))
    if backend is not None:
        experiment_metadata["openai_compatible"] = backend.settings.sanitized_dict()
    if native_eval and settings is not None:
        experiment_metadata["benchmark_native_eval_enabled"] = True
        experiment_metadata["benchmark_native_eval_backend"] = settings.sanitized_dict()
    instance = ExperimentInstance(
        name=instance.name,
        topic=instance.topic,
        literature=list(instance.literature),
        source_path=instance.source_path,
        metadata=experiment_metadata,
    )

    graph = run_baseline_experiment(
        instance,
        baseline_name=baseline_name,
        collaboration_backend=backend,
        max_rounds=max(1, max_rounds),
        stop_when_mature=True,
        external_baseline_config={},
    )
    native_evaluation = evaluate_benchmark_native(graph, settings=settings) if native_eval else None
    run_dir = write_run_artifacts(
        graph,
        output_root=Path(output_root) / "runs" / canonical_baseline_name(baseline_name),
        instance=instance,
        native_evaluation_payload=native_evaluation.as_dict() if native_evaluation is not None else None,
    )
    return build_run_manifest_row(
        row,
        baseline_name=baseline_name,
        run_dir=run_dir,
    )


def build_run_manifest_row(
    row: Mapping[str, Any],
    *,
    baseline_name: str,
    run_dir: str | Path,
) -> dict[str, Any]:
    return {
        "group_id": _normalize_str(row.get("group_id")),
        "benchmark": _normalize_str(row.get("benchmark")),
        "instance_name": _normalize_str(row.get("instance_name")),
        "partition_role": _normalize_str(row.get("partition_role")),
        "source_split": _normalize_str(row.get("source_split")),
        "baseline_name": canonical_baseline_name(baseline_name),
        "paper_baseline_name": _paper_baseline_name(baseline_name),
        "run_dir": str(Path(run_dir)),
        "benchmark_index": _normalize_int(row.get("benchmark_index")),
        "row_index": _normalize_int(row.get("row_index")),
        "benchmark_keyword": _normalize_str(row.get("benchmark_keyword")),
    }


def load_run_manifest_rows(manifest_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_index, raw_line in enumerate(read_text_file(manifest_path).splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{manifest_path} line {line_index} must contain a JSON object.")
        rows.append(dict(payload))
    return rows


def _extract_native_score(summary_payload: Mapping[str, Any]) -> float | None:
    native_payload = summary_payload.get("benchmark_native_evaluation", {})
    if not isinstance(native_payload, Mapping):
        return None
    summary = native_payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return None
    value = summary.get("available_average_normalized_10")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _iter_token_usage_payloads(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }.issubset(payload.keys()):
            return [payload]
        rows: list[Mapping[str, Any]] = []
        for value in payload.values():
            rows.extend(_iter_token_usage_payloads(value))
        return rows
    if isinstance(payload, list):
        rows = []
        for item in payload:
            rows.extend(_iter_token_usage_payloads(item))
        return rows
    return []


def _token_total(payload: Mapping[str, Any]) -> float:
    try:
        return float(payload.get("total_tokens", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _load_run_payload(manifest_row: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(_normalize_str(manifest_row.get("run_dir")))
    summary_payload = json.loads(read_text_file(run_dir / "summary.json"))
    graph_payload = json.loads(read_text_file(run_dir / "graph.json"))
    graph_metadata = graph_payload.get("metadata", {})
    actions = graph_payload.get("actions", [])
    if not isinstance(graph_metadata, Mapping):
        graph_metadata = {}
    if not isinstance(actions, list):
        actions = []
    runtime_log = graph_metadata.get("runtime_controller_log", [])
    if not isinstance(runtime_log, list):
        runtime_log = []
    usage_payloads = _iter_token_usage_payloads(graph_payload)
    return {
        "manifest_row": dict(manifest_row),
        "summary_payload": summary_payload,
        "graph_payload": graph_payload,
        "graph_metadata": dict(graph_metadata),
        "actions": list(actions),
        "runtime_log": list(runtime_log),
        "native_score": _extract_native_score(summary_payload),
        "executed_round_count": _normalize_int(summary_payload.get("executed_round_count"), default=0)
        or 0,
        "llm_call_count": len(usage_payloads),
        "total_tokens": sum(_token_total(row) for row in usage_payloads),
    }


def _mean(values: Sequence[float]) -> float:
    return round(float(statistics.mean(values)), 4) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return round(float(statistics.median(values)), 4) if values else 0.0


def _bootstrap_mean_ci(values: Sequence[float], *, seed: int = 0, samples: int = 1000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [round(float(values[0]), 4), round(float(values[0]), 4)]
    generator = random.Random(seed)
    means: list[float] = []
    values_list = list(values)
    for _ in range(samples):
        sample = [values_list[generator.randrange(len(values_list))] for _ in range(len(values_list))]
        means.append(float(statistics.mean(sample)))
    means.sort()
    lower_index = max(0, math.floor(0.025 * (len(means) - 1)))
    upper_index = min(len(means) - 1, math.floor(0.975 * (len(means) - 1)))
    return [round(means[lower_index], 4), round(means[upper_index], 4)]


def _instance_key(payload: Mapping[str, Any]) -> str:
    manifest_row = payload["manifest_row"]
    return "::".join(
        [
            _normalize_str(manifest_row.get("benchmark")),
            _normalize_str(manifest_row.get("instance_name")),
            _normalize_str(manifest_row.get("partition_role")),
        ]
    )


def _aggregate_scores(payloads: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    rounds_grouped: dict[str, list[float]] = {}
    calls_grouped: dict[str, list[float]] = {}
    tokens_grouped: dict[str, list[float]] = {}
    for payload in payloads:
        baseline_name = _normalize_str(payload["manifest_row"].get("paper_baseline_name"))
        native_score = payload.get("native_score")
        if native_score is None:
            continue
        grouped.setdefault(baseline_name, []).append(float(native_score))
        rounds_grouped.setdefault(baseline_name, []).append(
            float(payload.get("executed_round_count", 0.0) or 0.0)
        )
        calls_grouped.setdefault(baseline_name, []).append(
            float(payload.get("llm_call_count", 0.0) or 0.0)
        )
        tokens_grouped.setdefault(baseline_name, []).append(
            float(payload.get("total_tokens", 0.0) or 0.0)
        )
    metrics: dict[str, dict[str, float]] = {}
    for baseline_name, scores in sorted(grouped.items()):
        metrics[baseline_name] = {
            "instance_count": len(scores),
            "mean_score": _mean(scores),
            "median_score": _median(scores),
            "mean_executed_round_count": _mean(rounds_grouped.get(baseline_name, [])),
            "mean_llm_call_count": _mean(calls_grouped.get(baseline_name, [])),
            "mean_total_tokens": _mean(tokens_grouped.get(baseline_name, [])),
        }
    return metrics


def _aggregate_pairs(payloads: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int | list[float]]]:
    by_instance_baseline: dict[str, dict[str, list[float]]] = {}
    for payload in payloads:
        native_score = payload.get("native_score")
        if native_score is None:
            continue
        key = _instance_key(payload)
        baseline_name = _normalize_str(payload["manifest_row"].get("paper_baseline_name"))
        by_instance_baseline.setdefault(key, {}).setdefault(baseline_name, []).append(float(native_score))

    paired: dict[str, dict[str, float | int | list[float]]] = {}
    for instance_scores in by_instance_baseline.values():
        if "ours-eig" not in instance_scores:
            continue
        ours_score = float(statistics.mean(instance_scores["ours-eig"]))
        for baseline_name, scores in instance_scores.items():
            if baseline_name == "ours-eig":
                continue
            delta = float(statistics.mean(scores)) - ours_score
            bucket = paired.setdefault(
                baseline_name,
                {
                    "comparable_instance_count": 0,
                    "delta_values": [],
                },
            )
            bucket["comparable_instance_count"] = int(bucket["comparable_instance_count"]) + 1
            cast_values = bucket["delta_values"]
            assert isinstance(cast_values, list)
            cast_values.append(delta)

    finalized: dict[str, dict[str, float | int | list[float]]] = {}
    for baseline_name, bucket in sorted(paired.items()):
        deltas = [float(value) for value in bucket["delta_values"]]
        comparable_count = int(bucket["comparable_instance_count"])
        wins = sum(1 for delta in deltas if delta > 0.0)
        ties = sum(1 for delta in deltas if delta == 0.0)
        finalized[baseline_name] = {
            "comparable_instance_count": comparable_count,
            "mean_delta": _mean(deltas),
            "median_delta": _median(deltas),
            "win_rate": round(wins / comparable_count, 4) if comparable_count else 0.0,
            "tie_count": ties,
            "bootstrap_ci": _bootstrap_mean_ci(deltas),
        }
    return finalized


def _aggregate_controller_traces(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline_buckets: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        baseline_name = _normalize_str(payload["manifest_row"].get("paper_baseline_name"))
        bucket = baseline_buckets.setdefault(
            baseline_name,
            {
                "run_count": 0,
                "decision_count": 0,
                "fallback_count": 0,
                "selected_source_counts": {},
                "action_kind_counts": {},
                "action_round_counts": {},
                "stop_reason_counts": {},
                "total_rounds_without_commit": 0,
            },
        )
        bucket["run_count"] += 1

        stop_reason = _normalize_str(payload["summary_payload"].get("stop_reason"), default="unknown")
        bucket["stop_reason_counts"][stop_reason] = bucket["stop_reason_counts"].get(stop_reason, 0) + 1

        runtime_log = payload.get("runtime_log", [])
        if isinstance(runtime_log, list):
            bucket["decision_count"] += len(runtime_log)
            for row in runtime_log:
                if not isinstance(row, Mapping):
                    continue
                selected_source = _normalize_str(row.get("selected_source"), default="unknown")
                bucket["selected_source_counts"][selected_source] = (
                    bucket["selected_source_counts"].get(selected_source, 0) + 1
                )
                if bool(row.get("used_heuristic_fallback", False)):
                    bucket["fallback_count"] += 1

        actions = payload.get("actions", [])
        round_names_without_commit: set[str] = set()
        commit_rounds: set[str] = set()
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            kind = _normalize_str(action.get("kind"), default="unknown")
            round_name = _normalize_str(action.get("round_name"), default="unknown")
            bucket["action_kind_counts"][kind] = bucket["action_kind_counts"].get(kind, 0) + 1
            bucket["action_round_counts"][round_name] = bucket["action_round_counts"].get(round_name, 0) + 1
            if kind == "commit":
                commit_rounds.add(round_name)
            elif round_name:
                round_names_without_commit.add(round_name)
        bucket["total_rounds_without_commit"] += len(round_names_without_commit - commit_rounds)

    for bucket in baseline_buckets.values():
        decision_count = int(bucket["decision_count"])
        selected_source_counts = bucket["selected_source_counts"]
        critic_selected = int(selected_source_counts.get("critic", 0))
        bucket["controller_override_rate"] = (
            round(critic_selected / decision_count, 4) if decision_count else 0.0
        )
        bucket["materialized_override_rate"] = bucket["controller_override_rate"]
        bucket["heuristic_fallback_rate"] = (
            round(int(bucket["fallback_count"]) / decision_count, 4) if decision_count else 0.0
        )
    return dict(sorted(baseline_buckets.items()))


def summarize_packet_runs(run_manifest_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payloads = [_load_run_payload(row) for row in run_manifest_rows]
    readout_filters = {
        "critic_dev": lambda payload: _normalize_str(payload["manifest_row"].get("partition_role")) == "critic_dev",
        "critic_train": lambda payload: _normalize_str(payload["manifest_row"].get("partition_role")) == "critic_train",
        "pooled": lambda payload: True,
    }
    readouts: dict[str, Any] = {}
    for name, predicate in readout_filters.items():
        selected_payloads = [payload for payload in payloads if predicate(payload)]
        readouts[name] = {
            "instance_count": len({
                "::".join(
                    [
                        _normalize_str(payload["manifest_row"].get("benchmark")),
                        _normalize_str(payload["manifest_row"].get("instance_name")),
                        _normalize_str(payload["manifest_row"].get("partition_role")),
                    ]
                )
                for payload in selected_payloads
            }),
            "baseline_metrics": _aggregate_scores(selected_payloads),
            "paired_against_ours_eig": _aggregate_pairs(selected_payloads),
        }
    controller_trace_summary = _aggregate_controller_traces(payloads)
    return {
        "readouts": readouts,
        "controller_trace_summary": controller_trace_summary,
    }


def format_packet_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Controller Evaluation Packet Summary", ""]
    readouts = summary.get("readouts", {})
    if isinstance(readouts, Mapping):
        for readout_name, payload in readouts.items():
            lines.extend([f"## {readout_name}", ""])
            instance_count = payload.get("instance_count", 0) if isinstance(payload, Mapping) else 0
            lines.append(f"- instance_count: `{instance_count}`")
            baseline_metrics = payload.get("baseline_metrics", {}) if isinstance(payload, Mapping) else {}
            if isinstance(baseline_metrics, Mapping):
                lines.append("- baseline metrics:")
                for baseline_name, metrics in baseline_metrics.items():
                    if not isinstance(metrics, Mapping):
                        continue
                    lines.append(
                        f"  - `{baseline_name}`: mean `{metrics.get('mean_score', 0.0)}`, "
                        f"median `{metrics.get('median_score', 0.0)}`, "
                        f"rounds `{metrics.get('mean_executed_round_count', 0.0)}`, "
                        f"calls `{metrics.get('mean_llm_call_count', 0.0)}`, "
                        f"tokens `{metrics.get('mean_total_tokens', 0.0)}`"
                    )
            paired = payload.get("paired_against_ours_eig", {}) if isinstance(payload, Mapping) else {}
            if isinstance(paired, Mapping) and paired:
                lines.append("- paired against `ours-eig`:")
                for baseline_name, metrics in paired.items():
                    if not isinstance(metrics, Mapping):
                        continue
                    lines.append(
                        f"  - `{baseline_name}`: mean delta `{metrics.get('mean_delta', 0.0)}`, "
                        f"win rate `{metrics.get('win_rate', 0.0)}`, "
                        f"CI `{metrics.get('bootstrap_ci', [0.0, 0.0])}`"
                    )
            lines.append("")

    controller_trace_summary = summary.get("controller_trace_summary", {})
    if isinstance(controller_trace_summary, Mapping):
        lines.extend(["## Controller Traces", ""])
        for baseline_name, payload in controller_trace_summary.items():
            if not isinstance(payload, Mapping):
                continue
            lines.append(
                f"- `{baseline_name}`: decisions `{payload.get('decision_count', 0)}`, "
                f"override rate `{payload.get('controller_override_rate', 0.0)}`, "
                f"fallback rate `{payload.get('heuristic_fallback_rate', 0.0)}`"
            )
    lines.append("")
    return "\n".join(lines)


def write_packet_summary(output_root: str | Path, summary: Mapping[str, Any]) -> None:
    output_dir = Path(output_root)
    write_text_file(
        output_dir / "paired_summary.json",
        json.dumps(dict(summary), indent=2, ensure_ascii=False, default=str),
    )
    write_text_file(
        output_dir / "paired_summary.md",
        format_packet_summary_markdown(summary),
    )
    controller_trace_summary = summary.get("controller_trace_summary", {})
    write_text_file(
        output_dir / "controller_trace_summary.json",
        json.dumps(controller_trace_summary, indent=2, ensure_ascii=False, default=str),
    )
