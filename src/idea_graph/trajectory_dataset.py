from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fs_utils import read_text_file, write_text_file


@dataclass(frozen=True)
class PricingConfig:
    prompt_price_per_1m_tokens: float
    completion_price_per_1m_tokens: float


@dataclass(frozen=True)
class ExportResult:
    dataset_dir: Path
    run_count: int
    transition_count: int


def discover_run_dirs(input_roots: Sequence[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for root in input_roots:
        root_path = Path(root)
        if root_path.is_file():
            continue
        for summary_path in root_path.rglob("summary.json"):
            run_dir = summary_path.parent
            if (run_dir / "graph.json").exists():
                discovered.add(run_dir.resolve())
    return sorted(discovered)


def load_run_artifacts(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_payload = json.loads(read_text_file(run_dir / "summary.json"))
    graph_payload = json.loads(read_text_file(run_dir / "graph.json"))
    if not isinstance(summary_payload, dict):
        raise ValueError(f"{run_dir / 'summary.json'} does not contain a JSON object.")
    if not isinstance(graph_payload, dict):
        raise ValueError(f"{run_dir / 'graph.json'} does not contain a JSON object.")
    return summary_payload, graph_payload


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _action_source_counts(actions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in actions:
        source = str(action.get("source", "deterministic") or "deterministic").strip()
        counts[source] = counts.get(source, 0) + 1
    return counts


def _native_average(native_payload: Mapping[str, Any]) -> float | None:
    summary = native_payload.get("summary", {})
    if isinstance(summary, Mapping):
        average = _safe_float(summary.get("available_average_normalized_10"))
        if average is not None:
            return average
    metrics = native_payload.get("metrics", [])
    if isinstance(metrics, list):
        for item in metrics:
            if isinstance(item, Mapping) and str(item.get("key", "")).strip() == "average":
                return _safe_float(item.get("score"))
    return None


def _timestamp_span_seconds(values: Sequence[datetime]) -> float | None:
    if len(values) < 2:
        return None
    ordered = sorted(values)
    return float((ordered[-1] - ordered[0]).total_seconds())


def extract_trace_stats(
    graph_payload: Mapping[str, Any],
    *,
    pricing: PricingConfig | None = None,
) -> dict[str, Any]:
    metadata = graph_payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    llm_call_count = 0
    timestamps: list[datetime] = []

    agent_traces = metadata.get("agent_traces", [])
    if isinstance(agent_traces, list):
        for trace in agent_traces:
            if not isinstance(trace, Mapping):
                continue
            raw_response = trace.get("raw_response", {})
            if not isinstance(raw_response, Mapping):
                continue
            usage = raw_response.get("usage", {})
            if isinstance(usage, Mapping):
                prompt_tokens += _safe_int(usage.get("prompt_tokens"))
                completion_tokens += _safe_int(usage.get("completion_tokens"))
                total_tokens += _safe_int(usage.get("total_tokens"))
                llm_call_count += 1
            created = _parse_timestamp(raw_response.get("created"))
            if created is not None:
                timestamps.append(created)

    final_synthesis_trace = metadata.get("final_synthesis_trace")
    if isinstance(final_synthesis_trace, Mapping):
        raw_response = final_synthesis_trace.get("raw_response", {})
        if isinstance(raw_response, Mapping):
            usage = raw_response.get("usage", {})
            if isinstance(usage, Mapping):
                prompt_tokens += _safe_int(usage.get("prompt_tokens"))
                completion_tokens += _safe_int(usage.get("completion_tokens"))
                total_tokens += _safe_int(usage.get("total_tokens"))
                llm_call_count += 1
            created = _parse_timestamp(raw_response.get("created"))
            if created is not None:
                timestamps.append(created)

    actions = graph_payload.get("actions", [])
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            created = _parse_timestamp(action.get("timestamp"))
            if created is not None:
                timestamps.append(created)

    estimated_cost = None
    if pricing is not None:
        estimated_cost = (
            prompt_tokens * pricing.prompt_price_per_1m_tokens
            + completion_tokens * pricing.completion_price_per_1m_tokens
        ) / 1_000_000

    return {
        "llm_call_count": llm_call_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": estimated_cost,
        "wall_clock_seconds": _timestamp_span_seconds(timestamps),
    }


def build_run_manifest_row(
    run_dir: Path,
    summary_payload: Mapping[str, Any],
    graph_payload: Mapping[str, Any],
    *,
    pricing: PricingConfig | None = None,
) -> dict[str, Any]:
    metadata = graph_payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}

    actions = graph_payload.get("actions", [])
    if not isinstance(actions, list):
        actions = []
    nodes = graph_payload.get("nodes", {})
    if not isinstance(nodes, Mapping):
        nodes = {}
    edges = graph_payload.get("edges", [])
    if not isinstance(edges, list):
        edges = []

    trace_stats = extract_trace_stats(graph_payload, pricing=pricing)
    idea_evaluation = summary_payload.get("idea_evaluation", {})
    native_evaluation = summary_payload.get("benchmark_native_evaluation", {})

    if not isinstance(idea_evaluation, Mapping):
        idea_evaluation = {}
    if not isinstance(native_evaluation, Mapping):
        native_evaluation = {}

    category_scores = idea_evaluation.get("category_scores", {})
    if not isinstance(category_scores, Mapping):
        category_scores = {}

    baseline_name = str(metadata.get("baseline_name", "")).strip() or "unknown"
    benchmark = (
        str(metadata.get("benchmark", "")).strip()
        or str(native_evaluation.get("benchmark", "")).strip()
        or "unknown"
    )

    return {
        "run_dir": str(Path(run_dir).resolve()),
        "benchmark": benchmark,
        "instance_name": str(summary_payload.get("instance_name", run_dir.name)).strip() or run_dir.name,
        "baseline_name": baseline_name,
        "topic": str(summary_payload.get("topic", graph_payload.get("topic", ""))).strip(),
        "is_eig_run": baseline_name == "ours-eig",
        "has_graph": True,
        "has_summary": True,
        "executed_round_count": _safe_int(summary_payload.get("executed_round_count")),
        "action_count": len(actions),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "action_source_counts": dict(summary_payload.get("action_source_counts", {}) or _action_source_counts(actions)),
        "stopped_early": bool(summary_payload.get("stopped_early", False)),
        "matured_at_round": summary_payload.get("matured_at_round"),
        "final_local_overall": _safe_float(idea_evaluation.get("overall_score")),
        "final_local_alignment": _safe_float(category_scores.get("benchmark_alignment")),
        "final_native_average": _native_average(native_evaluation),
        "trace_llm_call_count": trace_stats["llm_call_count"],
        "trace_prompt_tokens": trace_stats["prompt_tokens"],
        "trace_completion_tokens": trace_stats["completion_tokens"],
        "trace_total_tokens": trace_stats["total_tokens"],
        "estimated_cost": trace_stats["estimated_cost"],
        "wall_clock_seconds": trace_stats["wall_clock_seconds"],
        "has_agent_traces": bool(metadata.get("agent_traces")),
        "has_final_synthesis_trace": isinstance(metadata.get("final_synthesis_trace"), Mapping),
        "has_override_trace": bool(metadata.get("utility_controller_overrides")),
        "has_native_eval": bool(native_evaluation),
        "has_local_eval": bool(idea_evaluation),
    }


def _edge_is_contradiction(edge: Mapping[str, Any]) -> bool:
    relation = str(edge.get("relation", "")).strip().lower()
    return "contradict" in relation


def _infer_resolution_times(graph_payload: Mapping[str, Any]) -> dict[str, datetime]:
    actions = graph_payload.get("actions", [])
    if not isinstance(actions, list):
        actions = []
    resolution_times: dict[str, datetime] = {}
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        kind = str(action.get("kind", "")).strip()
        if kind != "propose_repair":
            continue
        action_time = _parse_timestamp(action.get("timestamp"))
        if action_time is None:
            continue
        for target_id in action.get("target_ids", []):
            target_text = str(target_id).strip()
            if not target_text:
                continue
            if target_text not in resolution_times or action_time < resolution_times[target_text]:
                resolution_times[target_text] = action_time
    return resolution_times


def reconstruct_state_before_action(graph_payload: Mapping[str, Any], action_index: int) -> dict[str, Any]:
    actions = graph_payload.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError("graph_payload.actions must be a list.")
    if action_index < 0 or action_index >= len(actions):
        raise IndexError(f"Action index {action_index} is out of range.")

    action = actions[action_index]
    if not isinstance(action, Mapping):
        raise ValueError(f"Action at index {action_index} is not a JSON object.")
    action_time = _parse_timestamp(action.get("timestamp"))
    if action_time is None:
        raise ValueError(f"Action at index {action_index} does not have a valid timestamp.")

    nodes_payload = graph_payload.get("nodes", {})
    if not isinstance(nodes_payload, Mapping):
        nodes_payload = {}
    edges_payload = graph_payload.get("edges", [])
    if not isinstance(edges_payload, list):
        edges_payload = []

    resolution_times = _infer_resolution_times(graph_payload)

    included_nodes: dict[str, dict[str, Any]] = {}
    for node_id, node_payload in nodes_payload.items():
        if not isinstance(node_payload, Mapping):
            continue
        created_at = _parse_timestamp(node_payload.get("created_at"))
        if created_at is None or created_at <= action_time:
            included_nodes[str(node_id)] = dict(node_payload)

    included_edges: list[dict[str, Any]] = []
    unresolved_contradictions = 0
    support_edge_count = 0
    for edge_payload in edges_payload:
        if not isinstance(edge_payload, Mapping):
            continue
        created_at = _parse_timestamp(edge_payload.get("created_at"))
        if created_at is not None and created_at > action_time:
            continue
        source_id = str(edge_payload.get("source_id", "")).strip()
        target_id = str(edge_payload.get("target_id", "")).strip()
        if source_id and source_id not in included_nodes:
            continue
        if target_id and target_id not in included_nodes:
            continue
        edge_copy = dict(edge_payload)
        if _edge_is_contradiction(edge_payload):
            if bool(edge_payload.get("resolved", False)):
                resolved_at = resolution_times.get(target_id)
                edge_copy["resolved"] = bool(resolved_at is not None and action_time > resolved_at)
            if not bool(edge_copy.get("resolved", False)):
                unresolved_contradictions += 1
        if str(edge_payload.get("relation", "")).strip().lower() == "supports":
            support_edge_count += 1
        included_edges.append(edge_copy)

    return {
        "action_id": str(action.get("id", "")).strip(),
        "action_index": action_index,
        "action_timestamp": action.get("timestamp"),
        "nodes": included_nodes,
        "edges": included_edges,
        "node_count": len(included_nodes),
        "edge_count": len(included_edges),
        "contradiction_count": unresolved_contradictions,
        "support_edge_count": support_edge_count,
    }


def _round_summary_maps(summary_payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    rounds = summary_payload.get("rounds", [])
    if not isinstance(rounds, list):
        return {}, []
    round_map: dict[str, Any] = {}
    ordered_rounds: list[str] = []
    for item in rounds:
        if not isinstance(item, Mapping):
            continue
        round_name = str(item.get("round", "")).strip()
        if not round_name:
            continue
        round_map[round_name] = dict(item)
        ordered_rounds.append(round_name)
    return round_map, ordered_rounds


def _override_lookup(graph_payload: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    metadata = graph_payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return {}
    overrides = metadata.get("utility_controller_overrides", [])
    if not isinstance(overrides, list):
        return {}
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for item in overrides:
        if not isinstance(item, Mapping):
            continue
        round_name = str(item.get("round", "")).strip()
        role = str(item.get("role", "")).strip()
        if round_name and role:
            lookup[(round_name, role)] = dict(item)
    return lookup


def _action_step_trace(
    graph_payload: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = graph_payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    traces = metadata.get("agent_traces", [])
    if not isinstance(traces, list):
        traces = []
    action_role = str(action.get("role", "")).strip()
    action_round = str(action.get("round_name", "")).strip()
    for trace in traces:
        if not isinstance(trace, Mapping):
            continue
        if str(trace.get("role", "")).strip() != action_role:
            continue
        request_messages = trace.get("request_messages", [])
        request_text = json.dumps(request_messages, ensure_ascii=False) if isinstance(request_messages, list) else ""
        if action_round and action_round not in request_text:
            continue
        raw_response = trace.get("raw_response", {})
        if not isinstance(raw_response, Mapping):
            continue
        usage = raw_response.get("usage", {})
        if not isinstance(usage, Mapping):
            continue
        return {
            "prompt_tokens": _safe_int(usage.get("prompt_tokens")),
            "completion_tokens": _safe_int(usage.get("completion_tokens")),
            "total_tokens": _safe_int(usage.get("total_tokens")),
        }
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def build_transition_rows(
    run_dir: Path,
    summary_payload: Mapping[str, Any],
    graph_payload: Mapping[str, Any],
    *,
    snapshot_dir: Path,
) -> list[dict[str, Any]]:
    actions = graph_payload.get("actions", [])
    if not isinstance(actions, list) or not actions:
        return []

    run_path = Path(run_dir).resolve()
    round_map, ordered_rounds = _round_summary_maps(summary_payload)
    override_lookup = _override_lookup(graph_payload)

    native_evaluation = summary_payload.get("benchmark_native_evaluation", {})
    if not isinstance(native_evaluation, Mapping):
        native_evaluation = {}
    idea_evaluation = summary_payload.get("idea_evaluation", {})
    if not isinstance(idea_evaluation, Mapping):
        idea_evaluation = {}

    metadata = graph_payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}

    write_rows: list[dict[str, Any]] = []
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for action_index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            continue
        snapshot = reconstruct_state_before_action(graph_payload, action_index)
        snapshot_name = f"{run_path.name}-step-{action_index:03d}.json"
        snapshot_path = snapshot_dir / snapshot_name
        write_text_file(
            snapshot_path,
            json.dumps(snapshot, indent=2, ensure_ascii=False, default=str),
        )

        round_name = str(action.get("round_name", "")).strip()
        previous_round_summary = None
        if round_name in ordered_rounds:
            current_round_index = ordered_rounds.index(round_name)
            if current_round_index > 0:
                previous_round_summary = round_map.get(ordered_rounds[current_round_index - 1])

        override = override_lookup.get((round_name, str(action.get("role", "")).strip()), {})
        step_trace = _action_step_trace(graph_payload, action)

        write_rows.append(
            {
                "run_dir": str(run_path),
                "benchmark": str(metadata.get("benchmark", "")).strip() or str(native_evaluation.get("benchmark", "")).strip() or "unknown",
                "instance_name": str(summary_payload.get("instance_name", run_path.name)).strip() or run_path.name,
                "baseline_name": str(metadata.get("baseline_name", "")).strip() or "unknown",
                "topic": str(summary_payload.get("topic", graph_payload.get("topic", ""))).strip(),
                "step_index": action_index,
                "round_name": round_name,
                "role": str(action.get("role", "")).strip(),
                "selected_action_kind": str(action.get("kind", "")).strip(),
                "selected_action_targets": list(action.get("target_ids", []) or []),
                "selected_action_source": str(action.get("source", "")).strip() or "unknown",
                "before_state_snapshot": f"state_snapshots/{snapshot_name}",
                "before_state_node_count": snapshot["node_count"],
                "before_state_edge_count": snapshot["edge_count"],
                "before_state_contradiction_count": snapshot["contradiction_count"],
                "before_state_support_edge_count": snapshot["support_edge_count"],
                "previous_round_summary": previous_round_summary,
                "after_round_summary": round_map.get(round_name),
                "final_return_local": _safe_float(idea_evaluation.get("overall_score")),
                "final_return_native": _native_average(native_evaluation),
                "trace_prompt_tokens_step": step_trace["prompt_tokens"],
                "trace_completion_tokens_step": step_trace["completion_tokens"],
                "trace_total_tokens_step": step_trace["total_tokens"],
                "llm_override_applied": str(action.get("source", "")).strip() == "utility_controller_override",
                "llm_proposed_kind": override.get("llm_kind"),
                "llm_predicted_gain": _safe_float(override.get("llm_predicted_gain")),
                "override_predicted_gain": _safe_float(override.get("deterministic_predicted_gain")),
                "commit_target": None,
            }
        )

    return write_rows


def aggregate_dataset_profile(
    manifest_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    benchmark_counts: dict[str, int] = {}
    baseline_counts: dict[str, int] = {}
    eig_action_counts: list[int] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    known_costs: list[float] = []

    agent_trace_count = 0
    final_synthesis_count = 0
    override_trace_count = 0
    local_eval_count = 0
    native_eval_count = 0

    for row in manifest_rows:
        benchmark = str(row.get("benchmark", "unknown")).strip() or "unknown"
        baseline = str(row.get("baseline_name", "unknown")).strip() or "unknown"
        benchmark_counts[benchmark] = benchmark_counts.get(benchmark, 0) + 1
        baseline_counts[baseline] = baseline_counts.get(baseline, 0) + 1

        if bool(row.get("is_eig_run", False)):
            eig_action_counts.append(_safe_int(row.get("action_count")))

        total_prompt_tokens += _safe_int(row.get("trace_prompt_tokens"))
        total_completion_tokens += _safe_int(row.get("trace_completion_tokens"))
        total_tokens += _safe_int(row.get("trace_total_tokens"))

        estimated_cost = _safe_float(row.get("estimated_cost"))
        if estimated_cost is not None:
            known_costs.append(estimated_cost)

        agent_trace_count += 1 if bool(row.get("has_agent_traces", False)) else 0
        final_synthesis_count += 1 if bool(row.get("has_final_synthesis_trace", False)) else 0
        override_trace_count += 1 if bool(row.get("has_override_trace", False)) else 0
        local_eval_count += 1 if bool(row.get("has_local_eval", False)) else 0
        native_eval_count += 1 if bool(row.get("has_native_eval", False)) else 0

    run_count = len(manifest_rows)
    transition_count = len(transition_rows)
    usable_eig_run_count = sum(1 for row in manifest_rows if bool(row.get("is_eig_run", False)))

    def _fraction(count: int) -> float:
        if run_count == 0:
            return 0.0
        return count / run_count

    return {
        "run_count": run_count,
        "usable_run_count": run_count,
        "usable_eig_run_count": usable_eig_run_count,
        "transition_count": transition_count,
        "average_actions_per_usable_eig_run": (
            sum(eig_action_counts) / len(eig_action_counts) if eig_action_counts else 0.0
        ),
        "benchmark_counts": benchmark_counts,
        "baseline_counts": baseline_counts,
        "trace_coverage": {
            "agent_traces_fraction": _fraction(agent_trace_count),
            "final_synthesis_trace_fraction": _fraction(final_synthesis_count),
            "override_trace_fraction": _fraction(override_trace_count),
            "local_eval_fraction": _fraction(local_eval_count),
            "native_eval_fraction": _fraction(native_eval_count),
        },
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "mean_tokens_per_run": (total_tokens / run_count) if run_count else 0.0,
            "mean_tokens_per_transition": (total_tokens / transition_count) if transition_count else 0.0,
        },
        "cost": {
            "known_run_cost_count": len(known_costs),
            "estimated_total_cost": sum(known_costs) if known_costs else None,
            "mean_known_cost_per_run": (sum(known_costs) / len(known_costs)) if known_costs else None,
        },
    }


def _jsonl_lines(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n" for row in rows)


def _readme_text() -> str:
    return "\n".join(
        [
            "# Graph Critic Dataset Export",
            "",
            "This directory contains run-level manifests, transition examples,",
            "approximate pre-action graph snapshots, and dataset profiling statistics",
            "exported from saved pre-critic EIG runs.",
            "",
            "Files:",
            "- `run_manifest.jsonl`: one row per discovered run directory",
            "- `trajectory_examples.jsonl`: one row per exported graph action",
            "- `dataset_profile.json`: aggregate counts, coverage, token, and cost indicators",
            "- `state_snapshots/`: JSON snapshots of the reconstructed pre-action graph state",
            "",
            "Caveat: state reconstruction is timestamp-based and approximate. It is",
            "structurally faithful for offline critic training, but it is not a full",
            "replay engine.",
            "",
        ]
    )


def export_graph_critic_dataset(
    *,
    input_roots: Sequence[Path],
    output_dir: Path,
    dataset_name: str,
    baseline: str | None = None,
    benchmark: str | None = None,
    pricing: PricingConfig | None = None,
    limit_runs: int | None = None,
) -> ExportResult:
    run_dirs = discover_run_dirs(input_roots)
    if limit_runs is not None:
        run_dirs = run_dirs[: max(limit_runs, 0)]

    dataset_dir = Path(output_dir) / dataset_name
    snapshot_dir = dataset_dir / "state_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        summary_payload, graph_payload = load_run_artifacts(run_dir)
        manifest_row = build_run_manifest_row(run_dir, summary_payload, graph_payload, pricing=pricing)
        if baseline and manifest_row["baseline_name"] != baseline:
            continue
        if benchmark and manifest_row["benchmark"] != benchmark:
            continue
        manifest_rows.append(manifest_row)
        transition_rows.extend(
            build_transition_rows(run_dir, summary_payload, graph_payload, snapshot_dir=snapshot_dir)
        )

    profile = aggregate_dataset_profile(manifest_rows, transition_rows)

    write_text_file(dataset_dir / "run_manifest.jsonl", _jsonl_lines(manifest_rows))
    write_text_file(dataset_dir / "trajectory_examples.jsonl", _jsonl_lines(transition_rows))
    write_text_file(
        dataset_dir / "dataset_profile.json",
        json.dumps(profile, indent=2, ensure_ascii=False, default=str),
    )
    write_text_file(dataset_dir / "README.md", _readme_text())

    return ExportResult(
        dataset_dir=dataset_dir,
        run_count=len(manifest_rows),
        transition_count=len(transition_rows),
    )
