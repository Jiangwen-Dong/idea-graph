from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fs_utils import read_text_file, write_text_file
from .trajectory_dataset import build_run_manifest_row, load_run_artifacts

ROOT = Path(__file__).resolve().parents[2]
RUN_PIPELINE_SCRIPT = ROOT / "scripts" / "run_pipeline.py"

_INDEX_PATTERN = re.compile(r"-(\d+)$")
_AIIB_BENCHMARKS = {"AI_Idea_Bench_2025", "ai_idea_bench_2025"}
_LIVE_BENCHMARKS = {"liveideabench", "LiveIdeaBench"}


def load_split_registry_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_index, raw_line in enumerate(read_text_file(path).splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} line {line_index} must contain a JSON object.")
        rows.append(dict(payload))
    return rows


def select_critic_train_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    pool_name: str,
    group_ids: Sequence[str] | None = None,
    limit: int | None = None,
    required_usage: str = "train_online_critic",
) -> list[dict[str, Any]]:
    normalized_pool = str(pool_name).strip()
    if not normalized_pool:
        raise ValueError("pool_name must not be empty.")

    allowed_group_ids = {str(group_id).strip() for group_id in group_ids or [] if str(group_id).strip()}
    matched_group_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    for row in rows:
        row_pool = str(row.get("pool_name", "")).strip()
        row_role = str(row.get("partition_role", "")).strip()
        row_group_id = str(row.get("group_id", "")).strip()
        usages = row.get("allowed_usages", [])
        if not isinstance(usages, list):
            usages = []
        if row_pool != normalized_pool:
            continue
        if row_role != "critic_train":
            continue
        if required_usage and required_usage not in {str(item).strip() for item in usages}:
            continue
        if allowed_group_ids and row_group_id not in allowed_group_ids:
            continue
        if row_group_id:
            matched_group_ids.add(row_group_id)
        selected.append(dict(row))

    if allowed_group_ids:
        missing = sorted(allowed_group_ids - matched_group_ids)
        if missing:
            raise ValueError("Requested group_ids were not found in the selected critic_train pool: " + ", ".join(missing))

    ordered = sorted(
        selected,
        key=lambda row: (
            str(row.get("benchmark", "")).strip(),
            str(row.get("instance_name", "")).strip(),
            str(row.get("group_id", "")).strip(),
        ),
    )
    if limit is not None:
        return ordered[: max(int(limit), 0)]
    return ordered


def _parse_instance_index(instance_name: str) -> int:
    match = _INDEX_PATTERN.search(instance_name.strip())
    if match is None:
        raise ValueError(f"Could not parse benchmark index from instance_name '{instance_name}'.")
    return int(match.group(1))


def _normalize_benchmark_cli_name(benchmark: str) -> str:
    normalized = benchmark.strip()
    if normalized in _AIIB_BENCHMARKS:
        return "ai_idea_bench_2025"
    if normalized in _LIVE_BENCHMARKS:
        return "liveideabench"
    raise ValueError(f"Unsupported benchmark in split registry: {benchmark!r}")


def _infer_live_keyword(instance_name: str) -> str | None:
    text = instance_name.strip()
    if not text.startswith("liveideabench-"):
        return None
    prefix_stripped = text[len("liveideabench-") :]
    if "-" not in prefix_stripped:
        return None
    keyword, _ = prefix_stripped.rsplit("-", 1)
    return keyword or None


def build_run_pipeline_command(
    manifest_row: Mapping[str, Any],
    *,
    runs_dir: Path,
    python_executable: str | None = None,
    llm_config_path: Path | None = None,
    benchmark_root: Path | None = None,
    agent_backend: str = "openai-compatible",
) -> list[str]:
    command = [
        python_executable or sys.executable,
        str(RUN_PIPELINE_SCRIPT),
        "--benchmark",
        str(manifest_row["benchmark_cli_name"]),
        "--benchmark-index",
        str(manifest_row["benchmark_index"]),
        "--baseline",
        str(manifest_row["baseline_name"]),
        "--agent-backend",
        str(agent_backend),
        "--max-rounds",
        str(manifest_row["max_rounds"]),
        "--output-dir",
        str(Path(runs_dir).resolve()),
    ]
    if benchmark_root is not None:
        command.extend(["--benchmark-root", str(Path(benchmark_root).resolve())])
    if llm_config_path is not None:
        command.extend(["--llm-config", str(Path(llm_config_path).resolve())])
    if bool(manifest_row.get("native_eval", False)):
        command.append("--native-eval")
    return command


def build_episode_launch_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_name: str,
    max_rounds: int,
    native_eval: bool,
    runs_dir: Path | None = None,
    llm_config_path: Path | None = None,
    benchmark_root: Path | None = None,
    python_executable: str | None = None,
    agent_backend: str = "openai-compatible",
) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        benchmark = str(row.get("benchmark", "")).strip()
        instance_name = str(row.get("instance_name", "")).strip()
        manifest_row = dict(row)
        manifest_row["benchmark_cli_name"] = _normalize_benchmark_cli_name(benchmark)
        manifest_row["benchmark_index"] = _parse_instance_index(instance_name)
        manifest_row["benchmark_keyword"] = _infer_live_keyword(instance_name)
        manifest_row["baseline_name"] = str(baseline_name).strip()
        manifest_row["max_rounds"] = int(max_rounds)
        manifest_row["native_eval"] = bool(native_eval)
        if runs_dir is not None:
            command = build_run_pipeline_command(
                manifest_row,
                runs_dir=runs_dir,
                python_executable=python_executable,
                llm_config_path=llm_config_path,
                benchmark_root=benchmark_root,
                agent_backend=agent_backend,
            )
            manifest_row["command"] = command
            manifest_row["command_text"] = subprocess.list2cmdline(command)
        manifest_rows.append(manifest_row)
    return manifest_rows


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n" for row in rows)


def write_collection_artifacts(
    *,
    collection_dir: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    collection_config: Mapping[str, Any],
    collection_summary: Mapping[str, Any],
    execution_results: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    write_text_file(collection_dir / "launch_manifest.jsonl", _jsonl_text(manifest_rows))
    write_text_file(
        collection_dir / "collection_config.json",
        json.dumps(dict(collection_config), indent=2, ensure_ascii=False, default=str),
    )
    write_text_file(
        collection_dir / "collection_summary.json",
        json.dumps(dict(collection_summary), indent=2, ensure_ascii=False, default=str),
    )
    if execution_results is not None:
        write_text_file(collection_dir / "execution_results.jsonl", _jsonl_text(execution_results))


def _parse_run_dir_from_stdout(stdout_text: str) -> Path | None:
    lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line == "== Artifacts ==" and index + 1 < len(lines):
            return Path(lines[index + 1])
    if lines:
        candidate = Path(lines[-1])
        if candidate.exists():
            return candidate
    return None


def _find_existing_run_dir(runs_dir: Path, instance_name: str) -> Path | None:
    if not runs_dir.exists():
        return None
    suffix = f"-{instance_name}"
    candidates = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and path.name.endswith(suffix) and (path / "summary.json").exists() and (path / "graph.json").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _profile_run_dir(run_dir: Path) -> dict[str, Any]:
    summary_payload, graph_payload = load_run_artifacts(run_dir)
    manifest_row = build_run_manifest_row(run_dir, summary_payload, graph_payload)
    return {
        "trace_prompt_tokens": manifest_row["trace_prompt_tokens"],
        "trace_completion_tokens": manifest_row["trace_completion_tokens"],
        "trace_total_tokens": manifest_row["trace_total_tokens"],
        "estimated_cost": manifest_row["estimated_cost"],
        "final_local_overall": manifest_row["final_local_overall"],
        "final_local_alignment": manifest_row["final_local_alignment"],
        "final_native_average": manifest_row["final_native_average"],
        "executed_round_count": manifest_row["executed_round_count"],
        "action_count": manifest_row["action_count"],
    }


def execute_episode_collection(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    collection_dir: Path,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    runs_dir = collection_dir / "runs"
    logs_dir = collection_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for index, row in enumerate(manifest_rows, start=1):
        instance_name = str(row.get("instance_name", "")).strip()
        command = row.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError(f"Manifest row for '{instance_name}' is missing a runnable command.")

        existing_run_dir = _find_existing_run_dir(runs_dir, instance_name)
        stdout_log = logs_dir / f"{index:02d}-{instance_name}.stdout.txt"
        stderr_log = logs_dir / f"{index:02d}-{instance_name}.stderr.txt"
        result: dict[str, Any] = {
            "group_id": row.get("group_id"),
            "benchmark": row.get("benchmark"),
            "instance_name": instance_name,
            "status": "pending",
            "return_code": None,
            "stdout_log": str(stdout_log.resolve()),
            "stderr_log": str(stderr_log.resolve()),
            "run_dir": None,
        }

        if skip_existing and existing_run_dir is not None:
            result["status"] = "skipped_existing"
            result["run_dir"] = str(existing_run_dir.resolve())
            result.update(_profile_run_dir(existing_run_dir))
            write_text_file(stdout_log, "Skipped: existing run directory found.\n")
            write_text_file(stderr_log, "")
            results.append(result)
            continue

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        write_text_file(stdout_log, completed.stdout)
        write_text_file(stderr_log, completed.stderr)

        result["return_code"] = int(completed.returncode)
        resolved_run_dir = _parse_run_dir_from_stdout(completed.stdout)
        if resolved_run_dir is None or not resolved_run_dir.exists():
            resolved_run_dir = _find_existing_run_dir(runs_dir, instance_name)
        if resolved_run_dir is not None and resolved_run_dir.exists():
            result["run_dir"] = str(resolved_run_dir.resolve())
            result.update(_profile_run_dir(resolved_run_dir))

        result["status"] = "completed" if completed.returncode == 0 else "failed"
        results.append(result)
    return results


def build_collection_summary(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    execution_results: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "selected_group_count": len(manifest_rows),
        "mode": mode,
        "benchmark_counts": {},
    }
    benchmark_counts: dict[str, int] = {}
    for row in manifest_rows:
        benchmark = str(row.get("benchmark", "unknown")).strip() or "unknown"
        benchmark_counts[benchmark] = benchmark_counts.get(benchmark, 0) + 1
    summary["benchmark_counts"] = dict(sorted(benchmark_counts.items()))

    if execution_results is not None:
        status_counts: dict[str, int] = {}
        total_tokens = 0
        known_costs: list[float] = []
        for row in execution_results:
            status = str(row.get("status", "unknown")).strip() or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1
            total_tokens += int(row.get("trace_total_tokens", 0) or 0)
            estimated_cost = row.get("estimated_cost")
            if estimated_cost is not None:
                known_costs.append(float(estimated_cost))
        summary["status_counts"] = dict(sorted(status_counts.items()))
        summary["total_trace_tokens"] = total_tokens
        summary["estimated_total_cost"] = sum(known_costs) if known_costs else None
    return summary
