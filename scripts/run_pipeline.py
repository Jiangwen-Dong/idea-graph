from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.agent_backend import OpenAICompatibleCollaborationBackend
from idea_graph.benchmarks import (
    ai_idea_bench_2025_instance_from_record,
    download_ai_idea_bench_2025,
    download_liveideabench,
    get_ai_idea_bench_2025_record,
    get_liveideabench_record,
    liveideabench_instance_from_record,
)
from idea_graph.evaluation import evaluate_graph
from idea_graph.engine import run_experiment
from idea_graph.io import load_instance, write_run_artifacts
from idea_graph.instances import ExperimentInstance
from idea_graph.settings import OpenAICompatibleSettings


def print_progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic idea-graph pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "sample_instance.json",
        help="Path to a JSON instance file.",
    )
    parser.add_argument(
        "--benchmark",
        choices=["ai_idea_bench_2025", "liveideabench"],
        help="Use an official benchmark loader instead of a local instance JSON.",
    )
    parser.add_argument(
        "--benchmark-index",
        type=int,
        default=0,
        help="Benchmark selector when --benchmark is provided. Meaning depends on the benchmark.",
    )
    parser.add_argument(
        "--benchmark-keyword",
        help="Optional keyword filter for benchmarks that support keyword-based selection, such as liveideabench.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "data" / "benchmarks",
        help="Base cache directory for benchmark files.",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download official benchmark metadata automatically if it is not cached locally.",
    )
    parser.add_argument(
        "--include-paper-assets",
        action="store_true",
        help="Also download the official paper archive when downloading the benchmark. This is about 31 GB.",
    )
    parser.add_argument(
        "--allow-large-download",
        action="store_true",
        help="Required to confirm that you really want the large paper-asset download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="Directory where run artifacts will be written.",
    )
    parser.add_argument(
        "--agent-backend",
        choices=["deterministic", "openai-compatible"],
        default="deterministic",
        help="Multi-agent backend to use for seed generation, collaboration actions, and final synthesis.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum number of collaboration rounds to execute.",
    )
    parser.add_argument(
        "--disable-maturity-stop",
        action="store_true",
        help="Keep running until --max-rounds even if the maturity metric is reached earlier.",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        help="Path to a JSON config file for the OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--llm-base-url",
        help="Base URL for an OpenAI-compatible API endpoint.",
    )
    parser.add_argument(
        "--llm-model",
        help="Default model name for the OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--llm-provider",
        help="Provider family override such as openai or dashscope.",
    )
    parser.add_argument(
        "--llm-reasoning-mode",
        choices=["auto", "off", "on"],
        help="Reasoning mode hint for providers that expose a thinking toggle.",
    )
    parser.add_argument(
        "--llm-api-key",
        help="API key for the OpenAI-compatible backend. Prefer env vars when possible.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        default=None,
        help="Environment variable to read when --llm-api-key is not provided.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        help="Sampling temperature for the OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        help="Max completion tokens for the OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=int,
        help="HTTP timeout in seconds for the OpenAI-compatible backend.",
    )
    parser.add_argument(
        "--llm-json-mode",
        action="store_true",
        help="Request provider-level JSON mode if the endpoint supports it.",
    )
    return parser


def resolve_benchmark_root(base_root: Path, benchmark_name: str) -> Path:
    if base_root.name == benchmark_name:
        return base_root
    return base_root / benchmark_name


def build_collaboration_backend(args: argparse.Namespace) -> OpenAICompatibleCollaborationBackend | None:
    if args.agent_backend == "deterministic":
        return None

    payload: dict[str, object] = {}
    if args.llm_config:
        config_payload = json.loads(args.llm_config.read_text(encoding="utf-8"))
        if not isinstance(config_payload, dict):
            raise ValueError(f"LLM config file {args.llm_config} must contain a JSON object.")
        payload.update(config_payload)

    nested = payload.get("openai_compatible")
    if isinstance(nested, dict):
        merged = dict(nested)
    else:
        merged = dict(payload)

    if args.llm_base_url:
        merged["base_url"] = args.llm_base_url
    if args.llm_model:
        merged["model"] = args.llm_model
    if args.llm_provider:
        merged["provider"] = args.llm_provider
    if args.llm_reasoning_mode:
        merged["reasoning_mode"] = args.llm_reasoning_mode
    if args.llm_api_key:
        merged["api_key"] = args.llm_api_key
    elif args.llm_api_key_env and os.getenv(args.llm_api_key_env):
        merged["api_key"] = os.getenv(args.llm_api_key_env)
    if args.llm_api_key_env:
        merged["api_key_env"] = args.llm_api_key_env
    if args.llm_temperature is not None:
        merged["temperature"] = args.llm_temperature
    if args.llm_max_tokens is not None:
        merged["max_tokens"] = args.llm_max_tokens
    if args.llm_timeout_seconds is not None:
        merged["timeout_seconds"] = args.llm_timeout_seconds
    if args.llm_json_mode:
        merged["json_mode"] = True

    settings = OpenAICompatibleSettings.from_mapping(merged)
    return OpenAICompatibleCollaborationBackend(settings)


def main() -> None:
    args = build_parser().parse_args()
    if args.include_paper_assets and not args.allow_large_download:
        raise SystemExit(
            "--include-paper-assets is blocked by default because the archive is about 31 GB. "
            "Re-run with --allow-large-download if you really want it."
        )
    if args.benchmark == "ai_idea_bench_2025":
        benchmark_root = resolve_benchmark_root(args.benchmark_root, args.benchmark)
        if args.download_if_missing:
            download_ai_idea_bench_2025(
                benchmark_root,
                include_papers=args.include_paper_assets,
                extract_papers=False,
            )
        record = get_ai_idea_bench_2025_record(benchmark_root, args.benchmark_index)
        instance = ai_idea_bench_2025_instance_from_record(record, benchmark_root=benchmark_root)
    elif args.benchmark == "liveideabench":
        benchmark_root = resolve_benchmark_root(args.benchmark_root, args.benchmark)
        if args.include_paper_assets:
            raise SystemExit("liveideabench does not provide a separate paper-asset archive.")
        if args.download_if_missing:
            download_liveideabench(benchmark_root)
        record = get_liveideabench_record(
            benchmark_root,
            args.benchmark_index,
            keyword=args.benchmark_keyword,
        )
        instance = liveideabench_instance_from_record(
            record,
            benchmark_root=benchmark_root,
        )
    else:
        benchmark_root = None
        instance = load_instance(args.input)

    collaboration_backend = build_collaboration_backend(args)
    experiment_metadata = dict(instance.metadata)
    experiment_metadata["agent_backend"] = args.agent_backend
    experiment_metadata["max_rounds_requested"] = max(1, args.max_rounds)
    experiment_metadata["stop_when_mature"] = not args.disable_maturity_stop
    if collaboration_backend is not None:
        experiment_metadata["openai_compatible"] = collaboration_backend.settings.sanitized_dict()
    instance = ExperimentInstance(
        name=instance.name,
        topic=instance.topic,
        literature=list(instance.literature),
        source_path=instance.source_path,
        metadata=experiment_metadata,
    )
    graph = run_experiment(
        topic=instance.topic,
        literature=list(instance.literature),
        metadata=experiment_metadata,
        collaboration_backend=collaboration_backend,
        progress_callback=print_progress,
        max_rounds=max(1, args.max_rounds),
        stop_when_mature=not args.disable_maturity_stop,
    )
    run_dir = write_run_artifacts(
        graph,
        output_root=args.output_dir,
        instance=instance,
    )

    print("== Instance ==")
    print(instance.name)
    print()

    if args.benchmark:
        print("== Benchmark ==")
        print(f"Name: {args.benchmark}")
        metadata = instance.metadata
        actual_index = metadata.get("benchmark_index", metadata.get("row_index", args.benchmark_index))
        print(f"Requested selector: {args.benchmark_index}")
        print(f"Resolved benchmark index: {actual_index}")
        if args.benchmark_keyword:
            print(f"Keyword filter: {args.benchmark_keyword}")
        print(f"Root: {benchmark_root}")
        print()

    print("== Topic ==")
    print(graph.topic)
    print()

    print("== Backend ==")
    print(args.agent_backend)
    if collaboration_backend is not None:
        print(f"Base URL: {collaboration_backend.settings.base_url}")
        print(f"Model: {collaboration_backend.settings.model}")
    print()

    print("== Graph Summary ==")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print(f"Branches: {len(graph.branches)}")
    print(f"Actions: {len(graph.actions)}")
    print(f"Executed rounds: {graph.metadata.get('executed_round_count', len(graph.round_summaries))}")
    print(f"Matured at: {graph.matured_at_round or 'not reached'}")
    print(f"Stop reason: {graph.metadata.get('stop_reason', 'unknown')}")
    print()

    print("== Round Maturity ==")
    for round_name, snapshot in graph.round_summaries:
        print(
            f"{round_name}: support={snapshot.support_coverage}, "
            f"contradictions={snapshot.unresolved_contradiction_ratio}, "
            f"utility={snapshot.utility}, stable={snapshot.utility_stable}, "
            f"complete={snapshot.completeness}, mature={snapshot.is_mature}"
        )
    print()

    print("== Final Proposal ==")
    assert graph.final_proposal is not None
    print(f"Title: {graph.final_proposal.title or instance.name}")
    print(f"Problem: {graph.final_proposal.problem}")
    print(f"Motivation: {graph.final_proposal.motivation}")
    print(f"Proposed Method: {graph.final_proposal.method}")
    print(f"Experiment Plan: {graph.final_proposal.evaluation}")
    print()

    evaluation = evaluate_graph(graph)
    print("== Idea Evaluation ==")
    print(f"Overall: {evaluation.overall_score}/10")
    for category, score in evaluation.category_scores.items():
        print(f"{category}: {score}/10")
    print()

    backend_diagnostics = []
    if "seed_generation_error" in graph.metadata:
        backend_diagnostics.append(f"Seed generation fallback: {graph.metadata['seed_generation_error']}")
    for item in graph.metadata.get("action_errors", []):
        if isinstance(item, dict):
            backend_diagnostics.append(
                f"Action fallback [{item.get('round')}/{item.get('role')}]: {item.get('error')}"
            )
    if "final_synthesis_error" in graph.metadata:
        backend_diagnostics.append(f"Final synthesis fallback: {graph.metadata['final_synthesis_error']}")
    if backend_diagnostics:
        print("== Backend Diagnostics ==")
        for line in backend_diagnostics:
            print(line)
        print()

    print("== Artifacts ==")
    print(run_dir)


if __name__ == "__main__":
    main()
