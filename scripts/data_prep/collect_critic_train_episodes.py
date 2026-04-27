from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.critic_episode_collection import (
    build_collection_summary,
    build_episode_launch_manifest,
    execute_episode_collection,
    load_split_registry_rows,
    select_pool_rows,
    write_collection_artifacts,
)
from idea_graph.repo_paths import default_benchmark_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect real critic_train episodes from the frozen split registry."
    )
    parser.add_argument(
        "--split-registry",
        type=Path,
        required=True,
        help="Path to split_registry.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Collection output root or a direct collection directory.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Optional child directory name under --output-dir.",
    )
    parser.add_argument(
        "--pool-name",
        type=str,
        default="development_pool_v1",
        help="Pool name to select from the registry.",
    )
    parser.add_argument(
        "--partition-role",
        type=str,
        default="critic_train",
        help="Partition role to select from the registry.",
    )
    parser.add_argument(
        "--required-usage",
        type=str,
        default="train_online_critic",
        help="Required usage tag that must be present in allowed_usages.",
    )
    parser.add_argument(
        "--group-id",
        action="append",
        default=[],
        help="Optional group_id filter. Repeatable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on selected groups after filtering.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="ours-eig",
        help="Baseline name passed through to run_pipeline.py.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum rounds for each collected episode.",
    )
    parser.add_argument(
        "--native-eval",
        action="store_true",
        help="Enable benchmark-native scoring for collected episodes.",
    )
    parser.add_argument(
        "--agent-backend",
        choices=["deterministic", "openai-compatible"],
        default="openai-compatible",
        help="Generation backend passed through to run_pipeline.py.",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=None,
        help="LLM config passed through when using the openai-compatible backend.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=default_benchmark_root(ROOT),
        help="Benchmark cache root passed through to run_pipeline.py.",
    )
    parser.add_argument(
        "--runtime-protocol",
        choices=["sequential_v1", "parallel_graph_v2"],
        default="parallel_graph_v2",
        help="Runtime protocol passed through to run_pipeline.py.",
    )
    parser.add_argument(
        "--disable-maturity-stop",
        action="store_true",
        help="Disable maturity stop behavior in run_pipeline.py.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch the runs. Without this flag the script performs a dry run only.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip instances that already have a collected run under this collection root.",
    )
    return parser


def _resolve_collection_dir(output_dir: Path, collection_name: str | None) -> Path:
    if collection_name:
        return Path(output_dir) / collection_name
    return Path(output_dir)


def main() -> None:
    args = build_parser().parse_args()
    if args.execute and args.agent_backend == "openai-compatible" and args.llm_config is None:
        raise SystemExit("--execute with --agent-backend openai-compatible requires --llm-config.")

    collection_dir = _resolve_collection_dir(args.output_dir, args.collection_name)
    runs_dir = collection_dir / "runs"
    registry_rows = load_split_registry_rows(args.split_registry)
    selected_rows = select_pool_rows(
        registry_rows,
        pool_name=args.pool_name,
        partition_role=args.partition_role,
        group_ids=args.group_id,
        limit=args.limit,
        required_usage=args.required_usage,
    )
    manifest_rows = build_episode_launch_manifest(
        selected_rows,
        baseline_name=args.baseline,
        max_rounds=args.max_rounds,
        native_eval=args.native_eval,
        runs_dir=runs_dir,
        llm_config_path=args.llm_config,
        benchmark_root=args.benchmark_root,
        agent_backend=args.agent_backend,
        runtime_protocol=args.runtime_protocol,
        disable_maturity_stop=bool(args.disable_maturity_stop),
    )

    collection_config = {
        "split_registry": str(args.split_registry.resolve()),
        "collection_dir": str(collection_dir.resolve()),
        "pool_name": args.pool_name,
        "partition_role": args.partition_role,
        "required_usage": args.required_usage,
        "group_ids": list(args.group_id),
        "limit": args.limit,
        "baseline": args.baseline,
        "max_rounds": int(args.max_rounds),
        "native_eval": bool(args.native_eval),
        "agent_backend": args.agent_backend,
        "llm_config": str(args.llm_config.resolve()) if args.llm_config is not None else None,
        "benchmark_root": str(args.benchmark_root.resolve()),
        "runtime_protocol": args.runtime_protocol,
        "disable_maturity_stop": bool(args.disable_maturity_stop),
        "execute": bool(args.execute),
        "skip_existing": bool(args.skip_existing),
    }

    execution_results = None
    if args.execute:
        execution_results = execute_episode_collection(
            manifest_rows,
            collection_dir=collection_dir,
            skip_existing=bool(args.skip_existing),
        )
        summary = build_collection_summary(
            manifest_rows,
            mode="execute",
            execution_results=execution_results,
        )
    else:
        summary = build_collection_summary(manifest_rows, mode="dry_run")

    write_collection_artifacts(
        collection_dir=collection_dir,
        manifest_rows=manifest_rows,
        collection_config=collection_config,
        collection_summary=summary,
        execution_results=execution_results,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
