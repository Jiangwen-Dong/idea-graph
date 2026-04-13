from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import mkdtemp
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.critic_episode_collection import (
    build_episode_launch_manifest,
    load_split_registry_rows,
    select_pool_rows,
    select_critic_train_rows,
)
from idea_graph.fs_utils import write_text_file


class CriticEpisodeCollectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())
        self.registry_path = self.tmp_dir / "split_registry.jsonl"
        self.registry_rows = [
            {
                "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-13",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "ai-idea-bench-2025-13",
                "pool_name": "development_pool_v1",
                "partition_role": "critic_train",
                "allowed_usages": [
                    "train_offline_critic",
                    "train_online_critic",
                    "development_analysis",
                ],
            },
            {
                "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-9849",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "ai-idea-bench-2025-9849",
                "pool_name": "development_pool_v1",
                "partition_role": "critic_dev",
                "allowed_usages": [
                    "select_checkpoint",
                    "development_analysis",
                ],
            },
            {
                "group_id": "liveideabench::liveideabench-meteorology-0",
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-meteorology-0",
                "pool_name": "development_pool_v1",
                "partition_role": "critic_train",
                "allowed_usages": [
                    "train_offline_critic",
                    "train_online_critic",
                    "development_analysis",
                ],
            },
            {
                "group_id": "liveideabench::liveideabench-weather forecasting-47",
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-weather forecasting-47",
                "pool_name": "paper_eval_v1",
                "partition_role": "paper_eval",
                "allowed_usages": ["paper_final_eval"],
            },
            {
                "group_id": "liveideabench::liveideabench-periodic table-23",
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-periodic table-23",
                "pool_name": "development_pool_v1",
                "partition_role": "critic_train",
                "allowed_usages": ["development_analysis"],
            },
            {
                "group_id": "liveideabench::liveideabench-climate-5",
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-climate-5",
                "pool_name": "development_pool_v2_candidate_pool_v1",
                "partition_role": "critic_dev",
                "allowed_usages": ["development_analysis"],
            },
        ]
        registry_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.registry_rows)
        write_text_file(self.registry_path, registry_text)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_select_critic_train_rows_filters_pool_role_and_allowed_usage(self) -> None:
        rows = load_split_registry_rows(self.registry_path)
        selected = select_critic_train_rows(rows, pool_name="development_pool_v1")
        self.assertEqual(
            [str(row["group_id"]) for row in selected],
            [
                "AI_Idea_Bench_2025::ai-idea-bench-2025-13",
                "liveideabench::liveideabench-meteorology-0",
            ],
        )

    def test_select_critic_train_rows_respects_group_filter_and_limit(self) -> None:
        rows = load_split_registry_rows(self.registry_path)
        selected = select_critic_train_rows(
            rows,
            pool_name="development_pool_v1",
            group_ids=["liveideabench::liveideabench-meteorology-0"],
            limit=1,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected[0]["group_id"],
            "liveideabench::liveideabench-meteorology-0",
        )

    def test_select_pool_rows_supports_critic_dev_with_required_usage(self) -> None:
        rows = load_split_registry_rows(self.registry_path)
        selected = select_pool_rows(
            rows,
            pool_name="development_pool_v2_candidate_pool_v1",
            partition_role="critic_dev",
            required_usage="development_analysis",
        )
        self.assertEqual(
            [str(row["group_id"]) for row in selected],
            ["liveideabench::liveideabench-climate-5"],
        )

    def test_select_pool_rows_raises_for_missing_requested_group(self) -> None:
        rows = load_split_registry_rows(self.registry_path)
        with self.assertRaises(ValueError):
            select_pool_rows(
                rows,
                pool_name="development_pool_v1",
                partition_role="critic_train",
                group_ids=["missing::group-1"],
            )

    def test_build_episode_launch_manifest_parses_benchmark_selectors(self) -> None:
        manifest = build_episode_launch_manifest(
            [
                self.registry_rows[0],
                self.registry_rows[2],
            ],
            baseline_name="ours-eig",
            max_rounds=5,
            native_eval=False,
        )
        by_group = {str(row["group_id"]): row for row in manifest}
        self.assertEqual(
            by_group["AI_Idea_Bench_2025::ai-idea-bench-2025-13"]["benchmark_cli_name"],
            "ai_idea_bench_2025",
        )
        self.assertEqual(
            by_group["AI_Idea_Bench_2025::ai-idea-bench-2025-13"]["benchmark_index"],
            13,
        )
        self.assertEqual(
            by_group["liveideabench::liveideabench-meteorology-0"]["benchmark_cli_name"],
            "liveideabench",
        )
        self.assertEqual(
            by_group["liveideabench::liveideabench-meteorology-0"]["benchmark_index"],
            0,
        )
        self.assertEqual(
            by_group["liveideabench::liveideabench-meteorology-0"]["baseline_name"],
            "ours-eig",
        )

    def test_cli_dry_run_writes_collection_artifacts(self) -> None:
        output_root = self.tmp_dir / "collections"
        script_path = ROOT / "scripts" / "collect_critic_train_episodes.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--split-registry",
                str(self.registry_path),
                "--output-dir",
                str(output_root),
                "--collection-name",
                "smoke_collection",
                "--limit",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        collection_dir = output_root / "smoke_collection"
        self.assertTrue((collection_dir / "launch_manifest.jsonl").exists())
        self.assertTrue((collection_dir / "collection_config.json").exists())
        self.assertTrue((collection_dir / "collection_summary.json").exists())

        summary = json.loads((collection_dir / "collection_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["selected_group_count"], 1)
        self.assertEqual(summary["mode"], "dry_run")


if __name__ == "__main__":
    unittest.main()
