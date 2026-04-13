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

from idea_graph.critic_split_registry import build_split_registry, build_split_registry_stats


class CriticSplitRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())
        self.partition_rows = [
            {
                "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-13",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "ai-idea-bench-2025-13",
                "source_split": "train",
                "partition_role": "critic_train",
            },
            {
                "group_id": "liveideabench::liveideabench-weather forecasting-47",
                "benchmark": "liveideabench",
                "instance_name": "liveideabench-weather forecasting-47",
                "source_split": "validation",
                "partition_role": "critic_dev",
            },
        ]
        self.paper_eval_rows = [
            {
                "group_id": "AI_Idea_Bench_2025::heldout",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "heldout",
                "source_split": "validation",
                "partition_role": "paper_eval",
            }
        ]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_build_split_registry_marks_development_pool_roles(self) -> None:
        rows = build_split_registry(self.partition_rows, pool_name="development_pool_v1")
        by_group = {str(row["group_id"]): row for row in rows}
        self.assertEqual(
            by_group["AI_Idea_Bench_2025::ai-idea-bench-2025-13"]["pool_name"],
            "development_pool_v1",
        )
        self.assertIn(
            "train_offline_critic",
            by_group["AI_Idea_Bench_2025::ai-idea-bench-2025-13"]["allowed_usages"],
        )
        self.assertNotIn(
            "paper_final_eval",
            by_group["AI_Idea_Bench_2025::ai-idea-bench-2025-13"]["allowed_usages"],
        )
        self.assertIn(
            "select_checkpoint",
            by_group["liveideabench::liveideabench-weather forecasting-47"]["allowed_usages"],
        )

    def test_build_split_registry_marks_paper_eval_as_final_eval_only(self) -> None:
        rows = build_split_registry(self.paper_eval_rows, pool_name="paper_eval_v1")
        by_group = {str(row["group_id"]): row for row in rows}
        self.assertEqual(by_group["AI_Idea_Bench_2025::heldout"]["partition_role"], "paper_eval")
        self.assertEqual(
            by_group["AI_Idea_Bench_2025::heldout"]["allowed_usages"],
            ["paper_final_eval"],
        )

    def test_build_split_registry_stats_reports_pool_and_role_counts(self) -> None:
        rows = build_split_registry(self.partition_rows, pool_name="development_pool_v1")
        stats = build_split_registry_stats(rows)
        self.assertEqual(stats["row_count"], 2)
        self.assertEqual(stats["pool_counts"]["development_pool_v1"], 2)
        self.assertEqual(stats["role_counts"]["critic_train"], 1)
        self.assertEqual(stats["role_counts"]["critic_dev"], 1)

    def test_cli_builder_writes_registry_outputs(self) -> None:
        partition_path = self.tmp_dir / "partition_manifest.jsonl"
        partition_text = "".join(json.dumps(row) + "\n" for row in self.partition_rows)
        partition_path.write_text(partition_text, encoding="utf-8")

        output_dir = self.tmp_dir / "registry"
        script_path = ROOT / "scripts" / "build_critic_split_registry.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--partition-manifest",
                str(partition_path),
                "--pool-name",
                "development_pool_v1",
                "--output-dir",
                str(output_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertTrue((output_dir / "split_registry.jsonl").exists())
        self.assertTrue((output_dir / "split_registry_stats.json").exists())


if __name__ == "__main__":
    unittest.main()
