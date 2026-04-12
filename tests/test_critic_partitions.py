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

from idea_graph.critic_partitions import build_partition_manifest, build_partition_stats
from idea_graph.fs_utils import _windows_safe_path, write_text_file


class CriticPartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())
        self.g2_dir = self.tmp_dir / "g2_tiny"
        self.split_rows = [
            {
                "group_id": "AI_Idea_Bench_2025::a",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "a",
                "split": "train",
            },
            {
                "group_id": "AI_Idea_Bench_2025::b",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "b",
                "split": "validation",
            },
            {
                "group_id": "liveideabench::x",
                "benchmark": "liveideabench",
                "instance_name": "x",
                "split": "train",
            },
        ]

    def tearDown(self) -> None:
        shutil.rmtree(_windows_safe_path(self.tmp_dir), ignore_errors=True)

    def _write_split_manifest(self, rows: list[dict[str, object]]) -> None:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        write_text_file(self.g2_dir / "split_manifest.jsonl", text)

    def _read_jsonl(self, path: Path) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
        return rows

    def test_build_partition_manifest_maps_train_and_validation_deterministically(self) -> None:
        first = build_partition_manifest(self.split_rows)
        second = build_partition_manifest(list(reversed(self.split_rows)))
        self.assertEqual(first, second)

        by_group = {str(row["group_id"]): row for row in first}
        self.assertEqual(by_group["AI_Idea_Bench_2025::a"]["partition_role"], "critic_train")
        self.assertEqual(by_group["AI_Idea_Bench_2025::b"]["partition_role"], "critic_dev")
        self.assertEqual(by_group["liveideabench::x"]["partition_role"], "critic_train")
        self.assertEqual(by_group["AI_Idea_Bench_2025::a"]["source_split"], "train")
        self.assertEqual(by_group["AI_Idea_Bench_2025::b"]["source_split"], "validation")

    def test_build_partition_manifest_assigns_explicit_holdout_groups_to_paper_eval(self) -> None:
        manifest = build_partition_manifest(
            self.split_rows,
            holdout_groups={"AI_Idea_Bench_2025::a", "liveideabench::x"},
        )
        by_group = {str(row["group_id"]): row for row in manifest}
        self.assertEqual(by_group["AI_Idea_Bench_2025::a"]["partition_role"], "paper_eval")
        self.assertEqual(by_group["liveideabench::x"]["partition_role"], "paper_eval")
        self.assertEqual(by_group["AI_Idea_Bench_2025::b"]["partition_role"], "critic_dev")

    def test_build_partition_manifest_rejects_duplicate_group_assignment(self) -> None:
        duplicate_rows = [
            self.split_rows[0],
            dict(self.split_rows[0]),
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate group_id"):
            build_partition_manifest(duplicate_rows)

    def test_build_partition_stats_reports_role_and_benchmark_counts(self) -> None:
        manifest = build_partition_manifest(
            self.split_rows,
            holdout_groups={"AI_Idea_Bench_2025::a"},
        )
        stats = build_partition_stats(manifest)
        self.assertEqual(stats["group_count"], 3)
        self.assertTrue(stats["has_paper_eval"])
        self.assertEqual(stats["role_counts"]["paper_eval"], 1)
        self.assertEqual(stats["role_counts"]["critic_train"], 1)
        self.assertEqual(stats["role_counts"]["critic_dev"], 1)
        self.assertEqual(stats["benchmark_role_counts"]["AI_Idea_Bench_2025"]["paper_eval"], 1)
        self.assertEqual(stats["benchmark_role_counts"]["AI_Idea_Bench_2025"]["critic_dev"], 1)
        self.assertEqual(stats["benchmark_role_counts"]["liveideabench"]["critic_train"], 1)

    def test_cli_builder_writes_partition_manifest_stats_and_readme(self) -> None:
        self._write_split_manifest(self.split_rows)
        script_path = ROOT / "scripts" / "build_critic_partition_manifest.py"
        command = [
            sys.executable,
            str(script_path),
            "--g2-dataset-dir",
            str(self.g2_dir),
            "--holdout-group",
            "AI_Idea_Bench_2025::a",
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

        partition_rows = self._read_jsonl(self.g2_dir / "partition_manifest.jsonl")
        by_group = {str(row["group_id"]): row for row in partition_rows}
        self.assertEqual(by_group["AI_Idea_Bench_2025::a"]["partition_role"], "paper_eval")
        self.assertEqual(by_group["AI_Idea_Bench_2025::b"]["partition_role"], "critic_dev")
        self.assertEqual(by_group["liveideabench::x"]["partition_role"], "critic_train")

        stats = json.loads((self.g2_dir / "partition_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["group_count"], 3)
        self.assertTrue(stats["has_paper_eval"])

        readme = (self.g2_dir / "README.md").read_text(encoding="utf-8")
        self.assertIn("partition_manifest.jsonl", readme)
        self.assertIn("partition_stats.json", readme)

    def test_cli_builder_can_write_to_dedicated_output_dir(self) -> None:
        self._write_split_manifest(self.split_rows)
        script_path = ROOT / "scripts" / "build_critic_partition_manifest.py"
        output_root = self.tmp_dir / "partition_outputs"
        command = [
            sys.executable,
            str(script_path),
            "--g2-dataset-dir",
            str(self.g2_dir),
            "--output-dir",
            str(output_root),
            "--dataset-name",
            "tiny_partitions",
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

        dataset_dir = output_root / "tiny_partitions"
        self.assertTrue((dataset_dir / "partition_manifest.jsonl").exists())
        self.assertTrue((dataset_dir / "partition_stats.json").exists())
        self.assertTrue((dataset_dir / "README.md").exists())

    def test_cli_builder_uses_output_dir_even_without_dataset_name(self) -> None:
        self._write_split_manifest(self.split_rows)
        script_path = ROOT / "scripts" / "build_critic_partition_manifest.py"
        output_root = self.tmp_dir / "partition_outputs_direct"
        command = [
            sys.executable,
            str(script_path),
            "--g2-dataset-dir",
            str(self.g2_dir),
            "--output-dir",
            str(output_root),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

        self.assertTrue((output_root / "partition_manifest.jsonl").exists())
        self.assertTrue((output_root / "partition_stats.json").exists())
        self.assertTrue((output_root / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
