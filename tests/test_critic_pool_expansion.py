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

from idea_graph.critic_pool_expansion import build_expansion_partition_rows


class CriticPoolExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())
        self.candidates = [
            {
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-01",
                "partition_role": "critic_train",
            },
            {
                "benchmark": "liveideabench",
                "instance_name": "lib-01",
                "partition_role": "critic_dev",
            },
        ]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_build_expansion_partition_rows_rejects_overlap_with_blocked_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlaps blocked group"):
            build_expansion_partition_rows(
                self.candidates,
                blocked_group_ids={"AI_Idea_Bench_2025::aiib-01"},
            )

    def test_build_expansion_partition_rows_preserves_requested_partition_roles(self) -> None:
        rows = build_expansion_partition_rows(
            self.candidates,
            blocked_group_ids=set(),
        )
        by_group = {str(row["group_id"]): row for row in rows}
        self.assertEqual(by_group["AI_Idea_Bench_2025::aiib-01"]["partition_role"], "critic_train")
        self.assertEqual(by_group["AI_Idea_Bench_2025::aiib-01"]["source_split"], "train")
        self.assertEqual(by_group["liveideabench::lib-01"]["partition_role"], "critic_dev")
        self.assertEqual(by_group["liveideabench::lib-01"]["source_split"], "validation")

    def test_build_expansion_partition_rows_rejects_duplicate_candidates(self) -> None:
        duplicate_candidates = [
            self.candidates[0],
            dict(self.candidates[0]),
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate candidate group_id"):
            build_expansion_partition_rows(duplicate_candidates, blocked_group_ids=set())

    def test_build_expansion_partition_rows_rejects_unsupported_partition_role(self) -> None:
        invalid_candidates = [
            {
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-01",
                "partition_role": "paper_eval",
            }
        ]
        with self.assertRaisesRegex(ValueError, "Unsupported partition_role"):
            build_expansion_partition_rows(invalid_candidates, blocked_group_ids=set())

    def test_build_expansion_partition_rows_rejects_unsafe_group_id_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain"):
            build_expansion_partition_rows(
                [
                    {
                        "benchmark": "AI_Idea_Bench_2025::unsafe",
                        "instance_name": "aiib-01",
                        "partition_role": "critic_train",
                    }
                ],
                blocked_group_ids=set(),
            )
        with self.assertRaisesRegex(ValueError, "must not contain"):
            build_expansion_partition_rows(
                [
                    {
                        "benchmark": "AI_Idea_Bench_2025",
                        "instance_name": "aiib-01\nunsafe",
                        "partition_role": "critic_train",
                    }
                ],
                blocked_group_ids=set(),
            )

    def test_cli_builds_partition_and_split_registry_outputs(self) -> None:
        candidate_file = self.tmp_dir / "candidate_instances.json"
        candidate_file.write_text(json.dumps(self.candidates, indent=2), encoding="utf-8")

        blocked_registry_file = self.tmp_dir / "blocked_registry.jsonl"
        blocked_registry_file.write_text(
            json.dumps(
                {
                    "group_id": "AI_Idea_Bench_2025::blocked-registry",
                    "benchmark": "AI_Idea_Bench_2025",
                    "instance_name": "blocked-registry",
                    "partition_role": "critic_train",
                    "source_split": "train",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        blocked_candidate_file = self.tmp_dir / "blocked_candidates.json"
        blocked_candidate_file.write_text(
            json.dumps(
                [
                    {
                        "benchmark": "liveideabench",
                        "instance_name": "blocked-candidate",
                        "partition_role": "critic_dev",
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        output_dir = self.tmp_dir / "pool"
        script_path = ROOT / "scripts" / "build_critic_expansion_pool.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--candidate-file",
                str(candidate_file),
                "--blocked-split-registry",
                str(blocked_registry_file),
                "--blocked-candidate-file",
                str(blocked_candidate_file),
                "--pool-name",
                "expansion_pool_v1",
                "--output-dir",
                str(output_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

        partition_path = output_dir / "partition_manifest.jsonl"
        registry_path = output_dir / "split_registry.jsonl"
        stats_path = output_dir / "split_registry_stats.json"
        self.assertTrue(partition_path.exists())
        self.assertTrue(registry_path.exists())
        self.assertTrue(stats_path.exists())

        partition_rows = [
            json.loads(line)
            for line in partition_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        partition_group_ids = {str(row["group_id"]) for row in partition_rows}
        self.assertEqual(
            partition_group_ids,
            {"AI_Idea_Bench_2025::aiib-01", "liveideabench::lib-01"},
        )

    def test_cli_rejects_overlap_from_blocked_candidate_file_by_canonical_components(self) -> None:
        candidate_file = self.tmp_dir / "candidate_instances_overlap.json"
        candidate_file.write_text(
            json.dumps(
                [
                    {
                        "benchmark": "AI_Idea_Bench_2025",
                        "instance_name": "aiib-overlap",
                        "partition_role": "critic_train",
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        blocked_candidate_file = self.tmp_dir / "blocked_candidates_overlap.json"
        blocked_candidate_file.write_text(
            json.dumps(
                [
                    {
                        "group_id": "stale::group-id",
                        "benchmark": "AI_Idea_Bench_2025",
                        "instance_name": "aiib-overlap",
                        "partition_role": "critic_dev",
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        output_dir = self.tmp_dir / "pool_overlap"
        script_path = ROOT / "scripts" / "build_critic_expansion_pool.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--candidate-file",
                str(candidate_file),
                "--blocked-candidate-file",
                str(blocked_candidate_file),
                "--pool-name",
                "expansion_pool_v1",
                "--output-dir",
                str(output_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("overlaps blocked group", completed.stderr)

    def test_cli_blocklist_falls_back_to_raw_group_id_when_canonical_fields_are_unsafe(self) -> None:
        candidate_file = self.tmp_dir / "candidate_instances_raw_fallback.json"
        candidate_file.write_text(
            json.dumps(
                [
                    {
                        "benchmark": "AI_Idea_Bench_2025",
                        "instance_name": "aiib-safe",
                        "partition_role": "critic_train",
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        blocked_registry_file = self.tmp_dir / "blocked_registry_raw_fallback.jsonl"
        blocked_registry_file.write_text(
            json.dumps(
                {
                    "group_id": "legacy::blocked-safe",
                    "benchmark": "AI_Idea_Bench_2025::unsafe",
                    "instance_name": "unsafe-name",
                    "partition_role": "critic_train",
                    "source_split": "train",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        output_dir = self.tmp_dir / "pool_raw_fallback"
        script_path = ROOT / "scripts" / "build_critic_expansion_pool.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--candidate-file",
                str(candidate_file),
                "--blocked-split-registry",
                str(blocked_registry_file),
                "--pool-name",
                "expansion_pool_v1",
                "--output-dir",
                str(output_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertTrue((output_dir / "partition_manifest.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
