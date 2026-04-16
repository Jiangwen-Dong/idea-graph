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

from idea_graph.fs_utils import write_text_file


class CriticSplitOverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_build_split_override_rows_maps_train_and_dev(self) -> None:
        from idea_graph.critic_split_overrides import build_split_override_rows

        rows = build_split_override_rows(
            [
                {"group_id": "g-train", "partition_role": "critic_train"},
                {"group_id": "g-dev", "partition_role": "critic_dev"},
            ]
        )

        self.assertEqual(
            rows,
            [
                {"group_id": "g-dev", "split": "validation"},
                {"group_id": "g-train", "split": "train"},
            ],
        )

    def test_build_split_override_rows_excludes_paper_eval(self) -> None:
        from idea_graph.critic_split_overrides import build_split_override_rows

        rows = build_split_override_rows(
            [
                {"group_id": "g-paper", "partition_role": "paper_eval"},
                {"group_id": "g-train", "partition_role": "critic_train"},
            ]
        )

        self.assertEqual(rows, [{"group_id": "g-train", "split": "train"}])

    def test_build_split_override_rows_rejects_missing_group_id(self) -> None:
        from idea_graph.critic_split_overrides import build_split_override_rows

        with self.assertRaisesRegex(ValueError, "missing required group_id"):
            build_split_override_rows([{"partition_role": "critic_train"}])

    def test_build_split_override_rows_rejects_conflicting_group_assignments(self) -> None:
        from idea_graph.critic_split_overrides import build_split_override_rows

        with self.assertRaisesRegex(ValueError, "Conflicting split override"):
            build_split_override_rows(
                [
                    {"group_id": "g-shared", "partition_role": "critic_train"},
                    {"group_id": "g-shared", "partition_role": "critic_dev"},
                ]
            )

    def test_load_split_registry_rows_skips_blank_lines(self) -> None:
        from idea_graph.critic_split_overrides import load_split_registry_rows

        registry_path = self.tmp_dir / "registry_with_blanks.jsonl"
        write_text_file(
            registry_path,
            '\n{"group_id":"g-train","partition_role":"critic_train"}\n\n'
            '{"group_id":"g-dev","partition_role":"critic_dev"}\n',
        )

        rows = load_split_registry_rows(registry_path)

        self.assertEqual(
            rows,
            [
                {"group_id": "g-train", "partition_role": "critic_train"},
                {"group_id": "g-dev", "partition_role": "critic_dev"},
            ],
        )

    def test_load_split_registry_rows_rejects_non_object_json(self) -> None:
        from idea_graph.critic_split_overrides import load_split_registry_rows

        registry_path = self.tmp_dir / "registry_invalid.jsonl"
        write_text_file(registry_path, '"not-an-object"\n')

        with self.assertRaisesRegex(ValueError, "must contain a JSON object"):
            load_split_registry_rows(registry_path)

    def test_cli_writes_split_override_rows(self) -> None:
        registry_path = self.tmp_dir / "split_registry.jsonl"
        output_path = self.tmp_dir / "split_overrides.jsonl"
        script_path = ROOT / "scripts" / "build_critic_split_overrides.py"
        write_text_file(
            registry_path,
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in [
                    {"group_id": "g-paper", "partition_role": "paper_eval"},
                    {"group_id": "g-train", "partition_role": "critic_train"},
                    {"group_id": "g-dev", "partition_role": "critic_dev"},
                ]
            ),
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--split-registry",
                str(registry_path),
                "--output-path",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("Wrote 2 split overrides", completed.stdout)
        self.assertTrue(output_path.exists())

        rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(
            rows,
            [
                {"group_id": "g-dev", "split": "validation"},
                {"group_id": "g-train", "split": "train"},
            ],
        )

    def test_cli_reports_conflicting_group_assignments(self) -> None:
        registry_path = self.tmp_dir / "split_registry_conflict.jsonl"
        output_path = self.tmp_dir / "split_overrides_conflict.jsonl"
        script_path = ROOT / "scripts" / "build_critic_split_overrides.py"
        write_text_file(
            registry_path,
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in [
                    {"group_id": "g-shared", "partition_role": "critic_train"},
                    {"group_id": "g-shared", "partition_role": "critic_dev"},
                ]
            ),
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--split-registry",
                str(registry_path),
                "--output-path",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Conflicting split override", completed.stderr)


if __name__ == "__main__":
    unittest.main()
