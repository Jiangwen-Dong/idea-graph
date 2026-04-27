from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from test_relation_graph_critic_data import write_relation_graph_fixture


class TrainRelationGraphCriticCliTests(unittest.TestCase):
    def test_train_relation_graph_critic_cli_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix=f"{self._testMethodName}_") as tmp_dir:
            fixture_root = Path(tmp_dir)
            fixture = write_relation_graph_fixture(fixture_root)
            output_dir = fixture_root / "model_output"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train/train_relation_graph_critic.py",
                    "--candidate-dataset-dir",
                    str(fixture.candidate_dir),
                    "--g1-dataset-dir",
                    str(fixture.g1_dir),
                    "--partition-manifest",
                    str(fixture.partition_manifest),
                    "--output-dir",
                    str(output_dir),
                    "--text-backend",
                    "hash",
                    "--embedding-dim",
                    "8",
                    "--hidden-dim",
                    "16",
                    "--batch-size",
                    "2",
                    "--epochs",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=ROOT,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((output_dir / "model.pt").exists())
            self.assertTrue((output_dir / "metrics_all.json").exists())
            self.assertTrue((output_dir / "metrics_edit_only.json").exists())
            metadata = json.loads((output_dir / "metadata.json").read_text())
            self.assertEqual(metadata["text_backend"], "hash")

