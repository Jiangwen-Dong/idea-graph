from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.baselines import BASELINE_SPECS, _maybe_build_runtime_controller
from idea_graph.joint_controller_calibration import (
    JointControllerCalibration,
    JointControllerCalibrationError,
    apply_joint_controller_calibration,
    fit_joint_controller_calibration,
    load_joint_controller_calibration,
)
from idea_graph.models import IdeaGraph


class JointControllerCalibrationTests(unittest.TestCase):
    def test_fit_joint_controller_calibration_picks_joint_thresholds(self) -> None:
        calibration = fit_joint_controller_calibration(
            edit_examples=[
                {"override_margin": 0.02, "label": 0},
                {"override_margin": 0.09, "label": 1},
                {"override_margin": 0.12, "label": 1},
            ],
            commit_examples=[
                {"commit_probability": 0.45, "round_index": 2, "label": 0},
                {"commit_probability": 0.77, "round_index": 3, "label": 1},
                {"commit_probability": 0.81, "round_index": 4, "label": 1},
            ],
        )

        self.assertGreaterEqual(calibration.tau_override, 0.05)
        self.assertGreaterEqual(calibration.gamma_commit, 0.70)
        self.assertGreaterEqual(calibration.min_commit_round, 2)

    def test_fit_joint_controller_calibration_rejects_single_class_commit_labels(self) -> None:
        with self.assertRaises(JointControllerCalibrationError):
            fit_joint_controller_calibration(
                edit_examples=[{"override_margin": 0.10, "label": 1}],
                commit_examples=[{"commit_probability": 0.80, "round_index": 3, "label": 1}],
            )

    def test_apply_joint_controller_calibration_overrides_runtime_metadata(self) -> None:
        calibration = JointControllerCalibration(
            tau_override=0.11,
            tau_commit=0.08,
            gamma_commit=0.73,
            min_commit_round=3,
            guard_support_threshold=0.72,
            source="critic_dev",
            version="joint_controller_calibration_v1",
        )
        metadata = {
            "runtime_controller_tau_override": 0.05,
            "runtime_controller_tau_commit": 0.08,
            "runtime_controller_gamma_commit": 0.60,
            "runtime_controller_min_commit_round": 2,
            "runtime_controller_guard_support_threshold": 0.66,
        }

        applied = apply_joint_controller_calibration(metadata, calibration)

        self.assertEqual(applied["runtime_controller_tau_override"], 0.11)
        self.assertEqual(applied["runtime_controller_gamma_commit"], 0.73)
        self.assertEqual(applied["runtime_controller_min_commit_round"], 3)
        self.assertEqual(applied["runtime_controller_guard_support_threshold"], 0.72)

    def test_load_joint_controller_calibration_reads_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "joint_controller_calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "tau_override": 0.09,
                        "tau_commit": 0.08,
                        "gamma_commit": 0.71,
                        "min_commit_round": 3,
                        "guard_support_threshold": 0.69,
                        "source": "critic_dev",
                        "version": "joint_controller_calibration_v1",
                    }
                ),
                encoding="utf-8",
            )

            calibration = load_joint_controller_calibration(path)

            self.assertEqual(calibration.tau_override, 0.09)
            self.assertEqual(calibration.gamma_commit, 0.71)

    def test_runtime_controller_loader_applies_joint_calibration_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "joint_controller_calibration.json").write_text(
                json.dumps(
                    {
                        "tau_override": 0.13,
                        "tau_commit": 0.08,
                        "gamma_commit": 0.74,
                        "min_commit_round": 4,
                        "guard_support_threshold": 0.75,
                        "source": "critic_dev",
                        "version": "joint_controller_calibration_v1",
                    }
                ),
                encoding="utf-8",
            )
            graph = IdeaGraph(
                topic="topic",
                literature=["paper"],
                metadata={
                    "runtime_controller_kind": "relation_graph_two_head_critic",
                    "runtime_controller_model_dir": str(model_dir),
                    "runtime_controller_use_commit": True,
                    "runtime_controller_tau_override": 0.05,
                    "runtime_controller_tau_commit": 0.08,
                    "runtime_controller_gamma_commit": 0.60,
                    "runtime_controller_min_commit_round": 2,
                },
            )

            with patch(
                "idea_graph.baselines.load_relation_graph_two_head_runtime_bundle",
                return_value=object(),
            ):
                _, controller_metadata = _maybe_build_runtime_controller(
                    graph,
                    BASELINE_SPECS["ours-eig-critic-graph-twohead"],
                )

            self.assertIsNotNone(controller_metadata)
            config = controller_metadata["config"]
            self.assertEqual(config.tau_override, 0.13)
            self.assertEqual(config.gamma_commit, 0.74)
            self.assertEqual(config.min_commit_round, 4)
            self.assertEqual(config.guard_support_threshold, 0.75)

    def test_calibration_cli_writes_joint_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            edit_examples = root / "edit_examples.jsonl"
            commit_examples = root / "commit_examples.jsonl"
            output_path = root / "joint_controller_calibration.json"
            edit_examples.write_text(
                '{"override_margin": 0.03, "label": 0}\n'
                '{"override_margin": 0.10, "label": 1}\n',
                encoding="utf-8",
            )
            commit_examples.write_text(
                '{"commit_probability": 0.48, "round_index": 2, "label": 0}\n'
                '{"commit_probability": 0.79, "round_index": 3, "label": 1}\n',
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/calibrate_joint_controller.py",
                    "--edit-examples",
                    str(edit_examples),
                    "--commit-examples",
                    str(commit_examples),
                    "--output-path",
                    str(output_path),
                ],
                check=False,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], "joint_controller_calibration_v1")
            self.assertEqual(payload["tau_override"], 0.1)
            self.assertEqual(payload["gamma_commit"], 0.79)


if __name__ == "__main__":
    unittest.main()
