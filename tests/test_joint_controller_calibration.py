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
    build_joint_calibration_examples_from_packet,
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

    def test_build_joint_calibration_examples_from_packet_pairs_runs_and_preserves_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            heuristic_dir = root / "heuristic"
            critic_dir = root / "critic"
            heuristic_dir.mkdir()
            critic_dir.mkdir()
            (heuristic_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "benchmark_native_evaluation": {
                            "summary": {
                                "available_average_normalized_10": 6.2,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (heuristic_dir / "graph.json").write_text(
                json.dumps({"metadata": {"runtime_controller_log": []}}),
                encoding="utf-8",
            )
            (critic_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "benchmark_native_evaluation": {
                            "summary": {
                                "available_average_normalized_10": 7.4,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (critic_dir / "graph.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "runtime_controller_log": [
                                {
                                    "round": "Round2",
                                    "role": "MethodDesigner",
                                    "selected_source": "critic",
                                    "override_margin": 0.12,
                                    "heuristic_candidate_id": "heur-a",
                                    "selected_candidate_id": "critic-a",
                                    "heuristic_candidate": {
                                        "candidate_id": "heur-a",
                                        "kind": "attach_evidence",
                                    },
                                    "selected_candidate": {
                                        "candidate_id": "critic-a",
                                        "kind": "skip",
                                    },
                                },
                                {
                                    "round": "Round2",
                                    "role": "EvaluationDesigner",
                                    "selected_source": "heuristic",
                                    "override_margin": 0.02,
                                    "heuristic_candidate_id": "heur-b",
                                    "selected_candidate_id": "heur-b",
                                    "heuristic_candidate": {
                                        "candidate_id": "heur-b",
                                        "kind": "add_support_edge",
                                    },
                                    "selected_candidate": {
                                        "candidate_id": "heur-b",
                                        "kind": "add_support_edge",
                                    },
                                },
                            ],
                            "post_round_commit_rows": [
                                {
                                    "round_name": "Round1",
                                    "commit_probability": 0.41,
                                    "commit_supervision": {"available": True, "label": 0},
                                },
                                {
                                    "round_name": "Round2",
                                    "commit_probability": 0.78,
                                    "commit_supervision": {"available": True, "label": 1},
                                },
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )

            run_manifest = root / "run_manifest.jsonl"
            run_manifest.write_text(
                json.dumps(
                    {
                        "group_id": "g1",
                        "baseline_name": "ours-eig",
                        "run_dir": str(heuristic_dir),
                        "instance_name": "ai-idea-bench-2025-12",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "group_id": "g1",
                        "baseline_name": "ours-eig-critic-graph-twohead",
                        "run_dir": str(critic_dir),
                        "instance_name": "ai-idea-bench-2025-12",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            edit_examples, commit_examples = build_joint_calibration_examples_from_packet(
                run_manifest_path=run_manifest,
                heuristic_baseline="ours-eig",
                critic_baseline="ours-eig-critic-graph-twohead",
            )

            self.assertEqual(len(edit_examples), 2)
            self.assertEqual(edit_examples[0]["group_id"], "g1")
            self.assertEqual(edit_examples[0]["instance_name"], "ai-idea-bench-2025-12")
            self.assertEqual(edit_examples[0]["label"], 1)
            self.assertEqual(edit_examples[0]["selected_kind"], "skip")
            self.assertTrue(edit_examples[0]["selected_is_skip"])
            self.assertFalse(edit_examples[0]["heuristic_is_skip"])
            self.assertEqual(edit_examples[1]["label"], 0)
            self.assertEqual(edit_examples[1]["selected_source"], "heuristic")

            self.assertEqual(len(commit_examples), 2)
            self.assertEqual(commit_examples[0]["round_index"], 1)
            self.assertEqual(commit_examples[0]["label"], 0)
            self.assertEqual(commit_examples[1]["round_index"], 2)
            self.assertEqual(commit_examples[1]["label"], 1)
            self.assertAlmostEqual(commit_examples[1]["commit_probability"], 0.78)

    def test_build_joint_calibration_examples_from_packet_marks_bad_overrides_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            heuristic_dir = root / "heuristic"
            critic_dir = root / "critic"
            heuristic_dir.mkdir()
            critic_dir.mkdir()
            (heuristic_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "benchmark_native_evaluation": {
                            "summary": {
                                "available_average_normalized_10": 8.1,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (heuristic_dir / "graph.json").write_text(
                json.dumps({"metadata": {"runtime_controller_log": []}}),
                encoding="utf-8",
            )
            (critic_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "benchmark_native_evaluation": {
                            "summary": {
                                "available_average_normalized_10": 6.7,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (critic_dir / "graph.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "runtime_controller_log": [
                                {
                                    "round": "Round3",
                                    "role": "MechanismProposer",
                                    "selected_source": "critic",
                                    "override_margin": 0.15,
                                    "heuristic_candidate_id": "heur-c",
                                    "selected_candidate_id": "critic-c",
                                    "heuristic_candidate": {
                                        "candidate_id": "heur-c",
                                        "kind": "freeze_branch",
                                    },
                                    "selected_candidate": {
                                        "candidate_id": "critic-c",
                                        "kind": "propose_repair",
                                    },
                                }
                            ],
                            "post_round_commit_rows": [],
                        }
                    }
                ),
                encoding="utf-8",
            )

            run_manifest = root / "run_manifest.jsonl"
            run_manifest.write_text(
                json.dumps(
                    {
                        "group_id": "g2",
                        "baseline_name": "ours-eig",
                        "run_dir": str(heuristic_dir),
                        "instance_name": "ai-idea-bench-2025-21",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "group_id": "g2",
                        "baseline_name": "ours-eig-critic-graph-twohead",
                        "run_dir": str(critic_dir),
                        "instance_name": "ai-idea-bench-2025-21",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            edit_examples, commit_examples = build_joint_calibration_examples_from_packet(
                run_manifest_path=run_manifest,
                heuristic_baseline="ours-eig",
                critic_baseline="ours-eig-critic-graph-twohead",
            )

            self.assertEqual(len(edit_examples), 1)
            self.assertEqual(edit_examples[0]["label"], 0)
            self.assertEqual(edit_examples[0]["selected_source"], "critic")
            self.assertEqual(commit_examples, [])

    def test_calibration_cli_can_fit_from_packet_manifest_and_write_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            heuristic_dir = root / "heuristic"
            critic_dir = root / "critic"
            heuristic_dir.mkdir()
            critic_dir.mkdir()
            (heuristic_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "benchmark_native_evaluation": {
                            "summary": {
                                "available_average_normalized_10": 5.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (heuristic_dir / "graph.json").write_text(
                json.dumps({"metadata": {"runtime_controller_log": []}}),
                encoding="utf-8",
            )
            (critic_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "benchmark_native_evaluation": {
                            "summary": {
                                "available_average_normalized_10": 6.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (critic_dir / "graph.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "runtime_controller_log": [
                                {
                                    "round": "Round1",
                                    "role": "MethodDesigner",
                                    "selected_source": "critic",
                                    "override_margin": 0.12,
                                    "heuristic_candidate": {"candidate_id": "h1", "kind": "attach_evidence"},
                                    "selected_candidate": {"candidate_id": "c1", "kind": "propose_repair"},
                                },
                                {
                                    "round": "Round1",
                                    "role": "EvaluationDesigner",
                                    "selected_source": "heuristic",
                                    "override_margin": 0.03,
                                    "heuristic_candidate": {"candidate_id": "h2", "kind": "add_support_edge"},
                                    "selected_candidate": {"candidate_id": "h2", "kind": "add_support_edge"},
                                },
                            ],
                            "post_round_commit_rows": [
                                {
                                    "round_name": "Round1",
                                    "commit_probability": 0.36,
                                    "commit_supervision": {"available": True, "label": 0},
                                },
                                {
                                    "round_name": "Round2",
                                    "commit_probability": 0.82,
                                    "commit_supervision": {"available": True, "label": 1},
                                },
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            run_manifest = root / "run_manifest.jsonl"
            run_manifest.write_text(
                json.dumps(
                    {
                        "group_id": "g3",
                        "baseline_name": "ours-eig",
                        "run_dir": str(heuristic_dir),
                        "instance_name": "ai-idea-bench-2025-31",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "group_id": "g3",
                        "baseline_name": "ours-eig-critic-graph-twohead",
                        "run_dir": str(critic_dir),
                        "instance_name": "ai-idea-bench-2025-31",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "joint_controller_calibration.json"
            prepared_dir = root / "prepared"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/calibrate_joint_controller.py",
                    "--run-manifest",
                    str(run_manifest),
                    "--output-path",
                    str(output_path),
                    "--prepared-output-dir",
                    str(prepared_dir),
                ],
                check=False,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["tau_override"], 0.12)
            self.assertEqual(payload["gamma_commit"], 0.82)
            self.assertTrue((prepared_dir / "edit_examples.jsonl").exists())
            self.assertTrue((prepared_dir / "commit_examples.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
