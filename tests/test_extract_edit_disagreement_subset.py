from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ExtractEditDisagreementSubsetTests(unittest.TestCase):
    def test_extract_edit_disagreement_subset_aligns_paired_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            heuristic_dir = root / "heuristic"
            critic_dir = root / "critic"
            heuristic_dir.mkdir()
            critic_dir.mkdir()
            (heuristic_dir / "graph.json").write_text(
                json.dumps({"metadata": {"runtime_controller_log": []}}),
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
                                    "heuristic_candidate_id": "heur-a",
                                    "selected_candidate_id": "critic-a",
                                    "selected_source": "critic",
                                    "override_margin": 0.12,
                                    "heuristic_candidate": {"candidate_id": "heur-a", "kind": "attach_evidence"},
                                    "selected_candidate": {"candidate_id": "critic-a", "kind": "propose_repair"},
                                },
                                {
                                    "round": "Round2",
                                    "role": "EvaluationDesigner",
                                    "heuristic_candidate_id": "heur-b",
                                    "selected_candidate_id": "heur-b",
                                    "selected_source": "heuristic",
                                    "override_margin": 0.01,
                                    "heuristic_candidate": {"candidate_id": "heur-b", "kind": "add_support_edge"},
                                    "selected_candidate": {"candidate_id": "heur-b", "kind": "add_support_edge"},
                                },
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            (heuristic_dir / "summary.json").write_text(
                json.dumps({"stop_reason": "max_rounds_reached", "executed_round_count": 5}),
                encoding="utf-8",
            )
            (critic_dir / "summary.json").write_text(
                json.dumps({"stop_reason": "mature_at_Round3", "executed_round_count": 3}),
                encoding="utf-8",
            )

            run_manifest = root / "run_manifest.jsonl"
            run_manifest.write_text(
                json.dumps(
                    {
                        "group_id": "g1",
                        "baseline_name": "ours-eig",
                        "run_dir": str(heuristic_dir),
                        "instance_name": "ai-idea-bench-2025-125",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "group_id": "g1",
                        "baseline_name": "ours-eig-critic-graph-twohead",
                        "run_dir": str(critic_dir),
                        "instance_name": "ai-idea-bench-2025-125",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "disagreements.jsonl"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/data_prep/extract_edit_disagreement_subset.py",
                    "--run-manifest",
                    str(run_manifest),
                    "--output-path",
                    str(output_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["group_id"], "g1")
            self.assertEqual(rows[0]["round"], "Round2")
            self.assertEqual(rows[0]["role"], "MethodDesigner")
            self.assertEqual(rows[0]["override_margin"], 0.12)
            self.assertEqual(rows[0]["heuristic_candidate_id"], "heur-a")
            self.assertEqual(rows[0]["critic_candidate_id"], "critic-a")
            self.assertEqual(rows[0]["critic_executed_round_count"], 3)

    def test_extract_edit_disagreement_subset_resolves_repo_relative_run_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            outputs_root = root / "outputs" / "packet"
            heuristic_dir = outputs_root / "runs" / "ours-eig" / "case-a"
            critic_dir = outputs_root / "runs" / "ours-eig-critic-graph-twohead" / "case-a"
            heuristic_dir.mkdir(parents=True)
            critic_dir.mkdir(parents=True)
            (heuristic_dir / "graph.json").write_text(
                json.dumps({"metadata": {"runtime_controller_log": []}}),
                encoding="utf-8",
            )
            (critic_dir / "graph.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "runtime_controller_log": [
                                {
                                    "round": "Round3",
                                    "role": "MethodDesigner",
                                    "heuristic_candidate_id": "heur-c",
                                    "selected_candidate_id": "critic-c",
                                    "selected_source": "critic",
                                    "override_margin": 0.08,
                                    "heuristic_candidate": {"candidate_id": "heur-c", "kind": "attach_evidence"},
                                    "selected_candidate": {"candidate_id": "critic-c", "kind": "propose_repair"},
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            (heuristic_dir / "summary.json").write_text(
                json.dumps({"stop_reason": "max_rounds_reached", "executed_round_count": 5}),
                encoding="utf-8",
            )
            (critic_dir / "summary.json").write_text(
                json.dumps({"stop_reason": "mature_at_Round4", "executed_round_count": 4}),
                encoding="utf-8",
            )

            run_manifest = outputs_root / "run_manifest.jsonl"
            run_manifest.parent.mkdir(parents=True, exist_ok=True)
            run_manifest.write_text(
                json.dumps(
                    {
                        "group_id": "g2",
                        "baseline_name": "ours-eig",
                        "run_dir": "outputs/packet/runs/ours-eig/case-a",
                        "instance_name": "ai-idea-bench-2025-71",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "group_id": "g2",
                        "baseline_name": "ours-eig-critic-graph-twohead",
                        "run_dir": "outputs/packet/runs/ours-eig-critic-graph-twohead/case-a",
                        "instance_name": "ai-idea-bench-2025-71",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "disagreements.jsonl"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/data_prep/extract_edit_disagreement_subset.py",
                    "--run-manifest",
                    str(run_manifest),
                    "--output-path",
                    str(output_path),
                    "--repo-root",
                    str(root),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["group_id"], "g2")
            self.assertEqual(rows[0]["critic_executed_round_count"], 4)


if __name__ == "__main__":
    unittest.main()
