from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.fs_utils import _windows_safe_path, write_text_file
from idea_graph.candidate_slate_dataset import (
    build_candidate_dataset_rows,
    build_candidate_dataset_stats,
    build_graph_critic_candidate_dataset,
)
from idea_graph.critic_dataset import build_graph_critic_dataset


class CandidateSlateDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())
        self.g1_dataset_dir = self.tmp_dir / "g1"
        self.g1_dataset_dir.mkdir(parents=True, exist_ok=True)
        self._write_snapshot(self.g1_dataset_dir / "state_snapshots" / "run-a-step-000.json")
        self._write_snapshot(self.g1_dataset_dir / "terminal_state_snapshots" / "run-a-terminal.json")

    def tearDown(self) -> None:
        shutil.rmtree(_windows_safe_path(self.tmp_dir), ignore_errors=True)

    def _write_snapshot(self, path: Path) -> None:
        payload = {
            "action_id": "A001",
            "action_index": 0,
            "action_timestamp": "2026-04-09 07:24:16.928017+00:00",
            "nodes": {
                "N001": {
                    "id": "N001",
                    "type": "Problem",
                    "text": "Improve 3D language field reliability.",
                    "role": "MechanismProposer",
                    "branch_id": "B001",
                    "confidence": 0.72,
                    "evidence": [],
                    "status": "active",
                    "created_at": "2026-04-09 07:24:16.927013+00:00",
                    "provenance": [],
                },
                "N002": {
                    "id": "N002",
                    "type": "Method",
                    "text": "Fuse semantic anchors with Gaussian splats.",
                    "role": "MechanismProposer",
                    "branch_id": "B001",
                    "confidence": 0.64,
                    "evidence": [],
                    "status": "active",
                    "created_at": "2026-04-09 07:24:16.927013+00:00",
                    "provenance": [],
                },
                "N003": {
                    "id": "N003",
                    "type": "Method",
                    "text": "Alternate branch method for fallback testing.",
                    "role": "MechanismProposer",
                    "branch_id": "B009",
                    "confidence": 0.61,
                    "evidence": [],
                    "status": "active",
                    "created_at": "2026-04-09 07:24:16.927013+00:00",
                    "provenance": [],
                },
            },
            "edges": [
                {
                    "id": "E001",
                    "source_id": "N001",
                    "relation": "supports",
                    "target_id": "N002",
                    "role": "MechanismProposer",
                    "branch_id": "B001",
                    "resolved": False,
                    "created_at": "2026-04-09 07:24:16.927013+00:00",
                }
            ],
            "node_count": 3,
            "edge_count": 1,
            "contradiction_count": 0,
            "support_edge_count": 1,
        }
        write_text_file(path, json.dumps(payload, indent=2, ensure_ascii=False))

    def _critic_rows(self) -> list[dict[str, object]]:
        return [
            {
                "run_dir": "run_a",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-13",
                "topic": "3D language field modeling",
                "step_index": 0,
                "round_name": "Round1",
                "role": "MechanismProposer",
                "selected_action_kind": "add_support_edge",
                "selected_action_targets": ["N001", "N002"],
                "selected_action_payload": {"branch_id": "B001"},
                "selected_action_rationale": "Support problem with method.",
                "selected_action_branch_id": "B001",
                "before_state_snapshot": "state_snapshots/run-a-step-000.json",
                "state_literature": [
                    "Paper A: Structured collaboration for idea graphs.",
                    "Paper B: Benchmark-specific evaluation protocols.",
                ],
                "group_id": "AI_Idea_Bench_2025::aiib-13",
                "split": "train",
                "targets": {"weak_value_01": 0.63, "native_value_01": 0.82},
                "weak_local": {"available": True, "overall_01": 0.63},
                "native": {"available": True, "average_01": 0.82},
                "label_availability": {
                    "has_weak_local": True,
                    "has_native": True,
                    "has_native_average": True,
                },
            }
        ]

    def _payload_rich_critic_rows(self) -> list[dict[str, object]]:
        return [
            {
                "run_dir": "run_a",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-13",
                "topic": "3D language field modeling",
                "step_index": 1,
                "round_name": "Round2",
                "role": "MechanismProposer",
                "selected_action_kind": "attach_evidence",
                "selected_action_targets": ["N003"],
                "selected_action_payload": {"branch_id": "B009", "evidence": "Evidence from benchmark appendix"},
                "selected_action_rationale": "Attach branch-specific evidence.",
                "selected_action_branch_id": "B009",
                "before_state_snapshot": "state_snapshots/run-a-step-000.json",
                "state_literature": [
                    "Paper A: Structured collaboration for idea graphs.",
                    "Paper B: Benchmark-specific evaluation protocols.",
                ],
                "group_id": "AI_Idea_Bench_2025::aiib-13",
                "split": "train",
                "targets": {"weak_value_01": 0.63, "native_value_01": 0.82},
                "weak_local": {"available": True, "overall_01": 0.63},
                "native": {"available": True, "average_01": 0.82},
                "label_availability": {
                    "has_weak_local": True,
                    "has_native": True,
                    "has_native_average": True,
                },
            }
        ]

    def test_build_rows_emits_commit_and_single_logged_selected(self) -> None:
        candidate_rows, state_manifest = build_candidate_dataset_rows(
            g1_dataset_dir=self.g1_dataset_dir,
            critic_rows=self._critic_rows(),
        )
        self.assertEqual(len(state_manifest), 1)
        self.assertGreaterEqual(len(candidate_rows), 2)
        self.assertTrue(all(str(row.get("candidate_id", "")).strip() for row in candidate_rows))
        self.assertEqual(sum(1 for row in candidate_rows if bool(row["is_commit"])), 1)
        self.assertEqual(sum(1 for row in candidate_rows if bool(row["is_logged_selected"])), 1)

    def test_candidate_text_contains_target_type_and_text(self) -> None:
        candidate_rows, _ = build_candidate_dataset_rows(
            g1_dataset_dir=self.g1_dataset_dir,
            critic_rows=self._critic_rows(),
        )
        logged = next(row for row in candidate_rows if bool(row["is_logged_selected"]))
        text = str(logged["candidate_text"])
        self.assertIn("Problem: Improve 3D language field reliability.", text)
        self.assertIn("Method: Fuse semantic anchors with Gaussian splats.", text)

    def test_state_manifest_preserves_group_split_and_candidate_count(self) -> None:
        candidate_rows, state_manifest = build_candidate_dataset_rows(
            g1_dataset_dir=self.g1_dataset_dir,
            critic_rows=self._critic_rows(),
        )
        manifest_row = state_manifest[0]
        self.assertEqual(manifest_row["group_id"], "AI_Idea_Bench_2025::aiib-13")
        self.assertEqual(manifest_row["split"], "train")
        self.assertEqual(manifest_row["candidate_count"], len(candidate_rows))

    def test_dataset_stats_reports_counts_and_commit_fraction(self) -> None:
        candidate_rows, state_manifest = build_candidate_dataset_rows(
            g1_dataset_dir=self.g1_dataset_dir,
            critic_rows=self._critic_rows(),
        )
        stats = build_candidate_dataset_stats(candidate_rows, state_manifest)
        self.assertEqual(stats["state_count"], 1)
        self.assertEqual(stats["candidate_count"], len(candidate_rows))
        self.assertAlmostEqual(stats["commit_fraction"], 1.0 / len(candidate_rows))

    def test_builder_writes_candidate_schema_json(self) -> None:
        g2_dataset_dir = self.tmp_dir / "g2"
        output_dir = self.tmp_dir / "out"
        g2_dataset_dir.mkdir(parents=True, exist_ok=True)
        critic_rows = self._critic_rows()
        write_text_file(
            g2_dataset_dir / "critic_dataset.jsonl",
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in critic_rows),
        )

        result = build_graph_critic_candidate_dataset(
            g1_dataset_dir=self.g1_dataset_dir,
            g2_dataset_dir=g2_dataset_dir,
            output_dir=output_dir,
            dataset_name="g25-smoke",
        )
        self.assertEqual(result.state_count, 1)
        self.assertTrue((result.dataset_dir / "candidate_schema.json").exists())

    def test_logged_selected_candidate_preserves_payload_semantics(self) -> None:
        candidate_rows, _ = build_candidate_dataset_rows(
            g1_dataset_dir=self.g1_dataset_dir,
            critic_rows=self._payload_rich_critic_rows(),
        )
        logged = next(row for row in candidate_rows if bool(row["is_logged_selected"]))
        self.assertEqual(logged["candidate_kind"], "attach_evidence")
        self.assertEqual(logged["candidate_target_ids"], ["N003"])
        self.assertEqual(
            logged["candidate_payload"],
            {"branch_id": "B009", "evidence": "Evidence from benchmark appendix"},
        )

    def test_same_role_multi_branch_uses_target_branch_before_role_match(self) -> None:
        candidate_rows, _ = build_candidate_dataset_rows(
            g1_dataset_dir=self.g1_dataset_dir,
            critic_rows=self._payload_rich_critic_rows(),
        )
        logged = next(row for row in candidate_rows if bool(row["is_logged_selected"]))
        self.assertEqual(logged["candidate_payload"]["branch_id"], "B009")

    def test_terminal_rows_create_positive_commit_candidate(self) -> None:
        terminal_rows = [
            {
                "run_dir": "run_a",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-13",
                "topic": "3D language field modeling",
                "step_index": 1,
                "round_name": "Terminal",
                "role": "CommitController",
                "selected_action_kind": "commit",
                "selected_action_targets": [],
                "selected_action_payload": {"branch_id": "B001"},
                "selected_action_rationale": "Commit after the final graph state is complete.",
                "selected_action_branch_id": "B001",
                "before_state_snapshot": "terminal_state_snapshots/run-a-terminal.json",
                "state_literature": [
                    "Paper A: Structured collaboration for idea graphs.",
                    "Paper B: Benchmark-specific evaluation protocols.",
                ],
                "group_id": "AI_Idea_Bench_2025::aiib-13",
                "split": "validation",
                "state_kind": "terminal_commit",
                "commit_supervision": {"available": True, "label": 1, "source": "terminal_final_graph"},
                "targets": {"weak_value_01": 0.63, "native_value_01": 0.82},
                "weak_local": {"available": True, "overall_01": 0.63},
                "native": {"available": True, "average_01": 0.82},
                "label_availability": {
                    "has_weak_local": True,
                    "has_native": True,
                    "has_native_average": True,
                },
            }
        ]

        candidate_rows, state_manifest = build_candidate_dataset_rows(
            g1_dataset_dir=self.g1_dataset_dir,
            critic_rows=[],
            terminal_rows=terminal_rows,
        )

        self.assertEqual(len(state_manifest), 1)
        commit_rows = [row for row in candidate_rows if bool(row["is_commit"])]
        self.assertEqual(len(commit_rows), 1)
        self.assertTrue(commit_rows[0]["is_logged_selected"])
        self.assertTrue(commit_rows[0]["is_commit_positive_state"])
        self.assertEqual(commit_rows[0]["commit_supervision"]["label"], 1)

    def test_dataset_stats_reports_commit_positive_count(self) -> None:
        terminal_rows = [
            {
                **self._critic_rows()[0],
                "step_index": 1,
                "round_name": "Terminal",
                "role": "CommitController",
                "selected_action_kind": "commit",
                "selected_action_targets": [],
                "selected_action_payload": {"branch_id": "B001"},
                "before_state_snapshot": "terminal_state_snapshots/run-a-terminal.json",
                "state_kind": "terminal_commit",
                "commit_supervision": {"available": True, "label": 1, "source": "terminal_final_graph"},
            }
        ]
        candidate_rows, state_manifest = build_candidate_dataset_rows(
            g1_dataset_dir=self.g1_dataset_dir,
            critic_rows=self._critic_rows(),
            terminal_rows=terminal_rows,
        )

        stats = build_candidate_dataset_stats(candidate_rows, state_manifest)

        self.assertEqual(stats["terminal_state_count"], 1)
        self.assertEqual(stats["commit_positive_count"], 1)

    def test_graph_critic_builder_honors_split_overrides(self) -> None:
        g1_dataset_dir = self.tmp_dir / "g1_graph_critic"
        output_dir = self.tmp_dir / "out_graph_critic"
        g1_dataset_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = [
            {
                "run_dir": "run-a",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-01",
                "has_local_eval": True,
                "has_native_eval": True,
                "final_local_overall": 6.0,
                "final_native_average": 7.0,
            },
            {
                "run_dir": "run-b",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-02",
                "has_local_eval": True,
                "has_native_eval": True,
                "final_local_overall": 6.5,
                "final_native_average": 7.5,
            },
            {
                "run_dir": "run-c",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-03",
                "has_local_eval": True,
                "has_native_eval": True,
                "final_local_overall": 7.0,
                "final_native_average": 8.0,
            },
        ]
        transition_rows = [
            {
                "run_dir": "run-a",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-01",
            },
            {
                "run_dir": "run-b",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-02",
            },
            {
                "run_dir": "run-c",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-03",
            },
        ]
        split_override_rows = [
            {"group_id": "AI_Idea_Bench_2025::aiib-01", "split": "validation"},
            {"group_id": "AI_Idea_Bench_2025::aiib-03", "split": "train"},
        ]

        write_text_file(
            g1_dataset_dir / "run_manifest.jsonl",
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        )
        write_text_file(
            g1_dataset_dir / "trajectory_examples.jsonl",
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in transition_rows),
        )
        split_overrides_path = g1_dataset_dir / "split_overrides.jsonl"
        write_text_file(
            split_overrides_path,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_override_rows),
        )

        result = build_graph_critic_dataset(
            g1_dataset_dir=g1_dataset_dir,
            output_dir=output_dir,
            dataset_name="g2-override-smoke",
            split_overrides_path=split_overrides_path,
        )
        split_manifest_rows = [
            json.loads(line)
            for line in (result.dataset_dir / "split_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        split_by_group = {
            str(row["group_id"]): str(row["split"])
            for row in split_manifest_rows
        }
        self.assertEqual(split_by_group["AI_Idea_Bench_2025::aiib-01"], "validation")
        self.assertEqual(split_by_group["AI_Idea_Bench_2025::aiib-03"], "train")
        self.assertEqual(split_by_group["AI_Idea_Bench_2025::aiib-02"], "train")

    def test_graph_critic_builder_overrides_do_not_change_uncovered_default_splits(self) -> None:
        g1_dataset_dir = self.tmp_dir / "g1_graph_critic_uncovered"
        output_dir = self.tmp_dir / "out_graph_critic_uncovered"
        g1_dataset_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = [
            {
                "run_dir": "run-a",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-01",
                "has_local_eval": True,
                "has_native_eval": True,
                "final_local_overall": 6.0,
                "final_native_average": 7.0,
            },
            {
                "run_dir": "run-b",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-02",
                "has_local_eval": True,
                "has_native_eval": True,
                "final_local_overall": 6.5,
                "final_native_average": 7.5,
            },
            {
                "run_dir": "run-c",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-03",
                "has_local_eval": True,
                "has_native_eval": True,
                "final_local_overall": 7.0,
                "final_native_average": 8.0,
            },
        ]
        transition_rows = [
            {
                "run_dir": "run-a",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-01",
            },
            {
                "run_dir": "run-b",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-02",
            },
            {
                "run_dir": "run-c",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-03",
            },
        ]
        split_override_rows = [
            {"group_id": "AI_Idea_Bench_2025::aiib-01", "split": "validation"},
        ]

        write_text_file(
            g1_dataset_dir / "run_manifest.jsonl",
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        )
        write_text_file(
            g1_dataset_dir / "trajectory_examples.jsonl",
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in transition_rows),
        )
        split_overrides_path = g1_dataset_dir / "split_overrides.jsonl"
        write_text_file(
            split_overrides_path,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_override_rows),
        )

        result = build_graph_critic_dataset(
            g1_dataset_dir=g1_dataset_dir,
            output_dir=output_dir,
            dataset_name="g2-override-uncovered-defaults",
            split_overrides_path=split_overrides_path,
        )
        split_manifest_rows = [
            json.loads(line)
            for line in (result.dataset_dir / "split_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        split_by_group = {
            str(row["group_id"]): str(row["split"])
            for row in split_manifest_rows
        }

        self.assertEqual(split_by_group["AI_Idea_Bench_2025::aiib-01"], "validation")
        self.assertEqual(split_by_group["AI_Idea_Bench_2025::aiib-02"], "train")
        self.assertEqual(split_by_group["AI_Idea_Bench_2025::aiib-03"], "validation")


if __name__ == "__main__":
    unittest.main()
