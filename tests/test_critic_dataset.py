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
from idea_graph.critic_dataset import (
    assign_group_splits,
    build_critic_dataset_rows,
    build_dataset_stats,
    build_group_manifest,
    build_label_schema,
    load_g1_dataset,
)


class CriticDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())
        self.dataset_dir = self.tmp_dir / "g1_dataset"
        self._write_fixture_dataset()

    def tearDown(self) -> None:
        shutil.rmtree(_windows_safe_path(self.tmp_dir), ignore_errors=True)

    def _write_json(self, path: Path, payload: object) -> None:
        write_text_file(path, json.dumps(payload, indent=2, ensure_ascii=False))

    def _write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        write_text_file(path, text)

    def _manifest_rows(self) -> list[dict[str, object]]:
        return [
            {
                "run_dir": "run_a",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-13",
                "baseline_name": "ours-eig",
                "topic": "3D language field modeling",
                "local_category_scores": {"benchmark_alignment": 4.5, "graph_process": 8.0},
                "final_local_overall": 6.0,
                "final_local_alignment": 4.5,
                "native_metric_map": {"fps": {"score": 4.0, "max_score": 5.0, "available": True}},
                "final_native_average": 8.0,
                "has_local_eval": True,
                "has_native_eval": True,
            },
            {
                "run_dir": "run_b",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-13",
                "baseline_name": "ours-eig",
                "topic": "3D language field modeling",
                "local_category_scores": {"benchmark_alignment": 5.0, "graph_process": 8.2},
                "final_local_overall": 6.4,
                "final_local_alignment": 5.0,
                "native_metric_map": {"fps": {"score": 4.5, "max_score": 5.0, "available": True}},
                "final_native_average": 8.4,
                "has_local_eval": True,
                "has_native_eval": True,
            },
            {
                "run_dir": "run_c",
                "benchmark": "liveideabench",
                "instance_name": "meteorology-0",
                "baseline_name": "ours-eig",
                "topic": "meteorology",
                "local_category_scores": {"benchmark_alignment": 3.2, "graph_process": 7.6},
                "final_local_overall": 5.2,
                "final_local_alignment": 3.2,
                "native_metric_map": {},
                "final_native_average": None,
                "has_local_eval": True,
                "has_native_eval": False,
            },
        ]

    def _transition_rows(self) -> list[dict[str, object]]:
        return [
            {
                "run_dir": "run_a",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-13",
                "topic": "3D language field modeling",
                "step_index": 0,
                "selected_action_kind": "add_support_edge",
            },
            {
                "run_dir": "run_a",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-13",
                "topic": "3D language field modeling",
                "step_index": 1,
                "selected_action_kind": "propose_repair",
            },
            {
                "run_dir": "run_b",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-13",
                "topic": "3D language field modeling",
                "step_index": 0,
                "selected_action_kind": "attach_evidence",
            },
            {
                "run_dir": "run_c",
                "benchmark": "liveideabench",
                "instance_name": "meteorology-0",
                "topic": "meteorology",
                "step_index": 0,
                "selected_action_kind": "add_support_edge",
            },
        ]

    def _write_fixture_dataset(self) -> None:
        manifest_rows = self._manifest_rows()
        transition_rows = self._transition_rows()
        self._write_jsonl(self.dataset_dir / "run_manifest.jsonl", manifest_rows)
        self._write_jsonl(self.dataset_dir / "trajectory_examples.jsonl", transition_rows)
        self._write_json(
            self.dataset_dir / "dataset_profile.json",
            {
                "run_count": len(manifest_rows),
                "transition_count": len(transition_rows),
            },
        )

    def test_assign_group_splits_is_deterministic_and_group_local(self) -> None:
        manifest_rows, transition_rows = load_g1_dataset(self.dataset_dir)
        group_rows = build_group_manifest(manifest_rows, transition_rows)

        reversed_rows = list(reversed(group_rows))
        first = assign_group_splits(group_rows, validation_fraction=0.2)
        second = assign_group_splits(reversed_rows, validation_fraction=0.2)

        self.assertEqual(first, second)
        self.assertEqual([row["group_id"] for row in first], sorted(row["group_id"] for row in group_rows))
        self.assertTrue(all(row["split"] == "train" for row in first))

    def test_assign_group_splits_includes_validation_when_benchmark_has_three_groups(self) -> None:
        group_rows = [
            {"group_id": "AI_Idea_Bench_2025::a", "benchmark": "AI_Idea_Bench_2025", "instance_name": "a", "run_count": 1},
            {"group_id": "AI_Idea_Bench_2025::b", "benchmark": "AI_Idea_Bench_2025", "instance_name": "b", "run_count": 1},
            {"group_id": "AI_Idea_Bench_2025::c", "benchmark": "AI_Idea_Bench_2025", "instance_name": "c", "run_count": 1},
            {"group_id": "liveideabench::x", "benchmark": "liveideabench", "instance_name": "x", "run_count": 1},
            {"group_id": "liveideabench::y", "benchmark": "liveideabench", "instance_name": "y", "run_count": 1},
        ]
        first = assign_group_splits(group_rows, validation_fraction=0.2)
        second = assign_group_splits(list(reversed(group_rows)), validation_fraction=0.2)
        self.assertEqual(first, second)

        split_map = {row["group_id"]: row["split"] for row in first}
        self.assertEqual(split_map["AI_Idea_Bench_2025::a"], "train")
        self.assertEqual(split_map["AI_Idea_Bench_2025::b"], "train")
        self.assertEqual(split_map["AI_Idea_Bench_2025::c"], "validation")
        self.assertEqual(split_map["liveideabench::x"], "train")
        self.assertEqual(split_map["liveideabench::y"], "train")

    def test_build_rows_preserves_duplicates_and_packaged_labels(self) -> None:
        manifest_rows, transition_rows = load_g1_dataset(self.dataset_dir)
        group_rows = build_group_manifest(manifest_rows, transition_rows)
        split_rows = assign_group_splits(group_rows, validation_fraction=0.2)

        critic_rows = build_critic_dataset_rows(manifest_rows, transition_rows, split_rows)
        self.assertEqual(len(critic_rows), 4)

        ai_rows = [row for row in critic_rows if row["group_id"] == "AI_Idea_Bench_2025::aiib-13"]
        live_rows = [row for row in critic_rows if row["group_id"] == "liveideabench::meteorology-0"]
        self.assertEqual(len(ai_rows), 3)
        self.assertEqual(len(live_rows), 1)
        self.assertTrue(all(row["split"] == "train" for row in ai_rows + live_rows))

        self.assertTrue(all(row["group_run_count"] == 2 for row in ai_rows))
        ai_index_map = {
            (row["run_dir"], int(row["step_index"])): row["group_run_index"]
            for row in ai_rows
        }
        self.assertEqual(ai_index_map[("run_a", 0)], 0)
        self.assertEqual(ai_index_map[("run_a", 1)], 0)
        self.assertEqual(ai_index_map[("run_b", 0)], 1)

        run_a_step0 = next(
            row for row in critic_rows if row["run_dir"] == "run_a" and int(row["step_index"]) == 0
        )
        self.assertEqual(run_a_step0["selected_action_kind"], "add_support_edge")
        self.assertTrue(run_a_step0["weak_local"]["available"])
        self.assertAlmostEqual(run_a_step0["weak_local"]["overall_10"], 6.0)
        self.assertAlmostEqual(run_a_step0["weak_local"]["overall_01"], 0.6)
        self.assertAlmostEqual(run_a_step0["weak_local"]["benchmark_alignment_10"], 4.5)
        self.assertAlmostEqual(run_a_step0["weak_local"]["benchmark_alignment_01"], 0.45)
        self.assertAlmostEqual(run_a_step0["weak_local"]["category_scores"]["graph_process"], 8.0)
        self.assertTrue(run_a_step0["native"]["available"])
        self.assertEqual(run_a_step0["native"]["benchmark"], "AI_Idea_Bench_2025")
        self.assertAlmostEqual(run_a_step0["native"]["average_10"], 8.0)
        self.assertAlmostEqual(run_a_step0["native"]["average_01"], 0.8)
        self.assertAlmostEqual(run_a_step0["native"]["metrics"]["fps"]["score"], 4.0)
        self.assertTrue(run_a_step0["label_availability"]["has_weak_local"])
        self.assertTrue(run_a_step0["label_availability"]["has_native"])
        self.assertTrue(run_a_step0["label_availability"]["has_native_average"])
        self.assertAlmostEqual(run_a_step0["targets"]["weak_value_01"], 0.6)
        self.assertAlmostEqual(run_a_step0["targets"]["native_value_01"], 0.8)

        run_c_step0 = next(
            row for row in critic_rows if row["run_dir"] == "run_c" and int(row["step_index"]) == 0
        )
        self.assertFalse(run_c_step0["native"]["available"])
        self.assertEqual(run_c_step0["native"]["benchmark"], "liveideabench")
        self.assertIsNone(run_c_step0["native"]["average_10"])
        self.assertIsNone(run_c_step0["native"]["average_01"])
        self.assertEqual(run_c_step0["native"]["metrics"], {})
        self.assertTrue(run_c_step0["label_availability"]["has_weak_local"])
        self.assertFalse(run_c_step0["label_availability"]["has_native"])
        self.assertFalse(run_c_step0["label_availability"]["has_native_average"])
        self.assertAlmostEqual(run_c_step0["targets"]["weak_value_01"], 0.52)
        self.assertIsNone(run_c_step0["targets"]["native_value_01"])

    def test_dataset_stats_counts_and_coverage(self) -> None:
        manifest_rows, transition_rows = load_g1_dataset(self.dataset_dir)
        group_rows = build_group_manifest(manifest_rows, transition_rows)
        split_rows = assign_group_splits(group_rows, validation_fraction=0.2)
        critic_rows = build_critic_dataset_rows(manifest_rows, transition_rows, split_rows)

        stats = build_dataset_stats(critic_rows, split_rows)
        schema = build_label_schema()

        self.assertEqual(stats["run_count"], 3)
        self.assertEqual(stats["transition_count"], 4)
        self.assertEqual(stats["group_count"], 2)
        self.assertEqual(stats["train_group_count"], 2)
        self.assertEqual(stats["validation_group_count"], 0)
        self.assertEqual(stats["train_transition_count"], 4)
        self.assertEqual(stats["validation_transition_count"], 0)
        self.assertEqual(stats["benchmark_group_counts"]["AI_Idea_Bench_2025"], 1)
        self.assertEqual(stats["benchmark_group_counts"]["liveideabench"], 1)
        self.assertEqual(stats["benchmark_transition_counts"]["AI_Idea_Bench_2025"], 3)
        self.assertEqual(stats["benchmark_transition_counts"]["liveideabench"], 1)
        self.assertAlmostEqual(stats["label_coverage"]["weak_local_fraction"], 1.0)
        self.assertAlmostEqual(stats["label_coverage"]["native_fraction"], 0.75)
        self.assertAlmostEqual(stats["label_coverage"]["native_average_fraction"], 0.75)
        self.assertAlmostEqual(stats["duplicate_burden"]["mean_runs_per_group"], 1.5)
        self.assertEqual(stats["duplicate_burden"]["max_runs_per_group"], 2)

        self.assertEqual(schema["weak_local"]["available"], "bool")
        self.assertEqual(schema["weak_local"]["overall_10"], "float|null")
        self.assertEqual(schema["weak_local"]["category_scores"], "object")
        self.assertEqual(schema["native"]["average_01"], "float|null")
        self.assertEqual(schema["native"]["metrics"], "object")
        self.assertEqual(schema["label_availability"]["has_native"], "bool")
        self.assertEqual(schema["targets"]["native_value_01"], "float|null")

    def test_build_rows_uses_deterministic_group_run_index_even_if_manifest_order_changes(self) -> None:
        manifest_rows = list(reversed(self._manifest_rows()))
        transition_rows = self._transition_rows()
        group_rows = build_group_manifest(manifest_rows, transition_rows)
        split_rows = assign_group_splits(group_rows, validation_fraction=0.2)
        critic_rows = build_critic_dataset_rows(manifest_rows, transition_rows, split_rows)

        ai_rows = [row for row in critic_rows if row["group_id"] == "AI_Idea_Bench_2025::aiib-13"]
        ai_index_by_run = {row["run_dir"]: row["group_run_index"] for row in ai_rows}
        self.assertEqual(ai_index_by_run["run_a"], 0)
        self.assertEqual(ai_index_by_run["run_b"], 1)

    def test_build_rows_raises_for_missing_manifest_run_dir(self) -> None:
        manifest_rows = self._manifest_rows()
        transition_rows = self._transition_rows() + [
            {
                "run_dir": "run_missing",
                "benchmark": "AI_Idea_Bench_2025",
                "instance_name": "aiib-missing",
                "topic": "missing",
                "step_index": 0,
            }
        ]
        group_rows = build_group_manifest(manifest_rows, transition_rows)
        split_rows = assign_group_splits(group_rows, validation_fraction=0.2)

        with self.assertRaisesRegex(ValueError, "run_dir.*run_missing"):
            build_critic_dataset_rows(manifest_rows, transition_rows, split_rows)

    def test_build_rows_raises_for_missing_group_split(self) -> None:
        manifest_rows = self._manifest_rows()
        transition_rows = self._transition_rows()
        group_rows = build_group_manifest(manifest_rows, transition_rows)
        split_rows = [row for row in assign_group_splits(group_rows, validation_fraction=0.2) if row["group_id"] != "liveideabench::meteorology-0"]

        with self.assertRaisesRegex(ValueError, "group_id.*liveideabench::meteorology-0"):
            build_critic_dataset_rows(manifest_rows, transition_rows, split_rows)


if __name__ == "__main__":
    unittest.main()
