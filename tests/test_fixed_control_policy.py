from __future__ import annotations

import sys
from pathlib import Path
from tempfile import mkdtemp
import json
import shutil
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.fixed_control_policy import FixedControlPolicy, load_fixed_control_policy


class FixedControlPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_prefers_first_configured_utility_kind_for_round(self) -> None:
        policy = FixedControlPolicy(
            ordered_kind_priors={"Round2": ("add_dependency_edge", "attach_evidence")}
        )
        candidates = [
            {"candidate_id": "c0", "kind": "add_dependency_edge", "candidate_source": "parallel_selected"},
            {"candidate_id": "c1", "kind": "attach_evidence", "candidate_source": "utility_attach_evidence"},
            {"candidate_id": "c2", "kind": "add_dependency_edge", "candidate_source": "utility_add_dependency"},
            {"candidate_id": "c3", "kind": "skip", "candidate_source": "parallel_skip"},
        ]

        selected = policy.choose(
            round_name="Round2",
            role="MechanismProposer",
            candidate_specs=candidates,
        )

        self.assertEqual(selected["candidate_id"], "c2")
        self.assertEqual(selected["kind"], "add_dependency_edge")
        self.assertEqual(selected["candidate_source"], "utility_add_dependency")
        self.assertIsNot(selected, candidates[2])

    def test_ignores_role_and_uses_same_round_schedule(self) -> None:
        policy = FixedControlPolicy(ordered_kind_priors={"Round1": ("add_support_edge",)})
        candidates = [
            {"candidate_id": "c0", "kind": "add_support_edge", "candidate_source": "utility_add_support"},
            {"candidate_id": "c1", "kind": "skip", "candidate_source": "parallel_skip"},
        ]

        mechanism_selected = policy.choose(
            round_name="Round1",
            role="MechanismProposer",
            candidate_specs=candidates,
        )
        novelty_selected = policy.choose(
            round_name="Round1",
            role="NoveltyExaminer",
            candidate_specs=candidates,
        )

        self.assertEqual(mechanism_selected["candidate_id"], "c0")
        self.assertEqual(novelty_selected["candidate_id"], "c0")

    def test_falls_back_to_first_available_utility_candidate_when_round_target_missing(self) -> None:
        policy = FixedControlPolicy(ordered_kind_priors={"Round4": ("add_dependency_edge",)})
        candidates = [
            {"candidate_id": "c0", "kind": "attach_evidence", "candidate_source": "utility_attach_evidence"},
            {"candidate_id": "c1", "kind": "add_support_edge", "candidate_source": "utility_add_support"},
            {"candidate_id": "c2", "kind": "skip", "candidate_source": "parallel_skip"},
        ]

        selected = policy.choose(
            round_name="Round4",
            role="NoveltyExaminer",
            candidate_specs=candidates,
        )

        self.assertEqual(selected["candidate_id"], "c0")
        self.assertEqual(selected["kind"], "attach_evidence")
        self.assertEqual(selected["candidate_source"], "utility_attach_evidence")

    def test_falls_back_to_skip_when_no_utility_candidates_exist(self) -> None:
        policy = FixedControlPolicy(ordered_kind_priors={"Round3": ("propose_repair",)})
        candidates = [
            {"candidate_id": "c0", "kind": "freeze_branch", "candidate_source": "parallel_selected"},
            {"candidate_id": "c1", "kind": "skip", "candidate_source": "parallel_skip"},
        ]

        selected = policy.choose(
            round_name="Round3",
            role="EvaluationDesigner",
            candidate_specs=candidates,
        )

        self.assertEqual(selected["candidate_id"], "c1")
        self.assertEqual(selected["kind"], "skip")

    def test_loader_reads_round_level_json_shape(self) -> None:
        config_path = self.tmp_dir / "fixed_policy.json"
        config_path.write_text(
            json.dumps(
                {
                    "Round2": ["add_dependency_edge", "attach_evidence"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        policy = load_fixed_control_policy(config_path)
        selected = policy.choose(
            round_name="Round2",
            role="MechanismProposer",
            candidate_specs=[
                {"candidate_id": "c0", "kind": "attach_evidence", "candidate_source": "utility_attach_evidence"},
                {"candidate_id": "c1", "kind": "add_dependency_edge", "candidate_source": "utility_add_dependency"},
            ],
        )
        self.assertEqual(selected["candidate_id"], "c1")


if __name__ == "__main__":
    unittest.main()
