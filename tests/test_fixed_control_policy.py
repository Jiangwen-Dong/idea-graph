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

    def test_prefers_first_configured_kind_for_round_and_role(self) -> None:
        policy = FixedControlPolicy(
            ordered_kind_priors={
                ("Round2", "MechanismProposer"): ("attach_evidence", "add_support_edge"),
            }
        )
        candidates = [
            {"candidate_id": "c0", "kind": "add_support_edge"},
            {"candidate_id": "c1", "kind": "attach_evidence"},
            {"candidate_id": "c2", "kind": "skip"},
        ]

        selected = policy.choose(
            round_name="Round2",
            role="MechanismProposer",
            candidate_specs=candidates,
        )

        self.assertEqual(selected["candidate_id"], "c1")
        self.assertEqual(selected["kind"], "attach_evidence")
        self.assertIsNot(selected, candidates[1])

    def test_falls_back_to_skip_when_no_prior_kind_is_present(self) -> None:
        policy = FixedControlPolicy(
            ordered_kind_priors={("Round3", "EvaluationDesigner"): ("request_evidence",)}
        )
        candidates = [
            {"candidate_id": "c0", "kind": "add_support_edge"},
            {"candidate_id": "c1", "kind": "skip"},
        ]

        selected = policy.choose(
            round_name="Round3",
            role="EvaluationDesigner",
            candidate_specs=candidates,
        )

        self.assertEqual(selected["candidate_id"], "c1")
        self.assertEqual(selected["kind"], "skip")

    def test_falls_back_to_first_candidate_when_no_prior_or_skip_match(self) -> None:
        policy = FixedControlPolicy(
            ordered_kind_priors={("Round4", "NoveltyExaminer"): ("request_evidence", "commit")}
        )
        candidates = [
            {"candidate_id": "c0", "kind": "add_support_edge"},
            {"candidate_id": "c1", "kind": "attach_evidence"},
        ]

        selected = policy.choose(
            round_name="Round4",
            role="NoveltyExaminer",
            candidate_specs=candidates,
        )

        self.assertEqual(selected["candidate_id"], "c0")
        self.assertEqual(selected["kind"], "add_support_edge")

    def test_loader_reads_nested_json_shape(self) -> None:
        config_path = self.tmp_dir / "fixed_policy.json"
        config_path.write_text(
            json.dumps(
                {
                    "Round2": {
                        "MechanismProposer": ["attach_evidence", "add_support_edge"],
                    }
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
                {"candidate_id": "c0", "kind": "add_support_edge"},
                {"candidate_id": "c1", "kind": "attach_evidence"},
            ],
        )
        self.assertEqual(selected["candidate_id"], "c1")


if __name__ == "__main__":
    unittest.main()
