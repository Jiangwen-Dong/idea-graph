from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.commit_label_repair import (
    OutcomeGroundedCommitConfig,
    relabel_scored_commit_rows,
)


class CommitLabelRepairTests(unittest.TestCase):
    def test_relabel_scored_commit_rows_uses_future_gain_and_structure_filters(self) -> None:
        rows = [
            {
                "run_dir": "run-a",
                "state_id": "run-a::round1",
                "post_round_state_index": 0,
                "commit_supervision": {"available": True, "label": 0, "source": "maturity_snapshot"},
                "support_coverage": 0.40,
                "unresolved_contradiction_ratio": 0.30,
                "utility": 5.8,
                "outcome_local_overall": 5.2,
            },
            {
                "run_dir": "run-a",
                "state_id": "run-a::round2",
                "post_round_state_index": 1,
                "commit_supervision": {"available": True, "label": 0, "source": "maturity_snapshot"},
                "support_coverage": 0.86,
                "unresolved_contradiction_ratio": 0.0,
                "utility": 7.6,
                "outcome_local_overall": 6.82,
            },
            {
                "run_dir": "run-a",
                "state_id": "run-a::round3",
                "post_round_state_index": 2,
                "commit_supervision": {"available": True, "label": 0, "source": "maturity_snapshot"},
                "support_coverage": 0.84,
                "unresolved_contradiction_ratio": 0.0,
                "utility": 7.8,
                "outcome_local_overall": 6.90,
            },
            {
                "run_dir": "run-b",
                "state_id": "run-b::round1",
                "post_round_state_index": 0,
                "commit_supervision": {"available": True, "label": 1, "source": "maturity_snapshot"},
                "support_coverage": 0.42,
                "unresolved_contradiction_ratio": 0.25,
                "utility": 6.2,
                "outcome_local_overall": 6.75,
            },
            {
                "run_dir": "run-b",
                "state_id": "run-b::round2",
                "post_round_state_index": 1,
                "commit_supervision": {"available": True, "label": 1, "source": "maturity_snapshot"},
                "support_coverage": 0.44,
                "unresolved_contradiction_ratio": 0.20,
                "utility": 6.3,
                "outcome_local_overall": 6.88,
            },
        ]

        repaired_rows, audit = relabel_scored_commit_rows(
            rows,
            config=OutcomeGroundedCommitConfig(
                commit_margin=0.15,
                continue_margin=0.35,
                positive_support_threshold=0.75,
                positive_utility_threshold=7.0,
                positive_unresolved_threshold=0.05,
                minimum_positive_signals=2,
            ),
        )

        repaired_by_state = {row["state_id"]: row for row in repaired_rows}
        self.assertEqual(repaired_by_state["run-a::round1"]["commit_supervision"]["label"], 0)
        self.assertTrue(repaired_by_state["run-a::round1"]["commit_supervision"]["available"])
        self.assertEqual(repaired_by_state["run-a::round2"]["commit_supervision"]["label"], 1)
        self.assertTrue(repaired_by_state["run-a::round2"]["commit_supervision"]["available"])
        self.assertEqual(repaired_by_state["run-a::round3"]["commit_supervision"]["label"], 1)
        self.assertTrue(repaired_by_state["run-a::round3"]["commit_supervision"]["available"])
        self.assertFalse(repaired_by_state["run-b::round1"]["commit_supervision"]["available"])
        self.assertFalse(repaired_by_state["run-b::round2"]["commit_supervision"]["available"])
        self.assertEqual(audit["repaired_positive_count"], 2)
        self.assertEqual(audit["repaired_continue_count"], 1)
        self.assertEqual(audit["ambiguous_drop_count"], 2)
        self.assertEqual(audit["flipped_negative_to_positive_count"], 2)


if __name__ == "__main__":
    unittest.main()
