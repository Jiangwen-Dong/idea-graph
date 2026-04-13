from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.critic_replay import (
    CriticReplayBuffer,
    OnlineCriticEpisode,
    build_mixed_training_rows,
)


class CriticReplayTests(unittest.TestCase):
    def _make_transition(self, state_id: str, *, candidate_id: str) -> dict[str, object]:
        return {
            "state_id": state_id,
            "candidate_id": candidate_id,
            "selected_candidate_id": candidate_id,
            "heuristic_candidate_id": f"{candidate_id}::heuristic",
            "round_index": 2,
            "state_text": f"{state_id} state text",
            "candidate_text": f"{candidate_id} candidate text",
            "is_logged_selected": True,
            "is_commit": False,
            "is_commit_positive_state": False,
            "targets": {"weak_value_01": 0.7, "native_value_01": 0.8},
        }

    def _make_episode(self, *, partition_role: str, group_id: str = "g-train") -> OnlineCriticEpisode:
        return OnlineCriticEpisode(
            episode_id=f"episode::{group_id}",
            group_id=group_id,
            partition_role=partition_role,
            benchmark="AI_Idea_Bench_2025",
            run_dir=f"outputs/{group_id}",
            final_return=0.73,
            final_native_return=0.81,
            transitions=(
                self._make_transition("state-0", candidate_id="state-0::0"),
                self._make_transition("state-1", candidate_id="state-1::0"),
            ),
        )

    def test_append_episode_rejects_non_train_partition(self) -> None:
        buffer = CriticReplayBuffer()
        with self.assertRaisesRegex(ValueError, "critic_train"):
            buffer.append_episode(self._make_episode(partition_role="critic_dev"))

    def test_append_episode_exports_online_training_rows(self) -> None:
        buffer = CriticReplayBuffer()
        buffer.append_episode(self._make_episode(partition_role="critic_train"))
        rows = buffer.training_rows()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["source"] == "online" for row in rows))
        self.assertEqual({row["partition_role"] for row in rows}, {"critic_train"})
        self.assertEqual({row["group_id"] for row in rows}, {"g-train"})

    def test_episode_training_rows_preserve_candidate_text_fields(self) -> None:
        buffer = CriticReplayBuffer()
        buffer.append_episode(self._make_episode(partition_role="critic_train"))
        rows = buffer.training_rows()
        self.assertTrue(rows[0]["state_text"])
        self.assertTrue(rows[0]["candidate_text"])
        self.assertIn("is_logged_selected", rows[0])
        self.assertIn("is_commit", rows[0])

    def test_append_episode_rejects_rows_missing_candidate_slate_fields(self) -> None:
        episode = self._make_episode(partition_role="critic_train")
        broken_transition = dict(episode.transitions[0])
        broken_transition.pop("candidate_text")
        broken_episode = OnlineCriticEpisode(
            episode_id=episode.episode_id,
            group_id=episode.group_id,
            partition_role=episode.partition_role,
            benchmark=episode.benchmark,
            run_dir=episode.run_dir,
            final_return=episode.final_return,
            final_native_return=episode.final_native_return,
            transitions=(broken_transition,),
        )
        buffer = CriticReplayBuffer()
        with self.assertRaisesRegex(ValueError, "candidate_text"):
            buffer.append_episode(broken_episode)

    def test_build_mixed_training_rows_respects_ratio_and_filters_non_train_rows(self) -> None:
        offline_rows = [
            {"row_id": f"offline-{index}", "partition_role": "critic_train", "source": "offline"}
            for index in range(6)
        ]
        online_rows = [
            {"row_id": f"online-{index}", "partition_role": "critic_train", "source": "online"}
            for index in range(4)
        ]
        online_rows.append(
            {"row_id": "online-dev", "partition_role": "critic_dev", "source": "online"}
        )
        mixed = build_mixed_training_rows(
            offline_rows,
            online_rows,
            offline_fraction=0.6,
            max_examples=10,
            random_seed=0,
        )
        self.assertEqual(len(mixed), 10)
        self.assertEqual(sum(1 for row in mixed if row["source"] == "offline"), 6)
        self.assertEqual(sum(1 for row in mixed if row["source"] == "online"), 4)
        self.assertEqual({row["partition_role"] for row in mixed}, {"critic_train"})


if __name__ == "__main__":
    unittest.main()
