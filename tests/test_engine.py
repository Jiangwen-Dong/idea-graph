from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.agent_backend import ActionDecision
from idea_graph.engine import run_experiment
from idea_graph.models import FinalProposal, IdeaGraph


class InvalidActionBackend:
    name = "openai-compatible"

    def generate_seed(self, graph: IdeaGraph, role: str):
        raise RuntimeError("seed generation disabled for test")

    def choose_action(self, graph: IdeaGraph, round_name: str, role: str) -> ActionDecision:
        return ActionDecision(
            kind="add_support_edge",
            target_ids=[],
            payload={"branch_id": "B001"},
            rationale="invalid test action",
        )

    def synthesize_final_proposal(self, graph: IdeaGraph, subgraph: dict[str, object]) -> FinalProposal:
        return FinalProposal(
            problem="p",
            hypothesis="h",
            method="m",
            evaluation="e",
            significance="s",
            caveats="c",
        )


class EngineTests(unittest.TestCase):
    def test_invalid_llm_actions_fall_back_without_crashing(self) -> None:
        messages: list[str] = []
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            collaboration_backend=InvalidActionBackend(),
            progress_callback=messages.append,
        )

        self.assertIsNotNone(graph.final_proposal)
        self.assertEqual(len(graph.actions), 15)
        self.assertIn("seed_generation_error", graph.metadata)
        self.assertEqual(len(graph.metadata.get("action_errors", [])), 15)
        self.assertTrue(any("using deterministic fallback" in message for message in messages))

    def test_progress_callback_receives_round_updates(self) -> None:
        messages: list[str] = []
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            progress_callback=messages.append,
        )

        self.assertIsNotNone(graph.final_proposal)
        self.assertTrue(any("Round1 started" in message for message in messages))
        self.assertTrue(any("Run complete" in message for message in messages))
        self.assertIn("progress_log", graph.metadata)


if __name__ == "__main__":
    unittest.main()
