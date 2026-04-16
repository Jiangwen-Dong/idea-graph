import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.agent_backend import ActionDecision
from idea_graph.engine import build_seed_graphs, merge_seed_graphs
from idea_graph.models import IdeaGraph
from idea_graph.models import ParallelRoleRoundResult
from idea_graph.parallel_runtime import execute_parallel_role_round


def test_parallel_round_result_tracks_selected_actions_and_skips() -> None:
    result = ParallelRoleRoundResult(
        round_name="Round1",
        active_roles=("MechanismProposer", "EvaluationDesigner"),
        skipped_roles=("EvaluationDesigner",),
        selected_actions=(),
        termination_reason="continue",
    )

    assert result.round_name == "Round1"
    assert result.skipped_roles == ("EvaluationDesigner",)
    assert result.termination_reason == "continue"


def test_parallel_role_round_uses_frozen_snapshot_for_all_roles() -> None:
    observed_node_counts = []

    class FrozenSnapshotBackend:
        name = "frozen-snapshot-test"

        def generate_seed(self, graph, role):
            raise RuntimeError("seed generation not used")

        def choose_action(self, graph, round_name, role):
            observed_node_counts.append((role, len(graph.active_nodes())))
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": next(branch.id for branch in graph.branches.values() if branch.role == role)},
                rationale="freeze for snapshot test",
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    execute_parallel_role_round(
        graph,
        round_name="Round1",
        collaboration_backend=FrozenSnapshotBackend(),
        runtime_controller=None,
        runtime_controller_metadata=None,
        progress_callback=None,
    )

    counts = {count for _, count in observed_node_counts}
    assert len(counts) == 1
