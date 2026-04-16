import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.agent_backend import ActionDecision
from idea_graph.engine import build_seed_graphs, merge_seed_graphs
from idea_graph.models import IdeaGraph
from idea_graph.models import (
    ParallelCommitCheckRecord,
    ParallelEditPatchRecord,
    ParallelRoleDecisionRecord,
    ParallelRoleRoundResult,
)
from idea_graph.parallel_runtime import execute_parallel_role_round


def test_parallel_round_result_tracks_decisions_patches_and_commit_check() -> None:
    result = ParallelRoleRoundResult(
        round_name="Round1",
        active_roles=("MechanismProposer", "EvaluationDesigner"),
        skipped_roles=("EvaluationDesigner",),
        selected_role_decisions=(
            ParallelRoleDecisionRecord(
                role="MechanismProposer",
                kind="freeze_branch",
                target_ids=(),
                payload={"branch_id": "B001"},
                rationale="freeze",
            ),
        ),
        edit_patches=(
            ParallelEditPatchRecord(
                role="MechanismProposer",
                kind="freeze_branch",
                target_ids=(),
                payload={"branch_id": "B001"},
                is_empty=False,
            ),
        ),
        materialized_graph_actions=(),
        post_round_commit=ParallelCommitCheckRecord(
            round_name="Round1",
            state_kind="parallel_post_round",
            should_commit=False,
            source="maturity_snapshot",
        ),
    )

    assert result.round_name == "Round1"
    assert result.skipped_roles == ("EvaluationDesigner",)
    assert result.selected_role_decisions[0].role == "MechanismProposer"
    assert result.edit_patches[0].is_empty is False
    assert result.post_round_commit.should_commit is False


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


def test_parallel_round_materializes_non_skip_actions_in_role_order() -> None:
    class OrderedBackend:
        name = "ordered-backend"

        def generate_seed(self, graph, role):
            raise RuntimeError("not used")

        def choose_action(self, graph, round_name, role):
            branch_id = next(branch.id for branch in graph.branches.values() if branch.role == role)
            if role == "EvaluationDesigner":
                return ActionDecision(
                    kind="skip",
                    target_ids=[],
                    payload={"branch_id": branch_id},
                    rationale="skip",
                )
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": branch_id},
                rationale=role,
            )

        def synthesize_final_proposal(self, graph, subgraph):
            raise RuntimeError("not used")

    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    result = execute_parallel_role_round(
        graph,
        round_name="Round1",
        collaboration_backend=OrderedBackend(),
        runtime_controller=None,
        runtime_controller_metadata=None,
        progress_callback=None,
    )

    assert "EvaluationDesigner" in result.skipped_roles
    assert [decision.role for decision in result.selected_role_decisions] == sorted(
        decision.role for decision in result.selected_role_decisions
    )
    assert all(patch.is_empty for patch in result.edit_patches if patch.role == "EvaluationDesigner")
    assert [action.role for action in result.materialized_graph_actions] == sorted(
        action.role for action in result.materialized_graph_actions
    )
    assert result.post_round_commit.round_name == "Round1"
