import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.agent_backend import ActionDecision
from idea_graph.engine import build_seed_graphs, merge_seed_graphs
from idea_graph.models import IdeaGraph
from idea_graph.parallel_replay import (
    append_parallel_edit_rows,
    append_parallel_round_trace,
    build_parallel_edit_rows,
)


def test_append_parallel_round_trace_persists_round_payload_in_metadata() -> None:
    metadata = {}
    append_parallel_round_trace(
        metadata,
        {
            "round": "Round1",
            "active_roles": ["MechanismProposer"],
            "inactive_roles": ["EvaluationDesigner"],
            "selected_actions": [],
            "skipped_roles": ["MechanismProposer"],
            "graph_delta": {
                "node_count_before": 5,
                "node_count_after": 6,
            },
        },
    )

    traces = metadata.get("parallel_round_traces")
    assert isinstance(traces, list)
    assert traces[0]["round"] == "Round1"
    assert traces[0]["graph_delta"]["node_count_after"] == 6


def test_build_parallel_edit_rows_emits_skip_candidates_and_single_selection() -> None:
    graph = IdeaGraph(topic="topic", literature=["paper a"])
    build_seed_graphs(graph)
    merge_seed_graphs(graph)

    mechanism_branch_id = next(branch.id for branch in graph.branches.values() if branch.role == "MechanismProposer")
    evaluation_branch_id = next(branch.id for branch in graph.branches.values() if branch.role == "EvaluationDesigner")
    rows = build_parallel_edit_rows(
        graph,
        round_name="Round1",
        role_decisions=(
            (
                "MechanismProposer",
                ActionDecision(
                    kind="freeze_branch",
                    target_ids=[],
                    payload={"branch_id": mechanism_branch_id},
                    rationale="Teacher freezes the branch this round.",
                ),
            ),
            (
                "EvaluationDesigner",
                ActionDecision(
                    kind="skip",
                    target_ids=[],
                    payload={"branch_id": evaluation_branch_id},
                    rationale="Teacher skips editing this round.",
                ),
            ),
        ),
        runtime_protocol="parallel_graph_v2",
        label_source="parallel_protocol_teacher_v1",
    )

    assert len(rows) == 2
    evaluation_row = next(row for row in rows if row["role"] == "EvaluationDesigner")
    assert evaluation_row["schema_version"] == "parallel_edit_row_v1"
    assert evaluation_row["runtime_protocol"] == "parallel_graph_v2"
    assert evaluation_row["label_source"] == "parallel_protocol_teacher_v1"
    assert not any(candidate["candidate_kind"] == "commit" for candidate in evaluation_row["candidates"])
    assert any(candidate["candidate_kind"] == "skip" for candidate in evaluation_row["candidates"])
    assert sum(1 for candidate in evaluation_row["candidates"] if bool(candidate["is_selected"])) == 1
    selected = next(candidate for candidate in evaluation_row["candidates"] if bool(candidate["is_selected"]))
    assert selected["candidate_kind"] == "skip"
    assert evaluation_row["selected_candidate_id"] == selected["candidate_id"]


def test_append_parallel_edit_rows_extends_metadata_rows() -> None:
    metadata = {}
    append_parallel_edit_rows(
        metadata,
        [
            {
                "state_id": "parallel::Round1::MechanismProposer",
                "role": "MechanismProposer",
            }
        ],
    )
    append_parallel_edit_rows(
        metadata,
        [
            {
                "state_id": "parallel::Round1::EvaluationDesigner",
                "role": "EvaluationDesigner",
            }
        ],
    )

    rows = metadata.get("parallel_edit_rows")
    assert isinstance(rows, list)
    assert [row["role"] for row in rows] == ["MechanismProposer", "EvaluationDesigner"]
