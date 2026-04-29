import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.agent_backend import ActionDecision
from idea_graph.engine import run_experiment
from idea_graph.models import Branch, IdeaGraph, Node
from idea_graph.relation_graph_runtime_critic import RelationGraphRuntimeConfig
from idea_graph.role_activation import active_roles_for_round
from idea_graph.runtime_protocols import ROLE_ORDER_PRESETS
from idea_graph.sequential_runtime import execute_sequential_role_round


def _build_round2_eval_graph() -> IdeaGraph:
    graph = IdeaGraph(
        topic="topic",
        literature=["paper a"],
        metadata={
            "runtime_protocol": "sequential_graph_v2",
            "runtime_role_order_id": "order_a_canonical",
            "runtime_role_order": list(ROLE_ORDER_PRESETS["order_a_canonical"]),
        },
    )
    graph.branches = {
        "B001": Branch(id="B001", role="MechanismProposer"),
        "B002": Branch(id="B002", role="EvaluationDesigner"),
        "B003": Branch(id="B003", role="NoveltyExaminer"),
        "B004": Branch(id="B004", role="ImpactReframer"),
    }
    graph.nodes = {
        "N001": Node(
            id="N001",
            type="Problem",
            text="Problem node.",
            role="ImpactReframer",
            branch_id="B004",
            confidence=0.8,
            evidence=["problem evidence"],
        ),
        "N002": Node(
            id="N002",
            type="Hypothesis",
            text="Hypothesis node.",
            role="MechanismProposer",
            branch_id="B001",
            confidence=0.8,
            evidence=["hypothesis evidence"],
        ),
        "N003": Node(
            id="N003",
            type="Method",
            text="Method node.",
            role="MechanismProposer",
            branch_id="B001",
            confidence=0.8,
            evidence=["method evidence"],
        ),
        "N004": Node(
            id="N004",
            type="NoveltyClaim",
            text="Novelty claim node.",
            role="NoveltyExaminer",
            branch_id="B003",
            confidence=0.8,
            evidence=["novelty evidence"],
        ),
        "N005": Node(
            id="N005",
            type="EvalPlan",
            text="Eval plan node.",
            role="EvaluationDesigner",
            branch_id="B002",
            confidence=0.8,
        ),
    }
    return graph


def test_sequential_runtime_freezes_active_role_set_at_round_start() -> None:
    class FrozenActiveSetBackend:
        def __init__(self) -> None:
            self.calls = []

        def choose_action(self, graph, round_name, role):
            del round_name
            self.calls.append(role)
            branch_id = next(branch.id for branch in graph.branches.values() if branch.role == role)
            if role == "MechanismProposer":
                return ActionDecision(
                    kind="attach_evidence",
                    target_ids=["N005"],
                    payload={"branch_id": branch_id, "evidence": "mechanism supplied evidence"},
                    rationale="Ground the evaluation plan.",
                )
            return ActionDecision(
                kind="skip",
                target_ids=[],
                payload={"branch_id": branch_id},
                rationale="No further edit.",
            )

    graph = _build_round2_eval_graph()
    backend = FrozenActiveSetBackend()

    assert active_roles_for_round(graph, "Round2") == ("MechanismProposer", "EvaluationDesigner")

    result = execute_sequential_role_round(
        graph,
        round_name="Round2",
        collaboration_backend=backend,
        runtime_controller=None,
        runtime_controller_metadata=None,
        progress_callback=None,
    )

    assert result.active_roles == ("MechanismProposer", "EvaluationDesigner")
    assert backend.calls == ["MechanismProposer", "EvaluationDesigner"]
    assert active_roles_for_round(graph, "Round2") == ("MechanismProposer", "NoveltyExaminer")


def test_sequential_runtime_later_roles_see_earlier_same_round_mutations() -> None:
    class VisibilityBackend:
        def __init__(self) -> None:
            self.observed_eval_evidence = {}

        def choose_action(self, graph, round_name, role):
            del round_name
            eval_node = graph.nodes["N005"]
            self.observed_eval_evidence[role] = list(eval_node.evidence)
            branch_id = next(branch.id for branch in graph.branches.values() if branch.role == role)
            if role == "MechanismProposer":
                return ActionDecision(
                    kind="attach_evidence",
                    target_ids=["N005"],
                    payload={"branch_id": branch_id, "evidence": "mechanism supplied evidence"},
                    rationale="Ground the evaluation plan.",
                )
            return ActionDecision(
                kind="skip",
                target_ids=[],
                payload={"branch_id": branch_id},
                rationale="No further edit.",
            )

    graph = _build_round2_eval_graph()
    backend = VisibilityBackend()

    execute_sequential_role_round(
        graph,
        round_name="Round2",
        collaboration_backend=backend,
        runtime_controller=None,
        runtime_controller_metadata=None,
        progress_callback=None,
    )

    assert backend.observed_eval_evidence["MechanismProposer"] == []
    assert backend.observed_eval_evidence["EvaluationDesigner"] == ["mechanism supplied evidence"]


def test_sequential_runtime_runs_post_round_commit_once() -> None:
    class SkipBackend:
        def choose_action(self, graph, round_name, role):
            del graph, round_name
            branch_id = "B001" if role == "MechanismProposer" else "B002"
            return ActionDecision(
                kind="skip",
                target_ids=[],
                payload={"branch_id": branch_id},
                rationale="Skip for commit test.",
            )

    class CommitCountingController:
        def __init__(self) -> None:
            self.commit_calls = 0

        def score_commit_graph(self, graph, *, snapshot=None):
            del graph, snapshot
            self.commit_calls += 1
            return 0.0

    graph = _build_round2_eval_graph()
    controller = CommitCountingController()

    result = execute_sequential_role_round(
        graph,
        round_name="Round2",
        collaboration_backend=SkipBackend(),
        runtime_controller=controller,
        runtime_controller_metadata={
            "kind": "relation_graph_two_head_critic",
            "config": RelationGraphRuntimeConfig(
                use_edit=False,
                use_commit=True,
                min_commit_round=1,
                gamma_commit=0.50,
            ),
        },
        progress_callback=None,
    )

    assert controller.commit_calls == 1
    assert result.post_round_commit.round_name == "Round2"
    assert result.post_round_commit.state_kind == "sequential_post_round"


def test_sequential_runtime_run_experiment_emits_protocol_and_role_order_metadata() -> None:
    graph = run_experiment(
        topic="topic",
        literature=["paper a"],
        metadata={
            "runtime_protocol": "sequential_graph_v2",
            "runtime_role_order_id": "order_b_reverse",
            "runtime_role_order": list(ROLE_ORDER_PRESETS["order_b_reverse"]),
        },
        collaboration_backend=None,
        runtime_controller=None,
        runtime_controller_metadata=None,
        progress_callback=None,
        max_rounds=1,
        stop_when_mature=False,
    )

    expected_sequence = list(ROLE_ORDER_PRESETS["order_b_reverse"])
    traces = graph.metadata.get("parallel_round_traces")
    edit_rows = graph.metadata.get("parallel_edit_rows")
    post_rows = graph.metadata.get("post_round_commit_rows")

    assert isinstance(traces, list)
    assert isinstance(edit_rows, list)
    assert isinstance(post_rows, list)
    assert traces
    assert edit_rows
    assert post_rows

    trace = traces[0]
    assert trace["runtime_protocol"] == "sequential_graph_v2"
    assert trace["runtime_role_order_id"] == "order_b_reverse"
    assert trace["resolved_role_sequence"] == expected_sequence
    assert trace["active_roles"] == expected_sequence

    assert [row["role"] for row in edit_rows] == expected_sequence
    assert all(row["runtime_protocol"] == "sequential_graph_v2" for row in edit_rows)
    assert all(row["runtime_role_order_id"] == "order_b_reverse" for row in edit_rows)
    assert all(row["resolved_role_sequence"] == expected_sequence for row in edit_rows)
    assert all(row["state_kind"] == "sequential_pre_action" for row in edit_rows)

    post_row = post_rows[0]
    assert post_row["runtime_protocol"] == "sequential_graph_v2"
    assert post_row["runtime_role_order_id"] == "order_b_reverse"
    assert post_row["resolved_role_sequence"] == expected_sequence
    assert post_row["state_kind"] == "sequential_post_round"


def test_sequential_runtime_defaults_missing_role_order_metadata_for_direct_callers() -> None:
    graph = _build_round2_eval_graph()
    graph.metadata = {
        "runtime_protocol": "sequential_graph_v2",
        "runtime_role_order": list(ROLE_ORDER_PRESETS["order_b_reverse"]),
    }

    result = execute_sequential_role_round(
        graph,
        round_name="Round2",
        collaboration_backend=None,
        runtime_controller=None,
        runtime_controller_metadata=None,
        progress_callback=None,
    )

    assert result.active_roles == ("MechanismProposer", "EvaluationDesigner")
    assert graph.metadata["runtime_role_order_id"] == "order_a_canonical"
    assert graph.metadata["runtime_role_order"] == list(ROLE_ORDER_PRESETS["order_a_canonical"])
    assert all(row["runtime_role_order_id"] == "order_a_canonical" for row in result.edit_rows)
    assert all(
        row["resolved_role_sequence"] == ["MechanismProposer", "EvaluationDesigner"]
        for row in result.edit_rows
    )
    assert result.post_round_commit_rows[0]["runtime_role_order_id"] == "order_a_canonical"


def test_sequential_runtime_invalid_explicit_role_order_id_raises_value_error() -> None:
    graph = _build_round2_eval_graph()
    graph.metadata = {
        "runtime_protocol": "sequential_graph_v2",
        "runtime_role_order_id": "not_a_real_order",
        "runtime_role_order": list(ROLE_ORDER_PRESETS["order_a_canonical"]),
    }

    with pytest.raises(ValueError, match="Unknown role order 'not_a_real_order'"):
        execute_sequential_role_round(
            graph,
            round_name="Round2",
            collaboration_backend=None,
            runtime_controller=None,
            runtime_controller_metadata=None,
            progress_callback=None,
        )


def test_sequential_controller_trace_candidate_ids_align_with_replay_rows() -> None:
    class FreezeBackend:
        name = "freeze-backend"

        def choose_action(self, graph, round_name, role):
            del graph, round_name
            branch_id = "B001" if role == "MechanismProposer" else "B002"
            return ActionDecision(
                kind="freeze_branch",
                target_ids=[],
                payload={"branch_id": branch_id},
                rationale="backend selected freeze",
            )

    class SkipPolicyController:
        def choose(self, *, round_name, role, candidate_specs):
            del round_name, role
            return next(spec for spec in candidate_specs if str(spec.get("kind")) == "skip")

    graph = _build_round2_eval_graph()
    controller = SkipPolicyController()

    result = execute_sequential_role_round(
        graph,
        round_name="Round2",
        collaboration_backend=FreezeBackend(),
        runtime_controller=controller,
        runtime_controller_metadata={
            "kind": "fixed_control",
            "config": SimpleNamespace(use_edit=True, use_commit=False),
        },
        progress_callback=None,
    )

    assert all(row["state_id"].startswith("sequential::") for row in result.edit_rows)
    assert graph.metadata["runtime_controller_log"]
    assert all(
        str(entry["heuristic_candidate_id"]).startswith("sequential::")
        and str(entry["selected_candidate_id"]).startswith("sequential::")
        for entry in graph.metadata["runtime_controller_log"]
    )
    replay_candidate_ids = {
        candidate["candidate_id"]
        for row in result.edit_rows
        for candidate in row["candidates"]
    }
    assert all(
        entry["heuristic_candidate_id"] in replay_candidate_ids
        and entry["selected_candidate_id"] in replay_candidate_ids
        for entry in graph.metadata["runtime_controller_log"]
    )
