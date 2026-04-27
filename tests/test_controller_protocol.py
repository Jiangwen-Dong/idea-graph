import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.controller_protocol import (
    ACTIVE_EDIT_ACTIONS,
    CONTROLLER_ACTIONS,
    SIGNAL_NAMES,
    allowed_actions_for_role,
    is_active_edit_action,
    is_controller_action,
)


def test_active_protocol_actions_and_signals_are_submit_ready() -> None:
    assert set(ACTIVE_EDIT_ACTIONS) == {
        "add_support_edge",
        "attach_evidence",
        "add_dependency_edge",
        "add_contradiction_edge",
        "propose_repair",
        "skip",
    }
    assert set(CONTROLLER_ACTIONS) == {"commit"}
    assert set(SIGNAL_NAMES) == {
        "grounding",
        "contradiction_load",
        "completeness",
        "maturity",
    }

    assert is_active_edit_action("add_support_edge") is True
    assert is_active_edit_action("freeze_branch") is False
    assert is_active_edit_action("request_evidence") is False
    assert is_active_edit_action("mark_overlap") is False
    assert is_controller_action("commit") is True


def test_role_action_permissions_match_the_streamlined_protocol() -> None:
    for role in (
        "ImpactReframer",
        "MechanismProposer",
        "FeasibilityCritic",
        "NoveltyExaminer",
        "EvaluationDesigner",
    ):
        allowed = set(allowed_actions_for_role(role))
        assert allowed
        assert "skip" in allowed
        assert "commit" not in allowed

    assert "add_contradiction_edge" in allowed_actions_for_role("FeasibilityCritic")
    assert "add_contradiction_edge" in allowed_actions_for_role("NoveltyExaminer")
    assert "add_dependency_edge" in allowed_actions_for_role("EvaluationDesigner")
    assert "add_dependency_edge" in allowed_actions_for_role("MechanismProposer")
    assert "attach_evidence" in allowed_actions_for_role("ImpactReframer")
