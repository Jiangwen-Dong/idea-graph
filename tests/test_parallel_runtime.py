import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.models import ParallelRoleRoundResult


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
