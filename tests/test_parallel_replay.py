import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.parallel_replay import append_parallel_round_trace


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
