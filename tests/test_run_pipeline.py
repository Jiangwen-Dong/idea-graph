import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pipeline import build_parser


def test_run_pipeline_parser_accepts_runtime_protocol() -> None:
    parser = build_parser()

    args = parser.parse_args(["--runtime-protocol", "parallel_graph_v2"])

    assert args.runtime_protocol == "parallel_graph_v2"
