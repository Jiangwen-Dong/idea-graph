import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_pipeline
from scripts.run_pipeline import build_parser


def test_run_pipeline_parser_accepts_runtime_protocol() -> None:
    parser = build_parser()

    args = parser.parse_args(["--runtime-protocol", "parallel_graph_v2"])

    assert args.runtime_protocol == "parallel_graph_v2"


def test_run_pipeline_parser_does_not_default_to_tracked_sample_file() -> None:
    parser = build_parser()

    args = parser.parse_args([])

    assert args.input is None


def test_run_pipeline_requires_input_when_no_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = build_parser()
    args = parser.parse_args([])

    monkeypatch.setattr(run_pipeline.argparse.ArgumentParser, "parse_args", lambda self: args)

    with pytest.raises(SystemExit, match="Use --input <instance.json> or select --benchmark"):
        run_pipeline.main()
