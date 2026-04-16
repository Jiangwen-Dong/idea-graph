from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.fs_utils import _windows_safe_path, read_text_file
from idea_graph.instances import ExperimentInstance
from idea_graph.io import write_run_artifacts
from idea_graph.models import FinalProposal, IdeaGraph


class WriteRunArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(_windows_safe_path(self.tmp_dir), ignore_errors=True)

    def test_write_run_artifacts_handles_long_windows_paths(self) -> None:
        output_root = self.tmp_dir
        while len(str(output_root)) < 170:
            output_root = output_root / ("deep-output-root-segment-" * 4).strip("-")

        instance = ExperimentInstance(
            name="liveideabench periodic table 23 " * 4,
            topic="Ideation topic keyword: periodic table",
            literature=["Periodic relations can improve element-property prediction."],
            source_path="tests/fixtures/periodic.json",
        )
        graph = IdeaGraph(topic=instance.topic, literature=list(instance.literature), metadata={})
        graph.final_proposal = FinalProposal(
            title="Periodic-Relation Modeling for Element Property Prediction",
            problem="Property prediction underuses periodic structure.",
            existing_methods="Standard predictors do not explicitly encode periodic relations.",
            motivation="Periodic structure offers a natural inductive bias.",
            hypothesis="Relation-aware periodic modeling improves generalization.",
            method="Build a periodic-relation graph encoder for element property prediction.",
            evaluation="Compare against strong graph baselines on benchmark datasets.",
            significance="Better periodic generalization could improve material discovery.",
            caveats="Benefits may depend on benchmark coverage.",
        )

        run_dir = write_run_artifacts(
            graph,
            output_root=output_root,
            instance=instance,
        )

        summary_path = run_dir / "summary.json"
        self.assertIn("Periodic-Relation Modeling", read_text_file(summary_path))
        self.assertGreater(len(str(run_dir)), 260)


if __name__ == "__main__":
    unittest.main()
