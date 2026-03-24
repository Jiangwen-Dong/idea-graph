from __future__ import annotations

import shutil
import sys
from pathlib import Path
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.evaluation import evaluate_graph
from idea_graph.engine import run_experiment
from idea_graph.instances import ExperimentInstance
from idea_graph.io import write_run_artifacts


class EvaluationTests(unittest.TestCase):
    def _benchmark_metadata(self) -> dict[str, object]:
        return {
            "benchmark": "AI_Idea_Bench_2025",
            "benchmark_index": 15,
            "target_paper": "PanoPose",
            "motivation": (
                "Traditional SfM methods struggle with scale estimation in textureless scenes and straight-line camera motion."
            ),
            "method_summary": (
                "PanoPose combines a depth-net, a pose-net, rotation-only pre-training, and a fusion block "
                "to estimate scaled relative poses from panoramic images."
            ),
            "reference_titles": [
                "Unsupervised learning of depth and ego-motion from video",
                "Digging into self-supervised monocular depth estimation",
            ],
            "raw_record": {
                "summary": {
                    "split_topic": [
                        {"keyword": "scaled relative poses"},
                        {"keyword": "panoramic images"},
                    ],
                    "method": {
                        "datasets": (
                            "Experiments were conducted on several datasets, including "
                            "PanoSUNCG (synthetic indoor) and Mapillary Metropolis (real-world panoramic images)."
                        ),
                        "metrics": (
                            "Evaluation metrics include Relative Rotation Error (RRE), Relative Translation "
                            "Angle Error (RTAE), and Relative Scale Error (RSE)."
                        ),
                    },
                }
            },
            "literature_grounding": {
                "dataset_items": [
                    "PanoSUNCG (synthetic indoor)",
                    "Mapillary Metropolis (real-world panoramic images)",
                ],
                "metric_items": [
                    "Relative Rotation Error (RRE)",
                    "Relative Translation Angle Error (RTAE)",
                    "Relative Scale Error (RSE)",
                ],
                "existing_methods_summary": (
                    "Existing work uses self-supervised depth estimation and calibrated relative-pose methods."
                ),
            },
        }

    def _make_temp_output_root(self) -> Path:
        root = ROOT / "tests" / "_tmp_evaluation"
        root.mkdir(parents=True, exist_ok=True)
        run_root = root / f"case-{uuid4().hex}"
        run_root.mkdir(parents=True, exist_ok=True)
        return run_root

    def test_evaluate_graph_returns_benchmark_alignment_metrics_when_available(self) -> None:
        graph = run_experiment(
            topic="The topic of this paper is estimating scaled relative poses in panoramic images.",
            literature=[
                "Unsupervised learning of depth and ego-motion from video",
                "Digging into self-supervised monocular depth estimation",
            ],
            metadata=self._benchmark_metadata(),
            max_rounds=3,
            stop_when_mature=False,
        )

        evaluation = evaluate_graph(graph)
        metrics = {metric.key: metric for metric in evaluation.metrics}

        self.assertGreater(evaluation.overall_score, 0.0)
        self.assertIn("expert_style_quality", evaluation.category_scores)
        self.assertIn("benchmark_alignment", evaluation.category_scores)
        self.assertTrue(metrics["ground_truth_concordance"].available)
        self.assertGreater(metrics["experiment_alignment"].score, 0.0)
        self.assertGreater(metrics["topic_alignment"].score, 0.0)

    def test_ground_truth_concordance_is_unavailable_without_benchmark_oracle_fields(self) -> None:
        graph = run_experiment(
            topic="graph-based scientific ideation",
            literature=["paper a", "paper b", "paper c", "paper d"],
            max_rounds=3,
            stop_when_mature=False,
        )

        evaluation = evaluate_graph(graph)
        metric = next(metric for metric in evaluation.metrics if metric.key == "ground_truth_concordance")

        self.assertFalse(metric.available)
        self.assertEqual(metric.score, 0.0)

    def test_write_run_artifacts_emits_evaluation_files(self) -> None:
        output_root = self._make_temp_output_root()
        try:
            graph = run_experiment(
                topic="The topic of this paper is estimating scaled relative poses in panoramic images.",
                literature=[
                    "Unsupervised learning of depth and ego-motion from video",
                    "Digging into self-supervised monocular depth estimation",
                ],
                metadata=self._benchmark_metadata(),
                max_rounds=3,
                stop_when_mature=False,
            )
            instance = ExperimentInstance(
                name="eval-case",
                topic=graph.topic,
                literature=list(graph.literature),
                source_path="test",
                metadata=self._benchmark_metadata(),
            )

            run_dir = write_run_artifacts(graph, output_root=output_root, instance=instance)

            self.assertTrue((run_dir / "evaluation.json").exists())
            self.assertTrue((run_dir / "evaluation.md").exists())
            self.assertIn("idea_evaluation", (run_dir / "summary.json").read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(output_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
