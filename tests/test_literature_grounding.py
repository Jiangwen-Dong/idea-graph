from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.literature_grounding import build_literature_grounding


class LiteratureGroundingTests(unittest.TestCase):
    def test_benchmark_metadata_is_split_into_datasets_and_metrics(self) -> None:
        grounding = build_literature_grounding(
            literature=[
                "Unsupervised learning of depth and ego-motion from video",
                "Posenet: A convolutional network for real-time 6-dof camera relocalization",
            ],
            metadata={
                "target_paper": "PanoPose",
                "method_summary": (
                    "PanoPose comprises a depth-net and a pose-net, utilizing self-supervision through "
                    "image reconstruction based on estimated depth and relative pose."
                ),
                "raw_record": {
                    "summary": {
                        "method": {
                            "datasets": (
                                "Experiments were conducted on several datasets, including "
                                "PanoSUNCG (synthetic indoor), Mapillary Metropolis (real-world panoramic images), "
                                "360VO Dataset (synthetic urban scenes), and custom datasets Building and Campus "
                                "(collected with an Insta 360 ONE X2 camera)."
                            ),
                            "metrics": (
                                "Evaluation metrics include Relative Rotation Error (RRE), Relative Translation "
                                "Angle Error (RTAE), and Relative Scale Error (RSE). For global pose estimation, "
                                "Absolute Rotation Error (ARE) and Absolute Translation Error (ATE) are used."
                            ),
                            "targeted_designs_details": [
                                {
                                    "design_name": "Fusion Block",
                                    "description": "A fusion block integrates depth features into pose estimation.",
                                }
                            ],
                        }
                    }
                },
            },
        )

        self.assertEqual(grounding.source, "metadata_structured")
        self.assertIn("PanoSUNCG (synthetic indoor)", grounding.dataset_items)
        self.assertIn("Building and Campus (collected with an Insta 360 ONE X2 camera)", grounding.dataset_items)
        self.assertIn("Relative Rotation Error (RRE)", grounding.metric_items)
        self.assertIn("Absolute Translation Error (ATE)", grounding.metric_items)
        self.assertIn("Evaluate on", grounding.experiment_plan_summary)
        self.assertIn("Fusion Block", grounding.existing_methods_summary)


if __name__ == "__main__":
    unittest.main()
