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

    def test_reference_snippets_can_supply_safe_dataset_and_metric_grounding(self) -> None:
        grounding = build_literature_grounding(
            literature=["LERF", "3D Gaussian Splatting"],
            metadata={
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "LERF",
                            "evaluation": "Evaluate on the LERF dataset and report localization accuracy and IoU.",
                            "method": "LERF uses CLIP-aligned open-vocabulary querying in 3D scenes.",
                        },
                        {
                            "resolved_title": "3D Gaussian Splatting",
                            "evaluation": "Report accuracy on held-out 3D scene understanding tasks.",
                            "method": "3D Gaussian Splatting enables efficient radiance-field rendering.",
                        },
                    ]
                }
            },
        )

        self.assertIn("LERF dataset", grounding.dataset_items)
        self.assertTrue("accuracy" in grounding.metric_items or "IoU" in grounding.metric_items)
        self.assertIn("Evaluate on", grounding.experiment_plan_summary)

    def test_reference_titles_can_supply_safe_dataset_anchors_when_snippets_are_weak(self) -> None:
        grounding = build_literature_grounding(
            literature=[
                "Seeclick: Harnessing gui grounding for advanced visual gui agents",
                "On the effects of data scale on computer control agents",
                "Amex: Android multi-annotation expo dataset for mobile gui agents",
                "Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments",
            ],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "benchmark_input_packet": {
                    "benchmark": "AI_Idea_Bench_2025",
                    "topic": "The topic of this paper is improving GUI grounding and OOD generalization for GUI agents.",
                },
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "Seeclick: Harnessing gui grounding for advanced visual gui agents",
                            "method": "This innovative approach bypasses structured text and adapts to various GUI platforms.",
                        },
                        {
                            "resolved_title": "Amex: Android multi-annotation expo dataset for mobile gui agents",
                            "abstract": "Red boxes and brown tabs illustrate annotated GUI elements.",
                        },
                        {
                            "resolved_title": "Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments",
                            "evaluation": "Task Instruction (See examples above) input Agent (e.g., GPT-4V) a11y-tree screenshot keyboardmouse.",
                        },
                    ]
                },
            },
        )

        self.assertIn("Amex dataset", grounding.dataset_items)
        self.assertIn("Osworld benchmark", grounding.dataset_items)
        self.assertIn("Amex dataset", grounding.experiment_plan_summary)
        self.assertIn("Osworld benchmark", grounding.experiment_plan_summary)
        self.assertNotIn("grounding accuracy", grounding.metric_items)
        self.assertNotIn("success rate", grounding.metric_items)
        self.assertNotIn("error rate", grounding.metric_items)

    def test_dataset_items_filter_work_in_progress_eval_residue(self) -> None:
        grounding = build_literature_grounding(
            literature=[
                "MAmmoTH: Building math generalist models through hybrid instruction tuning",
                "Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks",
            ],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "benchmark_input_packet": {
                    "benchmark": "AI_Idea_Bench_2025",
                    "topic": "The topic of this paper is improving mathematical reasoning in Large Language Models.",
                },
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "MAmmoTH: Building math generalist models through hybrid instruction tuning",
                            "abstract": "Work in Progress 1 I NTRODUCTION This work focuses on mathematical reasoning, a critical capability of modern large language models.",
                            "evaluation": "Work in Progress Eval Dataset # Samples In-Domain.",
                        },
                        {
                            "resolved_title": "Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks",
                            "evaluation": "Our PoT+SC achieves the best-known results on several evaluated datasets.",
                        },
                    ]
                },
            },
        )

        self.assertFalse(any("work in progress" in item.casefold() for item in grounding.dataset_items))
        self.assertFalse(any("# samples" in item.casefold() for item in grounding.dataset_items))
        self.assertNotIn("Work in Progress", grounding.experiment_plan_summary)

    def test_safe_grounding_does_not_recover_hidden_target_benchmark_fields(self) -> None:
        grounding = build_literature_grounding(
            literature=["SeeClick"],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "SeeClick",
                            "abstract": "Visual GUI grounding for cross-platform interaction.",
                            "method": "Use screenshot-grounded interaction instead of structured text.",
                            "evaluation": "Evaluate on OSWorld and report success rate and error rate.",
                        }
                    ]
                },
                "benchmark_input_packet": {
                    "benchmark": "AI_Idea_Bench_2025",
                    "topic": "The topic of this paper is improving GUI grounding and OOD generalization for GUI agents.",
                    "reference_packet": [
                        {"title": "SeeClick", "snippet": "Visual grounding for GUI agents."}
                    ],
                },
            },
        )

        self.assertEqual(grounding.target_paper, "")
        self.assertNotIn("os atlas", " ".join(grounding.design_highlights).lower())
        self.assertNotIn("13 million", " ".join(grounding.dataset_items).lower())

    def test_safe_grounding_filters_noisy_reference_fragments_as_fake_datasets_or_metrics(self) -> None:
        grounding = build_literature_grounding(
            literature=["SeeClick"],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "SeeClick",
                            "abstract": "This innovative approach bypasses structured text and adapts to GUI platforms.",
                            "method": "This innovative approach bypasses structured text and adapts to GUI platforms.",
                            "evaluation": "This innovative approach bypasses structured text and adapts to GUI platforms.",
                        }
                    ]
                },
            },
        )

        self.assertEqual(grounding.dataset_items, [])
        self.assertEqual(grounding.metric_items, [])

    def test_metric_signal_matching_does_not_treat_various_as_iou(self) -> None:
        grounding = build_literature_grounding(
            literature=["SeeClick"],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "SeeClick",
                            "evaluation": "This method adapts to various GUI platforms without structured text.",
                        }
                    ]
                },
            },
        )

        self.assertNotIn("various GUI platforms", " ".join(grounding.metric_items))
        self.assertEqual(grounding.metric_items, [])

    def test_noisy_reference_snippet_dataset_fragments_are_filtered(self) -> None:
        grounding = build_literature_grounding(
            literature=["LiDAR HPS"],
            metadata={
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "FreeMotion",
                            "evaluation": (
                                "Experiments were conducted on paper introduces FreeMotion, a novel dataset "
                                "captured in diverse real scenarios with multi-modal and multi-view data. "
                                "Report Chamfer distance (SUCD)."
                            ),
                            "method": "Fuse LiDAR and inertial sensing for human pose estimation.",
                        }
                    ]
                }
            },
        )

        self.assertNotIn("paper introduces", grounding.experiment_plan_summary.lower())
        self.assertNotIn("freemotion", " ".join(grounding.dataset_items).lower())

    def test_keyword_only_liveideabench_gets_safe_weak_context_scaffold(self) -> None:
        grounding = build_literature_grounding(
            literature=[
                "Benchmark keyword: meteorology",
                "This benchmark row provides a keyword prompt rather than retrieved literature.",
                "Use the keyword as the ideation seed and treat benchmark idea text only as held-out metadata.",
            ],
            metadata={
                "benchmark": "liveideabench",
                "keyword": "meteorology",
                "benchmark_input_packet": {
                    "benchmark": "liveideabench",
                    "keyword": "meteorology",
                    "reference_packet": [],
                },
            },
        )

        self.assertEqual(grounding.source, "keyword_scaffold")
        self.assertTrue(grounding.design_highlights)
        self.assertIn("meteorology", grounding.existing_methods_summary.lower())
        self.assertTrue(any("forecast" in item.lower() for item in grounding.design_highlights))
        self.assertTrue(any(metric in grounding.metric_items for metric in ("RMSE", "MAE", "CRPS")))
        self.assertNotIn("Benchmark keyword:", grounding.existing_methods_summary)
        self.assertTrue(grounding.weak_context_scaffold)
        self.assertIn("divergence_axes", grounding.weak_context_scaffold)
        self.assertIn("method_instantiation", grounding.weak_context_scaffold)

    def test_periodic_table_keyword_avoids_generic_scaffold_placeholders(self) -> None:
        grounding = build_literature_grounding(
            literature=[
                "Benchmark keyword: periodic table",
                "This benchmark row provides a keyword prompt rather than retrieved literature.",
            ],
            metadata={
                "benchmark": "liveideabench",
                "keyword": "periodic table",
                "benchmark_input_packet": {
                    "benchmark": "liveideabench",
                    "keyword": "periodic table",
                    "reference_packet": [],
                },
            },
        )

        scaffold = grounding.weak_context_scaffold
        self.assertTrue(scaffold)
        self.assertNotEqual(scaffold.get("domain_family"), "general_science")
        self.assertTrue(any("periodic" in item.lower() or "element" in item.lower() for item in grounding.design_highlights))
        self.assertNotIn("keyword-specific case studies", " ".join(grounding.dataset_items).lower())
        self.assertNotIn("generic problem framing", grounding.experiment_plan_summary.lower())
        self.assertNotIn("insufficient mechanism grounding", grounding.existing_methods_summary.lower())
        self.assertIn("method_instantiation", scaffold)


if __name__ == "__main__":
    unittest.main()
