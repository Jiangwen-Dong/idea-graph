from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.evaluation import evaluate_graph
from idea_graph.models import FinalProposal, GraphAction, IdeaGraph, MaturitySnapshot


class EvaluationTests(unittest.TestCase):
    def _base_graph(self) -> IdeaGraph:
        graph = IdeaGraph(
            topic="The topic of this paper is multimodal distillation for long-video understanding.",
            literature=[],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "motivation": "Long-video models still struggle to transfer fine-grained temporal reasoning from strong teachers.",
                "method_summary": "Use teacher-guided temporal token selection to distill long-range reasoning into a compact student.",
                "paper_grounding": {
                    "reference_paper_snippets": [
                        {
                            "resolved_title": "Teacher-Guided Long-Video Distillation",
                            "abstract": "Teacher-guided distillation improves compact long-video models.",
                        }
                    ]
                },
                "literature_grounding": {
                    "existing_methods_summary": "Existing methods use distillation or token compression for long-video reasoning.",
                    "dataset_items": ["Video-MME"],
                    "metric_items": ["accuracy", "F1"],
                    "experiment_plan_summary": "Evaluate on Video-MME with accuracy and F1 against long-video baselines.",
                },
            },
        )
        graph.final_proposal = FinalProposal(
            title="Curriculum Temporal Distillation For Long-Video Models",
            problem="Compact long-video models still miss teacher-level temporal reasoning on long contexts.",
            existing_methods="Existing methods use token pruning or distillation, but they often compress away the teacher's temporal cues.",
            motivation="We need a distillation strategy that preserves temporal evidence rather than only logits.",
            hypothesis="Teacher-aligned temporal curriculum signals can transfer long-range reasoning more faithfully.",
            method="Distill a compact student with curriculum-weighted temporal evidence maps and teacher-guided segment sampling.",
            evaluation="Evaluate on Video-MME with accuracy and F1, compare against long-video distillation baselines, and run ablations on curriculum stages.",
            significance="This could improve efficient long-video understanding without full teacher-scale inference.",
            caveats="The method may depend on accurate teacher evidence maps and could underperform on very noisy videos.",
        )
        return graph

    def test_main_outcome_scores_are_output_only(self) -> None:
        graph_without_process = self._base_graph()
        graph_with_process = self._base_graph()
        graph_with_process.round_summaries.append(
            (
                "Round3",
                MaturitySnapshot(
                    support_coverage=0.95,
                    unresolved_contradiction_ratio=0.0,
                    utility=7.2,
                    utility_stable=True,
                    completeness=True,
                    is_mature=True,
                ),
            )
        )
        graph_with_process.actions.extend(
            [
                GraphAction(id="A001", round_name="Round1", role="MechanismProposer", kind="expand_branch", target_ids=["N001"]),
                GraphAction(id="A002", round_name="Round2", role="NoveltyExaminer", kind="attach_evidence", target_ids=["N002"]),
                GraphAction(id="A003", round_name="Round3", role="EvaluationDesigner", kind="freeze_branch", target_ids=[]),
            ]
        )

        eval_without = evaluate_graph(graph_without_process)
        eval_with = evaluate_graph(graph_with_process)

        self.assertEqual(
            eval_without.category_scores["expert_style_quality"],
            eval_with.category_scores["expert_style_quality"],
        )
        self.assertEqual(
            eval_without.category_scores["benchmark_alignment"],
            eval_with.category_scores["benchmark_alignment"],
        )
        self.assertEqual(eval_without.overall_score, eval_with.overall_score)
        self.assertNotEqual(
            eval_without.category_scores.get("graph_process", 0.0),
            eval_with.category_scores.get("graph_process", 0.0),
        )

    def test_overall_score_excludes_graph_process_category(self) -> None:
        graph = self._base_graph()
        graph.round_summaries.append(
            (
                "Round2",
                MaturitySnapshot(
                    support_coverage=1.0,
                    unresolved_contradiction_ratio=0.0,
                    utility=8.0,
                    utility_stable=True,
                    completeness=True,
                    is_mature=True,
                ),
            )
        )
        evaluation = evaluate_graph(graph)
        expert = evaluation.category_scores["expert_style_quality"]
        alignment = evaluation.category_scores["benchmark_alignment"]
        expected = round((expert + alignment) / 2.0, 2)
        self.assertEqual(expected, evaluation.overall_score)

    def test_topic_alignment_handles_sentence_final_keyword_tokens(self) -> None:
        graph = IdeaGraph(
            topic="Ideation topic keyword: meteorology",
            literature=["Benchmark keyword: meteorology"],
            metadata={
                "benchmark": "liveideabench",
                "keyword": "meteorology",
                "raw_record": {},
            },
        )
        graph.final_proposal = FinalProposal(
            title="Spatial-Temporal Disentanglement for Weather Pattern Forecasting",
            problem="Current weather prediction models struggle with uncertainty. This matters directly for meteorology.",
            existing_methods="Existing weather models often entangle spatial and temporal dynamics.",
            motivation="Reliable forecasting matters for disaster mitigation.",
            hypothesis="Separating spatial and temporal structure can improve meteorological forecasting.",
            method="Use a variational architecture that disentangles atmospheric dynamics for weather prediction.",
            evaluation="Evaluate forecasting accuracy and calibration on weather benchmarks.",
            significance="Improves meteorology forecasting.",
            caveats="May depend on data quality.",
        )

        evaluation = evaluate_graph(graph)
        topic_metric = next(metric for metric in evaluation.metrics if metric.key == "topic_alignment")

        self.assertGreater(topic_metric.score, 0.0)
        self.assertGreater(topic_metric.signals.get("keyword_signal", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
