from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.benchmark_scoring import (
    _NativeJudge,
    _extract_i2t_alignment,
    _extract_json_object,
    _has_valid_i2t_alignment,
    evaluate_benchmark_native,
)
from idea_graph.models import FinalProposal, IdeaGraph
from idea_graph.settings import OpenAICompatibleSettings


class BenchmarkScoringTests(unittest.TestCase):
    def _ai_idea_bench_graph(self) -> IdeaGraph:
        graph = IdeaGraph(
            topic="The topic of this paper is backdoor attacks on multimodal contrastive learning models.",
            literature=[],
            metadata={
                "benchmark": "AI_Idea_Bench_2025",
                "benchmark_index": 15,
                "benchmark_root": str(ROOT / "data" / "benchmarks" / "ai_idea_bench_2025"),
                "motivation": "Existing multimodal contrastive models remain vulnerable to backdoor attacks that can survive downstream defenses.",
                "method_summary": "Use dual embedding consistency objectives to preserve attack behavior after defensive fine-tuning.",
            },
        )
        graph.final_proposal = FinalProposal(
            title="Robust Backdoor Stress Testing For Multimodal Contrastive Models",
            problem="Current multimodal contrastive models can hide persistent backdoor behavior.",
            existing_methods="Existing attack analyses often fail after stronger model adaptation defenses.",
            motivation="We need a more durable backdoor formulation that reveals the true defense gap.",
            hypothesis="Embedding-aware trigger optimization can preserve malicious behavior under adaptation.",
            method="Optimize both image and text side consistency objectives for poisoned samples.",
            evaluation="Evaluate clean accuracy, attack success rate, and robustness after fine-tuning defenses.",
            significance="This would expose a harder security benchmark for multimodal models.",
            caveats="The attack could still overfit a narrow target label distribution.",
        )
        return graph

    def _liveideabench_graph(self) -> IdeaGraph:
        graph = IdeaGraph(
            topic="Ideation topic keyword: meteorology",
            literature=[],
            metadata={
                "benchmark": "liveideabench",
                "benchmark_root": str(ROOT / "data" / "benchmarks" / "liveideabench"),
                "keyword": "meteorology",
            },
        )
        graph.final_proposal = FinalProposal(
            title="Citizen Weather Mesh",
            problem="Local weather events remain under-observed at fine spatial granularity.",
            motivation="Dense local sensing could improve short-term forecasts and hazard response.",
            hypothesis="Crowd-sourced phone sensing plus uncertainty-aware fusion can improve nowcasting.",
            method="Fuse phone, satellite, and station data with a reliability-weighted forecasting model.",
            evaluation="Compare against standard nowcasting baselines on severe-weather and microclimate tasks.",
            significance="This could improve local forecasting where traditional sensing is sparse.",
            caveats="Phone sensing is noisy and may create geographic participation bias.",
        )
        return graph

    def test_ai_idea_bench_native_scoring_reports_unavailable_metrics_without_judge(self) -> None:
        evaluation = evaluate_benchmark_native(self._ai_idea_bench_graph())
        metrics = {metric.key: metric for metric in evaluation.metrics}

        self.assertEqual(evaluation.protocol_name, "ai_idea_bench_2025_public_protocol_v1")
        self.assertFalse(metrics["i2i_motivation"].available)
        self.assertFalse(metrics["ic"].available)
        self.assertEqual(evaluation.summary["available_metric_count"], 0.0)

    def test_liveideabench_native_scoring_reports_keyword_rubric(self) -> None:
        evaluation = evaluate_benchmark_native(self._liveideabench_graph())
        metrics = {metric.key: metric for metric in evaluation.metrics}

        self.assertEqual(evaluation.protocol_name, "liveideabench_public_rubric_v1")
        self.assertFalse(metrics["originality"].available)
        self.assertEqual(evaluation.benchmark, "liveideabench")

    def test_extract_json_object_accepts_valid_prefix_before_extra_text(self) -> None:
        payload = _extract_json_object('{"score": 4, "rationale": "ok"} trailing commentary')
        self.assertEqual(payload["score"], 4)

    def test_extract_json_object_repairs_trailing_commas(self) -> None:
        payload = _extract_json_object('{"scores":{"a":1,},"overall_average":7,}')
        self.assertEqual(payload["overall_average"], 7)

    def test_extract_i2t_alignment_accepts_flat_schema(self) -> None:
        payload = {
            "motivation_alignment": 4,
            "motivation_comments": "Strong topic fit.",
            "experiment_alignment": 5,
            "experiment_comments": "Clear experiment-topic match.",
        }

        result = _extract_i2t_alignment(payload)

        self.assertEqual(result["motivation_alignment"], 4.0)
        self.assertEqual(result["experiment_alignment"], 5.0)
        self.assertEqual(result["motivation_comments"], "Strong topic fit.")
        self.assertEqual(result["experiment_comments"], "Clear experiment-topic match.")
        self.assertTrue(_has_valid_i2t_alignment(payload))

    def test_native_judge_retries_invalid_i2t_schema_once(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def create_chat_completion(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    content = '{"motivation_alignment": 0, "experiment_alignment": 0}'
                else:
                    content = (
                        '{"motivation":{"alignment":4,"comments":"Topic fit is clear."},'
                        '"experiment_plan":{"alignment":5,"comments":"Experiment plan matches the topic."}}'
                    )
                return SimpleNamespace(
                    content=content,
                    raw_response={"choices": [{"message": {"content": content}}]},
                )

        fake_client = FakeClient()
        settings = OpenAICompatibleSettings.from_mapping(
            {
                "base_url": "https://example.com/v1",
                "api_key": "test-key",
                "model": "qwen3-8b",
                "provider": "dashscope",
                "reasoning_mode": "auto",
                "max_retries": 1,
            }
        )

        with patch("idea_graph.benchmark_scoring.OpenAICompatibleChatClient", return_value=fake_client):
            judge = _NativeJudge(settings)
            payload = judge.score_json_validated(
                system_prompt="system",
                user_prompt="user",
                validator=_has_valid_i2t_alignment,
                repair_instruction="Return the exact topic-alignment JSON schema.",
                max_tokens=200,
            )

        repaired = _extract_i2t_alignment(payload)
        self.assertEqual(fake_client.calls, 2)
        self.assertEqual(repaired["motivation_alignment"], 4.0)
        self.assertEqual(repaired["experiment_alignment"], 5.0)


if __name__ == "__main__":
    unittest.main()
