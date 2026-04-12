from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.text_critic import (
    CandidateExample,
    build_split_audit,
    build_training_examples,
    evaluate_state_rankings,
    train_text_critic,
)


class _StubModel:
    def score(self, texts: list[str]) -> list[float]:
        output: list[float] = []
        for text in texts:
            if "best" in text.lower():
                output.append(0.95)
            elif "mid" in text.lower():
                output.append(0.45)
            else:
                output.append(0.05)
        return output


class _CaptureModel:
    def __init__(self) -> None:
        self.last_texts: list[str] = []

    def score(self, texts: list[str]) -> list[float]:
        self.last_texts = list(texts)
        return [0.5 for _ in texts]


class _BadLengthModel:
    def score(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [0.1 for _ in range(len(texts) - 1)]


class TextCriticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "state_id": "state-train-1",
                "candidate_id": "state-train-1::candidate:0000",
                "split": "train",
                "group_id": "group-train",
                "is_logged_selected": True,
                "state_text": "benchmark=AI_Idea_Bench_2025|round=Round1|role=TaskFramer",
                "candidate_text": "kind=add_support_edge|targets=Problem:best link",
                "targets": {"weak_value_01": 0.61, "native_value_01": 0.84},
            },
            {
                "state_id": "state-train-1",
                "candidate_id": "state-train-1::candidate:0001",
                "split": "train",
                "group_id": "group-train",
                "is_logged_selected": False,
                "state_text": "benchmark=AI_Idea_Bench_2025|round=Round1|role=TaskFramer",
                "candidate_text": "kind=commit|targets=<none>|note=weak option",
                "targets": {"weak_value_01": 0.61, "native_value_01": 0.84},
            },
            {
                "state_id": "state-train-2",
                "candidate_id": "state-train-2::candidate:0000",
                "split": "train",
                "group_id": "group-train",
                "is_logged_selected": True,
                "state_text": "benchmark=liveideabench|round=Round2|role=MechanismProposer",
                "candidate_text": "kind=attach_evidence|targets=Method:best evidence",
                "targets": {"weak_value_01": 0.52, "native_value_01": None},
            },
            {
                "state_id": "state-train-2",
                "candidate_id": "state-train-2::candidate:0001",
                "split": "train",
                "group_id": "group-train",
                "is_logged_selected": False,
                "state_text": "benchmark=liveideabench|round=Round2|role=MechanismProposer",
                "candidate_text": "kind=add_node|targets=Method:low baseline",
                "targets": {"weak_value_01": 0.52, "native_value_01": None},
            },
            {
                "state_id": "state-val-1",
                "candidate_id": "state-val-1::candidate:0000",
                "split": "validation",
                "group_id": "group-val",
                "is_logged_selected": True,
                "state_text": "benchmark=liveideabench|round=Round1|role=TaskFramer",
                "candidate_text": "kind=attach_evidence|targets=Problem:best citation",
                "targets": {"weak_value_01": 0.58, "native_value_01": None},
            },
            {
                "state_id": "state-val-1",
                "candidate_id": "state-val-1::candidate:0001",
                "split": "validation",
                "group_id": "group-val",
                "is_logged_selected": False,
                "state_text": "benchmark=liveideabench|round=Round1|role=TaskFramer",
                "candidate_text": "kind=commit|targets=<none>|note=mid",
                "targets": {"weak_value_01": 0.58, "native_value_01": None},
            },
        ]

    def test_build_training_examples_preserves_split_and_state_id(self) -> None:
        examples = build_training_examples(self.rows)
        by_candidate_id = {example.candidate_id: example for example in examples}
        self.assertEqual(len(examples), len(self.rows))
        self.assertEqual(by_candidate_id["state-train-1::candidate:0000"].split, "train")
        self.assertEqual(by_candidate_id["state-val-1::candidate:0001"].split, "validation")
        self.assertEqual(by_candidate_id["state-val-1::candidate:0001"].state_id, "state-val-1")

    def test_train_text_critic_fits_and_scores_candidate_texts(self) -> None:
        examples = build_training_examples(self.rows)
        train_examples = [example for example in examples if example.split == "train"]
        model = train_text_critic(train_examples)
        scores = model.score(
            [
                "benchmark=liveideabench [SEP] kind=commit|targets=<none>",
                "benchmark=liveideabench [SEP] kind=attach_evidence|targets=Problem:best citation",
            ]
        )
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(isinstance(score, float) for score in scores))

    def test_train_text_critic_rejects_sample_weight_length_mismatch(self) -> None:
        examples = build_training_examples(self.rows)
        train_examples = [example for example in examples if example.split == "train"]
        with self.assertRaises(ValueError):
            train_text_critic(train_examples, sample_weights=[1.0])

    def test_evaluate_state_rankings_reports_top1_and_mrr(self) -> None:
        validation_examples = [
            CandidateExample(
                state_id="s1",
                candidate_id="s1::0",
                split="validation",
                label=1,
                state_text="state one",
                candidate_text="best action",
                group_id="g1",
                weak_value_01=0.7,
                native_value_01=0.8,
            ),
            CandidateExample(
                state_id="s1",
                candidate_id="s1::1",
                split="validation",
                label=0,
                state_text="state one",
                candidate_text="weak action",
                group_id="g1",
                weak_value_01=0.7,
                native_value_01=0.8,
            ),
            CandidateExample(
                state_id="s2",
                candidate_id="s2::0",
                split="validation",
                label=0,
                state_text="state two",
                candidate_text="best decoy",
                group_id="g2",
                weak_value_01=0.5,
                native_value_01=None,
            ),
            CandidateExample(
                state_id="s2",
                candidate_id="s2::1",
                split="validation",
                label=1,
                state_text="state two",
                candidate_text="mid winner",
                group_id="g2",
                weak_value_01=0.5,
                native_value_01=None,
            ),
        ]
        metrics = evaluate_state_rankings(_StubModel(), validation_examples)
        self.assertEqual(metrics["state_count"], 2)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["mean_reciprocal_rank"], 0.75)

    def test_model_text_path_strips_source_and_rationale_markers_from_candidate_text(self) -> None:
        capture = _CaptureModel()
        validation_examples = [
            CandidateExample(
                state_id="s-leak",
                candidate_id="s-leak::0",
                split="validation",
                label=1,
                state_text="state context",
                candidate_text="kind=attach_evidence|source=legacy_policy|rationale=shortcut|targets=Problem:signal",
                group_id="g-leak",
                weak_value_01=0.4,
                native_value_01=None,
            ),
            CandidateExample(
                state_id="s-leak",
                candidate_id="s-leak::1",
                split="validation",
                label=0,
                state_text="state context",
                candidate_text="kind=commit|source=special_commit|targets=<none>",
                group_id="g-leak",
                weak_value_01=0.4,
                native_value_01=None,
            ),
        ]
        evaluate_state_rankings(capture, validation_examples)
        self.assertEqual(len(capture.last_texts), 2)
        self.assertTrue(all("source=" not in text for text in capture.last_texts))
        self.assertTrue(all("rationale=" not in text for text in capture.last_texts))

    def test_evaluate_state_rankings_raises_on_state_with_zero_positive_labels(self) -> None:
        bad_examples = [
            CandidateExample(
                state_id="s0",
                candidate_id="s0::0",
                split="validation",
                label=0,
                state_text="state",
                candidate_text="kind=commit",
                group_id="g0",
                weak_value_01=None,
                native_value_01=None,
            ),
            CandidateExample(
                state_id="s0",
                candidate_id="s0::1",
                split="validation",
                label=0,
                state_text="state",
                candidate_text="kind=attach_evidence",
                group_id="g0",
                weak_value_01=None,
                native_value_01=None,
            ),
        ]
        with self.assertRaises(ValueError):
            evaluate_state_rankings(_StubModel(), bad_examples)

    def test_evaluate_state_rankings_raises_on_state_with_multiple_positive_labels(self) -> None:
        bad_examples = [
            CandidateExample(
                state_id="s1",
                candidate_id="s1::0",
                split="validation",
                label=1,
                state_text="state",
                candidate_text="kind=commit",
                group_id="g1",
                weak_value_01=None,
                native_value_01=None,
            ),
            CandidateExample(
                state_id="s1",
                candidate_id="s1::1",
                split="validation",
                label=1,
                state_text="state",
                candidate_text="kind=attach_evidence",
                group_id="g1",
                weak_value_01=None,
                native_value_01=None,
            ),
        ]
        with self.assertRaises(ValueError):
            evaluate_state_rankings(_StubModel(), bad_examples)

    def test_evaluate_state_rankings_raises_on_score_length_mismatch(self) -> None:
        examples = [
            CandidateExample(
                state_id="s2",
                candidate_id="s2::0",
                split="validation",
                label=1,
                state_text="state",
                candidate_text="kind=commit",
                group_id="g2",
                weak_value_01=None,
                native_value_01=None,
            ),
            CandidateExample(
                state_id="s2",
                candidate_id="s2::1",
                split="validation",
                label=0,
                state_text="state",
                candidate_text="kind=attach_evidence",
                group_id="g2",
                weak_value_01=None,
                native_value_01=None,
            ),
        ]
        with self.assertRaises(ValueError):
            evaluate_state_rankings(_BadLengthModel(), examples)

    def test_build_split_audit_reports_group_counts_and_overlap(self) -> None:
        examples = build_training_examples(self.rows)
        train_examples = [example for example in examples if example.split == "train"]
        validation_examples = [example for example in examples if example.split == "validation"]
        audit = build_split_audit(train_examples, validation_examples)
        self.assertEqual(audit["train_group_count"], 1)
        self.assertEqual(audit["validation_group_count"], 1)
        self.assertEqual(audit["group_overlap_count"], 0)


if __name__ == "__main__":
    unittest.main()
