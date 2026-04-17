from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
for candidate in (str(SRC), str(TESTS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from idea_graph.models import Branch, Edge, IdeaGraph, Node
from idea_graph.relation_graph_critic_model import RelationGraphCritic
from idea_graph.relation_graph_critic_data import (  # type: ignore[attr-defined]
    HashTextEmbeddingBackend,
    RelationGraphVocabularies,
    build_relation_graph_runtime_batch,
    build_relation_graph_vocabularies,
)
from idea_graph.relation_graph_runtime_critic import (  # type: ignore[attr-defined]
    RelationGraphRuntimeConfig,
    load_relation_graph_runtime_bundle,
    select_relation_graph_critic_candidate,
)
from test_relation_graph_critic_data import write_relation_graph_fixture


class RecordingHashTextEmbeddingBackend(HashTextEmbeddingBackend):
    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim=dim)
        self.encoded_texts: list[str] = []

    def encode(self, texts):  # type: ignore[no-untyped-def]
        self.encoded_texts = [str(text) for text in texts]
        return super().encode(texts)


class RelationGraphRuntimeCriticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())
        self.fixture = write_relation_graph_fixture(self.tmp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_build_relation_graph_vocabularies_is_deterministic(self) -> None:
        first = build_relation_graph_vocabularies(
            candidate_dataset_dir=self.fixture.candidate_dir,
            g1_dataset_dir=self.fixture.g1_dir,
            partition_manifest_path=self.fixture.partition_manifest,
        )
        second = build_relation_graph_vocabularies(
            candidate_dataset_dir=self.fixture.candidate_dir,
            g1_dataset_dir=self.fixture.g1_dir,
            partition_manifest_path=self.fixture.partition_manifest,
        )

        self.assertEqual(first.node_type_to_id, second.node_type_to_id)
        self.assertEqual(first.role_to_id, second.role_to_id)
        self.assertEqual(first.edge_type_to_id, second.edge_type_to_id)
        self.assertEqual(first.candidate_kind_to_id, second.candidate_kind_to_id)
        self.assertEqual(first.edge_type_to_id, {"depends_on": 0, "supports": 1, "unknown": 2})
        self.assertEqual(first.candidate_kind_to_id, {"add_support_edge": 0, "commit": 1, "unknown": 2})
        self.assertIn("unknown", first.node_type_to_id)
        self.assertIn("unknown", first.role_to_id)

    def test_runtime_batch_builder_filters_commit_and_sorts_graph_structure(self) -> None:
        graph = IdeaGraph(topic="runtime", literature=[], metadata={})
        graph.branches["B001"] = Branch(id="B001", role="MechanismProposer")
        graph.nodes["N200"] = Node(
            id="N200",
            type="Method",
            text="Method node text.",
            role="MechanismProposer",
            branch_id="B001",
            confidence=0.7,
            evidence=["method-evidence"],
        )
        graph.nodes["N100"] = Node(
            id="N100",
            type="Hypothesis",
            text="Hypothesis node text.",
            role="MechanismProposer",
            branch_id="B001",
            confidence=0.9,
            evidence=["hypothesis-evidence"],
        )
        graph.edges.extend(
            [
                Edge(
                    id="E999",
                    source_id="N200",
                    relation="supports",
                    target_id="N100",
                    role="MechanismProposer",
                    branch_id="B001",
                ),
                Edge(
                    id="E050",
                    source_id="N100",
                    relation="depends_on",
                    target_id="N200",
                    role="MechanismProposer",
                    branch_id="B001",
                ),
                Edge(
                    id="E001",
                    source_id="N100",
                    relation="supports",
                    target_id="N404",
                    role="MechanismProposer",
                    branch_id="B001",
                ),
            ]
        )
        vocab = build_relation_graph_vocabularies(
            candidate_dataset_dir=self.fixture.candidate_dir,
            g1_dataset_dir=self.fixture.g1_dir,
            partition_manifest_path=self.fixture.partition_manifest,
        )
        backend = RecordingHashTextEmbeddingBackend(dim=8)

        runtime_batch = build_relation_graph_runtime_batch(
            graph=graph,
            candidate_specs=[
                {
                    "candidate_id": "runtime-commit",
                    "kind": "commit",
                    "target_ids": [],
                    "payload": {"branch_id": "B001"},
                    "rationale": "teacher-only rationale",
                    "candidate_source": "runtime-commit",
                },
                {
                    "candidate_id": "runtime-edit",
                    "kind": "add_support_edge",
                    "target_ids": ["N200", "N404"],
                    "payload": {"branch_id": "B001", "evidence": "new-signal"},
                    "rationale": "teacher-only rationale",
                    "candidate_source": "runtime-edit",
                },
            ],
            text_backend=backend,
            vocabularies=vocab,
            use_commit=False,
        )

        self.assertEqual(len(runtime_batch.examples), 1)
        example = runtime_batch.examples[0]
        self.assertEqual(example.target_node_indices, [1])
        self.assertEqual(example.edge_index, [(0, 1), (1, 0)])
        self.assertFalse(any("kind=commit" in text for text in backend.encoded_texts))
        self.assertTrue(all("source=" not in text for text in backend.encoded_texts))
        self.assertTrue(all("rationale=" not in text for text in backend.encoded_texts))

    def test_runtime_batch_builder_keeps_rows_with_vocab_misses_and_emits_fallback_signal(self) -> None:
        graph = IdeaGraph(topic="runtime", literature=[], metadata={})
        graph.branches["B900"] = Branch(id="B900", role="NovelRole")
        graph.nodes["N777"] = Node(
            id="N777",
            type="NovelType",
            text="Out-of-vocab node text.",
            role="NovelRole",
            branch_id="B900",
            confidence=0.1,
        )
        graph.nodes["N778"] = Node(
            id="N778",
            type="NovelType",
            text="Another out-of-vocab node text.",
            role="NovelRole",
            branch_id="B900",
            confidence=0.2,
        )
        graph.edges.append(
            Edge(
                id="E777",
                source_id="N777",
                relation="novel_relation",
                target_id="N778",
                role="NovelRole",
                branch_id="B900",
            )
        )
        vocab = build_relation_graph_vocabularies(
            candidate_dataset_dir=self.fixture.candidate_dir,
            g1_dataset_dir=self.fixture.g1_dir,
            partition_manifest_path=self.fixture.partition_manifest,
        )

        runtime_batch = build_relation_graph_runtime_batch(
            graph=graph,
            candidate_specs=[
                {
                    "candidate_id": "runtime-oov",
                    "kind": "novel_kind",
                    "target_ids": ["N777", "N404"],
                    "payload": {"branch_id": "B900"},
                    "candidate_source": "runtime-oov",
                },
            ],
            text_backend=HashTextEmbeddingBackend(dim=8),
            vocabularies=vocab,
            use_commit=True,
        )

        self.assertEqual(len(runtime_batch.examples), 1)
        example = runtime_batch.examples[0]
        node_type_unknown_id = vocab.node_type_to_id["unknown"]
        role_unknown_id = vocab.role_to_id["unknown"]
        edge_type_unknown_id = vocab.edge_type_to_id["unknown"]
        candidate_kind_unknown_id = vocab.candidate_kind_to_id["unknown"]
        self.assertNotEqual(node_type_unknown_id, vocab.node_type_to_id["Hypothesis"])
        self.assertNotEqual(role_unknown_id, vocab.role_to_id["MechanismProposer"])
        self.assertNotEqual(edge_type_unknown_id, vocab.edge_type_to_id["supports"])
        self.assertNotEqual(candidate_kind_unknown_id, vocab.candidate_kind_to_id["add_support_edge"])
        self.assertEqual(example.node_type_ids, [node_type_unknown_id, node_type_unknown_id])
        self.assertEqual(example.node_role_ids, [role_unknown_id, role_unknown_id])
        self.assertEqual(example.edge_type_ids, [edge_type_unknown_id])
        self.assertEqual(example.candidate_kind_id, candidate_kind_unknown_id)
        self.assertEqual(runtime_batch.fallback_row_mask.tolist(), [True])
        diagnostics = runtime_batch.diagnostics[0]
        self.assertEqual(diagnostics.candidate_id, "runtime-oov")
        self.assertEqual(diagnostics.missing_node_types, ("NovelType",))
        self.assertEqual(diagnostics.missing_roles, ("NovelRole",))
        self.assertEqual(diagnostics.missing_edge_types, ("novel_relation",))
        self.assertTrue(diagnostics.missing_candidate_kind)
        self.assertEqual(diagnostics.missing_target_ids, ("N404",))
        self.assertTrue(diagnostics.used_vocab_fallback)


class _StubRuntimeBundle:
    def __init__(self, *, scores: list[float], vocabularies) -> None:  # type: ignore[no-untyped-def]
        self.scores = list(scores)
        self.vocabularies = vocabularies
        self.text_backend = HashTextEmbeddingBackend(dim=8)
        self.score_call_count = 0

    def build_runtime_batch(self, *, graph, candidate_specs, use_commit):  # type: ignore[no-untyped-def]
        return build_relation_graph_runtime_batch(
            graph=graph,
            candidate_specs=candidate_specs,
            text_backend=self.text_backend,
            vocabularies=self.vocabularies,
            use_commit=use_commit,
        )

    def runtime_token_status(self, runtime_batch):  # type: ignore[no-untyped-def]
        return {"ok": True, "reason": "", "candidate_ids": ()}

    def score_runtime_batch(self, batch):  # type: ignore[no-untyped-def]
        self.score_call_count += 1
        return list(self.scores)


def _build_runtime_graph() -> IdeaGraph:
    graph = IdeaGraph(topic="runtime topic", literature=["paper a"], metadata={})
    graph.branches["B001"] = Branch(id="B001", role="MechanismProposer")
    graph.nodes["N001"] = Node(
        id="N001",
        type="Hypothesis",
        text="Hypothesis node text.",
        role="MechanismProposer",
        branch_id="B001",
        confidence=0.8,
    )
    graph.nodes["N002"] = Node(
        id="N002",
        type="Method",
        text="Method node text.",
        role="MechanismProposer",
        branch_id="B001",
        confidence=0.8,
    )
    return graph


class RelationGraphRuntimeSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())
        self.fixture = write_relation_graph_fixture(self.tmp_dir / "fixture")
        self.vocabularies = build_relation_graph_vocabularies(
            candidate_dataset_dir=self.fixture.candidate_dir,
            g1_dataset_dir=self.fixture.g1_dir,
            partition_manifest_path=self.fixture.partition_manifest,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_relation_graph_runtime_bundle_rebuilds_vocab_context_and_scores_batch(self) -> None:
        model_dir = self.tmp_dir / "artifact"
        model_dir.mkdir(parents=True, exist_ok=True)
        training_config = {
            "candidate_dataset_dir": str(self.fixture.candidate_dir.resolve()),
            "g1_dataset_dir": str(self.fixture.g1_dir.resolve()),
            "partition_manifest": str(self.fixture.partition_manifest.resolve()),
            "text_backend": "hash",
            "text_model_name": None,
            "embedding_dim": 8,
            "hidden_dim": 16,
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1e-3,
        }
        (model_dir / "training_config.json").write_text(
            json.dumps(training_config, indent=2),
            encoding="utf-8",
        )
        (model_dir / "metadata.json").write_text(
            json.dumps({"text_backend": "hash", "hidden_dim": 16}, indent=2),
            encoding="utf-8",
        )

        model = RelationGraphCritic(
            text_dim=8,
            hidden_dim=16,
            node_type_count=len(self.vocabularies.node_type_to_id),
            role_count=len(self.vocabularies.role_to_id),
            edge_type_count=len(self.vocabularies.edge_type_to_id),
            candidate_kind_count=len(self.vocabularies.candidate_kind_to_id),
        )
        torch.save(model.state_dict(), model_dir / "model.pt")

        bundle = load_relation_graph_runtime_bundle(model_dir)

        self.assertEqual(bundle.vocabs.node_type_to_id, self.vocabularies.node_type_to_id)
        self.assertEqual(bundle.vocabs.role_to_id, self.vocabularies.role_to_id)
        self.assertIn("Hypothesis", bundle.vocabs.node_type_to_id)
        self.assertIn("MechanismProposer", bundle.vocabs.role_to_id)
        self.assertEqual(bundle.text_backend.__class__.__name__, "HashTextEmbeddingBackend")
        self.assertIn(bundle.device.type, {"cpu", "cuda"})
        self.assertTrue((model_dir / "vocabularies.json").exists())

        runtime_batch = build_relation_graph_runtime_batch(
            graph=_build_runtime_graph(),
            candidate_specs=[
                {
                    "candidate_id": "runtime-candidate",
                    "kind": "add_support_edge",
                    "target_ids": ["N001", "N002"],
                    "payload": {"branch_id": "B001"},
                },
            ],
            text_backend=bundle.text_backend,
            vocabularies=bundle.vocabs,
            use_commit=False,
        )
        scores = bundle.score_runtime_batch(runtime_batch.batch)
        self.assertEqual(len(scores), 1)
        self.assertIsInstance(scores[0], float)

    def test_load_relation_graph_runtime_bundle_uses_artifact_local_vocab_snapshot(self) -> None:
        model_dir = self.tmp_dir / "artifact-with-vocab-snapshot"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "training_config.json").write_text(
            json.dumps(
                {
                    "candidate_dataset_dir": str((self.tmp_dir / "missing-candidates").resolve()),
                    "g1_dataset_dir": str((self.tmp_dir / "missing-g1").resolve()),
                    "partition_manifest": str((self.tmp_dir / "missing-manifest.jsonl").resolve()),
                    "text_backend": "hash",
                    "text_model_name": None,
                    "embedding_dim": 8,
                    "hidden_dim": 16,
                    "batch_size": 2,
                    "epochs": 1,
                    "learning_rate": 1e-3,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (model_dir / "metadata.json").write_text(
            json.dumps({"text_backend": "hash", "hidden_dim": 16}, indent=2),
            encoding="utf-8",
        )
        (model_dir / "vocabularies.json").write_text(
            json.dumps(
                {
                    "node_type_to_id": self.vocabularies.node_type_to_id,
                    "role_to_id": self.vocabularies.role_to_id,
                    "edge_type_to_id": self.vocabularies.edge_type_to_id,
                    "candidate_kind_to_id": self.vocabularies.candidate_kind_to_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        model = RelationGraphCritic(
            text_dim=8,
            hidden_dim=16,
            node_type_count=len(self.vocabularies.node_type_to_id),
            role_count=len(self.vocabularies.role_to_id),
            edge_type_count=len(self.vocabularies.edge_type_to_id),
            candidate_kind_count=len(self.vocabularies.candidate_kind_to_id),
        )
        torch.save(model.state_dict(), model_dir / "model.pt")

        bundle = load_relation_graph_runtime_bundle(model_dir)

        self.assertEqual(bundle.vocabs.node_type_to_id, self.vocabularies.node_type_to_id)
        self.assertEqual(bundle.vocabs.role_to_id, self.vocabularies.role_to_id)
        self.assertEqual(bundle.vocabs.edge_type_to_id, self.vocabularies.edge_type_to_id)
        self.assertEqual(bundle.vocabs.candidate_kind_to_id, self.vocabularies.candidate_kind_to_id)

    def test_load_relation_graph_runtime_bundle_pads_legacy_unknown_bucket_parameters(self) -> None:
        model_dir = self.tmp_dir / "artifact-legacy-unknown-compat"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "training_config.json").write_text(
            json.dumps(
                {
                    "candidate_dataset_dir": str(self.fixture.candidate_dir.resolve()),
                    "g1_dataset_dir": str(self.fixture.g1_dir.resolve()),
                    "partition_manifest": str(self.fixture.partition_manifest.resolve()),
                    "text_backend": "hash",
                    "text_model_name": None,
                    "embedding_dim": 8,
                    "hidden_dim": 16,
                    "batch_size": 2,
                    "epochs": 1,
                    "learning_rate": 1e-3,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (model_dir / "metadata.json").write_text(
            json.dumps({"text_backend": "hash", "hidden_dim": 16}, indent=2),
            encoding="utf-8",
        )

        legacy_model = RelationGraphCritic(
            text_dim=8,
            hidden_dim=16,
            node_type_count=len(self.vocabularies.node_type_to_id) - 1,
            role_count=len(self.vocabularies.role_to_id) - 1,
            edge_type_count=len(self.vocabularies.edge_type_to_id) - 1,
            candidate_kind_count=len(self.vocabularies.candidate_kind_to_id) - 1,
        )
        torch.save(legacy_model.state_dict(), model_dir / "model.pt")

        bundle = load_relation_graph_runtime_bundle(model_dir)

        node_unknown_id = self.vocabularies.node_type_to_id["unknown"]
        role_unknown_id = self.vocabularies.role_to_id["unknown"]
        edge_unknown_id = self.vocabularies.edge_type_to_id["unknown"]
        kind_unknown_id = self.vocabularies.candidate_kind_to_id["unknown"]
        zero_vector = torch.zeros(
            16,
            device=bundle.model.node_type_embed.weight.device,
        )
        self.assertTrue(
            torch.allclose(bundle.model.node_type_embed.weight[node_unknown_id], zero_vector)
        )
        self.assertTrue(
            torch.allclose(bundle.model.role_embed.weight[role_unknown_id], zero_vector)
        )
        self.assertTrue(
            torch.allclose(bundle.model.candidate_kind_embed.weight[kind_unknown_id], zero_vector)
        )
        for layer in bundle.model.layers:
            self.assertTrue(
                torch.allclose(
                    layer.edge_linears[edge_unknown_id].weight,
                    torch.zeros_like(layer.edge_linears[edge_unknown_id].weight),
                )
            )

    def test_load_relation_graph_runtime_bundle_rejects_non_legacy_vocab_shape_drift(self) -> None:
        model_dir = self.tmp_dir / "artifact-non-legacy-drift"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "training_config.json").write_text(
            json.dumps(
                {
                    "candidate_dataset_dir": str(self.fixture.candidate_dir.resolve()),
                    "g1_dataset_dir": str(self.fixture.g1_dir.resolve()),
                    "partition_manifest": str(self.fixture.partition_manifest.resolve()),
                    "text_backend": "hash",
                    "text_model_name": None,
                    "embedding_dim": 8,
                    "hidden_dim": 16,
                    "batch_size": 2,
                    "epochs": 1,
                    "learning_rate": 1e-3,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (model_dir / "metadata.json").write_text(
            json.dumps({"text_backend": "hash", "hidden_dim": 16}, indent=2),
            encoding="utf-8",
        )

        legacy_model = RelationGraphCritic(
            text_dim=8,
            hidden_dim=16,
            node_type_count=len(self.vocabularies.node_type_to_id) - 2,
            role_count=len(self.vocabularies.role_to_id) - 1,
            edge_type_count=len(self.vocabularies.edge_type_to_id) - 1,
            candidate_kind_count=len(self.vocabularies.candidate_kind_to_id) - 1,
        )
        torch.save(legacy_model.state_dict(), model_dir / "model.pt")

        with self.assertRaisesRegex(ValueError, "does not match runtime vocab shape"):
            load_relation_graph_runtime_bundle(model_dir)

    def test_select_relation_graph_critic_candidate_overrides_when_margin_is_large(self) -> None:
        graph = _build_runtime_graph()
        decision = select_relation_graph_critic_candidate(
            graph,
            round_name="Round3",
            role="MechanismProposer",
            state_features={
                "round_index": 3,
                "support_coverage": 0.72,
                "unresolved_contradiction_ratio": 0.0,
            },
            candidate_specs=[
                {
                    "candidate_id": "heuristic",
                    "kind": "add_support_edge",
                    "target_ids": ["N001", "N002"],
                    "payload": {"branch_id": "B001"},
                },
                {
                    "candidate_id": "critic-best",
                    "kind": "add_support_edge",
                    "target_ids": ["N002", "N001"],
                    "payload": {"branch_id": "B001"},
                },
            ],
            heuristic_candidate_id="heuristic",
            runtime_bundle=_StubRuntimeBundle(scores=[0.40, 0.88], vocabularies=self.vocabularies),
            config=RelationGraphRuntimeConfig(tau_override=0.05, use_commit=False),
        )

        self.assertEqual(decision.policy_decision.selected_candidate_id, "critic-best")
        self.assertEqual(decision.policy_decision.selected_source, "critic")
        self.assertFalse(decision.policy_decision.used_heuristic_fallback)

    def test_select_relation_graph_critic_candidate_scores_safe_subset_when_only_some_rows_are_unsafe(self) -> None:
        graph = _build_runtime_graph()
        bundle = _StubRuntimeBundle(scores=[0.40, 0.88], vocabularies=self.vocabularies)
        decision = select_relation_graph_critic_candidate(
            graph,
            round_name="Round3",
            role="MechanismProposer",
            state_features={
                "round_index": 3,
                "support_coverage": 0.72,
                "unresolved_contradiction_ratio": 0.0,
            },
            candidate_specs=[
                {
                    "candidate_id": "heuristic",
                    "kind": "add_support_edge",
                    "target_ids": ["N001", "N002"],
                    "payload": {"branch_id": "B001"},
                },
                {
                    "candidate_id": "critic-safe",
                    "kind": "add_support_edge",
                    "target_ids": ["N002", "N001"],
                    "payload": {"branch_id": "B001"},
                },
                {
                    "candidate_id": "critic-unsafe",
                    "kind": "brand_new_action_kind",
                    "target_ids": ["N001", "N404"],
                    "payload": {"branch_id": "B001"},
                },
            ],
            heuristic_candidate_id="heuristic",
            runtime_bundle=bundle,
            config=RelationGraphRuntimeConfig(tau_override=0.05, use_commit=False),
        )

        self.assertEqual(decision.policy_decision.selected_candidate_id, "critic-safe")
        self.assertEqual(decision.policy_decision.selected_source, "critic")
        self.assertFalse(decision.policy_decision.used_heuristic_fallback)
        self.assertEqual(bundle.score_call_count, 1)
        scored = {str(row["candidate_id"]): row for row in decision.scored_candidates}
        self.assertEqual(scored["critic-unsafe"]["critic_score"], float("-inf"))
        self.assertEqual(scored["critic-unsafe"]["controller_fallback_reason"], "unmapped_runtime_token")
        self.assertEqual(scored["critic-safe"]["critic_score"], 0.88)
        self.assertEqual(scored["heuristic"]["critic_score"], 0.40)

    def test_select_relation_graph_critic_candidate_falls_back_when_heuristic_row_is_unsafe(self) -> None:
        graph = _build_runtime_graph()
        bundle = _StubRuntimeBundle(scores=[0.99], vocabularies=self.vocabularies)
        decision = select_relation_graph_critic_candidate(
            graph,
            round_name="Round3",
            role="MechanismProposer",
            state_features={
                "round_index": 3,
                "support_coverage": 0.72,
                "unresolved_contradiction_ratio": 0.0,
            },
            candidate_specs=[
                {
                    "candidate_id": "heuristic",
                    "kind": "brand_new_action_kind",
                    "target_ids": ["N001", "N404"],
                    "payload": {"branch_id": "B001"},
                },
                {
                    "candidate_id": "critic-safe",
                    "kind": "add_support_edge",
                    "target_ids": ["N002", "N001"],
                    "payload": {"branch_id": "B001"},
                },
            ],
            heuristic_candidate_id="heuristic",
            runtime_bundle=bundle,
            config=RelationGraphRuntimeConfig(tau_override=0.05, use_commit=False),
        )

        self.assertEqual(decision.policy_decision.selected_candidate_id, "heuristic")
        self.assertEqual(decision.policy_decision.selected_source, "heuristic")
        self.assertTrue(decision.policy_decision.used_heuristic_fallback)
        self.assertEqual(decision.selected_spec["controller_fallback_reason"], "unmapped_runtime_token")
        self.assertEqual(bundle.score_call_count, 0)

    def test_select_relation_graph_critic_candidate_preserves_gain_metadata(self) -> None:
        graph = _build_runtime_graph()
        decision = select_relation_graph_critic_candidate(
            graph,
            round_name="Round3",
            role="MechanismProposer",
            state_features={
                "round_index": 3,
                "support_coverage": 0.70,
                "unresolved_contradiction_ratio": 0.0,
            },
            candidate_specs=[
                {
                    "candidate_id": "heuristic",
                    "kind": "add_support_edge",
                    "target_ids": ["N001", "N002"],
                    "payload": {"branch_id": "B001"},
                    "predicted_gain": 0.50,
                    "support_gain": 0.00,
                    "contradiction_gain": 0.00,
                    "maturity_gain": 1.0,
                    "after_is_mature": True,
                    "after_subgraph": {"is_mature": False},
                },
                {
                    "candidate_id": "critic-safe",
                    "kind": "add_support_edge",
                    "target_ids": ["N002", "N001"],
                    "payload": {"branch_id": "B001"},
                    "predicted_gain": 0.75,
                    "support_gain": 0.20,
                    "contradiction_gain": 0.00,
                    "maturity_gain": 0.0,
                    "after_is_mature": False,
                    "after_subgraph": {"is_mature": True},
                },
            ],
            heuristic_candidate_id="heuristic",
            runtime_bundle=_StubRuntimeBundle(scores=[0.45, 0.80], vocabularies=self.vocabularies),
            config=RelationGraphRuntimeConfig(tau_override=0.05, use_commit=False),
        )

        self.assertEqual(decision.selected_spec["candidate_id"], "critic-safe")
        self.assertEqual(decision.policy_decision.selected_candidate_id, "critic-safe")
        scored = {str(row["candidate_id"]): row for row in decision.scored_candidates}
        self.assertEqual(scored["critic-safe"]["predicted_gain"], 0.75)
        self.assertEqual(scored["critic-safe"]["support_gain"], 0.20)
        self.assertEqual(scored["critic-safe"]["contradiction_gain"], 0.00)
        self.assertEqual(scored["critic-safe"]["maturity_gain"], 0.0)
        self.assertTrue(scored["heuristic"]["after_is_mature"])
        self.assertFalse(scored["critic-safe"]["after_is_mature"])

    def test_select_relation_graph_critic_candidate_disables_commit_by_default(self) -> None:
        graph = _build_runtime_graph()
        decision = select_relation_graph_critic_candidate(
            graph,
            round_name="Round3",
            role="MechanismProposer",
            state_features={"round_index": 3},
            candidate_specs=[
                {
                    "candidate_id": "heuristic",
                    "kind": "add_support_edge",
                    "target_ids": ["N001", "N002"],
                    "payload": {"branch_id": "B001"},
                },
                {
                    "candidate_id": "commit-now",
                    "kind": "commit",
                    "target_ids": [],
                    "payload": {"branch_id": "B001"},
                },
            ],
            heuristic_candidate_id="heuristic",
            runtime_bundle=_StubRuntimeBundle(scores=[0.20], vocabularies=self.vocabularies),
            config=RelationGraphRuntimeConfig(),
        )

        self.assertEqual(decision.policy_decision.selected_candidate_id, "heuristic")
        self.assertTrue(all(str(row.get("kind", "")) != "commit" for row in decision.scored_candidates))

    def test_select_relation_graph_critic_candidate_records_predicted_gain_guard_fallback(self) -> None:
        graph = _build_runtime_graph()
        decision = select_relation_graph_critic_candidate(
            graph,
            round_name="Round3",
            role="MechanismProposer",
            state_features={
                "round_index": 3,
                "support_coverage": 0.60,
                "unresolved_contradiction_ratio": 0.0,
            },
            candidate_specs=[
                {
                    "candidate_id": "heuristic",
                    "kind": "add_support_edge",
                    "target_ids": ["N001", "N002"],
                    "payload": {"branch_id": "B001"},
                    "predicted_gain": 0.80,
                },
                {
                    "candidate_id": "critic-low-gain",
                    "kind": "add_support_edge",
                    "target_ids": ["N002", "N001"],
                    "payload": {"branch_id": "B001"},
                    "predicted_gain": 0.0,
                },
            ],
            heuristic_candidate_id="heuristic",
            runtime_bundle=_StubRuntimeBundle(scores=[0.10, 0.90], vocabularies=self.vocabularies),
            config=RelationGraphRuntimeConfig(tau_override=0.05, use_commit=False),
        )

        self.assertEqual(decision.policy_decision.selected_candidate_id, "heuristic")
        self.assertEqual(decision.policy_decision.selected_source, "heuristic")
        self.assertTrue(decision.policy_decision.used_heuristic_fallback)
        self.assertEqual(decision.selected_spec["controller_fallback_reason"], "predicted_gain_guard")

    def test_select_relation_graph_critic_candidate_blocks_low_signal_kind_swap(self) -> None:
        graph = _build_runtime_graph()
        vocabularies = RelationGraphVocabularies(
            node_type_to_id=dict(self.vocabularies.node_type_to_id),
            role_to_id=dict(self.vocabularies.role_to_id),
            edge_type_to_id=dict(self.vocabularies.edge_type_to_id),
            candidate_kind_to_id={
                "request_evidence": 0,
                "add_support_edge": 1,
                "unknown": 2,
            },
        )
        decision = select_relation_graph_critic_candidate(
            graph,
            round_name="Round3",
            role="MechanismProposer",
            state_features={
                "round_index": 3,
                "support_coverage": 0.60,
                "unresolved_contradiction_ratio": 0.0,
            },
            candidate_specs=[
                {
                    "candidate_id": "heuristic",
                    "kind": "request_evidence",
                    "target_ids": ["N001"],
                    "payload": {"branch_id": "B001"},
                    "predicted_gain": 0.0,
                },
                {
                    "candidate_id": "critic-low-signal",
                    "kind": "add_support_edge",
                    "target_ids": ["N002", "N001"],
                    "payload": {"branch_id": "B001"},
                    "predicted_gain": 0.0,
                },
            ],
            heuristic_candidate_id="heuristic",
            runtime_bundle=_StubRuntimeBundle(scores=[0.10, 0.90], vocabularies=vocabularies),
            config=RelationGraphRuntimeConfig(tau_override=0.05, use_commit=False),
        )

        self.assertEqual(decision.policy_decision.selected_candidate_id, "heuristic")
        self.assertEqual(decision.policy_decision.selected_source, "heuristic")
        self.assertTrue(decision.policy_decision.used_heuristic_fallback)
        self.assertEqual(decision.selected_spec["controller_fallback_reason"], "low_signal_kind_swap_guard")


if __name__ == "__main__":
    unittest.main()
