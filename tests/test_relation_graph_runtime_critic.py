from __future__ import annotations

import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
for candidate in (str(SRC), str(TESTS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from idea_graph.models import Branch, Edge, IdeaGraph, Node
from idea_graph.relation_graph_critic_data import (  # type: ignore[attr-defined]
    HashTextEmbeddingBackend,
    build_relation_graph_runtime_batch,
    build_relation_graph_vocabularies,
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


if __name__ == "__main__":
    unittest.main()
