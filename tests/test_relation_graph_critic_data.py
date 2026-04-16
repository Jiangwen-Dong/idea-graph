from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.fs_utils import _windows_safe_path, write_text_file
from idea_graph.relation_graph_critic_data import (  # type: ignore[import-not-found]
    HashTextEmbeddingBackend,
    build_relation_graph_dataset,
)


@dataclass(frozen=True)
class RelationGraphFixture:
    candidate_dir: Path
    g1_dir: Path
    partition_manifest: Path


class RecordingHashTextEmbeddingBackend(HashTextEmbeddingBackend):
    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim=dim)
        self.encoded_texts: list[str] = []

    def encode(self, texts):  # type: ignore[no-untyped-def]
        self.encoded_texts = [str(text) for text in texts]
        return super().encode(texts)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    write_text_file(path, "\n".join(json.dumps(row) for row in rows))


def write_relation_graph_fixture(root: Path) -> RelationGraphFixture:
    candidate_dir = root / "g25"
    g1_dir = root / "g1"
    partition_manifest = root / "partition_manifest.jsonl"

    train_run = "C:/tmp/run-train"
    dev_run = "C:/tmp/run-dev"
    repeated_text = "shared evidence anchor"

    candidate_rows = [
        {
            "state_id": f"{train_run}::step:0000::Round1::MechanismProposer",
            "candidate_id": f"{train_run}::step:0000::Round1::MechanismProposer::candidate:0000",
            "group_id": "AI_Idea_Bench_2025::train-case",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "train-case",
            "run_dir": train_run,
            "step_index": 0,
            "round_name": "Round1",
            "role": "MechanismProposer",
            "state_kind": "pre_action",
            "state_text": repeated_text,
            "candidate_count": 2,
            "candidate_kind": "add_support_edge",
            "candidate_target_ids": ["N001", " N002 "],
            "candidate_text": repeated_text,
            "is_commit": False,
            "is_logged_selected": True,
            "is_commit_positive_state": False,
        },
        {
            "state_id": f"{train_run}::step:0001::Terminal::CommitController",
            "candidate_id": f"{train_run}::step:0001::Terminal::CommitController::candidate:0000",
            "group_id": "AI_Idea_Bench_2025::train-case",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "train-case",
            "run_dir": train_run,
            "step_index": 1,
            "round_name": "Terminal",
            "role": "CommitController",
            "state_kind": "terminal_commit",
            "state_text": repeated_text,
            "candidate_count": 1,
            "candidate_kind": "commit",
            "candidate_target_ids": [],
            "candidate_text": repeated_text,
            "is_commit": True,
            "is_logged_selected": True,
            "is_commit_positive_state": True,
        },
        {
            "state_id": f"{dev_run}::step:0000::Round1::MechanismProposer",
            "candidate_id": f"{dev_run}::step:0000::Round1::MechanismProposer::candidate:0000",
            "group_id": "AI_Idea_Bench_2025::dev-case",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "dev-case",
            "run_dir": dev_run,
            "step_index": 0,
            "round_name": "Round1",
            "role": "MechanismProposer",
            "state_kind": "pre_action",
            "state_text": repeated_text,
            "candidate_count": 2,
            "candidate_kind": "add_support_edge",
            "candidate_target_ids": ["N001", " N002 "],
            "candidate_text": repeated_text,
            "is_commit": False,
            "is_logged_selected": True,
            "is_commit_positive_state": False,
        },
        {
            "state_id": f"{dev_run}::step:0001::Terminal::CommitController",
            "candidate_id": f"{dev_run}::step:0001::Terminal::CommitController::candidate:0000",
            "group_id": "AI_Idea_Bench_2025::dev-case",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "dev-case",
            "run_dir": dev_run,
            "step_index": 1,
            "round_name": "Terminal",
            "role": "CommitController",
            "state_kind": "terminal_commit",
            "state_text": repeated_text,
            "candidate_count": 1,
            "candidate_kind": "commit",
            "candidate_target_ids": [],
            "candidate_text": repeated_text,
            "is_commit": True,
            "is_logged_selected": True,
            "is_commit_positive_state": True,
        },
    ]

    trajectory_rows = [
        {
            "run_dir": train_run,
            "step_index": 0,
            "round_name": "Round1",
            "role": "MechanismProposer",
            "before_state_snapshot": "state_snapshots/train-step-000.json",
        },
        {
            "run_dir": dev_run,
            "step_index": 0,
            "round_name": "Round1",
            "role": "MechanismProposer",
            "before_state_snapshot": "state_snapshots/dev-step-000.json",
        },
    ]
    terminal_rows = [
        {
            "run_dir": train_run,
            "step_index": 1,
            "round_name": "Terminal",
            "role": "CommitController",
            "before_state_snapshot": "terminal_state_snapshots/train-terminal.json",
        },
        {
            "run_dir": dev_run,
            "step_index": 1,
            "round_name": "Terminal",
            "role": "CommitController",
            "before_state_snapshot": "terminal_state_snapshots/dev-terminal.json",
        },
    ]
    partition_rows = [
        {
            "group_id": "AI_Idea_Bench_2025::train-case",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "train-case",
            "source_split": "train",
            "partition_role": "critic_train",
        },
        {
            "group_id": "AI_Idea_Bench_2025::dev-case",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "dev-case",
            "source_split": "validation",
            "partition_role": "critic_dev",
        },
    ]

    snapshot_payload = {
        "node_count": 3,
        "edge_count": 2,
        "contradiction_count": 0,
        "support_edge_count": 1,
        "nodes": {
            "N002": {
                "id": "N002",
                "type": "Method",
                "text": repeated_text,
                "role": "MechanismProposer",
                "branch_id": "B001",
                "confidence": 0.7,
                "evidence": [repeated_text],
                "status": "active",
            },
            "N001": {
                "id": "N001",
                "type": "Hypothesis",
                "text": repeated_text,
                "role": "MechanismProposer",
                "branch_id": "B001",
                "confidence": 0.8,
                "evidence": [repeated_text],
                "status": "active",
            },
            "N003": {
                "id": "N003",
                "type": "EvalPlan",
                "text": repeated_text,
                "role": "EvaluationDesigner",
                "branch_id": "B002",
                "confidence": 0.6,
                "evidence": [],
                "status": "active",
            },
        },
        "edges": [
            {
                "id": "E002",
                "source_id": "N002",
                "target_id": "N003",
                "relation": "supports",
                "resolved": False,
            },
            {
                "id": "E001",
                "source_id": "N001",
                "target_id": "N002",
                "relation": "depends_on",
                "resolved": True,
            },
        ],
    }

    candidate_dir.mkdir(parents=True, exist_ok=True)
    g1_dir.mkdir(parents=True, exist_ok=True)
    (g1_dir / "state_snapshots").mkdir(parents=True, exist_ok=True)
    (g1_dir / "terminal_state_snapshots").mkdir(parents=True, exist_ok=True)
    _write_jsonl(_windows_safe_path(candidate_dir / "candidate_dataset.jsonl"), candidate_rows)
    _write_jsonl(_windows_safe_path(g1_dir / "trajectory_examples.jsonl"), trajectory_rows)
    _write_jsonl(_windows_safe_path(g1_dir / "terminal_state_manifest.jsonl"), terminal_rows)
    _write_jsonl(_windows_safe_path(partition_manifest), partition_rows)
    write_text_file(g1_dir / "state_snapshots" / "train-step-000.json", json.dumps(snapshot_payload))
    write_text_file(g1_dir / "terminal_state_snapshots" / "train-terminal.json", json.dumps(snapshot_payload))
    write_text_file(g1_dir / "state_snapshots" / "dev-step-000.json", json.dumps(snapshot_payload))
    write_text_file(g1_dir / "terminal_state_snapshots" / "dev-terminal.json", json.dumps(snapshot_payload))
    return RelationGraphFixture(
        candidate_dir=candidate_dir,
        g1_dir=g1_dir,
        partition_manifest=partition_manifest,
    )


class RelationGraphCriticDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(mkdtemp())
        self.fixture = write_relation_graph_fixture(self.tmp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_build_relation_graph_dataset_uses_partition_roles_and_target_indices(self) -> None:
        dataset = build_relation_graph_dataset(
            candidate_dataset_dir=self.fixture.candidate_dir,
            g1_dataset_dir=self.fixture.g1_dir,
            partition_manifest_path=self.fixture.partition_manifest,
            text_backend=HashTextEmbeddingBackend(dim=8),
        )

        self.assertEqual(len(dataset.train_examples), 2)
        self.assertEqual(len(dataset.dev_examples), 2)
        example = dataset.train_examples[0]
        self.assertEqual(example.group_id, "AI_Idea_Bench_2025::train-case")
        self.assertEqual(example.target_node_indices, [0, 1])
        self.assertEqual(example.node_text_embeddings.shape, (3, 8))
        self.assertEqual(example.edge_index, [(0, 1), (1, 2)])
        self.assertEqual(example.edge_type_ids, [0, 1])

    def test_build_relation_graph_dataset_reuses_cached_text_embeddings(self) -> None:
        backend = HashTextEmbeddingBackend(dim=8)
        dataset = build_relation_graph_dataset(
            candidate_dataset_dir=self.fixture.candidate_dir,
            g1_dataset_dir=self.fixture.g1_dir,
            partition_manifest_path=self.fixture.partition_manifest,
            text_backend=backend,
        )

        self.assertEqual(backend.encode_call_count, 1)
        self.assertTrue(np.isfinite(dataset.train_examples[0].state_text_embedding).all())

    def test_build_relation_graph_dataset_strips_leaky_candidate_metadata_before_embedding(self) -> None:
        candidate_rows = [
            json.loads(line)
            for line in (self.fixture.candidate_dir / "candidate_dataset.jsonl").read_text().splitlines()
            if line.strip()
        ]
        candidate_rows[0]["candidate_text"] = "kind=add_support_edge | targets=core-target"
        candidate_rows[2]["candidate_text"] = (
            "kind=add_support_edge | targets=core-target | "
            "rationale=teacher-only hint | source=utility_add_support"
        )
        _write_jsonl(_windows_safe_path(self.fixture.candidate_dir / "candidate_dataset.jsonl"), candidate_rows)

        backend = RecordingHashTextEmbeddingBackend(dim=8)
        dataset = build_relation_graph_dataset(
            candidate_dataset_dir=self.fixture.candidate_dir,
            g1_dataset_dir=self.fixture.g1_dir,
            partition_manifest_path=self.fixture.partition_manifest,
            text_backend=backend,
        )

        np.testing.assert_allclose(
            dataset.train_examples[0].candidate_text_embedding,
            dataset.dev_examples[0].candidate_text_embedding,
        )
        self.assertTrue(all("source=" not in text for text in backend.encoded_texts))
        self.assertTrue(all("rationale=" not in text for text in backend.encoded_texts))

    def test_build_relation_graph_dataset_handles_empty_node_snapshots(self) -> None:
        empty_snapshot = {
            "node_count": 0,
            "edge_count": 0,
            "contradiction_count": 0,
            "support_edge_count": 0,
            "nodes": {},
            "edges": [],
        }
        write_text_file(
            self.fixture.g1_dir / "state_snapshots" / "train-step-000.json",
            json.dumps(empty_snapshot),
        )

        dataset = build_relation_graph_dataset(
            candidate_dataset_dir=self.fixture.candidate_dir,
            g1_dataset_dir=self.fixture.g1_dir,
            partition_manifest_path=self.fixture.partition_manifest,
            text_backend=HashTextEmbeddingBackend(dim=8),
        )

        self.assertEqual(dataset.train_examples[0].node_text_embeddings.shape, (0, 8))

    def test_build_relation_graph_dataset_reads_parallel_state_snapshots(self) -> None:
        candidate_rows = [
            json.loads(line)
            for line in (self.fixture.candidate_dir / "candidate_dataset.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in candidate_rows:
            state_id = str(row["state_id"])
            if "::Terminal::" in state_id:
                continue
            run_dir = str(row["run_dir"])
            row["state_id"] = f"{run_dir}::parallel::Round1::{row['role']}"
            row["before_state_snapshot"] = "parallel_state_snapshots/train-parallel-000.json"
            if "dev-case" in str(row["group_id"]):
                row["state_id"] = f"{run_dir}::parallel::Round1::{row['role']}"
                row["before_state_snapshot"] = "parallel_state_snapshots/dev-parallel-000.json"
        _write_jsonl(_windows_safe_path(self.fixture.candidate_dir / "candidate_dataset.jsonl"), candidate_rows)

        parallel_rows = [
            {
                "run_dir": "C:/tmp/run-train",
                "state_id": "C:/tmp/run-train::parallel::Round1::MechanismProposer",
                "round_name": "Round1",
                "role": "MechanismProposer",
                "before_state_snapshot": "parallel_state_snapshots/train-parallel-000.json",
            },
            {
                "run_dir": "C:/tmp/run-dev",
                "state_id": "C:/tmp/run-dev::parallel::Round1::MechanismProposer",
                "round_name": "Round1",
                "role": "MechanismProposer",
                "before_state_snapshot": "parallel_state_snapshots/dev-parallel-000.json",
            },
        ]
        (self.fixture.g1_dir / "parallel_state_snapshots").mkdir(parents=True, exist_ok=True)
        _write_jsonl(_windows_safe_path(self.fixture.g1_dir / "parallel_edit_examples.jsonl"), parallel_rows)
        write_text_file(self.fixture.g1_dir / "parallel_state_snapshots" / "train-parallel-000.json", json.dumps({
            "node_count": 3,
            "edge_count": 2,
            "contradiction_count": 0,
            "support_edge_count": 1,
            "nodes": {
                "N001": {"id": "N001", "type": "Hypothesis", "text": "shared evidence anchor", "role": "MechanismProposer", "branch_id": "B001", "confidence": 0.8, "evidence": [], "status": "active"},
                "N002": {"id": "N002", "type": "Method", "text": "shared evidence anchor", "role": "MechanismProposer", "branch_id": "B001", "confidence": 0.7, "evidence": [], "status": "active"},
                "N003": {"id": "N003", "type": "EvalPlan", "text": "shared evidence anchor", "role": "EvaluationDesigner", "branch_id": "B002", "confidence": 0.6, "evidence": [], "status": "active"},
            },
            "edges": [
                {"id": "E001", "source_id": "N001", "target_id": "N002", "relation": "depends_on", "resolved": True},
                {"id": "E002", "source_id": "N002", "target_id": "N003", "relation": "supports", "resolved": False},
            ],
        }))
        write_text_file(self.fixture.g1_dir / "parallel_state_snapshots" / "dev-parallel-000.json", json.dumps({
            "node_count": 3,
            "edge_count": 2,
            "contradiction_count": 0,
            "support_edge_count": 1,
            "nodes": {
                "N001": {"id": "N001", "type": "Hypothesis", "text": "shared evidence anchor", "role": "MechanismProposer", "branch_id": "B001", "confidence": 0.8, "evidence": [], "status": "active"},
                "N002": {"id": "N002", "type": "Method", "text": "shared evidence anchor", "role": "MechanismProposer", "branch_id": "B001", "confidence": 0.7, "evidence": [], "status": "active"},
                "N003": {"id": "N003", "type": "EvalPlan", "text": "shared evidence anchor", "role": "EvaluationDesigner", "branch_id": "B002", "confidence": 0.6, "evidence": [], "status": "active"},
            },
            "edges": [
                {"id": "E001", "source_id": "N001", "target_id": "N002", "relation": "depends_on", "resolved": True},
                {"id": "E002", "source_id": "N002", "target_id": "N003", "relation": "supports", "resolved": False},
            ],
        }))

        dataset = build_relation_graph_dataset(
            candidate_dataset_dir=self.fixture.candidate_dir,
            g1_dataset_dir=self.fixture.g1_dir,
            partition_manifest_path=self.fixture.partition_manifest,
            text_backend=HashTextEmbeddingBackend(dim=8),
        )

        self.assertEqual(len(dataset.train_examples), 2)
        self.assertEqual(len(dataset.dev_examples), 2)
