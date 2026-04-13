# Relation-Aware Graph Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline-only relation-aware graph critic with frozen node and candidate text embeddings, train it on `development_pool_v2_combined_g25`, and compare it against the refreshed text scorer and the first graph-feature scorer on the frozen `critic_dev` groups.

**Architecture:** A new dataset loader will convert `G1` state snapshots plus `G2.5` candidate rows into per-state graph examples with cached text embeddings, typed nodes, typed edges, and target-node indices. A small PyTorch model will run two layers of relation-aware message passing, pool target and one-hop neighborhood nodes, and rank candidates within each state using a state-local softmax loss. The full stage remains offline and uses the same frozen train/dev partition and offline gate as the current baselines.

**Tech Stack:** Python 3.10, PyTorch, sentence-transformers, scikit-learn-compatible JSON artifact writing, pytest

---

## File Structure

### New Files

- `src/idea_graph/relation_graph_critic_data.py`
  - Load frozen `G2.5` candidate rows and `G1` snapshots
  - Build vocab ids for node types, roles, edge relations, and candidate kinds
  - Cache frozen text embeddings for node text, candidate text, and state text
  - Produce typed graph examples and PyTorch-ready mini-batches
- `src/idea_graph/relation_graph_critic_model.py`
  - Define the relation-aware graph encoder
  - Define target pooling, neighborhood pooling, and scoring head
- `src/idea_graph/relation_graph_critic_train.py`
  - Define state-local ranking loss
  - Define training loop, validation loop, and ranking metrics
  - Save model bundle and metrics
- `scripts/train_relation_graph_critic.py`
  - CLI entrypoint for offline training on the frozen split
- `tests/test_relation_graph_critic_data.py`
  - Unit tests for dataset construction, partition safety, and embedding caching
- `tests/test_relation_graph_critic_model.py`
  - Unit tests for graph batching, forward pass shapes, and target-aware pooling
- `tests/test_relation_graph_critic_train.py`
  - Unit tests for ranking loss, all-candidate metrics, and edit-only metrics
- `tests/test_train_relation_graph_critic.py`
  - CLI smoke test for artifact writing

### Modified Files

- `pyproject.toml`
  - Add `torch` and `sentence-transformers`
- `docs/experiment_execution_log.md`
  - Record the offline relation-aware graph critic run and the gate decision
- `docs/eig_graph_critic_plan.md`
  - Record the stronger graph line and whether it clears the offline gate

## Execution Rules

- Keep the offline roots fixed:
  - `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g25`
  - `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g1`
  - `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g2_partitions/partition_manifest.jsonl`
- Keep runtime controller integration out of scope.
- Keep learned runtime `commit` out of scope.
- Report both:
  - all-candidate validation metrics
  - edit-only validation metrics

### Task 1: Add The Dataset And Embedding Pipeline

**Files:**
- Modify: `pyproject.toml`
- Create: `src/idea_graph/relation_graph_critic_data.py`
- Test: `tests/test_relation_graph_critic_data.py`

- [ ] **Step 1: Write the failing data-loader tests**

```python
from pathlib import Path

import numpy as np

from idea_graph.relation_graph_critic_data import (
    HashTextEmbeddingBackend,
    build_relation_graph_dataset,
)


def test_build_relation_graph_dataset_uses_partition_roles_and_target_indices(tmp_path: Path) -> None:
    fixture = write_relation_graph_fixture(tmp_path)
    dataset = build_relation_graph_dataset(
        candidate_dataset_dir=fixture.candidate_dir,
        g1_dataset_dir=fixture.g1_dir,
        partition_manifest_path=fixture.partition_manifest,
        text_backend=HashTextEmbeddingBackend(dim=8),
    )
    assert len(dataset.train_examples) == 4
    assert len(dataset.dev_examples) == 4
    example = dataset.train_examples[0]
    assert example.group_id == "AI_Idea_Bench_2025::train-case"
    assert example.target_node_indices == [0, 1]
    assert example.node_text_embeddings.shape == (3, 8)
    assert example.edge_index == [(0, 1), (1, 2)]
    assert example.edge_type_ids == [0, 1]


def test_build_relation_graph_dataset_reuses_cached_text_embeddings(tmp_path: Path) -> None:
    fixture = write_relation_graph_fixture(tmp_path)
    backend = HashTextEmbeddingBackend(dim=8)
    dataset = build_relation_graph_dataset(
        candidate_dataset_dir=fixture.candidate_dir,
        g1_dataset_dir=fixture.g1_dir,
        partition_manifest_path=fixture.partition_manifest,
        text_backend=backend,
    )
    assert backend.encode_call_count == 1
    assert np.isfinite(dataset.train_examples[0].state_text_embedding).all()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_relation_graph_critic_data.py -q
```

Expected:

```text
E   ImportError: cannot import name 'build_relation_graph_dataset'
```

- [ ] **Step 3: Add dependencies and write the minimal dataset implementation**

Update `pyproject.toml`:

```toml
dependencies = [
    "scikit-learn>=1.5,<2",
    "torch>=2.4,<3",
    "sentence-transformers>=3,<4",
]
```

Create `src/idea_graph/relation_graph_critic_data.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from .fs_utils import read_text_file
from .online_text_critic import build_partition_role_lookup, load_partition_manifest_rows
from .candidate_slate_dataset import state_id_from_transition


class TextEmbeddingBackend(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class HashTextEmbeddingBackend:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.encode_call_count = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.encode_call_count += 1
        rows: list[np.ndarray] = []
        for text in texts:
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            values = np.frombuffer(seed, dtype=np.uint8).astype(np.float32)
            tiled = np.resize(values, self.dim)
            rows.append((tiled / 255.0).astype(np.float32))
        return np.stack(rows, axis=0) if rows else np.zeros((0, self.dim), dtype=np.float32)


@dataclass(frozen=True)
class RelationGraphCandidateExample:
    state_id: str
    candidate_id: str
    group_id: str
    split: str
    label: int
    is_commit: bool
    candidate_kind_id: int
    candidate_text_embedding: np.ndarray
    state_text_embedding: np.ndarray
    node_text_embeddings: np.ndarray
    node_type_ids: list[int]
    node_role_ids: list[int]
    node_confidence: list[float]
    node_evidence_count: list[float]
    edge_index: list[tuple[int, int]]
    edge_type_ids: list[int]
    edge_resolved: list[float]
    target_node_indices: list[int]


@dataclass(frozen=True)
class RelationGraphDataset:
    train_examples: list[RelationGraphCandidateExample]
    dev_examples: list[RelationGraphCandidateExample]
    metadata: dict[str, Any]


def build_relation_graph_dataset(
    *,
    candidate_dataset_dir: Path,
    g1_dataset_dir: Path,
    partition_manifest_path: Path,
    text_backend: TextEmbeddingBackend,
) -> RelationGraphDataset:
    candidate_rows = _load_jsonl(Path(candidate_dataset_dir) / "candidate_dataset.jsonl")
    snapshot_lookup = _build_snapshot_lookup(Path(g1_dataset_dir))
    partition_lookup = build_partition_role_lookup(load_partition_manifest_rows(Path(partition_manifest_path)))
    node_type_vocab: dict[str, int] = {}
    role_vocab: dict[str, int] = {}
    edge_type_vocab: dict[str, int] = {}
    candidate_kind_vocab: dict[str, int] = {}
    embedding_cache: dict[str, np.ndarray] = {}
    train_examples: list[RelationGraphCandidateExample] = []
    dev_examples: list[RelationGraphCandidateExample] = []
    for row in candidate_rows:
        group_id = str(row["group_id"])
        split = partition_lookup[group_id]
        if split == "paper_eval":
            continue
        snapshot = snapshot_lookup[str(row["state_id"])]
        example = _build_relation_graph_example(
            row=row,
            snapshot=snapshot,
            split=split,
            node_type_vocab=node_type_vocab,
            role_vocab=role_vocab,
            edge_type_vocab=edge_type_vocab,
            candidate_kind_vocab=candidate_kind_vocab,
            embedding_cache=embedding_cache,
            text_backend=text_backend,
        )
        if split == "critic_train":
            train_examples.append(example)
        else:
            dev_examples.append(example)
    return RelationGraphDataset(
        train_examples=train_examples,
        dev_examples=dev_examples,
        metadata={
            "node_type_count": len(node_type_vocab),
            "role_count": len(role_vocab),
            "edge_type_count": len(edge_type_vocab),
            "candidate_kind_count": len(candidate_kind_vocab),
        },
    )
```

Implementation details to include in the real code:

- load `candidate_dataset.jsonl`
- load `trajectory_examples.jsonl` plus `terminal_state_manifest.jsonl`
- map `state_id -> snapshot`
- cache embeddings by exact text string
- ignore `paper_eval`
- preserve leakage-safe partition roles
- map target node ids into stable node indices per state

- [ ] **Step 4: Run the dataset tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_relation_graph_critic_data.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit the dataset pipeline**

```powershell
git add pyproject.toml src/idea_graph/relation_graph_critic_data.py tests/test_relation_graph_critic_data.py
git commit -m "feat: add relation graph critic dataset pipeline"
```

### Task 2: Add The Relation-Aware Graph Encoder

**Files:**
- Create: `src/idea_graph/relation_graph_critic_model.py`
- Modify: `src/idea_graph/relation_graph_critic_data.py`
- Test: `tests/test_relation_graph_critic_model.py`

- [ ] **Step 1: Write the failing model tests**

```python
import torch

from idea_graph.relation_graph_critic_data import (
    HashTextEmbeddingBackend,
    build_relation_graph_dataset,
    collate_relation_graph_examples,
)
from idea_graph.relation_graph_critic_model import RelationGraphCritic


def test_relation_graph_critic_forward_returns_one_score_per_candidate(tmp_path) -> None:
    fixture = write_relation_graph_fixture(tmp_path)
    dataset = build_relation_graph_dataset(
        candidate_dataset_dir=fixture.candidate_dir,
        g1_dataset_dir=fixture.g1_dir,
        partition_manifest_path=fixture.partition_manifest,
        text_backend=HashTextEmbeddingBackend(dim=8),
    )
    batch = collate_relation_graph_examples(dataset.train_examples[:4])
    model = RelationGraphCritic(
        text_dim=8,
        hidden_dim=16,
        node_type_count=batch.node_type_vocab_size,
        role_count=batch.role_vocab_size,
        edge_type_count=batch.edge_type_vocab_size,
        candidate_kind_count=batch.candidate_kind_vocab_size,
    )
    scores = model(batch)
    assert tuple(scores.shape) == (4,)


def test_relation_graph_critic_changes_score_when_target_indices_change(tmp_path) -> None:
    fixture = write_relation_graph_fixture(tmp_path)
    dataset = build_relation_graph_dataset(
        candidate_dataset_dir=fixture.candidate_dir,
        g1_dataset_dir=fixture.g1_dir,
        partition_manifest_path=fixture.partition_manifest,
        text_backend=HashTextEmbeddingBackend(dim=8),
    )
    batch = collate_relation_graph_examples(dataset.train_examples[:2])
    model = RelationGraphCritic(
        text_dim=8,
        hidden_dim=16,
        node_type_count=batch.node_type_vocab_size,
        role_count=batch.role_vocab_size,
        edge_type_count=batch.edge_type_vocab_size,
        candidate_kind_count=batch.candidate_kind_vocab_size,
    )
    left = model(batch).detach().clone()
    batch.target_mask = batch.target_mask.roll(shifts=1, dims=1)
    right = model(batch).detach().clone()
    assert not torch.allclose(left, right)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_relation_graph_critic_model.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'idea_graph.relation_graph_critic_model'
```

- [ ] **Step 3: Implement batching plus the relation-aware model**

Add to `src/idea_graph/relation_graph_critic_data.py`:

```python
@dataclass
class RelationGraphBatch:
    node_text_embeddings: torch.Tensor
    state_text_embeddings: torch.Tensor
    candidate_text_embeddings: torch.Tensor
    node_type_ids: torch.Tensor
    node_role_ids: torch.Tensor
    node_scalars: torch.Tensor
    edge_index: torch.Tensor
    edge_type_ids: torch.Tensor
    edge_resolved: torch.Tensor
    candidate_kind_ids: torch.Tensor
    target_mask: torch.Tensor
    neighbor_mask: torch.Tensor
    graph_mask: torch.Tensor
    candidate_state_index: torch.Tensor
    labels: torch.Tensor
    is_commit: torch.Tensor
    node_type_vocab_size: int
    role_vocab_size: int
    edge_type_vocab_size: int
    candidate_kind_vocab_size: int


def collate_relation_graph_examples(
    examples: Sequence[RelationGraphCandidateExample],
) -> RelationGraphBatch:
    max_nodes = max(len(example.node_type_ids) for example in examples)
    total_edges = sum(len(example.edge_index) for example in examples)
    node_text_embeddings = torch.zeros((len(examples), max_nodes, examples[0].node_text_embeddings.shape[1]))
    node_type_ids = torch.zeros((len(examples), max_nodes), dtype=torch.long)
    node_role_ids = torch.zeros((len(examples), max_nodes), dtype=torch.long)
    node_scalars = torch.zeros((len(examples), max_nodes, 2), dtype=torch.float32)
    graph_mask = torch.zeros((len(examples), max_nodes), dtype=torch.bool)
    target_mask = torch.zeros((len(examples), max_nodes), dtype=torch.bool)
    neighbor_mask = torch.zeros((len(examples), max_nodes), dtype=torch.bool)
    candidate_text_embeddings = torch.zeros((len(examples), examples[0].candidate_text_embedding.shape[0]))
    state_text_embeddings = torch.zeros((len(examples), examples[0].state_text_embedding.shape[0]))
    candidate_kind_ids = torch.zeros((len(examples),), dtype=torch.long)
    labels = torch.zeros((len(examples),), dtype=torch.float32)
    is_commit = torch.zeros((len(examples),), dtype=torch.float32)
    edge_index = torch.zeros((2, max(total_edges, 1)), dtype=torch.long)
    edge_type_ids = torch.zeros((max(total_edges, 1),), dtype=torch.long)
    edge_resolved = torch.zeros((max(total_edges, 1),), dtype=torch.float32)
    candidate_state_index = torch.arange(len(examples), dtype=torch.long)
    edge_cursor = 0
    for example_index, example in enumerate(examples):
        node_count = len(example.node_type_ids)
        node_text_embeddings[example_index, :node_count] = torch.from_numpy(example.node_text_embeddings)
        node_type_ids[example_index, :node_count] = torch.tensor(example.node_type_ids, dtype=torch.long)
        node_role_ids[example_index, :node_count] = torch.tensor(example.node_role_ids, dtype=torch.long)
        node_scalars[example_index, :node_count, 0] = torch.tensor(example.node_confidence, dtype=torch.float32)
        node_scalars[example_index, :node_count, 1] = torch.tensor(example.node_evidence_count, dtype=torch.float32)
        graph_mask[example_index, :node_count] = True
        for target_index in example.target_node_indices:
            target_mask[example_index, target_index] = True
        candidate_text_embeddings[example_index] = torch.from_numpy(example.candidate_text_embedding)
        state_text_embeddings[example_index] = torch.from_numpy(example.state_text_embedding)
        candidate_kind_ids[example_index] = example.candidate_kind_id
        labels[example_index] = float(example.label)
        is_commit[example_index] = float(example.is_commit)
        for local_edge_index, (src, dst) in enumerate(example.edge_index):
            edge_index[:, edge_cursor + local_edge_index] = torch.tensor([src, dst], dtype=torch.long)
            edge_type_ids[edge_cursor + local_edge_index] = example.edge_type_ids[local_edge_index]
            edge_resolved[edge_cursor + local_edge_index] = example.edge_resolved[local_edge_index]
        edge_cursor += len(example.edge_index)
    return RelationGraphBatch(
        node_text_embeddings=node_text_embeddings,
        state_text_embeddings=state_text_embeddings,
        candidate_text_embeddings=candidate_text_embeddings,
        node_type_ids=node_type_ids,
        node_role_ids=node_role_ids,
        node_scalars=node_scalars,
        edge_index=edge_index[:, :edge_cursor] if edge_cursor else edge_index[:, :1],
        edge_type_ids=edge_type_ids[:edge_cursor] if edge_cursor else edge_type_ids[:1],
        edge_resolved=edge_resolved[:edge_cursor] if edge_cursor else edge_resolved[:1],
        candidate_kind_ids=candidate_kind_ids,
        target_mask=target_mask,
        neighbor_mask=neighbor_mask,
        graph_mask=graph_mask,
        candidate_state_index=candidate_state_index,
        labels=labels,
        is_commit=is_commit,
        node_type_vocab_size=int(node_type_ids.max().item()) + 1,
        role_vocab_size=int(node_role_ids.max().item()) + 1,
        edge_type_vocab_size=int(edge_type_ids.max().item()) + 1 if edge_cursor else 1,
        candidate_kind_vocab_size=int(candidate_kind_ids.max().item()) + 1,
    )
```

Create `src/idea_graph/relation_graph_critic_model.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class RelationMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_type_count: int) -> None:
        super().__init__()
        self.edge_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(edge_type_count)]
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_states, edge_index, edge_type_ids, local_stats):
        src, dst = edge_index
        messages = torch.zeros_like(node_states)
        for relation_id, linear in enumerate(self.edge_linears):
            relation_mask = edge_type_ids == relation_id
            if relation_mask.any():
                messages.index_add_(
                    0,
                    dst[relation_mask],
                    linear(node_states[src[relation_mask]]),
                )
        updated = self.update(torch.cat([node_states, messages, local_stats], dim=-1))
        return self.norm(node_states + updated)


class RelationGraphCritic(nn.Module):
    def __init__(
        self,
        *,
        text_dim: int,
        hidden_dim: int,
        node_type_count: int,
        role_count: int,
        edge_type_count: int,
        candidate_kind_count: int,
    ) -> None:
        super().__init__()
        self.node_type_embed = nn.Embedding(node_type_count, 16)
        self.role_embed = nn.Embedding(role_count, 16)
        self.candidate_kind_embed = nn.Embedding(candidate_kind_count, 16)
        self.node_project = nn.Sequential(
            nn.Linear(text_dim + 16 + 16 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [RelationMessageLayer(hidden_dim, edge_type_count) for _ in range(2)]
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4 + text_dim * 2 + 16 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        node_inputs = torch.cat(
            [
                batch.node_text_embeddings,
                self.node_type_embed(batch.node_type_ids),
                self.role_embed(batch.node_role_ids),
                batch.node_scalars,
            ],
            dim=-1,
        )
        node_states = self.node_project(node_inputs)
        local_stats = torch.cat(
            [
                batch.target_mask.unsqueeze(-1).float(),
                batch.graph_mask.unsqueeze(-1).float(),
                batch.node_scalars[..., :1],
            ],
            dim=-1,
        )
        for layer in self.layers:
            flat_states = node_states.reshape(-1, node_states.shape[-1])
            flat_stats = local_stats.reshape(-1, local_stats.shape[-1])
            node_count = node_states.shape[1]
            state_offsets = (
                torch.arange(node_states.shape[0], device=node_states.device).repeat_interleave(batch.edge_index.shape[1])
                * node_count
            )
            flat_edge_index = batch.edge_index.repeat(1, node_states.shape[0]) + state_offsets.unsqueeze(0)
            updated = layer(flat_states, flat_edge_index, batch.edge_type_ids.repeat(node_states.shape[0]), flat_stats)
            node_states = updated.reshape_as(node_states)
        graph_summary = (node_states * batch.graph_mask.unsqueeze(-1)).sum(dim=1) / batch.graph_mask.sum(dim=1, keepdim=True).clamp_min(1)
        target_summary = (node_states * batch.target_mask.unsqueeze(-1)).sum(dim=1) / batch.target_mask.sum(dim=1, keepdim=True).clamp_min(1)
        neighbor_summary = (node_states * batch.neighbor_mask.unsqueeze(-1)).sum(dim=1) / batch.neighbor_mask.sum(dim=1, keepdim=True).clamp_min(1)
        action_summary = torch.cat(
            [
                batch.candidate_text_embeddings,
                self.candidate_kind_embed(batch.candidate_kind_ids),
                batch.is_commit.unsqueeze(-1),
            ],
            dim=-1,
        )
        score_inputs = torch.cat(
            [
                graph_summary,
                target_summary,
                neighbor_summary,
                action_summary,
                batch.state_text_embeddings,
            ],
            dim=-1,
        )
        return self.scorer(score_inputs).squeeze(-1)
```

Implementation details to include in the real code:

- use typed edge-specific message transforms
- pool target nodes with `target_mask`
- pool one-hop neighbors with `neighbor_mask`
- pool graph summary with `graph_mask`
- concatenate graph, target, neighborhood, candidate text, state text, candidate kind, and commit flag

- [ ] **Step 4: Run the model tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_relation_graph_critic_model.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit the model**

```powershell
git add src/idea_graph/relation_graph_critic_data.py src/idea_graph/relation_graph_critic_model.py tests/test_relation_graph_critic_model.py
git commit -m "feat: add relation-aware graph critic model"
```

### Task 3: Add Ranking Loss And Offline Evaluation

**Files:**
- Create: `src/idea_graph/relation_graph_critic_train.py`
- Test: `tests/test_relation_graph_critic_train.py`

- [ ] **Step 1: Write the failing training and metric tests**

```python
import torch

from idea_graph.relation_graph_critic_train import (
    compute_state_ranking_loss,
    evaluate_relation_graph_rankings,
)


def test_compute_state_ranking_loss_rewards_positive_candidate() -> None:
    scores = torch.tensor([3.0, 1.0, -2.0, -3.0], dtype=torch.float32)
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    state_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss = compute_state_ranking_loss(scores, labels, state_index)
    assert float(loss) < 0.4


def test_evaluate_relation_graph_rankings_reports_all_and_edit_only_metrics() -> None:
    metrics = evaluate_relation_graph_rankings(
        state_rows=[
            {
                "state_id": "s1",
                "candidate_id": "c1",
                "label": 1,
                "score": 0.9,
                "is_commit": False,
            },
            {
                "state_id": "s1",
                "candidate_id": "c2",
                "label": 0,
                "score": 0.2,
                "is_commit": True,
            },
        ]
    )
    assert metrics["all"]["top1_accuracy"] == 1.0
    assert metrics["edit_only"]["top1_accuracy"] == 1.0
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_relation_graph_critic_train.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'idea_graph.relation_graph_critic_train'
```

- [ ] **Step 3: Implement the trainer and evaluator**

Create `src/idea_graph/relation_graph_critic_train.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from .fs_utils import write_text_file


def compute_state_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    state_index: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for state_id in state_index.unique(sorted=True):
        mask = state_index == state_id
        state_scores = scores[mask]
        state_labels = labels[mask]
        positive_index = torch.argmax(state_labels)
        losses.append(nn.functional.cross_entropy(state_scores.unsqueeze(0), positive_index.unsqueeze(0)))
    return torch.stack(losses).mean()


def evaluate_relation_graph_rankings(state_rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in state_rows:
        by_state.setdefault(str(row["state_id"]), []).append(dict(row))

    def _score(rows: list[dict[str, Any]]) -> dict[str, float | int]:
        top1_hits = 0
        reciprocal_ranks: list[float] = []
        for state_id in sorted(by_state):
            state_candidates = [row for row in by_state[state_id] if row in rows]
            if not state_candidates:
                continue
            ranked = sorted(state_candidates, key=lambda item: (-float(item["score"]), str(item["candidate_id"])))
            positive_rank = next(
                rank for rank, row in enumerate(ranked, start=1) if int(row["label"]) == 1
            )
            reciprocal_ranks.append(1.0 / positive_rank)
            if positive_rank == 1:
                top1_hits += 1
        state_count = len(reciprocal_ranks)
        return {
            "state_count": state_count,
            "top1_accuracy": top1_hits / state_count if state_count else 0.0,
            "mean_reciprocal_rank": sum(reciprocal_ranks) / state_count if state_count else 0.0,
        }

    all_rows = list(state_rows)
    edit_rows = [row for row in state_rows if not bool(row["is_commit"])]
    return {"all": _score(all_rows), "edit_only": _score(edit_rows)}


@dataclass(frozen=True)
class RelationGraphTrainingArtifacts:
    metrics_all: dict[str, float | int]
    metrics_edit_only: dict[str, float | int]
    metadata: dict[str, Any]
```

Implementation details to include in the real code:

- train with `AdamW`
- early stop on dev MRR
- save best checkpoint by all-candidate dev MRR
- report:
  - `all.top1_accuracy`
  - `all.mean_reciprocal_rank`
  - `edit_only.top1_accuracy`
  - `edit_only.mean_reciprocal_rank`

- [ ] **Step 4: Run the training tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_relation_graph_critic_train.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit the trainer**

```powershell
git add src/idea_graph/relation_graph_critic_train.py tests/test_relation_graph_critic_train.py
git commit -m "feat: add relation graph critic training loop"
```

### Task 4: Add The Offline Training CLI And Artifact Contract

**Files:**
- Create: `scripts/train_relation_graph_critic.py`
- Modify: `src/idea_graph/relation_graph_critic_data.py`
- Modify: `src/idea_graph/relation_graph_critic_train.py`
- Test: `tests/test_train_relation_graph_critic.py`

- [ ] **Step 1: Write the failing CLI smoke test**

```python
import json
import subprocess
import sys
from pathlib import Path


def test_train_relation_graph_critic_cli_writes_artifacts(tmp_path: Path) -> None:
    fixture = write_relation_graph_fixture(tmp_path)
    output_dir = tmp_path / "model_output"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_relation_graph_critic.py",
            "--candidate-dataset-dir",
            str(fixture.candidate_dir),
            "--g1-dataset-dir",
            str(fixture.g1_dir),
            "--partition-manifest",
            str(fixture.partition_manifest),
            "--output-dir",
            str(output_dir),
            "--text-backend",
            "hash",
            "--embedding-dim",
            "8",
            "--epochs",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "model.pt").exists()
    assert (output_dir / "metrics_all.json").exists()
    assert (output_dir / "metrics_edit_only.json").exists()
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["text_backend"] == "hash"
```

- [ ] **Step 2: Run the CLI test and verify it fails**

Run:

```powershell
python -m pytest tests/test_train_relation_graph_critic.py -q
```

Expected:

```text
E   FileNotFoundError: scripts/train_relation_graph_critic.py
```

- [ ] **Step 3: Implement the CLI and artifact writing**

Create `scripts/train_relation_graph_critic.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.relation_graph_critic_data import (
    HashTextEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
    build_relation_graph_dataset,
)
from idea_graph.relation_graph_critic_train import train_relation_graph_critic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the offline relation-aware graph critic.")
    parser.add_argument("--candidate-dataset-dir", type=Path, required=True)
    parser.add_argument("--g1-dataset-dir", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-backend", choices=["sentence-transformer", "hash"], default="sentence-transformer")
    parser.add_argument("--text-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = (
        HashTextEmbeddingBackend(dim=args.embedding_dim)
        if args.text_backend == "hash"
        else SentenceTransformerEmbeddingBackend(args.text_model_name)
    )
    dataset = build_relation_graph_dataset(
        candidate_dataset_dir=args.candidate_dataset_dir,
        g1_dataset_dir=args.g1_dataset_dir,
        partition_manifest_path=args.partition_manifest,
        text_backend=backend,
    )
    artifacts = train_relation_graph_critic(
        dataset=dataset,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        text_backend_name=args.text_backend,
        text_model_name=args.text_model_name,
    )
    print(json.dumps({"all": artifacts.metrics_all, "edit_only": artifacts.metrics_edit_only}, indent=2))


if __name__ == "__main__":
    main()
```

Artifact contract to implement:

- `model.pt`
- `metrics_all.json`
- `metrics_edit_only.json`
- `metadata.json`
- `training_config.json`

- [ ] **Step 4: Run the CLI smoke test and the focused critic suite**

Run:

```powershell
python -m pytest tests/test_train_relation_graph_critic.py tests/test_relation_graph_critic_data.py tests/test_relation_graph_critic_model.py tests/test_relation_graph_critic_train.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 5: Commit the CLI**

```powershell
git add scripts/train_relation_graph_critic.py src/idea_graph/relation_graph_critic_data.py src/idea_graph/relation_graph_critic_train.py tests/test_train_relation_graph_critic.py
git commit -m "feat: add relation graph critic training cli"
```

### Task 5: Run The Frozen Offline Gate And Record The Decision

**Files:**
- Modify: `docs/experiment_execution_log.md`
- Modify: `docs/eig_graph_critic_plan.md`
- Verify: `outputs/graph_critic_models/development_pool_v2_relation_graph_v1`

- [ ] **Step 1: Run a tiny hash-backend smoke training**

Run:

```powershell
python scripts/train_relation_graph_critic.py `
  --candidate-dataset-dir outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g25 `
  --g1-dataset-dir outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g1 `
  --partition-manifest outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g2_partitions/partition_manifest.jsonl `
  --output-dir outputs/graph_critic_models/development_pool_v2_relation_graph_smoke `
  --text-backend hash `
  --embedding-dim 64 `
  --hidden-dim 64 `
  --batch-size 16 `
  --epochs 2
```

Expected:

```text
metrics_all.json and metrics_edit_only.json are written under the smoke output directory
```

- [ ] **Step 2: Run the full MiniLM-backed frozen offline gate**

Run:

```powershell
python scripts/train_relation_graph_critic.py `
  --candidate-dataset-dir outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g25 `
  --g1-dataset-dir outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g1 `
  --partition-manifest outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g2_partitions/partition_manifest.jsonl `
  --output-dir outputs/graph_critic_models/development_pool_v2_relation_graph_v1 `
  --text-backend sentence-transformer `
  --text-model-name sentence-transformers/all-MiniLM-L6-v2 `
  --hidden-dim 128 `
  --batch-size 16 `
  --epochs 10 `
  --lr 1e-3
```

Expected:

```text
model.pt, metrics_all.json, metrics_edit_only.json, metadata.json, and training_config.json are written
```

- [ ] **Step 3: Compare against the frozen baselines**

Inspect:

```powershell
Get-Content outputs/graph_critic_models/development_pool_v2_text_warmstart_v1/metrics.json
Get-Content outputs/graph_critic_models/development_pool_v2_graph_feature_v1/metrics.json
Get-Content outputs/graph_critic_models/development_pool_v2_relation_graph_v1/metrics_all.json
Get-Content outputs/graph_critic_models/development_pool_v2_relation_graph_v1/metrics_edit_only.json
```

Record the comparison table:

```text
text_warmstart_v1
- top1_accuracy = copy the exact value from `metrics.json`
- mean_reciprocal_rank = copy the exact value from `metrics.json`

graph_feature_v1
- top1_accuracy = copy the exact value from `metrics.json`
- mean_reciprocal_rank = copy the exact value from `metrics.json`

relation_graph_v1 all-candidate
- top1_accuracy = copy the exact value from `metrics_all.json`
- mean_reciprocal_rank = copy the exact value from `metrics_all.json`

relation_graph_v1 edit-only
- top1_accuracy = copy the exact value from `metrics_edit_only.json`
- mean_reciprocal_rank = copy the exact value from `metrics_edit_only.json`
```

Promotion rule:

- if `relation_graph_v1` beats `text_warmstart_v1` on frozen all-candidate validation MRR and top-1, proceed to the next controller planning step
- otherwise keep the graph critic offline and iterate on representation quality

- [ ] **Step 4: Update the tracked experiment docs**

Append to `docs/experiment_execution_log.md`:

```md
## 2026-04-13: Relation-Aware Graph Critic Offline Gate

- trained offline artifact:
  `outputs/graph_critic_models/development_pool_v2_relation_graph_v1`
- frozen comparison roots:
  - `development_pool_v2_combined_g25`
  - `development_pool_v2_combined_g1`
  - `development_pool_v2_combined_g2_partitions/partition_manifest.jsonl`
- recorded:
  - all-candidate dev top-1 / MRR
  - edit-only dev top-1 / MRR
  - delta vs text warm start
  - delta vs graph-feature baseline
- decision:
  - runtime graph critic remains blocked
  - or graph critic is ready for the next controller gate
```

Append to `docs/eig_graph_critic_plan.md`:

```md
### Stage G5.2: Relation-Aware Offline Graph Critic

- artifact:
  `outputs/graph_critic_models/development_pool_v2_relation_graph_v1`
- metrics:
  - all-candidate top-1: copy the exact value from `metrics_all.json`
  - all-candidate MRR: copy the exact value from `metrics_all.json`
  - edit-only top-1: copy the exact value from `metrics_edit_only.json`
  - edit-only MRR: copy the exact value from `metrics_edit_only.json`
- decision:
  - promoted to next controller planning stage
  - or held offline for further graph-representation refinement
```

- [ ] **Step 5: Run final verification and commit**

Run:

```powershell
python -m pytest tests/test_relation_graph_critic_data.py tests/test_relation_graph_critic_model.py tests/test_relation_graph_critic_train.py tests/test_train_relation_graph_critic.py tests/test_graph_feature_critic.py tests/test_text_critic.py tests/test_online_text_critic.py tests/test_runtime_critic.py tests/test_critic_policy.py tests/test_critic_replay.py tests/test_critic_episode_collection.py tests/test_critic_dataset.py tests/test_critic_partitions.py tests/test_critic_split_registry.py tests/test_candidate_slate_dataset.py -q
```

Expected:

```text
all tests passed
```

Commit:

```powershell
git add docs/experiment_execution_log.md docs/eig_graph_critic_plan.md
git commit -m "feat: add relation-aware graph critic offline gate"
```

## Self-Review Checklist

- Spec coverage:
  - frozen offline roots are covered in Tasks 1 and 5
  - relation-aware message passing is covered in Task 2
  - state-local ranking loss is covered in Task 3
  - offline-only artifact writing is covered in Task 4
  - frozen offline comparison and gate decision are covered in Task 5
- Placeholder scan:
  - no `TBD`, `TODO`, or deferred implementation markers remain
- Type consistency:
  - data loader exports `RelationGraphCandidateExample` and `RelationGraphBatch`
  - model consumes `RelationGraphBatch`
  - trainer consumes the dataset/model pair and writes the artifact contract used by the docs task
