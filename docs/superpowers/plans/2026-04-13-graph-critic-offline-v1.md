# Graph Critic Offline V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first true offline graph critic for `EIG`, using graph-structured state encoding rather than flattened state text, and compare it against the current text critic on frozen `critic_dev`.

**Architecture:** Reuse the current `G2.5` candidate-slate supervision and the existing `critic_train` / `critic_dev` split story, but replace flattened `state_text [SEP] candidate_text` scoring with a typed-graph encoder. The first version should stay lightweight: a small relational message-passing encoder in PyTorch, candidate-action embeddings over action kind plus target nodes, and state-group ranking loss. No runtime deployment and no learned `commit` in this slice.

**Tech Stack:** Python 3.10+, PyTorch CPU, existing `idea_graph` candidate-slate dataset, `pytest`

---

## Scope

This slice should:

- build a graph-structured offline dataset
- train a lightweight graph critic
- evaluate it on frozen `critic_dev`
- compare it against the current text critic

This slice should **not**:

- deploy the graph critic into the runtime loop
- add learned `commit`
- claim paper-ready graph-critic superiority

## File Map

### New Files

- Create: `src/idea_graph/graph_critic_dataset.py`
  - graph-structured example builder from `G1` snapshots plus `G2.5` candidate rows
- Create: `src/idea_graph/graph_critic_model.py`
  - lightweight typed graph encoder and candidate scorer
- Create: `src/idea_graph/graph_critic_train.py`
  - training and evaluation helpers
- Create: `scripts/train_graph_critic.py`
  - offline training entrypoint
- Create: `tests/test_graph_critic_dataset.py`
- Create: `tests/test_graph_critic_model.py`
- Create: `tests/test_graph_critic_train.py`

### Files To Modify

- Modify: `pyproject.toml`
  - add explicit `torch` dependency for reproducible setup
- Modify: `docs/eig_graph_critic_plan.md`
- Modify: `docs/experiment_execution_log.md`

## Task 1: Build The Graph-Critic Dataset Layer

**Files:**
- Create: `src/idea_graph/graph_critic_dataset.py`
- Create: `tests/test_graph_critic_dataset.py`

- [ ] **Step 1.1: Write failing dataset tests**

Add `tests/test_graph_critic_dataset.py`:

```python
def test_build_graph_critic_examples_preserves_split_safety(tmp_path: Path) -> None:
    dataset = build_graph_critic_examples(
        candidate_dataset_dir=tmp_path / "g25",
        g1_dataset_dir=tmp_path / "g1",
        partition_manifest_path=tmp_path / "partition_manifest.jsonl",
    )
    assert dataset.split_counts["critic_train"] > 0
    assert dataset.split_counts["critic_dev"] > 0
    assert dataset.group_overlap_count == 0


def test_example_contains_typed_graph_and_candidate_targets(tmp_path: Path) -> None:
    dataset = build_graph_critic_examples(...)
    example = dataset.examples[0]
    assert example.node_type_ids
    assert example.edge_index
    assert example.edge_type_ids
    assert example.action_kind_id >= 0
    assert example.target_node_indices
```

- [ ] **Step 1.2: Run the dataset tests to verify failure**

Run:
`python -m pytest tests/test_graph_critic_dataset.py -q`

Expected:
- import failure because `graph_critic_dataset.py` does not exist yet

- [ ] **Step 1.3: Define the graph example schema**

Create in `src/idea_graph/graph_critic_dataset.py`:

```python
@dataclass(frozen=True)
class GraphCriticExample:
    state_id: str
    candidate_id: str
    group_id: str
    split: str
    label: int
    node_token_ids: tuple[tuple[int, ...], ...]
    node_type_ids: tuple[int, ...]
    node_role_ids: tuple[int, ...]
    edge_index: tuple[tuple[int, int], ...]
    edge_type_ids: tuple[int, ...]
    edge_resolved_flags: tuple[int, ...]
    action_kind_id: int
    target_node_indices: tuple[int, ...]
```

- [ ] **Step 1.4: Build the dataset from existing artifacts**

Use:
- `outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g25_commit_enriched`
- `outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g1_commit_enriched`
- `outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions/partition_manifest.jsonl`

Implement:

```python
def build_graph_critic_examples(
    *,
    candidate_dataset_dir: Path,
    g1_dataset_dir: Path,
    partition_manifest_path: Path,
) -> GraphCriticDataset:
    ...
```

Rules:
- preserve existing `critic_train` / `critic_dev`
- exclude `paper_eval`
- use the current `before_state_snapshot` reconstruction path

- [ ] **Step 1.5: Re-run the dataset tests**

Run:
`python -m pytest tests/test_graph_critic_dataset.py -q`

## Task 2: Implement The Lightweight Graph Critic Model

**Files:**
- Create: `src/idea_graph/graph_critic_model.py`
- Create: `tests/test_graph_critic_model.py`

- [ ] **Step 2.1: Write failing model-forward tests**

Add `tests/test_graph_critic_model.py`:

```python
def test_graph_critic_forward_returns_one_score_per_candidate() -> None:
    model = GraphCriticModel(
        vocab_size=128,
        node_type_count=8,
        role_count=8,
        edge_type_count=8,
        action_kind_count=16,
        hidden_dim=32,
    )
    batch = make_graph_critic_batch(candidate_count=3)
    scores = model(batch)
    assert scores.shape == (3,)


def test_graph_critic_uses_target_node_indices() -> None:
    model = GraphCriticModel(...)
    batch = make_graph_critic_batch(candidate_count=2, distinct_targets=True)
    scores = model(batch)
    assert float(scores[0]) != float(scores[1])
```

- [ ] **Step 2.2: Run the model tests to verify failure**

Run:
`python -m pytest tests/test_graph_critic_model.py -q`

- [ ] **Step 2.3: Add explicit PyTorch dependency**

Modify `pyproject.toml`:

```toml
dependencies = [
    "scikit-learn>=1.5,<2",
    "torch>=2.1,<3",
]
```

- [ ] **Step 2.4: Implement the graph critic v1**

Create in `src/idea_graph/graph_critic_model.py`:

```python
class GraphCriticModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        node_type_count: int,
        role_count: int,
        edge_type_count: int,
        action_kind_count: int,
        hidden_dim: int = 64,
    ) -> None:
        ...

    def forward(self, batch: GraphCriticBatch) -> torch.Tensor:
        ...
```

Recommended forward path:

```python
# 1. node text -> EmbeddingBag
# 2. add node type and role embeddings
# 3. one or two rounds of relation-aware message passing
# 4. graph embedding = mean pooled node embedding
# 5. action embedding = action kind embedding + mean target-node embedding
# 6. score = MLP([graph_embedding ; action_embedding])
```

Keep it small and deterministic:
- `hidden_dim = 64`
- `message_passing_steps = 1`

- [ ] **Step 2.5: Re-run the model tests**

Run:
`python -m pytest tests/test_graph_critic_model.py -q`

## Task 3: Train And Evaluate Offline

**Files:**
- Create: `src/idea_graph/graph_critic_train.py`
- Create: `scripts/train_graph_critic.py`
- Create: `tests/test_graph_critic_train.py`

- [ ] **Step 3.1: Write failing training tests**

Add `tests/test_graph_critic_train.py`:

```python
def test_train_graph_critic_reports_dev_metrics(tmp_path: Path) -> None:
    result = train_graph_critic(
        dataset=make_small_graph_critic_dataset(),
        output_dir=tmp_path,
        hidden_dim=32,
        max_epochs=2,
        learning_rate=1e-3,
    )
    assert result.metrics["state_count"] > 0
    assert "top1_accuracy" in result.metrics
    assert "mean_reciprocal_rank" in result.metrics


def test_train_graph_critic_rejects_group_overlap() -> None:
    with pytest.raises(ValueError, match="group overlap"):
        train_graph_critic(dataset=make_overlapping_split_dataset(), output_dir=Path("dummy"))
```

- [ ] **Step 3.2: Run the training tests to verify failure**

Run:
`python -m pytest tests/test_graph_critic_train.py -q`

- [ ] **Step 3.3: Implement grouped ranking loss**

Create in `src/idea_graph/graph_critic_train.py`:

```python
def grouped_cross_entropy_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    positive_index = int(labels.argmax().item())
    return F.cross_entropy(scores.unsqueeze(0), torch.tensor([positive_index], device=scores.device))
```

Training rule:
- group candidates by `state_id`
- require exactly one positive candidate per state
- optimize state-level ranking, not independent binary BCE

- [ ] **Step 3.4: Implement training and evaluation helpers**

Add:

```python
@dataclass(frozen=True)
class GraphCriticTrainResult:
    metrics: dict[str, float | int]
    model_path: Path
    metadata_path: Path


def train_graph_critic(...): ...
def evaluate_graph_critic(...): ...
```

Required outputs:
- `model.pt`
- `metrics.json`
- `metadata.json`

- [ ] **Step 3.5: Add the CLI**

Create `scripts/train_graph_critic.py` with inputs:

```text
--candidate-dataset-dir
--g1-dataset-dir
--partition-manifest
--output-dir
--hidden-dim
--max-epochs
--learning-rate
```

- [ ] **Step 3.6: Re-run the training tests**

Run:
`python -m pytest tests/test_graph_critic_train.py -q`

## Task 4: Compare Against The Text Critic

**Files:**
- Modify: `docs/eig_graph_critic_plan.md`
- Modify: `docs/experiment_execution_log.md`

- [ ] **Step 4.1: Train the first offline graph critic**

Run:

```powershell
python scripts/train_graph_critic.py `
  --candidate-dataset-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g25_commit_enriched `
  --g1-dataset-dir outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g1_commit_enriched `
  --partition-manifest outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions/partition_manifest.jsonl `
  --output-dir outputs/graph_critic_models/current_benchmarked_ours_eig_full_g5_graph_v1 `
  --hidden-dim 64 `
  --max-epochs 10 `
  --learning-rate 1e-3
```

- [ ] **Step 4.2: Compare with the current text critic**

Reference text critic:
- `outputs/graph_critic_models/current_benchmarked_ours_eig_full_g46_text_online_real_train_v1/metrics.json`

Record:
- top-1 accuracy
- mean reciprocal rank
- split audit

- [ ] **Step 4.3: Apply the deployment gate**

Decision:
- only move the graph critic into runtime pilot work if it beats the text
  critic on frozen `critic_dev`
- otherwise keep it offline and treat it as an informative negative / neutral
  result

## Final Verification Checklist

- [ ] Graph-structured dataset builds from current artifacts without split leakage
- [ ] Graph-critic forward pass works
- [ ] Offline training produces dev metrics
- [ ] Graph critic is compared against the text critic on frozen `critic_dev`
- [ ] Runtime deployment is blocked unless offline graph critic wins
