# Graph Critic Controller Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the leakage-safe relation-aware graph critic into the `EIG` runtime controller as a conservative edit reranker, then run the frozen 4-case AIIB controller gate before any larger benchmark spend.

**Architecture:** Reuse the current offline graph-critic artifact instead of retraining first. Add a runtime adapter that reconstructs the trained vocab context from the frozen dataset roots recorded in `training_config.json`, builds a live relation-graph batch from the current `IdeaGraph` plus candidate edits, scores those edits with the learned graph model, and passes the resulting scores through the existing safe controller policy. If the narrow runtime gate is unsatisfying, stop and branch to targeted robustness-data expansion.

**Tech Stack:** Python 3.10, PyTorch, sentence-transformers, existing `IdeaGraph` runtime, pytest, PowerShell batch execution

---

## File Structure

### New Files

- `src/idea_graph/relation_graph_runtime_critic.py`
  - load the saved relation-graph critic artifact
  - rebuild the offline vocab context from the frozen dataset roots
  - build a runtime batch from the live graph plus candidate specs
  - select a runtime edit candidate through the shared safe critic policy
- `tests/test_relation_graph_runtime_critic.py`
  - unit tests for runtime bundle loading, runtime batch construction,
    leakage-safe candidate embedding, and heuristic fallback behavior
- `docs/superpowers/specs/2026-04-13-graph-critic-controller-gate-design.md`
  - design note for this stage

### Modified Files

- `src/idea_graph/relation_graph_critic_data.py`
  - expose reusable vocab-building helpers and a runtime batch builder
- `src/idea_graph/baselines.py`
  - register `ours-eig-critic-graph`
  - load the graph-critic runtime bundle when the new baseline is requested
- `src/idea_graph/engine.py`
  - dispatch graph-critic runtime selection and persist richer trace metadata
- `tests/test_benchmark_mode_and_baselines.py`
  - verify the new baseline metadata and runtime-controller wiring
- `tests/test_engine.py`
  - verify runtime graph-controller traces and heuristic fallback behavior
- `docs/experiment_execution_log.md`
  - record the controller gate result
- `docs/eig_graph_critic_plan.md`
  - record promotion or fallback after the gate

## Execution Rules

- keep learned `commit` disabled for this stage
- keep the same maturity-sensitive safe override policy used by the text critic
- use only the frozen 4-case AIIB packet:
  - `13`
  - `3883`
  - `7909`
  - `9849`
- do not expand the dataset until the first graph-controller gate is observed
- if runtime bundle loading or runtime token mapping fails, fall back to the
  heuristic action and log the reason

### Task 1: Expose Reusable Vocab And Runtime-Batch Helpers

**Files:**
- Modify: `src/idea_graph/relation_graph_critic_data.py`
- Create: `tests/test_relation_graph_runtime_critic.py`
- Check: `tests/test_relation_graph_critic_data.py`

- [ ] **Step 1: Write the failing runtime-batch tests**

```python
from pathlib import Path

from idea_graph.models import Branch, Edge, IdeaGraph, Node
from idea_graph.relation_graph_critic_data import (
    HashTextEmbeddingBackend,
    build_relation_graph_vocabs,
    build_runtime_relation_graph_batch,
)
from test_relation_graph_critic_data import write_relation_graph_fixture


def _make_runtime_graph() -> IdeaGraph:
    graph = IdeaGraph(topic="runtime topic", literature=["paper a"], metadata={})
    graph.branches["B001"] = Branch(id="B001", role="MechanismProposer")
    graph.nodes["N001"] = Node(
        id="N001",
        type="Hypothesis",
        text="Hypothesis node.",
        role="MechanismProposer",
        branch_id="B001",
        confidence=0.8,
    )
    graph.nodes["N002"] = Node(
        id="N002",
        type="Method",
        text="Method node.",
        role="MechanismProposer",
        branch_id="B001",
        confidence=0.7,
    )
    graph.edges.append(
        Edge(
            id="E001",
            source_id="N001",
            target_id="N002",
            relation="depends_on",
            resolved=True,
        )
    )
    return graph


def test_build_relation_graph_vocabs_captures_runtime_ontology(tmp_path: Path) -> None:
    fixture = write_relation_graph_fixture(tmp_path)
    vocabs = build_relation_graph_vocabs(
        candidate_dataset_dir=fixture.candidate_dir,
        g1_dataset_dir=fixture.g1_dir,
        partition_manifest_path=fixture.partition_manifest,
    )
    assert "Hypothesis" in vocabs.node_type_to_id
    assert "MechanismProposer" in vocabs.role_to_id
    assert "depends_on" in vocabs.edge_type_to_id
    assert "add_support_edge" in vocabs.candidate_kind_to_id


def test_build_runtime_relation_graph_batch_maps_targets_and_strips_leaky_suffixes(tmp_path: Path) -> None:
    fixture = write_relation_graph_fixture(tmp_path)
    vocabs = build_relation_graph_vocabs(
        candidate_dataset_dir=fixture.candidate_dir,
        g1_dataset_dir=fixture.g1_dir,
        partition_manifest_path=fixture.partition_manifest,
    )
    graph = _make_runtime_graph()
    backend = HashTextEmbeddingBackend(dim=8)
    batch, normalized = build_runtime_relation_graph_batch(
        graph=graph,
        candidate_specs=[
            {
                "candidate_id": "c0",
                "kind": "add_support_edge",
                "target_ids": ["N001", "N002"],
                "payload": {"branch_id": "B001"},
                "rationale": "teacher-only reason",
                "candidate_source": "utility_add_support",
            }
        ],
        vocabs=vocabs,
        text_backend=backend,
        use_commit=False,
    )
    assert normalized[0]["candidate_id"] == "c0"
    assert int(batch.target_mask[0].sum().item()) == 2
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_relation_graph_runtime_critic.py -q
```

Expected:

```text
E   ImportError: cannot import name 'build_relation_graph_vocabs'
```

- [ ] **Step 3: Implement the reusable vocab helper and runtime batch builder**

Add to `src/idea_graph/relation_graph_critic_data.py`:

```python
@dataclass(frozen=True)
class RelationGraphVocabularies:
    node_type_to_id: dict[str, int]
    role_to_id: dict[str, int]
    edge_type_to_id: dict[str, int]
    candidate_kind_to_id: dict[str, int]


def build_relation_graph_vocabs(
    *,
    candidate_dataset_dir: Path,
    g1_dataset_dir: Path,
    partition_manifest_path: Path,
) -> RelationGraphVocabularies:
    candidate_rows = _load_jsonl(Path(candidate_dataset_dir) / "candidate_dataset.jsonl")
    snapshot_lookup = _build_snapshot_lookup(Path(g1_dataset_dir))
    partition_lookup = build_partition_role_lookup(
        load_partition_manifest_rows(Path(partition_manifest_path))
    )
    node_type_vocab: dict[str, int] = {}
    role_vocab: dict[str, int] = {}
    edge_type_vocab: dict[str, int] = {}
    candidate_kind_vocab: dict[str, int] = {}
    for row in candidate_rows:
        group_id = str(row.get("group_id", "")).strip()
        if partition_lookup.get(group_id) == "paper_eval":
            continue
        snapshot = snapshot_lookup.get(str(row.get("state_id", "")).strip())
        if snapshot is None:
            continue
        for node_id in sorted(snapshot.get("nodes", {})):
            node_payload = dict(snapshot["nodes"][node_id])
            _intern_id(node_type_vocab, node_payload.get("type", "unknown"))
            _intern_id(role_vocab, node_payload.get("role", "unknown"))
        for edge_payload in snapshot.get("edges", []):
            _intern_id(edge_type_vocab, edge_payload.get("relation", "unknown"))
        _intern_id(candidate_kind_vocab, row.get("candidate_kind", "unknown"))
    return RelationGraphVocabularies(
        node_type_to_id=node_type_vocab,
        role_to_id=role_vocab,
        edge_type_to_id=edge_type_vocab,
        candidate_kind_to_id=candidate_kind_vocab,
    )


def build_runtime_relation_graph_batch(
    *,
    graph: IdeaGraph,
    candidate_specs: Sequence[Mapping[str, Any]],
    vocabs: RelationGraphVocabularies,
    text_backend: TextEmbeddingBackend,
    use_commit: bool,
) -> tuple[RelationGraphBatch, list[dict[str, object]]]:
    normalized_specs: list[dict[str, object]] = []
    for index, raw_spec in enumerate(candidate_specs):
        kind = str(raw_spec.get("kind", "")).strip()
        if kind == "commit" and not use_commit:
            continue
        normalized_specs.append(
            {
                **dict(raw_spec),
                "candidate_id": str(raw_spec.get("candidate_id", f"candidate:{index:03d}")).strip()
                or f"candidate:{index:03d}",
            }
        )
    examples: list[RelationGraphCandidateExample] = []
    for spec in normalized_specs:
        candidate_text = _strip_leaky_candidate_segments(flatten_candidate_text(graph, spec))
        state_text = flatten_graph_state_text(graph)
        # Build one state-local example per candidate using the current live
        # graph, then collate them into a single batch for runtime scoring.
        example = RelationGraphCandidateExample(
            state_id="runtime-state",
            candidate_id=str(spec["candidate_id"]),
            group_id="runtime",
            split="runtime",
            label=0,
            is_commit=str(spec.get("kind", "")).strip() == "commit",
            candidate_kind_id=candidate_kind_id,
            candidate_text_embedding=candidate_embedding,
            state_text_embedding=state_embedding,
            node_text_embeddings=node_text_embeddings,
            node_type_ids=node_type_ids,
            node_role_ids=node_role_ids,
            node_confidence=node_confidence,
            node_evidence_count=node_evidence_count,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_resolved=edge_resolved,
            target_node_indices=target_node_indices,
        )
        examples.append(example)
    return collate_relation_graph_examples(examples), normalized_specs
```

Implementation requirements:

- runtime candidate text must call `_strip_leaky_candidate_segments(...)`
- active nodes must be sorted by node id
- active edges must be sorted by `(source_id, relation, target_id, id)`
- if any target id is missing from the live graph, just ignore that target id
- if a runtime token is missing from the vocab maps, keep the row but mark it
  for later heuristic fallback in Task 2

- [ ] **Step 4: Run the data-path tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_relation_graph_runtime_critic.py tests/test_relation_graph_critic_data.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 5: Commit**

```powershell
git add src/idea_graph/relation_graph_critic_data.py tests/test_relation_graph_runtime_critic.py
git commit -m "feat: add runtime graph critic batch builder"
```

### Task 2: Add The Runtime Relation-Graph Critic Loader And Selector

**Files:**
- Create: `src/idea_graph/relation_graph_runtime_critic.py`
- Test: `tests/test_relation_graph_runtime_critic.py`

- [ ] **Step 1: Write the failing runtime-selector tests**

Add to `tests/test_relation_graph_runtime_critic.py`:

```python
import json
from pathlib import Path

import torch

from idea_graph.models import Branch, IdeaGraph, Node
from idea_graph.relation_graph_critic_model import RelationGraphCritic
from idea_graph.relation_graph_runtime_critic import (
    RelationGraphRuntimeConfig,
    load_relation_graph_runtime_bundle,
    select_relation_graph_critic_candidate,
)
from test_relation_graph_critic_data import write_relation_graph_fixture


class _StubRuntimeBundle:
    def __init__(self, scores: list[float]) -> None:
        self.scores = list(scores)

    def score_runtime_batch(self, batch):
        return list(self.scores)

    def runtime_token_status(self, normalized_specs):
        return {"ok": True, "reason": ""}


class _RejectingRuntimeBundle(_StubRuntimeBundle):
    def runtime_token_status(self, normalized_specs):
        return {"ok": False, "reason": "unmapped_runtime_token"}


def _make_runtime_graph() -> IdeaGraph:
    graph = IdeaGraph(topic="runtime topic", literature=["paper a"], metadata={})
    graph.branches["B001"] = Branch(id="B001", role="MechanismProposer")
    graph.nodes["N001"] = Node(id="N001", type="Hypothesis", text="Hypothesis", role="MechanismProposer", branch_id="B001")
    graph.nodes["N002"] = Node(id="N002", type="Method", text="Method", role="MechanismProposer", branch_id="B001")
    return graph


def test_load_relation_graph_runtime_bundle_rebuilds_vocab_context(tmp_path: Path) -> None:
    fixture = write_relation_graph_fixture(tmp_path / "fixture")
    model_dir = tmp_path / "artifact"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "training_config.json").write_text(
        json.dumps(
            {
                "candidate_dataset_dir": str(fixture.candidate_dir.resolve()),
                "g1_dataset_dir": str(fixture.g1_dir.resolve()),
                "partition_manifest": str(fixture.partition_manifest.resolve()),
                "text_backend": "hash",
                "text_model_name": None,
                "embedding_dim": 8,
                "hidden_dim": 16,
                "batch_size": 2,
                "epochs": 1,
                "learning_rate": 1e-3,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "metadata.json").write_text(
        json.dumps({"text_backend": "hash", "hidden_dim": 16}),
        encoding="utf-8",
    )
    dummy = RelationGraphCritic(
        text_dim=8,
        hidden_dim=16,
        node_type_count=2,
        role_count=2,
        edge_type_count=2,
        candidate_kind_count=2,
    )
    torch.save(dummy.state_dict(), model_dir / "model.pt")
    bundle = load_relation_graph_runtime_bundle(model_dir)
    assert "Hypothesis" in bundle.vocabs.node_type_to_id
    assert "MechanismProposer" in bundle.vocabs.role_to_id
    assert bundle.device.type in {"cpu", "cuda"}


def test_select_relation_graph_critic_candidate_overrides_when_margin_is_large(tmp_path: Path) -> None:
    graph = _make_runtime_graph()
    decision = select_relation_graph_critic_candidate(
        graph,
        round_name="Round3",
        role="MechanismProposer",
        state_features={"round_index": 3, "support_coverage": 0.72, "unresolved_contradiction_ratio": 0.0},
        candidate_specs=[
            {"candidate_id": "heuristic", "kind": "attach_evidence", "target_ids": ["N001"], "payload": {"branch_id": "B001"}},
            {"candidate_id": "critic-best", "kind": "add_support_edge", "target_ids": ["N001", "N002"], "payload": {"branch_id": "B001"}},
        ],
        heuristic_candidate_id="heuristic",
        runtime_bundle=_StubRuntimeBundle([0.40, 0.88]),
        config=RelationGraphRuntimeConfig(tau_override=0.05, use_commit=False),
    )
    assert decision.policy_decision.selected_candidate_id == "critic-best"


def test_select_relation_graph_critic_candidate_falls_back_on_unmapped_runtime_tokens(tmp_path: Path) -> None:
    graph = _make_runtime_graph()
    decision = select_relation_graph_critic_candidate(
        graph,
        round_name="Round3",
        role="MechanismProposer",
        state_features={"round_index": 3, "support_coverage": 0.72, "unresolved_contradiction_ratio": 0.0},
        candidate_specs=[
            {"candidate_id": "heuristic", "kind": "attach_evidence", "target_ids": ["N001"], "payload": {"branch_id": "B001"}},
            {"candidate_id": "critic-best", "kind": "brand_new_action_kind", "target_ids": ["N001"], "payload": {"branch_id": "B001"}},
        ],
        heuristic_candidate_id="heuristic",
        runtime_bundle=_RejectingRuntimeBundle([0.40, 0.88]),
        config=RelationGraphRuntimeConfig(tau_override=0.05, use_commit=False),
    )
    assert decision.policy_decision.selected_candidate_id == "heuristic"
    assert decision.policy_decision.used_heuristic_fallback
    assert decision.selected_spec["controller_fallback_reason"] == "unmapped_runtime_token"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_relation_graph_runtime_critic.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'idea_graph.relation_graph_runtime_critic'
```

- [ ] **Step 3: Implement the runtime bundle loader and selector**

Create `src/idea_graph/relation_graph_runtime_critic.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .critic_policy import SafeCriticPolicyConfig, ScoredCandidate, choose_critic_action
from .models import IdeaGraph
from .relation_graph_critic_data import (
    HashTextEmbeddingBackend,
    RelationGraphBatch,
    RelationGraphVocabularies,
    SentenceTransformerEmbeddingBackend,
    build_relation_graph_vocabs,
    build_runtime_relation_graph_batch,
)
from .relation_graph_critic_model import RelationGraphCritic


@dataclass(frozen=True)
class RelationGraphRuntimeConfig:
    tau_override: float = 0.05
    tau_commit: float = 0.08
    gamma_commit: float = 0.60
    min_commit_round: int = 2
    use_commit: bool = False
    guard_support_threshold: float = 0.66
    guard_support_gain_floor: float = 0.10
    guard_requires_contradiction_progress: bool = False


@dataclass(frozen=True)
class RelationGraphRuntimeDecision:
    selected_spec: dict[str, object]
    scored_candidates: tuple[dict[str, object], ...]
    policy_decision: Any


class LoadedRelationGraphRuntimeCritic:
    def __init__(
        self,
        *,
        model: RelationGraphCritic,
        vocabs: RelationGraphVocabularies,
        text_backend: Any,
        device: torch.device,
    ) -> None:
        self.model = model
        self.vocabs = vocabs
        self.text_backend = text_backend
        self.device = device

    def score_runtime_batch(self, batch: RelationGraphBatch) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            scores = self.model(batch.to(self.device)).detach().cpu().tolist()
        return [float(value) for value in scores]


def load_relation_graph_runtime_bundle(model_dir: Path | str) -> LoadedRelationGraphRuntimeCritic:
    resolved = Path(model_dir).resolve()
    training_config = json.loads((resolved / "training_config.json").read_text(encoding="utf-8"))
    metadata = json.loads((resolved / "metadata.json").read_text(encoding="utf-8"))
    vocabs = build_relation_graph_vocabs(
        candidate_dataset_dir=Path(training_config["candidate_dataset_dir"]),
        g1_dataset_dir=Path(training_config["g1_dataset_dir"]),
        partition_manifest_path=Path(training_config["partition_manifest"]),
    )
    text_backend = (
        HashTextEmbeddingBackend(dim=int(training_config["embedding_dim"]))
        if training_config["text_backend"] == "hash"
        else SentenceTransformerEmbeddingBackend(str(training_config["text_model_name"]))
    )
    model = RelationGraphCritic(
        text_dim=int(training_config["embedding_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        node_type_count=len(vocabs.node_type_to_id),
        role_count=len(vocabs.role_to_id),
        edge_type_count=len(vocabs.edge_type_to_id),
        candidate_kind_count=len(vocabs.candidate_kind_to_id),
    )
    state_dict = torch.load(resolved / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return LoadedRelationGraphRuntimeCritic(
        model=model,
        vocabs=vocabs,
        text_backend=text_backend,
        device=device,
    )
```

Implementation requirements:

- if runtime token mapping fails, set `controller_fallback_reason` and fall
  back to heuristic before scoring
- runtime candidate rows must preserve:
  - `predicted_gain`
  - `support_gain`
  - `contradiction_gain`
  - `maturity_gain`
  - `after_is_mature`
- the selector must reuse `choose_critic_action(...)`

- [ ] **Step 4: Run the runtime-selector tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_relation_graph_runtime_critic.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 5: Commit**

```powershell
git add src/idea_graph/relation_graph_runtime_critic.py tests/test_relation_graph_runtime_critic.py
git commit -m "feat: add runtime relation graph critic selector"
```

### Task 3: Wire The Graph Critic Into Baselines And Engine Traces

**Files:**
- Modify: `src/idea_graph/baselines.py`
- Modify: `src/idea_graph/engine.py`
- Modify: `tests/test_benchmark_mode_and_baselines.py`
- Modify: `tests/test_engine.py`

- [ ] **Step 1: Write the failing baseline and engine tests**

Add to `tests/test_benchmark_mode_and_baselines.py`:

```python
def test_graph_critic_baseline_is_registered() -> None:
    assert "ours-eig-critic-graph" in BASELINE_SPECS
    spec = BASELINE_SPECS["ours-eig-critic-graph"]
    assert spec.runtime_controller == "relation_graph_critic_rerank"
```

Add to `tests/test_engine.py`:

```python
class RuntimeGraphCriticStub:
    def __init__(self, selected_candidate_id: str) -> None:
        self.selected_candidate_id = selected_candidate_id

    def score_runtime_batch(self, batch):
        return [0.10, 0.90]


def test_engine_records_graph_runtime_controller_trace(self) -> None:
    graph = run_experiment(
        topic="test topic",
        literature=["paper a"],
        metadata={"runtime_controller_enabled": True},
        collaboration_backend=None,
        max_rounds=1,
        runtime_controller=RuntimeGraphCriticStub("critic-best"),
        runtime_controller_metadata={
            "kind": "relation_graph_critic_rerank",
            "config": RelationGraphRuntimeConfig(tau_override=0.05, use_commit=False),
        },
    )
    controller_log = graph.metadata.get("runtime_controller_log")
    assert controller_log
    assert controller_log[0]["controller_kind"] == "relation_graph_critic_rerank"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_benchmark_mode_and_baselines.py tests/test_engine.py -q
```

Expected:

```text
FAIL because 'ours-eig-critic-graph' is missing and engine only supports text reranking
```

- [ ] **Step 3: Implement baseline wiring and engine dispatch**

Update `src/idea_graph/baselines.py`:

```python
DEFAULT_RELATION_GRAPH_CRITIC_MODEL_DIR = (
    ROOT
    / "outputs"
    / "graph_critic_models"
    / "development_pool_v2_relation_graph_sanitized_v1"
)

BASELINE_SPECS["ours-eig-critic-graph"] = BaselineSpec(
    name="ours-eig-critic-graph",
    display_name="Ours (EIG + Graph Critic)",
    strategy="evolving_graph",
    description="Evolving Idea Graph with the leakage-safe relation-aware graph critic as a conservative edit reranker.",
    prompt_style="ours",
    runtime_controller="relation_graph_critic_rerank",
)
```

Update `_maybe_build_runtime_controller(...)`:

```python
if baseline.runtime_controller == "relation_graph_critic_rerank":
    model_dir = Path(
        str(
            graph.metadata.get("runtime_controller_model_dir")
            or DEFAULT_RELATION_GRAPH_CRITIC_MODEL_DIR
        )
    )
    controller = load_relation_graph_runtime_bundle(model_dir)
    config = RelationGraphRuntimeConfig(
        tau_override=float(graph.metadata.get("runtime_controller_tau_override", 0.05)),
        tau_commit=float(graph.metadata.get("runtime_controller_tau_commit", 0.08)),
        gamma_commit=float(graph.metadata.get("runtime_controller_gamma_commit", 0.60)),
        min_commit_round=int(graph.metadata.get("runtime_controller_min_commit_round", 2)),
        use_commit=bool(graph.metadata.get("runtime_controller_use_commit", False)),
        guard_support_threshold=float(graph.metadata.get("runtime_controller_guard_support_threshold", 0.66)),
        guard_support_gain_floor=float(graph.metadata.get("runtime_controller_guard_support_gain_floor", 0.10)),
        guard_requires_contradiction_progress=bool(graph.metadata.get("runtime_controller_guard_requires_contradiction_progress", False)),
    )
    return controller, {"kind": "relation_graph_critic_rerank", "config": config, "model_dir": str(model_dir.resolve())}
```

Update `src/idea_graph/engine.py` inside `_select_ranked_action(...)`:

```python
controller_kind = str(runtime_controller_metadata.get("kind", "")).strip()
if controller_kind == "text_critic_rerank":
    controller_decision = select_text_critic_candidate(
        graph,
        round_name=round_name,
        role=role,
        state_features=controller_state,
        candidate_specs=valid_candidates,
        heuristic_candidate_id=str(heuristic_selected_candidate.get("candidate_id", "")).strip(),
        model=runtime_controller,
        config=controller_config,
    )
elif controller_kind == "relation_graph_critic_rerank":
    controller_decision = select_relation_graph_critic_candidate(
        graph,
        round_name=round_name,
        role=role,
        state_features=controller_state,
        candidate_specs=valid_candidates,
        heuristic_candidate_id=str(heuristic_selected_candidate.get("candidate_id", "")).strip(),
        runtime_bundle=runtime_controller,
        config=controller_config,
    )
else:
    raise ValueError(f"Unsupported runtime controller kind: {controller_kind}")
```

Extend `_record_runtime_controller_trace(...)` with:

```python
"controller_kind": controller_decision.get("controller_kind"),
"selected_critic_score": selected_candidate.get("critic_score"),
"heuristic_candidate_source": heuristic_candidate.get("candidate_source"),
"fallback_reason": selected_candidate.get("controller_fallback_reason"),
```

- [ ] **Step 4: Run the integration tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_benchmark_mode_and_baselines.py tests/test_engine.py tests/test_relation_graph_runtime_critic.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 5: Commit**

```powershell
git add src/idea_graph/baselines.py src/idea_graph/engine.py tests/test_benchmark_mode_and_baselines.py tests/test_engine.py
git commit -m "feat: wire graph critic into controller baseline"
```

### Task 4: Run The Frozen 4-Case AIIB Graph-Controller Gate

**Files:**
- Create: `outputs/m2_aiib_g6_graph_controller_gate_v1/paired_summary.md`
- Modify: `docs/experiment_execution_log.md`
- Modify: `docs/eig_graph_critic_plan.md`

- [ ] **Step 1: Run the heuristic `ours-eig` side of the packet**

Run:

```powershell
$cases = 13,3883,7909,9849
foreach ($case in $cases) {
  python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index $case --baseline ours-eig --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --max-rounds 5 --native-eval --output-root outputs/m2_aiib_g6_graph_controller_gate_v1
}
```

Expected:

```text
4 completed run directories under outputs/m2_aiib_g6_graph_controller_gate_v1
```

- [ ] **Step 2: Run the graph-controller side of the same packet**

Run:

```powershell
$cases = 13,3883,7909,9849
foreach ($case in $cases) {
  python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index $case --baseline ours-eig-critic-graph --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --max-rounds 5 --native-eval --output-root outputs/m2_aiib_g6_graph_controller_gate_v1
}
```

Expected:

```text
4 additional completed run directories under outputs/m2_aiib_g6_graph_controller_gate_v1
```

- [ ] **Step 3: Build the paired summary**

Write `outputs/m2_aiib_g6_graph_controller_gate_v1/paired_summary.md` with these section headers, and populate every numeric field from the eight saved run artifacts:

```md
# AIIB Frozen 4-Case Graph-Controller Gate

## Cases

- `13`
- `3883`
- `7909`
- `9849`

## Paired Result Table

| Case | Ours Native | Graph Critic Native | Delta | Notes |
| --- | ---: | ---: | ---: | --- |

## Mean Result

- `ours-eig` mean AIIB native
- `ours-eig-critic-graph` mean AIIB native
- mean delta

## Controller Trace Readout

- runtime-controller trace entries
- critic-selected edits
- heuristic fallbacks
- notable fallback reasons:
  - one bullet per observed reason

## Decision

- satisfying / unsatisfying
- one-sentence explanation
```

- [ ] **Step 4: Update the tracked docs**

Append to `docs/experiment_execution_log.md`:

```md
## 2026-04-13: Frozen 4-Case AIIB Graph-Controller Gate

- artifact:
  `outputs/m2_aiib_g6_graph_controller_gate_v1`
- baselines:
  - `ours-eig`
  - `ours-eig-critic-graph`
- cases:
  - `13`
  - `3883`
  - `7909`
  - `9849`
- recorded:
  - per-case native deltas
  - mean native delta
  - runtime-controller trace counts
  - fallback counts and reasons
- decision:
  - promoted to larger packet
  - or blocked pending robustness-data expansion
```

Append to `docs/eig_graph_critic_plan.md`:

```md
### Stage G6.1: Graph-Critic Controller Gate

- artifact:
  `outputs/m2_aiib_g6_graph_controller_gate_v1`
- compared:
  - `ours-eig`
  - `ours-eig-critic-graph`
- decision:
  - promoted to larger controller packet
  - or held for targeted robustness-data expansion
```

- [ ] **Step 5: Run the final regression packet**

Run:

```powershell
python -m pytest tests/test_relation_graph_runtime_critic.py tests/test_relation_graph_critic_data.py tests/test_relation_graph_critic_model.py tests/test_relation_graph_critic_train.py tests/test_train_relation_graph_critic.py tests/test_graph_feature_critic.py tests/test_text_critic.py tests/test_runtime_critic.py tests/test_critic_policy.py tests/test_engine.py tests/test_benchmark_mode_and_baselines.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 6: Commit**

```powershell
git add docs/experiment_execution_log.md docs/eig_graph_critic_plan.md
git commit -m "feat: run graph critic controller gate"
```

## Gate Outcome Branch

### Go

Proceed to a larger controller packet only if:

- mean AIIB native is at least neutral versus `ours-eig`
- no new early-maturity pathology appears
- no catastrophic single-case drop appears
- graph-critic overrides are non-trivial and interpretable

### No-Go

If the gate is unsatisfying:

- do **not** scale generation yet
- freeze the gate artifact and diagnosis
- use the recorded disagreement / fallback states to drive the next
  robustness-data expansion plan

## Self-Review Checklist

- Spec coverage:
  - runtime graph-critic loading is covered in Task 2
  - baseline and engine integration are covered in Task 3
  - the frozen 4-case AIIB gate is covered in Task 4
  - fallback branching is covered in `Gate Outcome Branch`
- Placeholder scan:
  - no `TODO`, `TBD`, or deferred implementation markers remain
- Type consistency:
  - runtime config, runtime decision, and runtime bundle names stay consistent
  - `ours-eig-critic-graph` is the only new runtime baseline name
