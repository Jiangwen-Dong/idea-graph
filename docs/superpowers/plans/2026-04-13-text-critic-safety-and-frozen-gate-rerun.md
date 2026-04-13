# Text Critic Safety And Frozen Gate Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the current `ours-eig-critic-text` controller by adding maturity-sensitive override safety, then rerun the same frozen 4-case AIIB gate with trace-persistent artifacts.

**Architecture:** Keep the current text critic and heuristic stopping policy. Do not add learned `commit`. Instead, enrich the critic-policy inputs with candidate-level post-action gains already computed by the engine, then block critic overrides that create fragile maturity jumps without enough support or contradiction reduction. After the patch, rerun the exact same 4-case packet and compare `ours-eig` versus `ours-eig-critic-text`.

**Tech Stack:** Python 3.10+, existing `idea_graph` runtime controller, `pytest`, OpenAI-compatible LLM backend, JSON/Markdown run artifacts

---

## Current Status

- Task 1 implemented and verified in the main checkout.
- Task 2 implemented and verified in the main checkout.
- Task 3 local controller-regression verification completed:
  `python -m pytest tests/test_engine.py tests/test_runtime_critic.py tests/test_benchmark_mode_and_baselines.py tests/test_online_text_critic.py tests/test_critic_policy.py tests/test_critic_replay.py tests/test_critic_episode_collection.py tests/test_critic_split_registry.py -q`
  passed with `81 passed`.
- Task 4, the API-backed frozen 4-case AIIB gate rerun, has now been
  completed after the safety patch.
- Task 4 artifact:
  `outputs/m2_aiib_g48_controller_gate_v2/paired_summary.md`.
- Task 4 decision:
  the safety patch fixed the obvious `3883` Round2 early-stop pathology, but
  the native-score result remains mixed, so the text critic should stay a
  controller pilot rather than become the main paper system.
- Subagent conflict rule used in this pass:
  one read-only subagent only; all edits and subprocesses were run by the main
  session.

## Scope

This slice is intentionally narrow.

It should:

- keep the current text critic
- keep heuristic stopping
- improve only maturity-sensitive safety around learned reranking
- rerun the frozen 4-case AIIB controller gate

It should **not**:

- add learned `commit`
- redesign the main EIG method
- start the graph critic yet

## File Map

### Files To Modify

- Modify: `src/idea_graph/critic_policy.py`
  - add maturity-sensitive override guardrails
- Modify: `src/idea_graph/runtime_critic.py`
  - pass candidate-level gain metadata into the policy layer
- Modify: `src/idea_graph/engine.py`
  - pass current state maturity features into runtime-controller selection
- Modify: `tests/test_critic_policy.py`
- Modify: `tests/test_runtime_critic.py`
- Modify: `tests/test_engine.py`
- Modify: `docs/experiment_execution_log.md`
- Modify: `docs/eig_graph_critic_plan.md`

### Run Artifacts To Produce

- `outputs/m2_aiib_g48_controller_gate_v2`
- `outputs/m2_aiib_g48_controller_gate_v2/paired_summary.md`

## Task 1: Expose The Right Safety Signals To The Policy

**Files:**
- Modify: `src/idea_graph/critic_policy.py`
- Modify: `src/idea_graph/runtime_critic.py`
- Modify: `src/idea_graph/engine.py`
- Modify: `tests/test_critic_policy.py`
- Modify: `tests/test_runtime_critic.py`

- [ ] **Step 1.1: Write failing tests for maturity-sensitive metadata**

Add to `tests/test_critic_policy.py`:

```python
def test_policy_blocks_fragile_maturity_jump_without_support_gain() -> None:
    decision = choose_critic_action(
        state={
            "round_index": 3,
            "support_coverage": 0.66,
            "unresolved_contradiction_ratio": 0.0,
        },
        critic_candidates=[
            ScoredCandidate(
                candidate_id="critic-fragile",
                score=0.82,
                predicted_gain=1.10,
                support_gain=0.00,
                contradiction_gain=0.00,
                maturity_gain=1.0,
                after_is_mature=True,
            ),
        ],
        heuristic_candidate=ScoredCandidate(
            candidate_id="heuristic-safe",
            score=0.72,
            predicted_gain=0.74,
            support_gain=0.25,
            contradiction_gain=0.00,
            maturity_gain=0.0,
            after_is_mature=False,
        ),
        config=SafeCriticPolicyConfig(
            tau_override=0.05,
            guard_support_threshold=0.66,
            guard_support_gain_floor=0.10,
        ),
    )
    assert decision.selected_candidate_id == "heuristic-safe"
    assert decision.used_heuristic_fallback is True
```

Add to `tests/test_runtime_critic.py`:

```python
def test_runtime_critic_preserves_candidate_gain_metadata() -> None:
    decision = select_text_critic_candidate(
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
                "kind": "attach_evidence",
                "target_ids": ["N001"],
                "payload": {"branch_id": "B001", "evidence": "weak"},
                "rationale": "heuristic",
                "predicted_gain": 0.50,
                "support_gain": 0.00,
                "contradiction_gain": 0.00,
                "maturity_gain": 1.0,
                "after_subgraph": {"is_mature": True},
            },
            {
                "candidate_id": "critic-safe",
                "kind": "attach_evidence",
                "target_ids": ["N002"],
                "payload": {"branch_id": "B001", "evidence": "strong"},
                "rationale": "critic",
                "predicted_gain": 0.75,
                "support_gain": 0.20,
                "contradiction_gain": 0.00,
                "maturity_gain": 0.0,
                "after_subgraph": {"is_mature": False},
            },
        ],
        heuristic_candidate_id="heuristic",
        model=_KeywordScoreModel({"weak": 0.45, "strong": 0.80}),
        config=TextCriticRuntimeConfig(tau_override=0.05, use_commit=False),
    )
    assert decision.selected_spec["candidate_id"] == "critic-safe"
```

- [ ] **Step 1.2: Run the targeted tests to verify failure**

Run:
`python -m pytest tests/test_critic_policy.py tests/test_runtime_critic.py -q`

Expected:
- failing tests because `ScoredCandidate` and `select_text_critic_candidate(...)` do not yet carry the needed safety metadata

- [ ] **Step 1.3: Extend the scored-candidate schema**

Modify `src/idea_graph/critic_policy.py`:

```python
@dataclass(frozen=True)
class ScoredCandidate:
    candidate_id: str
    score: float
    is_commit: bool = False
    confidence: float | None = None
    predicted_gain: float = 0.0
    support_gain: float = 0.0
    contradiction_gain: float = 0.0
    maturity_gain: float = 0.0
    after_is_mature: bool = False
```

- [ ] **Step 1.4: Pass candidate-level gain metadata from runtime reranking**

Modify `src/idea_graph/runtime_critic.py`:

```python
def select_text_critic_candidate(
    graph: IdeaGraph,
    *,
    round_name: str,
    role: str,
    state_features: Mapping[str, Any],
    candidate_specs: Sequence[Mapping[str, Any]],
    heuristic_candidate_id: str,
    model: Any,
    config: TextCriticRuntimeConfig,
) -> TextCriticRuntimeDecision:
    ...
    scored_policy_candidates.append(
        ScoredCandidate(
            candidate_id=candidate_id,
            score=score_value,
            is_commit=str(spec.get("kind", "")).strip() == "commit",
            confidence=score_value,
            predicted_gain=float(spec.get("predicted_gain", 0.0) or 0.0),
            support_gain=float(spec.get("support_gain", 0.0) or 0.0),
            contradiction_gain=float(spec.get("contradiction_gain", 0.0) or 0.0),
            maturity_gain=float(spec.get("maturity_gain", 0.0) or 0.0),
            after_is_mature=bool((spec.get("after_subgraph") or {}).get("is_mature", False)),
        )
    )
```

- [ ] **Step 1.5: Pass current state maturity features from the engine**

Modify `src/idea_graph/engine.py` inside `_select_ranked_action(...)`:

```python
controller_state = {
    "round_index": _parse_round_index(round_name),
    "support_coverage": reference_snapshot.support_coverage,
    "unresolved_contradiction_ratio": reference_snapshot.unresolved_contradiction_ratio,
    "completeness": reference_snapshot.completeness,
    "is_mature": reference_snapshot.is_mature,
}
```

Then pass:

```python
controller_decision = select_text_critic_candidate(
    graph,
    round_name=round_name,
    role=role,
    state_features=controller_state,
    candidate_specs=valid_candidates,
    heuristic_candidate_id=...,
    model=runtime_controller,
    config=controller_config,
)
```

- [ ] **Step 1.6: Re-run the targeted tests**

Run:
`python -m pytest tests/test_critic_policy.py tests/test_runtime_critic.py -q`

Expected:
- targeted safety-metadata tests pass

## Task 2: Add Maturity-Sensitive Fallback Rules

**Files:**
- Modify: `src/idea_graph/critic_policy.py`
- Modify: `tests/test_critic_policy.py`
- Modify: `tests/test_engine.py`

- [ ] **Step 2.1: Write failing tests for maturity-sensitive fallback**

Add to `tests/test_critic_policy.py`:

```python
def test_policy_allows_override_when_support_gain_is_real() -> None:
    decision = choose_critic_action(
        state={
            "round_index": 3,
            "support_coverage": 0.70,
            "unresolved_contradiction_ratio": 0.0,
        },
        critic_candidates=[
            ScoredCandidate(
                candidate_id="critic-grounded",
                score=0.84,
                predicted_gain=0.90,
                support_gain=0.20,
                contradiction_gain=0.00,
                maturity_gain=1.0,
                after_is_mature=True,
            ),
        ],
        heuristic_candidate=ScoredCandidate(
            candidate_id="heuristic",
            score=0.73,
            predicted_gain=0.60,
            support_gain=0.00,
            contradiction_gain=0.00,
            maturity_gain=0.0,
            after_is_mature=False,
        ),
        config=SafeCriticPolicyConfig(
            tau_override=0.05,
            guard_support_threshold=0.66,
            guard_support_gain_floor=0.10,
        ),
    )
    assert decision.selected_candidate_id == "critic-grounded"
    assert decision.selected_source == "critic"
```

Add to `tests/test_engine.py`:

```python
def test_runtime_controller_trace_is_saved_with_llm_backend(self) -> None:
    graph = run_experiment(...)
    controller_log = graph.metadata.get("runtime_controller_log")
    assert controller_log
    assert "selected_source" in controller_log[0]
```

- [ ] **Step 2.2: Run the safety tests to verify failure**

Run:
`python -m pytest tests/test_critic_policy.py tests/test_engine.py -q`

- [ ] **Step 2.3: Add explicit guard fields to the policy config**

Modify `src/idea_graph/critic_policy.py`:

```python
@dataclass(frozen=True)
class SafeCriticPolicyConfig:
    min_commit_round: int = 2
    tau_override: float = 0.05
    tau_commit: float = 0.08
    gamma_commit: float = 0.60
    guard_support_threshold: float = 0.66
    guard_support_gain_floor: float = 0.10
    guard_requires_contradiction_progress: bool = False
```

- [ ] **Step 2.4: Implement fragile-maturity detection**

Inside `src/idea_graph/critic_policy.py`, add:

```python
def _is_fragile_maturity_override(
    *,
    state: Mapping[str, object],
    candidate: ScoredCandidate,
    config: SafeCriticPolicyConfig,
) -> bool:
    support_coverage = float(state.get("support_coverage", 0.0) or 0.0)
    if support_coverage < float(config.guard_support_threshold):
        return False
    if not candidate.after_is_mature:
        return False
    if candidate.maturity_gain <= 0.0:
        return False
    if candidate.support_gain >= float(config.guard_support_gain_floor):
        return False
    if bool(config.guard_requires_contradiction_progress) and candidate.contradiction_gain > 0.0:
        return False
    return True
```

Then inside `choose_critic_action(...)`, before accepting a non-commit critic override:

```python
if best_edit is not None:
    override_margin = float(best_edit.score - heuristic_candidate.score)
    if override_margin >= float(config.tau_override):
        if _is_fragile_maturity_override(state=state, candidate=best_edit, config=config):
            return CriticPolicyDecision(
                selected_candidate_id=heuristic_candidate.candidate_id,
                selected_source="heuristic",
                used_heuristic_fallback=True,
                commit_allowed=commit_allowed,
                commit_requested=commit_requested,
                override_margin=override_margin,
                commit_margin=commit_margin,
            )
        ...
```

- [ ] **Step 2.5: Re-run the safety tests**

Run:
`python -m pytest tests/test_critic_policy.py tests/test_runtime_critic.py tests/test_engine.py -q`

Expected:
- maturity-sensitive fallback tests pass
- LLM-backed controller traces remain present

## Task 3: Run The Full Local Verification Packet

**Files:**
- Modify: `docs/experiment_execution_log.md`
- Modify: `docs/eig_graph_critic_plan.md`

- [ ] **Step 3.1: Run the controller-related regression suite**

Run:
`python -m pytest tests/test_engine.py tests/test_runtime_critic.py tests/test_benchmark_mode_and_baselines.py tests/test_online_text_critic.py tests/test_critic_policy.py tests/test_critic_replay.py tests/test_critic_episode_collection.py tests/test_critic_split_registry.py -q`

Expected:
- all controller-related tests pass

- [ ] **Step 3.2: Record the safety patch in the docs**

Update:
- `docs/experiment_execution_log.md`
- `docs/eig_graph_critic_plan.md`

Record:
- what the maturity-sensitive fallback does
- which tests cover it
- why this patch exists

## Task 4: Rerun The Frozen 4-Case AIIB Gate

**Files:**
- Modify: `docs/experiment_execution_log.md`
- Modify: `docs/eig_graph_critic_plan.md`
- Create: `outputs/m2_aiib_g48_controller_gate_v2/paired_summary.md`

- [ ] **Step 4.1: Launch the heuristic comparison runs**

Run:

```powershell
$cases = 13,3883,7909,9849
foreach ($case in $cases) {
  python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index $case --baseline ours-eig --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --max-rounds 5 --native-eval --output-root outputs/m2_aiib_g48_controller_gate_v2
}
```

- [ ] **Step 4.2: Launch the text-critic comparison runs**

Run:

```powershell
$cases = 13,3883,7909,9849
foreach ($case in $cases) {
  python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index $case --baseline ours-eig-critic-text --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --max-rounds 5 --native-eval --output-root outputs/m2_aiib_g48_controller_gate_v2
}
```

- [ ] **Step 4.3: Summarize the paired result**

Compute and record:
- mean local overall
- mean local alignment
- mean AIIB native average
- per-case stop round
- whether `runtime_controller_log` is present in the critic runs

Write the readout to:
`outputs/m2_aiib_g48_controller_gate_v2/paired_summary.md`

- [ ] **Step 4.4: Apply the stop/go rule**

Decision:

- if mean native score is neutral or positive and no clear new maturity
  regression appears, keep the text critic as a valid controller pilot
- otherwise freeze it as a mixed / negative pilot and stop spending large
  effort on it

## Final Verification Checklist

- [ ] Candidate-level gain metadata reaches the critic policy
- [ ] Mature-but-fragile overrides are blocked
- [ ] Controller traces persist in LLM-backed runs
- [ ] Controller-related regression suite passes
- [ ] Frozen 4-case AIIB gate is rerun and summarized
- [ ] Text critic is explicitly frozen as positive / mixed / negative pilot
