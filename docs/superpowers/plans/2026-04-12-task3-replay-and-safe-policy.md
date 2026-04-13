# Task 3 Replay And Safe Critic Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first conservative online-control layer for the learned text critic by introducing a replay buffer and a safe policy that can compare critic scores against the heuristic controller without touching held-out evaluation groups.

**Architecture:** Keep the existing warm-start scorer unchanged as the learned core. Add two thin layers around it: a replay module that stores train-group online episodes in a clean, append-only format, and a safe policy module that decides when the critic may override the heuristic or trigger `commit`. This keeps the controller reviewer-safe: the scorer stays simple, online updates stay batched, and all risky stopping behavior is filtered through explicit guardrails.

**Tech Stack:** Python 3.10+, existing `idea_graph` text-critic stack, JSONL artifacts, `pytest`, `scikit-learn`

---

## Scope

This slice is intentionally narrow. It does **not** yet run a large online adaptation batch. It only adds the infrastructure that the next pilot depends on:

1. an online episode / replay record format
2. conservative policy rules for `edit` override and `commit`
3. targeted tests for split safety and threshold behavior

The first adaptation run should happen only after this slice is implemented, reviewed, and verified.

## File Map

### New Files

- Create: `src/idea_graph/critic_replay.py`
  - episode schema
  - replay append / load helpers
  - offline/online batch-mixing helpers
- Create: `src/idea_graph/critic_policy.py`
  - conservative action-selection policy
  - commit gating and heuristic fallback
- Create: `tests/test_critic_replay.py`
- Create: `tests/test_critic_policy.py`

### Files To Extend

- Modify: `src/idea_graph/online_text_critic.py`
  - shared types and helpers used by replay / policy integration
- Modify: `docs/eig_graph_critic_plan.md`
  - mark Task 3 as the active implementation slice once code is landed
- Modify: `docs/experiment_execution_log.md`
  - record tests, artifacts, and the first controller-safety checkpoint

## Task A: Define And Test The Replay Contract

**Files:**
- Create: `src/idea_graph/critic_replay.py`
- Create: `tests/test_critic_replay.py`

- [ ] **Step A1: Write the failing replay tests**

```python
def test_append_episode_rejects_non_train_partition() -> None:
    episode = make_episode(partition_role="critic_dev")
    buffer = CriticReplayBuffer()
    with pytest.raises(ValueError, match="critic_train"):
        buffer.append_episode(episode)


def test_build_mixed_training_rows_respects_ratio_and_keeps_partitions_clean() -> None:
    offline_rows = [make_offline_row(i) for i in range(6)]
    online_rows = [make_online_row(i) for i in range(4)]
    mixed = build_mixed_training_rows(
        offline_rows,
        online_rows,
        offline_fraction=0.6,
        max_examples=10,
        random_seed=0,
    )
    assert len(mixed) == 10
    assert sum(1 for row in mixed if row["source"] == "offline") == 6
    assert sum(1 for row in mixed if row["source"] == "online") == 4
    assert {row["partition_role"] for row in mixed} == {"critic_train"}
```

- [ ] **Step A2: Run the replay tests to verify failure**

Run:
`python -m pytest tests/test_critic_replay.py -q`

Expected:
- import failure because `critic_replay.py` does not exist yet

- [ ] **Step A3: Implement the replay schema**

```python
@dataclass(frozen=True)
class OnlineCriticEpisode:
    episode_id: str
    group_id: str
    partition_role: str
    benchmark: str
    run_dir: str
    final_return: float | None
    final_native_return: float | None
    transitions: tuple[dict[str, object], ...]


class CriticReplayBuffer:
    def __init__(self) -> None:
        self._episodes: list[OnlineCriticEpisode] = []

    def append_episode(self, episode: OnlineCriticEpisode) -> None:
        if episode.partition_role != "critic_train":
            raise ValueError("Replay accepts only critic_train episodes.")
        self._episodes.append(episode)
```

- [ ] **Step A4: Add offline/online row mixing**

```python
def build_mixed_training_rows(
    offline_rows: Sequence[Mapping[str, object]],
    online_rows: Sequence[Mapping[str, object]],
    *,
    offline_fraction: float,
    max_examples: int,
    random_seed: int,
) -> list[dict[str, object]]:
    ...
```

Rules:
- only `critic_train` rows may enter the mixed batch
- keep source tags: `offline` or `online`
- deterministic sampling with `random_seed`

- [ ] **Step A5: Re-run the replay tests**

Run:
`python -m pytest tests/test_critic_replay.py -q`

Expected:
- all replay tests pass

## Task B: Define And Test The Safe Policy

**Files:**
- Create: `src/idea_graph/critic_policy.py`
- Create: `tests/test_critic_policy.py`

- [ ] **Step B1: Write the failing policy tests**

```python
def test_commit_blocked_before_min_round() -> None:
    decision = choose_critic_action(
        state=make_state(round_index=0),
        critic_candidates=[make_commit_candidate(score=0.95), make_edit_candidate(score=0.90)],
        heuristic_candidate=make_edit_candidate(score=0.70),
        config=SafeCriticPolicyConfig(min_commit_round=2),
    )
    assert decision.selected_candidate_id != "commit"
    assert decision.commit_allowed is False


def test_heuristic_fallback_when_override_margin_is_small() -> None:
    decision = choose_critic_action(
        state=make_state(round_index=3),
        critic_candidates=[make_edit_candidate(candidate_id="critic-best", score=0.61)],
        heuristic_candidate=make_edit_candidate(candidate_id="heuristic-best", score=0.60),
        config=SafeCriticPolicyConfig(tau_override=0.05),
    )
    assert decision.selected_candidate_id == "heuristic-best"
    assert decision.used_heuristic_fallback is True
```

- [ ] **Step B2: Run the policy tests to verify failure**

Run:
`python -m pytest tests/test_critic_policy.py -q`

Expected:
- import failure because `critic_policy.py` does not exist yet

- [ ] **Step B3: Implement the conservative policy config**

```python
@dataclass(frozen=True)
class SafeCriticPolicyConfig:
    min_commit_round: int = 2
    tau_override: float = 0.05
    tau_commit: float = 0.08
    gamma_commit: float = 0.60
```

- [ ] **Step B4: Implement action selection**

```python
def choose_critic_action(
    *,
    state: Mapping[str, object],
    critic_candidates: Sequence[ScoredCandidate],
    heuristic_candidate: ScoredCandidate,
    config: SafeCriticPolicyConfig,
) -> CriticPolicyDecision:
    ...
```

Rules:
- `commit` is blocked when `round_index < min_commit_round`
- `commit` requires both:
  - `commit_score - best_edit_score >= tau_commit`
  - `commit_confidence >= gamma_commit`
- non-commit critic override requires:
  - `critic_best_score - heuristic_score >= tau_override`
- otherwise return the heuristic action

- [ ] **Step B5: Re-run the policy tests**

Run:
`python -m pytest tests/test_critic_policy.py -q`

Expected:
- all policy tests pass

## Task C: Wire Minimal Integration Helpers

**Files:**
- Modify: `src/idea_graph/online_text_critic.py`

- [ ] **Step C1: Add small integration helpers only**

Add narrow helpers instead of a full runner:

```python
def partition_rows_for_role(
    candidate_rows: Sequence[Mapping[str, Any]],
    partition_lookup: Mapping[str, str],
    *,
    partition_role: str,
) -> list[dict[str, Any]]:
    ...
```

Use:
- replay construction
- future adaptation runner input filtering

- [ ] **Step C2: Keep warm-start behavior unchanged**

Verification:
- no existing public warm-start function changes signature
- current warm-start tests still pass

## Task D: Verify The Full Task 3 Safety Slice

**Files:**
- Modify: `docs/eig_graph_critic_plan.md`
- Modify: `docs/experiment_execution_log.md`

- [ ] **Step D1: Run the targeted packet**

Run:
`python -m pytest tests/test_critic_replay.py tests/test_critic_policy.py tests/test_online_text_critic.py -q`

Expected:
- all targeted tests pass

- [ ] **Step D2: Run the broader graph-critic packet**

Run:
`python -m pytest tests/test_trajectory_dataset.py tests/test_critic_dataset.py tests/test_candidate_slate_dataset.py tests/test_critic_partitions.py tests/test_text_critic.py tests/test_online_text_critic.py tests/test_critic_replay.py tests/test_critic_policy.py -q`

Expected:
- no regression in the offline dataset / warm-start stack

- [ ] **Step D3: Update docs**

Record:
- controller-safety layer implemented
- no `paper_eval` exposure
- Task 3 ready for the first tiny online adaptation pilot

## Decision Gate After This Slice

Proceed to the first online adaptation run only if:

- replay accepts only `critic_train` episodes
- the safe policy never commits before the minimum round gate
- heuristic fallback works when critic margins are weak
- the expanded graph-critic test packet passes

If any of these fail, do not launch adaptation. Fix the controller-safety layer first.
