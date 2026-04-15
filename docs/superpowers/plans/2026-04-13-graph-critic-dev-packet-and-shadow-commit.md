# Graph Critic Dev Packet And Shadow Commit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add controller observability plus shadow-commit logging, then run a medium development-only packet that is large enough to judge whether the graph critic is promising without contaminating final paper-eval benchmarks.

**Architecture:** Keep the current relation-aware graph critic fixed. First patch runtime observability so each controller decision can be traced through ranking, materialization, maturity, and shadow stopping. Then rerun a small sentinel regression packet plus one 12-instance development packet drawn exactly from `development_pool_v2_candidate_pool_v1`, and evaluate the graph critic on both end-to-end quality and controller-process metrics.

**Tech Stack:** Python 3.10, existing `idea_graph` runtime, pytest, PowerShell, OpenAI-compatible backend, Markdown/JSON artifact analysis

---

## File Structure

### New Files

- `docs/superpowers/plans/2026-04-13-graph-critic-dev-packet-and-shadow-commit.md`
  - concrete execution protocol for the next controller stage
- `outputs/m2_graph_critic_dev_packet_v1/README.md`
  - packet-local artifact index after runs complete
- `outputs/m2_graph_critic_dev_packet_v1/packet_summary.md`
  - benchmark-level and pooled comparison summary after analysis
- `outputs/m2_graph_critic_dev_packet_v1/packet_summary.json`
  - machine-readable comparison payload after analysis

### Modified Files

- `src/idea_graph/engine.py`
  - add decision-materialization logging
  - add maturity-term breakdown logging
  - add shadow-commit logging
- `src/idea_graph/io.py`
  - surface controller and shadow-commit summaries in `summary.json`
- `tests/test_engine.py`
  - verify new runtime-controller observability fields
- `docs/eig_graph_critic_plan.md`
  - record the planned G6.2 development packet
- `docs/experiment_execution_log.md`
  - record the planned packet and exact development instances

## Packet Definition

### Sentinel Regression Set

Keep the current 4-case AIIB controller packet as a regression gate:

- `ai_idea_bench_2025`
  - `13`
  - `3883`
  - `7909`
  - `9849`

Purpose:

- verify that the next patch does not reintroduce the `3883` early-stop failure
- preserve comparability with `outputs/m2_aiib_g6_graph_controller_gate_v1`

### Development-Only Medium Packet

Use the exact frozen candidate instances from:

- `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_candidate_pool_v1/candidate_instances.json`

This gives a 12-instance development packet:

- `AI_Idea_Bench_2025`
  - `25`
  - `30`
  - `36`
  - `82`
  - `87`
  - `95`
  - `110`
  - `125`
- `liveideabench`
  - `hurricanes-118`
  - `phycology-140`
  - `galaxies-163`
  - `global positioning system-191`

Pool rule:

- this packet is **development-only**
- do not use these runs as frozen final paper evidence for the learned controller

## Runtime Changes Required Before The Packet

### Observability Additions

For every controller decision, persist:

- heuristic selected candidate id / kind / predicted gain
- critic selected candidate id / kind / score
- whether the critic-selected candidate became the executed action
- if not, why not:
  - LLM proposal kept
  - heuristic fallback
  - validation failure
  - application failure
- selected action source after execution
- maturity snapshot used before the decision
- maturity snapshot after the round

### Maturity-Term Breakdown

For every completed round, persist a breakdown of the maturity rule:

- `support_coverage`
- `unresolved_contradiction_ratio`
- `completeness`
- `coherence`
- `evidence`
- `utility`
- `utility_stable`
- `benchmark_specific_ready`
- `benchmark_high_confidence_ready`
- `high_confidence_mature`
- final `is_mature`

Purpose:

- distinguish "bad ranking" from "bad stopping"
- explain whether a stop came from the normal maturity path or the high-confidence shortcut

### Shadow Commit

Do **not** enable live commit yet.

Instead, add a shadow-commit probe that logs:

- whether `commit` would have been available in the runtime slate
- whether a commit would have passed the critic policy thresholds
- round index at which shadow commit would have fired
- best edit score vs best commit score
- whether later executed rounds actually produced additional utility gain

Purpose:

- measure wasted late rounds without mixing in a new live stopping policy yet

## Artifact Roots

### Sentinel Packet Root

- `outputs/m2_graph_critic_dev_packet_v1/sentinel_regression`

### Development Packet Roots

- `outputs/m2_graph_critic_dev_packet_v1/dev_packet/ours-eig`
- `outputs/m2_graph_critic_dev_packet_v1/dev_packet/ours-eig-critic-graph`

### Analysis Root

- `outputs/m2_graph_critic_dev_packet_v1`

Required final artifacts:

- `outputs/m2_graph_critic_dev_packet_v1/README.md`
- `outputs/m2_graph_critic_dev_packet_v1/packet_summary.md`
- `outputs/m2_graph_critic_dev_packet_v1/packet_summary.json`

## Evaluation Metrics

### End-To-End Quality

Per run:

- benchmark native average normalized to 10
- local overall score
- local benchmark alignment
- executed round count
- stop reason

Aggregates:

- pooled mean by baseline
- benchmark-specific mean by baseline
- paired delta by instance

### Controller-Process Metrics

Per run:

- runtime decision count
- critic-selected count
- heuristic-selected count
- materialized override count
- materialized override rate
- kind-match rate between controller-selected kind and executed action kind
- shadow-commit candidate count
- shadow-commit would-fire count

Aggregates:

- pooled and per-benchmark materialized override rate
- pooled and per-benchmark kind-match rate
- role-wise selected-source counts

### Stop-Policy Metrics

Per run:

- first round where `is_mature=True`
- whether `utility_stable=True` at stop
- whether `high_confidence_mature=True` at stop
- support / evidence / utility at stop
- whether later rounds existed in the paired heuristic run and improved utility

## Decision Rules

### Stage A: Sentinel Safety Gate

The next patch fails immediately if either graph-controller regression run shows:

- stop before `Round3` while `utility_stable=False`
- AIIB native delta less than `-0.75` versus the paired `ours-eig` run

Interpretation:

- this guards directly against another `3883`-style premature maturity failure

### Stage B: Development Packet Promotion Gate

Promote to a larger learned-controller packet only if all of the following hold:

1. Sentinel safety gate passes.
2. On the 12-instance development packet:
   - pooled mean native delta is at least `-0.05`
   - `AI_Idea_Bench_2025` mean native delta is non-negative
   - no benchmark family mean is worse than `-0.10`
3. Materialized override rate is at least `0.15` or at least `2x` the current graph-controller pilot rate (`0.0566`).
4. Shadow-commit analysis does not show repeated "should have committed early" failures concentrated at `Round4` / `Round5`.

### Mixed Outcome

If sentinel safety passes but promotion fails:

- keep the graph critic in development-only status
- use the new logs to decide between:
  - maturity-rule tightening
  - controller materialization cleanup
  - targeted robustness-data expansion

## Execution Order

### Task 1: Add Observability And Shadow Commit

**Files:**
- Modify: `src/idea_graph/engine.py`
- Modify: `src/idea_graph/io.py`
- Modify: `tests/test_engine.py`

- [ ] Add controller-materialization fields to the runtime trace and summary payloads.
- [ ] Add round-level maturity-term breakdown fields.
- [ ] Add shadow-commit logging without changing live controller behavior.
- [ ] Run:

```powershell
python -m pytest tests/test_engine.py tests/test_benchmark_mode_and_baselines.py tests/test_relation_graph_runtime_critic.py -q
```

- [ ] Commit:

```powershell
git add src/idea_graph/engine.py src/idea_graph/io.py tests/test_engine.py
git commit -m "feat: add controller observability and shadow commit logging"
```

### Task 2: Rerun Sentinel Regression Packet

**Root:**
- `outputs/m2_graph_critic_dev_packet_v1/sentinel_regression`

- [ ] Run paired sentinel regression for:

```powershell
$aiibCases = 13,3883,7909,9849
$baselines = 'ours-eig','ours-eig-critic-graph'
foreach ($baseline in $baselines) {
  foreach ($idx in $aiibCases) {
    python scripts/run_pipeline.py `
      --benchmark ai_idea_bench_2025 `
      --benchmark-index $idx `
      --baseline $baseline `
      --agent-backend openai-compatible `
      --llm-config configs/openai_compatible.example.json `
      --max-rounds 5 `
      --native-eval `
      --output-dir outputs/m2_graph_critic_dev_packet_v1/sentinel_regression
  }
}
```

- [ ] Verify the sentinel gate before spending on the 12-instance packet.

### Task 3: Run The 12-Instance Development Packet

**Roots:**
- `outputs/m2_graph_critic_dev_packet_v1/dev_packet/ours-eig`
- `outputs/m2_graph_critic_dev_packet_v1/dev_packet/ours-eig-critic-graph`

- [ ] Run the AIIB development subset:

```powershell
$aiibCases = 25,30,36,82,87,95,110,125
$baselines = 'ours-eig','ours-eig-critic-graph'
foreach ($baseline in $baselines) {
  foreach ($idx in $aiibCases) {
    python scripts/run_pipeline.py `
      --benchmark ai_idea_bench_2025 `
      --benchmark-index $idx `
      --baseline $baseline `
      --agent-backend openai-compatible `
      --llm-config configs/openai_compatible.example.json `
      --max-rounds 5 `
      --native-eval `
      --output-dir ("outputs/m2_graph_critic_dev_packet_v1/dev_packet/" + $baseline)
  }
}
```

- [ ] Run the LiveIdeaBench development subset:

```powershell
$liveRows = @(
  @{ Index = 118; Keyword = 'hurricanes' },
  @{ Index = 140; Keyword = 'phycology' },
  @{ Index = 163; Keyword = 'galaxies' },
  @{ Index = 191; Keyword = 'global positioning system' }
)
$baselines = 'ours-eig','ours-eig-critic-graph'
foreach ($baseline in $baselines) {
  foreach ($row in $liveRows) {
    python scripts/run_pipeline.py `
      --benchmark liveideabench `
      --benchmark-index $row.Index `
      --benchmark-keyword $row.Keyword `
      --baseline $baseline `
      --agent-backend openai-compatible `
      --llm-config configs/openai_compatible.example.json `
      --max-rounds 5 `
      --native-eval `
      --output-dir ("outputs/m2_graph_critic_dev_packet_v1/dev_packet/" + $baseline)
  }
}
```

### Task 4: Build Packet Analysis

**Files:**
- Create: `outputs/m2_graph_critic_dev_packet_v1/README.md`
- Create: `outputs/m2_graph_critic_dev_packet_v1/packet_summary.md`
- Create: `outputs/m2_graph_critic_dev_packet_v1/packet_summary.json`

- [ ] Aggregate:
  - pooled means
  - AIIB-only means
  - LiveIdeaBench-only means
  - paired per-instance deltas
  - controller-process metrics
  - shadow-commit metrics
  - sentinel gate result
- [ ] Write a short human-readable outcome:
  - `go`
  - `mixed`
  - `no-go`

### Task 5: Update Tracker Docs

**Files:**
- Modify: `docs/eig_graph_critic_plan.md`
- Modify: `docs/experiment_execution_log.md`

- [ ] Record:
  - exact dev packet membership
  - artifact roots
  - sentinel gate result
  - dev packet result
  - next recommendation

## Why This Plan Is Fast But Defensible

- It avoids the false confidence of another 4-case-only judgment.
- It keeps development data and future paper-eval data cleanly separated.
- It does not mix a live `commit` policy into the graph-critic diagnosis yet.
- It adds exactly the observability we need to decide whether failures come
  from ranking, materialization, or maturity.

## Self-Review

- Spec coverage:
  - exact 12-case development packet is defined
  - exact sentinel regression packet is preserved
  - shadow commit is specified as logging-only
  - decision rules are explicit
- Placeholder scan:
  - no `TODO` / `TBD`
  - artifact roots and benchmark instances are concrete
- Type consistency:
  - baseline names stay `ours-eig` and `ours-eig-critic-graph`
  - development pool references match `development_pool_v2_candidate_pool_v1`
