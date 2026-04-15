# Online Graph Critic Broad-Gate And Paper-Eval Launch Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the earlier `Tier C0 / C1 / C2` controller-evaluation ladder with one broader development-only online gate, then move directly to untouched paper-eval pool freezing and main learned-controller benchmarking if that gate passes.

**Architecture:** Reuse the current runtime controller and benchmark runner, but stop splitting the next decision across tiny online packets. Build one manifest over the full current combined `G2` development partition universe, run paired `ours-eig` versus `ours-eig-critic-graph` on all `59` frozen development groups, then summarize three readouts from the same run set: held-out `critic_dev` as the primary signal, pooled `59` as robustness evidence, and `critic_train` as diagnostic context. If that broad gate is acceptable, freeze the controller and immediately expand and freeze the untouched `paper_eval` pool for the main table.

**Tech Stack:** Python 3.10+, existing `run_pipeline.py`, JSON/JSONL manifests, benchmark-native evaluation, PowerShell, `pytest`, Markdown/JSON packet summaries

---

## Scope

This plan is the concrete execution layer for the revised fast path.

This slice should:

- stop using separate `Tier C0` and `Tier C1` stages as blockers
- use one larger frozen development-only online gate instead
- preserve a clean held-out signal by reporting `critic_dev` inside the broad gate
- freeze or reject the learned controller before any untouched `paper_eval` run
- move to `paper_eval` immediately after a successful broad gate

This slice should not:

- spend more time on another tiny 4-case regression packet
- treat the current `36`-group v3 candidate pool alone as the whole story
- retune the controller after seeing paper-eval outputs
- present any development-only packet as the final paper table

## Naming Bridge

Paper-facing labels:

- `ours-eig`
- `ours-eig-graph-critic`
- optional supporting ablation: `ours-eig-critic-text`

Current runtime baseline names:

- `ours-eig`
- `ours-eig-critic-graph`
- `ours-eig-critic-text`

Rule:

- keep the runtime execution baseline name as `ours-eig-critic-graph`
- map it to the paper label `ours-eig-graph-critic` only in summaries, tables,
  and paper artifacts

## Broad Development Gate Definition

### Source

- `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g2_partitions/partition_manifest.jsonl`

### Membership

Use all frozen development groups currently available in the combined `G2`
partition layer:

- total groups: `59`
- benchmark mix:
  - `39` `AI_Idea_Bench_2025`
  - `20` `LiveIdeaBench`
- partition-role mix:
  - `47` `critic_train`
  - `12` `critic_dev`

Benchmark-role breakdown:

- `AI_Idea_Bench_2025`
  - `33` `critic_train`
  - `6` `critic_dev`
- `LiveIdeaBench`
  - `14` `critic_train`
  - `6` `critic_dev`

### Readout Policy

Run one broad packet and summarize three views from the same saved run set.

Primary readout:

- held-out `critic_dev` groups only (`12`)

Secondary readout:

- pooled all-group development result (`59`)

Diagnostic readout:

- `critic_train` groups only (`47`)

Interpretation rule:

- the held-out `critic_dev` readout is the real promotion signal
- the pooled readout is robustness evidence
- the `critic_train` readout is diagnostic only

### Optional Ambiguity Rerun

Only run extra repeated online reruns if the broad gate is ambiguous.

Trigger examples:

- primary `critic_dev` delta is close to zero
- win rate and mean delta disagree strongly
- stop behavior changes but quality deltas remain unclear
- AIIB and LiveIdeaBench trends disagree on the held-out slice

If triggered:

- rerun all `12` held-out `critic_dev` groups `3x` per compared system

## Stage E0 Paper-Eval Freeze Target

After a successful broad development gate, immediately build and freeze:

- `paper_eval_candidate_pool_v2`

Recommended first launch target:

- `64` `AI_Idea_Bench_2025`
- `48` `LiveIdeaBench`

Rules:

- zero overlap with:
  - `development_pool_v1`
  - `development_pool_v2_candidate_pool_v1`
  - `development_pool_v3_candidate_pool_v1`
- freeze membership before any learned-controller paper-eval run
- keep the exact frozen manifest with benchmark and instance names under a
  dedicated artifact root

## Output Roots

### Broad-Gate Packet Root

- `outputs/controller_eval_packets/graph_critic_scaleup_v2`

Required artifacts:

- `README.md`
- `broad_dev_gate_59.jsonl`
- `packet_stats.json`

### Broad-Gate Run Root

- `outputs/m2_graph_critic_online_scaleup_v2`

Required artifacts:

- `runs/`
- `run_manifest.jsonl`
- `paired_summary.md`
- `paired_summary.json`
- `controller_trace_summary.json`
- `freeze_decision.md`

### Paper-Eval Freeze Root

- `outputs/graph_critic_datasets/02_active_graph_critic/paper_eval_candidate_pool_v2`

Required artifacts:

- `candidate_instances.json`
- `README.md`
- `pool_stats.json`

## Current Status

Landed on the current main workspace:

- `src/idea_graph/controller_eval_packets.py`
- `scripts/build_controller_eval_packets.py`
- `tests/test_controller_eval_packets.py`
- `src/idea_graph/controller_eval_runtime.py`
- `scripts/run_controller_eval_packet.py`
- `scripts/summarize_controller_eval_packet.py`
- `tests/test_controller_eval_runner.py`
- `src/idea_graph/paper_eval_pool.py`
- `scripts/build_paper_eval_freeze_pool.py`
- `tests/test_paper_eval_freeze_pool.py`
- packet builder output root:
  - `outputs/controller_eval_packets/graph_critic_scaleup_v2`
- verified emitted counts:
  - total groups: `59`
  - `critic_train`: `47`
  - `critic_dev`: `12`
  - `AI_Idea_Bench_2025`: `39`
  - `LiveIdeaBench`: `20`
- no-API dry-run root:
  - `outputs/m2_graph_critic_online_scaleup_v2_dryrun`
  - planned run-manifest rows: `118`

Active next execution frontier:

- launch the real broad-gate paired run on all `59` groups
- summarize the saved run set
- write `freeze_decision.md`
- build `paper_eval_candidate_pool_v2` only after the broad-gate freeze memo
  says `go`

## File Map

### Files Already Landed

- `src/idea_graph/controller_eval_packets.py`
  - broad-gate manifest shaping from the frozen combined `G2` partition
    manifest
- `scripts/build_controller_eval_packets.py`
  - CLI writer for `broad_dev_gate_59.jsonl`, `README.md`, and
    `packet_stats.json`
- `tests/test_controller_eval_packets.py`
  - broad-gate count, selector, duplicate, and CLI artifact checks

### Files Already Landed For The Runtime/Freeze Stack

- `src/idea_graph/controller_eval_runtime.py`
  - shared loader, runner, score aggregation, bootstrap, and trace-summary
    helpers for packet evaluation
- `scripts/run_controller_eval_packet.py`
  - execute one packet manifest for one or more baselines through the existing
    pipeline
- `scripts/summarize_controller_eval_packet.py`
  - compute the held-out, pooled, and diagnostic readouts from saved runs
- `tests/test_controller_eval_runner.py`
  - verify packet loading, runtime summaries, and summarizer CLI artifacts
- `scripts/build_paper_eval_freeze_pool.py`
  - materialize the expanded untouched `paper_eval_candidate_pool_v2`
- `tests/test_paper_eval_freeze_pool.py`
  - verify `paper_eval_candidate_pool_v2` is disjoint from all development pools

### Files To Create After The Real Broad Gate

- `outputs/m2_graph_critic_online_scaleup_v2/freeze_decision.md`
  - freeze-or-no-go memo from the held-out, pooled, and diagnostic readouts

### Files To Modify

- `docs/superpowers/specs/2026-04-14-graph-critic-scaleup-and-main-eval-design.md`
  - reflect the single broad gate strategy
- `docs/paper_experiment_plan.md`
  - align the paper-facing next steps with the broad gate
- `docs/experiment_execution_log.md`
  - record the strategy change and the new immediate run order

## Task 1: Materialize The Broad Development Gate Manifest

Status:

- this task is already landed on the current main workspace
- keep the checklist below as the regression contract, not the active frontier

**Files:**
- Create: `src/idea_graph/controller_eval_packets.py`
- Create: `scripts/build_controller_eval_packets.py`
- Create: `tests/test_controller_eval_packets.py`
- Create: `outputs/controller_eval_packets/graph_critic_scaleup_v2/README.md`
- Create: `outputs/controller_eval_packets/graph_critic_scaleup_v2/broad_dev_gate_59.jsonl`
- Create: `outputs/controller_eval_packets/graph_critic_scaleup_v2/packet_stats.json`

- [ ] **Step 1.1: Add a packet-builder test for the single broad gate**

Verify:

- the packet builder reads
  `development_pool_v3_combined_g2_partitions/partition_manifest.jsonl`
- emitted broad-gate count is exactly `59`
- role counts are exactly:
  - `critic_train = 47`
  - `critic_dev = 12`
- benchmark counts are exactly:
  - `AI_Idea_Bench_2025 = 39`
  - `LiveIdeaBench = 20`

- [ ] **Step 1.2: Implement the broad-gate packet builder**

Run:

```powershell
python scripts/build_controller_eval_packets.py --output-root outputs/controller_eval_packets/graph_critic_scaleup_v2
```

Expected:

- `broad_dev_gate_59.jsonl` exists
- `packet_stats.json` includes:
  - `group_count = 59`
  - `role_counts = {'critic_train': 47, 'critic_dev': 12}`
  - `benchmark_counts = {'AI_Idea_Bench_2025': 39, 'liveideabench': 20}`

- [ ] **Step 1.3: Run the packet-builder verification**

Run:

```powershell
python -m pytest tests/test_controller_eval_packets.py -q
```

Expected:

- pass

## Task 2: Finish The Shared Runtime And Add The Broad-Gate Runner

Status:

- this task is implemented
- verification:
  - `python -m pytest tests/test_controller_eval_runner.py -q`
  - `python -m pytest tests/test_controller_eval_packets.py tests/test_controller_eval_runner.py tests/test_paper_eval_freeze_pool.py -q`
- no-API dry run:
  - `python scripts/run_controller_eval_packet.py --packet-manifest outputs/controller_eval_packets/graph_critic_scaleup_v2/broad_dev_gate_59.jsonl --baselines ours-eig ours-eig-critic-graph --output-root outputs/m2_graph_critic_online_scaleup_v2_dryrun --max-rounds 5 --dry-run`
  - emitted `118` planned rows

**Files:**
- Modify: `src/idea_graph/controller_eval_runtime.py`
- Create: `scripts/run_controller_eval_packet.py`
- Create: `scripts/summarize_controller_eval_packet.py`
- Modify: `tests/test_controller_eval_runner.py`

- [ ] **Step 2.1: Add a manifest-runner smoke test**

Verify:

- the runner loads one packet row from `broad_dev_gate_59.jsonl`
- the runner preserves:
  - `benchmark`
  - `instance_name`
  - `partition_role`
- optional partition-role filtering preserves only the requested rows
- the runner writes a normalized `run_manifest.jsonl` before generation starts

- [ ] **Step 2.2: Make the shared runtime helper green**

Required helper behavior:

- `load_packet_rows(...)` enforces unique `group_id` rows and optional
  `partition_role` filtering
- `packet_row_to_benchmark_args(...)` normalizes:
  - `AI_Idea_Bench_2025 -> ai_idea_bench_2025`
  - `liveideabench -> liveideabench`
- `execute_packet_run(...)` calls the existing benchmark runner directly and
  returns a normalized manifest row
- `summarize_packet_runs(...)` emits:
  - held-out `critic_dev` readout
  - pooled readout
  - `critic_train` diagnostic readout
  - controller trace aggregation
  - paired-delta and bootstrap summaries

- [ ] **Step 2.3: Implement the broad-gate runner**

Required runner behavior:

- accept:
  - `--packet-manifest`
  - `--baselines`
  - `--llm-config`
  - `--output-root`
  - `--max-rounds`
  - `--native-eval`
  - optional `--partition-role-filter`
- reuse `execute_packet_run(...)` from
  `src/idea_graph/controller_eval_runtime.py` instead of shelling out through
  PowerShell
- write all run provenance under one broad-gate root

- [ ] **Step 2.4: Implement the broad-gate summarizer**

Required summary sections:

- held-out `critic_dev` summary
- pooled `59`-group summary
- `critic_train` diagnostic summary
- benchmark-family stratification:
  - AIIB
  - LiveIdeaBench
- controller-process diagnostics:
  - action counts by kind
  - action counts by round
  - controller override rate
  - materialized override rate
- heuristic fallback rate
- stop-round distribution
- total rounds without commit

- [ ] **Step 2.5: Run the runner/summarizer verification**

Run:

```powershell
python -m pytest tests/test_controller_eval_packets.py tests/test_controller_eval_runner.py tests/test_benchmark_mode_and_baselines.py tests/test_critic_policy.py tests/test_relation_graph_runtime_critic.py -q
```

Expected:

- pass

## Task 3: Run The Broad Development Gate

Status:

- this is now the active next step

**Files:**
- Create: `outputs/m2_graph_critic_online_scaleup_v2/runs/`
- Create: `outputs/m2_graph_critic_online_scaleup_v2/run_manifest.jsonl`

- [ ] **Step 3.1: Launch paired broad-gate evaluation**

Run:

```powershell
python scripts/run_controller_eval_packet.py --packet-manifest outputs/controller_eval_packets/graph_critic_scaleup_v2/broad_dev_gate_59.jsonl --baselines ours-eig ours-eig-critic-graph --llm-config configs/openai_compatible.example.json --output-root outputs/m2_graph_critic_online_scaleup_v2 --max-rounds 5 --native-eval
```

Expected:

- `59 x 2 = 118` completed run directories
- `run_manifest.jsonl` contains `118` rows

- [ ] **Step 3.2: Summarize the broad gate**

Run:

```powershell
python scripts/summarize_controller_eval_packet.py --input-root outputs/m2_graph_critic_online_scaleup_v2 --write-root outputs/m2_graph_critic_online_scaleup_v2
```

Expected:

- `paired_summary.md`
- `paired_summary.json`
- `controller_trace_summary.json`

Required promotion readouts:

- primary:
  - `critic_dev = 12`
- secondary:
  - pooled `59`
- diagnostic:
  - `critic_train = 47`

## Task 4: Freeze Decision After The Broad Gate

**Files:**
- Create: `outputs/m2_graph_critic_online_scaleup_v2/freeze_decision.md`

- [ ] **Step 4.1: Write the freeze-or-no-go memo**

The memo must answer:

- is the held-out `critic_dev` readout at least near-neutral versus `ours-eig`?
- does the controller materially affect execution?
- is there any clear early-stop or maturity pathology?
- does the pooled `59`-group readout avoid a clear negative trend?
- does the `critic_train` versus `critic_dev` gap look acceptable?

- [ ] **Step 4.2: Trigger an ambiguity rerun only if needed**

Run only if the memo cannot decide:

```powershell
python scripts/run_controller_eval_packet.py --packet-manifest outputs/controller_eval_packets/graph_critic_scaleup_v2/broad_dev_gate_59.jsonl --partition-role-filter critic_dev --baselines ours-eig ours-eig-critic-graph --llm-config configs/openai_compatible.example.json --output-root outputs/m2_graph_critic_online_scaleup_v2_repeat --max-rounds 5 --native-eval
```

Expected:

- `12 x 2 = 24` completed run directories
- only needed if the first broad gate is borderline

## Task 5: Expand And Freeze The Untouched Paper-Eval Pool

**Files:**
- Create: `scripts/build_paper_eval_freeze_pool.py`
- Create: `tests/test_paper_eval_freeze_pool.py`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/paper_eval_candidate_pool_v2/candidate_instances.json`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/paper_eval_candidate_pool_v2/README.md`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/paper_eval_candidate_pool_v2/pool_stats.json`

- [ ] **Step 5.1: Add a pool-disjointness test**

Verify:

- `paper_eval_candidate_pool_v2` has zero overlap with:
  - `development_pool_v1`
  - `development_pool_v2_candidate_pool_v1`
  - `development_pool_v3_candidate_pool_v1`

- [ ] **Step 5.2: Build the frozen paper-eval pool**

Run:

```powershell
python scripts/build_paper_eval_freeze_pool.py --output-root outputs/graph_critic_datasets/02_active_graph_critic/paper_eval_candidate_pool_v2 --target-aiib 64 --target-live 48
```

Expected:

- `candidate_instances.json` exists
- minimum frozen target:
  - `64` `AI_Idea_Bench_2025`
  - `48` `LiveIdeaBench`

- [ ] **Step 5.3: Verify the frozen paper-eval pool**

Run:

```powershell
python -m pytest tests/test_paper_eval_freeze_pool.py -q
```

Expected:

- pass

## Task 6: Launch Main Learned-Controller Evaluation

**Files:**
- Update or create the future main evaluation artifact root after pool freeze

- [ ] **Step 6.1: Launch the learned-controller main table only after the freeze memo says `go`**

Main comparison set:

- `direct`
- `self-refine`
- `ai-researcher`
- `ours-eig`
- `ours-eig-critic-graph`

- [ ] **Step 6.2: Keep the paper framing explicit**

Reporting rule:

- runtime label in artifacts may remain `ours-eig-critic-graph`
- paper tables and text must use `ours-eig-graph-critic`

## Notes

- This revised plan supersedes the earlier smaller-stage interpretation in the
  previous version of this file.
- Historical `Tier C0` and `Tier C1` artifacts remain useful diagnosis logs,
  but they are no longer separate blocking stages.
- The current untouched pool with only `10` proposed cases is still too small,
  so `paper_eval_candidate_pool_v2` is mandatory before the learned-controller
  main table.
