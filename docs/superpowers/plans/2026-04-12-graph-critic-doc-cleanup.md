# Graph Critic Documentation Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the active documentation around the new EIG graph-critic research track while preserving prior heuristic-controller records as archived history.

**Architecture:** Add one canonical forward-looking graph-critic plan, update the active protocol/evaluation/experiment docs to reference it, and move superseded planning notes out of the active reading path. No code behavior changes are included in this cleanup.

**Tech Stack:** Markdown documentation, PowerShell file moves, repository-local docs.

---

### Task 1: Add The Canonical Graph-Critic Plan

**Files:**
- Create: `docs/eig_graph_critic_plan.md`

- [x] **Step 1: Write the plan document**

Create a focused Markdown plan with these concrete sections:

```markdown
# EIG Graph Critic Plan

## Purpose

## Core Insight

## Optimization View

## Model Sketch

## Supervision

## Calibration

## Evaluation And Ablations

## Staged Roadmap

## Open Risks
```

- [x] **Step 2: Self-check**

Verify the plan clearly separates the new learned critic track from prior heuristic EIG runs and does not claim final experimental results.

### Task 2: Update Active Docs

**Files:**
- Modify: `docs/README.md`
- Modify: `docs/paper_protocol.md`
- Modify: `docs/evaluation.md`
- Modify: `docs/paper_experiment_plan.md`
- Modify: `docs/paper_experiment_tracker.md`
- Modify: `docs/experiment_execution_log.md`

- [x] **Step 1: Update the docs index**

Make `docs/eig_graph_critic_plan.md` part of the active reading order and remove superseded active-map references.

- [x] **Step 2: Update the protocol**

Describe the forward method as `EIG with a learned graph critic`; keep the heuristic controller as a prototype/ablation.

- [x] **Step 3: Update evaluation policy**

Add graph-critic-specific ablations and calibration checks while keeping benchmark-native and human review as primary evaluation.

- [x] **Step 4: Update experiment plan and tracker**

Mark the full `R009` launch as paused until graph-critic planning and trajectory export are complete.

- [x] **Step 5: Add a concise execution-log entry**

Record that the method track is shifting from heuristic maturity thresholds to a learned critic.

### Task 3: Archive Superseded Active Planning Notes

**Files:**
- Move: `docs/paper_experiment_map.md` to `docs/archive/paper_experiment_map_pre_critic.md`
- Move: `docs/r009_aiib_launch_plan.md` to `docs/archive/r009_aiib_launch_plan_pre_critic.md`
- Modify: `docs/archive/README.md`

- [x] **Step 1: Move the two superseded docs**

Use non-destructive file moves so history remains available.

- [x] **Step 2: Update archive index**

Add the moved files to the archive README and explain they are pre-critic planning records.

### Task 4: Verify The Cleanup

**Files:**
- Read-only verification

- [x] **Step 1: Check docs layout**

Run:

```powershell
Get-ChildItem docs -Recurse | Select-Object FullName
```

Expected: active docs include the graph-critic plan; archived docs include the pre-critic map and R009 launch plan.

- [x] **Step 2: Check git status**

Run:

```powershell
git status --short
```

Expected: documentation edits plus the already-existing code/test changes from the previous worktree state; no accidental output deletions.
