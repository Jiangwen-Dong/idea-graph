# Documentation Guide

This directory is organized into a small active set plus an archive.

The goal is to keep the paper-facing workflow easy to follow: protocol first,
then the current method plan, then evaluation and experiment tracking.

## Active Docs

- `paper_protocol.md`
  Canonical paper-facing task, baseline, and evaluation protocol.
- `reproducibility.md`
  Current release-hygiene, split, artifact, baseline, and worktree policy for
  a clean supplementary-code snapshot.
- `eig_graph_critic_plan.md`
  Historical-to-current graph-critic roadmap. The parallel-v2 plans and specs
  are now the active implementation path.
- `evaluation.md`
  Canonical evaluation policy for benchmark-native scoring, human review, and
  critic-specific ablations.
- `benchmarks.md`
  Short operational note on the two supported benchmark integrations.
- `critic_pools.md`
  Canonical note on development-only critic pools versus untouched future
  paper-evaluation pools.
- `paper_experiment_plan.md`
  The detailed claim-driven experiment roadmap.
- `paper_experiment_tracker.md`
  The compact run tracker and current milestone status.
- `experiment_execution_log.md`
  Chronological engineering and experiment log.

## Supporting Plans

- `superpowers/plans/2026-04-12-graph-critic-doc-cleanup.md`
  Documentation cleanup plan for the graph-critic transition.
- `superpowers/specs/2026-04-12-critic-train-and-paper-eval-split-design.md`
  Canonical split design for `critic_train`, `critic_dev`, and future
  `paper_eval` usage.
- `superpowers/plans/2026-04-12-task3-replay-and-safe-policy.md`
  Implementation plan for replay buffering and the conservative critic policy.
- `superpowers/plans/2026-04-12-task4-online-adaptation-runner.md`
  Implementation plan for offline-plus-online text-critic adaptation.
- `superpowers/plans/2026-04-12-split-registry-and-paper-eval-candidates.md`
  Implementation plan for the frozen split registry and untouched paper-eval
  candidate pool.
- `superpowers/plans/2026-04-12-critic-train-episode-collection.md`
  Implementation plan for real train-pool episode collection.
- `superpowers/specs/2026-04-11-eig-robustness-cleanup-design.md`
  Historical focused design note for the pre-critic EIG robustness cleanup.
- `superpowers/plans/2026-04-11-eig-robustness-cleanup.md`
  Historical implementation plan for the same cleanup.

## Archive

- `archive/`
  Older pilot notes, one-off redesign memos, and superseded planning documents
  that are kept only for historical reference.
- `archive/README.md`
  Short guide to the archived notes.

## Recommended Reading Order

1. `paper_protocol.md`
2. `eig_graph_critic_plan.md`
3. `evaluation.md`
4. `critic_pools.md`
5. `paper_experiment_plan.md`
6. `paper_experiment_tracker.md`
7. `experiment_execution_log.md`

## Current Status

- The active runtime path is `parallel_graph_v2`, not the older sequential EIG
  loop.
- The heuristic parallel-v2 teacher is the current bootstrap controller for
  replay collection and ablation.
- The tracked critic split is `data/splits/parallel_v2`:
  - `300` critic-train groups
  - `100` critic-dev groups
  - balanced `150/50` train/dev rows per benchmark
- The tracked paper-eval split is also in `data/splits/parallel_v2`:
  - `256` frozen paper-eval groups
  - `128` AI Idea Bench 2025 groups
  - `128` LiveIdeaBench groups
  - zero overlap with critic train/dev according to the tracked audit
- The old sequential/text-critic controller notes are preserved as historical
  development context, but new experiments should use the parallel-v2 protocol
  unless explicitly reproducing an older ablation.

## Practical Use

- If you want the paper-facing rules, read `paper_protocol.md` and
  `evaluation.md`.
- If you want the active method implementation, read the parallel-v2 spec and
  plans under `superpowers/specs/` and `superpowers/plans/`.
- If you want the next experiments, read `paper_experiment_plan.md` and
  `paper_experiment_tracker.md`.
- If you want the frozen splits and release policy, read `reproducibility.md`
  and `critic_pools.md`.
- If you want the engineering history, read `experiment_execution_log.md`.
