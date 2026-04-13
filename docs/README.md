# Documentation Guide

This directory is organized into a small active set plus an archive.

The goal is to keep the paper-facing workflow easy to follow: protocol first,
then the current method plan, then evaluation and experiment tracking.

## Active Docs

- `paper_protocol.md`
  Canonical paper-facing task, baseline, and evaluation protocol.
- `eig_graph_critic_plan.md`
  Canonical forward method plan for EIG with a learned graph critic.
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

- `M0` and the small `M1` pilot are complete.
- The current reference small-`M1` packet is:
  - `outputs/quality_batches/20260411-000159-refreshed-m1-mini-synthesis-cleanup-v2-native`
- The current graph-critic stack now includes:
  - a frozen `development_pool_v1`
  - a split registry over the active 11-group development pool
  - a first untouched `paper_eval_candidate_pool_v1`
  - a real `critic_train` episode collection packet:
    `outputs/graph_critic_online_episodes/development_pool_v1_critic_train_qwen_v1`
  - an adapted text critic:
    `outputs/graph_critic_models/current_benchmarked_ours_eig_full_g46_text_online_real_train_v1`
- The first controller-in-the-loop 4-case AIIB gate is complete:
  - `outputs/m2_aiib_g48_controller_gate_v1`
  - paired summary:
    `outputs/m2_aiib_g48_controller_gate_v1/paired_summary.md`
- Current controller conclusion:
  - the learned text critic improves held-out action ranking on `critic_dev`
  - but the first end-to-end 4-case gate is mixed and slightly negative, so
    `ours-eig` remains the main benchmarked method
- The next planned milestone is a trace-persistent rerun of the same frozen
  4-case AIIB controller gate with stronger maturity-sensitive safety, not a
  broader controller packet yet.

## Practical Use

- If you want the paper-facing rules, read `paper_protocol.md` and
  `evaluation.md`.
- If you want the next method implementation, read `eig_graph_critic_plan.md`.
- If you want the next experiments, read `paper_experiment_plan.md` and
  `paper_experiment_tracker.md`.
- If you want the engineering history, read `experiment_execution_log.md`.
