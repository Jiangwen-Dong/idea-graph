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
- `paper_experiment_plan.md`
  The detailed claim-driven experiment roadmap.
- `paper_experiment_tracker.md`
  The compact run tracker and current milestone status.
- `experiment_execution_log.md`
  Chronological engineering and experiment log.

## Supporting Plans

- `superpowers/plans/2026-04-12-graph-critic-doc-cleanup.md`
  Documentation cleanup plan for the graph-critic transition.
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
4. `paper_experiment_plan.md`
5. `paper_experiment_tracker.md`
6. `experiment_execution_log.md`

## Current Status

- `M0` and the small `M1` pilot are complete.
- The current reference small-`M1` packet is:
  - `outputs/quality_batches/20260411-000159-refreshed-m1-mini-synthesis-cleanup-v2-native`
- The pre-critic `R009` AIIB launch is paused.
- The next planned milestone is `G0/G1`: graph-critic documentation,
  trajectory export, and critic dataset construction.

## Practical Use

- If you want the paper-facing rules, read `paper_protocol.md` and
  `evaluation.md`.
- If you want the next method implementation, read `eig_graph_critic_plan.md`.
- If you want the next experiments, read `paper_experiment_plan.md` and
  `paper_experiment_tracker.md`.
- If you want the engineering history, read `experiment_execution_log.md`.
