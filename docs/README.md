# Documentation Guide

This directory is organized into a small active set plus an archive.

The goal is to keep the paper-facing workflow easy to follow: protocol first,
then experiment plan, then tracker, then execution details.

## Active Docs

- `paper_protocol.md`
  Canonical paper-facing task, baseline, and evaluation protocol.
- `evaluation.md`
  Canonical evaluation policy for benchmark-native scoring, human review, and
  local development metrics.
- `benchmarks.md`
  Short operational note on the two supported benchmark integrations.
- `paper_experiment_plan.md`
  The detailed claim-driven experiment roadmap.
- `paper_experiment_tracker.md`
  The compact run tracker and current milestone status.
- `paper_experiment_map.md`
  The short paper-facing experiment map and next-step summary.
- `experiment_execution_log.md`
  Chronological engineering and experiment log.
- `r009_aiib_launch_plan.md`
  Concrete launch note for the larger `AI_Idea_Bench_2025` `R009` slice.
- `superpowers/specs/2026-04-11-eig-robustness-cleanup-design.md`
  Focused design note for the pre-`R009` EIG robustness cleanup.
- `superpowers/plans/2026-04-11-eig-robustness-cleanup.md`
  Step-by-step implementation plan for the same cleanup.

## Archive

- `archive/`
  Older pilot notes, one-off redesign memos, and superseded planning documents
  that are kept only for historical reference.
- `archive/README.md`
  Short guide to the archived notes.

## Recommended Reading Order

1. `paper_protocol.md`
2. `evaluation.md`
3. `paper_experiment_plan.md`
4. `paper_experiment_tracker.md`
5. `paper_experiment_map.md`
6. `experiment_execution_log.md`

## Current Status

- `M0` is complete.
- The current reference small-`M1` packet is:
  - `outputs/quality_batches/20260411-000159-refreshed-m1-mini-synthesis-cleanup-v2-native`
- The next planned milestone is `M2`, starting with the larger
  `AI_Idea_Bench_2025` slice.

## Practical Use

- If you want the paper-facing rules, read `paper_protocol.md` and
  `evaluation.md`.
- If you want the next experiments, read `paper_experiment_plan.md`,
  `paper_experiment_tracker.md`, `paper_experiment_map.md`, and
  `r009_aiib_launch_plan.md`.
- If you want the engineering history, read `experiment_execution_log.md`.
