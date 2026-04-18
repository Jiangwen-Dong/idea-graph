# Parallel V2 Splits

This directory stores the lightweight, tracked split manifests for the
parallel EIG v2 critic and paper evaluation protocol. Large generated runs,
model checkpoints, exported replay datasets, and harvested episode outputs
remain under ignored `outputs/` or external artifact roots.

## Critic Train/Dev

- `critic_train_dev_registry.jsonl` contains 400 development groups for
  graph-critic supervision.
- `critic_train_dev_stats.json` records the fixed distribution: 300
  `critic_train` groups and 100 `critic_dev` groups, balanced as 150/50 per
  benchmark for AI Idea Bench 2025 and LiveIdeaBench.
- `critic_train_dev_split_overrides.jsonl` maps `critic_train -> train` and
  `critic_dev -> validation` for two-head critic dataset construction.

The critic train/dev split is used for heuristic-teacher replay collection,
two-head critic training, checkpoint selection, and commit-head calibration.

- `frozen_dev_joint_controller_calibration.json` is the tracked frozen-dev
  calibration artifact used by the self-contained
  `ours-eig-critic-calibrated` and `ours-eig-critic-no-edit` launch paths.
  The raw `ours-eig-critic-graph-twohead` controller explicitly disables this
  calibration so the raw/calibrated comparison does not depend on CLI flags.

## Paper Eval

- `paper_eval_v2_registry.jsonl` contains the frozen 256-sample paper-eval
  pool: 128 AI Idea Bench 2025 groups and 128 LiveIdeaBench groups.
- `paper_eval_v2_stats.json` records the paper-eval distribution.
- `paper_eval_v2_disjointness_audit.json` records zero overlap with the
  blocked critic train/dev groups used to build this pool.

Paper-eval groups must not be used for graph-critic training, checkpoint
selection, heuristic-teacher improvement, threshold calibration, or any other
development feedback loop.
