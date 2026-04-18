# Controller Held-Out Experiment Launch Plan

## Scope

This note freezes the next execution stages for the parallel-v2 learned controller track after the corrected dev12 calibration audit.

The goals are:

- choose between raw and calibrated two-head controllers without leaking into paper-eval
- run the frozen paper-eval main table with a stable learned-controller choice
- run controller-specific ablations on a deterministic held-out subset

## Stage 1: Dev Freeze Gate

Use a disjoint `critic_dev` packet that excludes the 12 groups already used in:

- `C:\eig_p2v2_calib_dev12_refit`

Recommended gate:

- split source: `critic_dev`
- size: `48` groups
- composition:
  - `24` AI Idea Bench 2025
  - `24` LiveIdeaBench
- methods:
  - `Ours-Heuristic-Parallel`
  - `Ours-Critic-Graph`
  - `Ours-Critic-Calibrated`
- runtime:
  - `parallel_graph_v2`
  - `max_rounds=5`
  - `stop_when_mature=True`
  - benchmark-native evaluation enabled
  - same model and prompt config across methods

Promotion rule:

- promote `Ours-Critic-Calibrated` to the paper main-table learned controller only if it beats `Ours-Critic-Graph` by at least `0.05` mean benchmark-native score on this held-out gate
- otherwise keep `Ours-Critic-Graph` as the main learned controller and report calibration as an ablation

Current launch note:

- the raw `ours-eig-critic-graph-twohead` launch path now disables calibration explicitly
- the calibrated `ours-eig-critic-calibrated` and commit-only `ours-eig-critic-no-edit`
  launch paths resolve the tracked frozen-dev artifact:
  `data/splits/parallel_v2/frozen_dev_joint_controller_calibration.json`

## Stage 2: Paper-Eval Main Table

Run the frozen split:

- manifest: `data/splits/parallel_v2/paper_eval_v2_registry.jsonl`
- size: `256` groups
- composition:
  - `128` AI Idea Bench 2025
  - `128` LiveIdeaBench
- shards:
  - `4` shards
  - `64` groups per shard
  - each shard balanced as `32` AIIB + `32` LiveIdeaBench

Main-table methods:

- `direct`
- `self-refine`
- `ai-researcher`
- `scipip`
- `virsci`
- `Ours-Heuristic-Parallel`
- selected learned controller from Stage 1

Optional shadow run:

- run the non-promoted learned controller too, but keep it out of the main table if Stage 1 did not justify promotion
- calibrated shadow runs can now use the named baseline directly without an extra CLI calibration override

Record for every method:

- pooled benchmark-native score
- AI Idea Bench 2025 score
- LiveIdeaBench score
- mean executed rounds
- mean LLM call count
- mean total tokens

## Stage 3: Controller Ablation Table

Use a deterministic held-out subset sampled from the frozen paper-eval pool.

Recommended subset:

- size: `128` groups
- composition:
  - `64` AI Idea Bench 2025
  - `64` LiveIdeaBench

Methods:

- `Ours-Heuristic-Parallel`
- `Ours-Critic-Graph`
- `Ours-Critic-Calibrated`
- `Ours-Critic-No-Commit`
- `Ours-Critic-No-Edit`

Interpretation:

- `critic-graph` vs `heuristic`: learned control matters
- `critic-calibrated` vs `critic-graph`: frozen-dev calibration changes the quality-cost frontier
- `critic-no-commit` vs `critic-graph`: learned stopping matters
- `critic-no-edit` vs `heuristic`: learned commit alone matters

## Stage 4: Supplementary Diagnostics

Run on dev-only packets, not paper-eval:

- commit reliability curve
- stop-round histogram
- override-rate summary
- fallback-rate summary
- skip-rate summary
- per-benchmark breakdown

These are mechanism diagnostics, not headline evidence.

## Stage 5: Human Blind Evaluation

After the main automatic table is frozen, sample `40-60` paper-eval cases and compare:

- strongest external baseline
- `Ours-Heuristic-Parallel`
- selected learned controller

Score:

- novelty
- significance
- feasibility
- clarity
- context adherence
- overall preference

## Launch Order

1. Finish method-plan and runtime support for:
   - `Ours-Critic-Calibrated`
   - `Ours-Critic-No-Commit`
   - `Ours-Critic-No-Edit`
2. Build the disjoint `critic_dev` freeze-gate manifest.
3. Run the Stage 1 gate.
4. Freeze the promoted learned controller.
5. Launch the 256-case main table.
6. Launch the 128-case controller ablation subset.
7. Run human blind evaluation.
8. Build final paper tables and supplementary figures.
