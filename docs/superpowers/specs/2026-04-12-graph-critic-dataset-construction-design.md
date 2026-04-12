# Graph Critic Dataset Construction Design

## Goal

Implement `G2`: convert exported `G1` trajectories into a critic-ready dataset
with:

- leakage-safe train/validation splits
- explicit separation between weak local labels and benchmark-native labels
- a stable label schema that can support both:
  - `G3` text-only critics
  - `G4` graph-structured critics

The key requirement is to avoid mixing different label qualities too early.
`G2` should package them cleanly, not collapse them into one hidden score.

## Why This Stage Exists

`G1` already exports:

- run-level manifests
- transition examples
- state snapshots
- token/runtime/cost profiling

But those exports are still raw artifacts. They are not yet a training dataset
because:

- train/validation leakage is not controlled
- labels from different evaluators are not separated
- repeated reruns of the same benchmark instance are not grouped
- the output schema is not yet normalized for critic training

`G2` is therefore the bridge between raw trajectory export and critic-model
training.

## Design Principles

### 1. Split By Benchmark Instance, Not By Transition

All trajectories from the same benchmark case must stay in the same split.

Reason:

- transitions from the same instance are highly correlated
- multiple reruns of the same instance often share topic, references, and final
  target structure
- splitting at the transition level would produce severe leakage

The split unit is:

`group_id = benchmark + "::" + instance_name`

### 2. Separate Label Namespaces

The dataset should store two independent supervision namespaces:

- `weak_local`
- `native`

Reason:

- local evaluator scores are useful for warm-start supervision and debugging,
  but they are not benchmark-faithful final evidence
- native benchmark scores are stronger, but coverage can be incomplete and
  semantics differ across benchmarks
- collapsing them into one scalar too early would make the training story hard
  to justify

### 3. Preserve Missingness Explicitly

If a run lacks native evaluation or some native submetrics, the dataset should
store:

- the missing field as `null`
- a corresponding availability flag

Reason:

- later critic training may use weak labels on some rows and native labels on
  others
- paper-facing analysis needs honest coverage accounting

### 4. Keep Raw Labels And Simple Normalized Targets

`G2` should store:

- raw labels
- simple normalized scalar targets

It should not yet decide the final multi-task loss or score fusion strategy.

Reason:

- `G2` is dataset construction, not model design
- the first critic baselines should be able to choose their target head without
  rewriting the dataset

## In Scope

- read `G1` exports from a dataset directory
- build benchmark-instance-level split assignments
- assign all runs and transitions to train/validation without leakage
- package weak and native labels separately
- export critic-ready JSONL rows plus split/coverage statistics
- record duplicate-run counts and group sizes for later reweighting

## Out Of Scope

- critic model training
- commit-vs-continue hindsight relabeling
- pairwise negative sampling beyond basic metadata support
- learned target fusion
- test split creation for the final paper

The first `G2` implementation should stay simple and make `G3` easy.

## Input Contract

`G2` consumes one `G1` dataset directory containing:

- `run_manifest.jsonl`
- `trajectory_examples.jsonl`
- `dataset_profile.json`
- `state_snapshots/`

The builder should assume `G1` has already completed and should not rescan raw
run directories.

## Split Strategy

### Group Definition

Every row inherits:

- `benchmark`
- `instance_name`

Construct:

`group_id = benchmark + "::" + instance_name`

All rows with the same `group_id` must map to the same split.

### Default Split Policy

For each benchmark independently:

1. collect unique `group_id`s
2. sort them deterministically
3. assign approximately `80%` to train and `20%` to validation
4. guarantee at least one validation group if the benchmark has at least `3`
   groups

This keeps the split deterministic and benchmark-balanced.

### Duplicate Runs

If the same `group_id` has multiple reruns:

- keep all runs
- keep all transitions
- record:
  - `group_run_count`
  - `group_run_index`

Reason:

- reruns provide useful variation in action quality
- dropping them too early wastes data
- later training can reweight by group if needed

## Label Package

### Weak Local Labels

Each transition row should carry:

- `weak_local.available`
- `weak_local.overall_10`
- `weak_local.overall_01`
- `weak_local.benchmark_alignment_10`
- `weak_local.benchmark_alignment_01`
- `weak_local.category_scores`

Notes:

- `overall_score` is the main weak scalar target
- `benchmark_alignment` is useful as a secondary diagnostic or auxiliary target
- `graph_process` should be preserved in metadata, but it should not be the
  primary critic target because it is partly self-referential to the graph
  itself

### Native Labels

Each transition row should carry:

- `native.available`
- `native.benchmark`
- `native.average_10`
- `native.average_01`
- `native.metrics`

Normalization rule:

- map all native benchmark averages to a common `0-10` and `0-1` representation

For:

- `AI_Idea_Bench_2025`
  use `available_average_normalized_10`
- `LiveIdeaBench`
  use the native `average` metric when available

### Label Availability

Each transition row should also expose:

- `label_availability.has_weak_local`
- `label_availability.has_native`
- `label_availability.has_native_average`

This makes later training and paper reporting straightforward.

## First Training Targets

`G2` should expose, but not yet fuse, these scalar targets:

- `targets.weak_value_01 = weak_local.overall_01`
- `targets.native_value_01 = native.average_01`

No combined target should be produced in the first implementation.

Reason:

- the first baseline critic can be trained either on weak labels or native
  labels
- later work can add:
  - multi-task heads
  - curriculum from weak to native labels
  - calibration on native-only subsets

## Output Layout

Recommended output directory:

`outputs/graph_critic_datasets/<dataset_name>_g2/`

Files:

- `critic_dataset.jsonl`
  - one row per transition example with split and label package
- `split_manifest.jsonl`
  - one row per `group_id`
- `label_schema.json`
  - machine-readable description of the exported labels
- `dataset_stats.json`
  - aggregate counts, coverage, split sizes, duplicate-run counts, and label
    availability
- `README.md`
  - brief description of the dataset and caveats

## Critic Dataset Row Schema

Each row should contain:

- `run_dir`
- `group_id`
- `split`
- `benchmark`
- `instance_name`
- `baseline_name`
- `topic`
- `step_index`
- `round_name`
- `role`
- `selected_action_kind`
- `selected_action_targets`
- `selected_action_source`
- `before_state_snapshot`
- `before_state_node_count`
- `before_state_edge_count`
- `before_state_contradiction_count`
- `before_state_support_edge_count`
- `previous_round_summary`
- `after_round_summary`
- `group_run_count`
- `group_run_index`
- `weak_local`
- `native`
- `label_availability`
- `targets`

The row should preserve the original transition information and only add split
and label packaging.

## Split Manifest Schema

Each split row should contain:

- `group_id`
- `benchmark`
- `instance_name`
- `split`
- `run_count`
- `transition_count`
- `has_any_weak_local`
- `has_any_native`
- `mean_weak_value_01`
- `mean_native_value_01`

This file is useful for leakage checks and split auditing.

## Dataset Statistics

`dataset_stats.json` should report:

### Scale

- total run count
- total transition count
- unique group count
- train group count
- validation group count
- train transition count
- validation transition count

### Benchmark Breakdown

- per-benchmark group counts
- per-benchmark transition counts

### Label Coverage

- fraction with weak labels
- fraction with native labels
- fraction with native average

### Duplicate Burden

- mean runs per group
- max runs per group
- histogram of group sizes if easy to produce

This is important for later reviewer-facing discussion of dataset redundancy and
training overhead.

## Why This Design Is NeurIPS-Reasonable

This design gives a clean methodological story:

- the graph critic is trained on relational decision traces
- leakage is controlled at the benchmark-instance level
- weak supervision and benchmark-faithful supervision are explicitly separated
- overhead and label coverage are measured rather than hidden

It is also intentionally conservative:

- no hand-designed fusion target
- no premature commit relabeling
- no benchmark-specific training hacks

That makes the first learned critic easier to justify.

## Implementation Shape

Add:

- `src/idea_graph/critic_dataset.py`
- `scripts/build_graph_critic_dataset.py`
- `tests/test_critic_dataset.py`

Responsibilities:

- load `G1` dataset files
- construct deterministic group-level splits
- attach label namespaces and normalized targets
- write critic-ready outputs and statistics

## Testing Requirements

The first implementation should test:

1. group-level split assignment is deterministic
2. all rows from the same `group_id` share the same split
3. duplicate runs are preserved and counted correctly
4. weak and native label namespaces are exported separately
5. native-label missingness is preserved correctly
6. normalized scalar targets are computed correctly
7. dataset statistics match the constructed rows

## Immediate Next Step

After this spec is approved:

1. write the `G2` implementation plan
2. implement `critic_dataset.py` with tests first
3. run a small builder smoke test on the saved `R009D` `G1` dataset
