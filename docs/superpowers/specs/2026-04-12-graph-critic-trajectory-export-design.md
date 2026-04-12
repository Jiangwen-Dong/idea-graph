# Graph Critic Trajectory Export Design

## Goal

Implement `G1`: export a graph-critic training dataset from saved pre-critic run
artifacts, while also recording profiling indicators needed for later
reviewer-facing overhead analysis.

The exporter must support two uses at once:

1. build state-action supervision for the graph critic
2. quantify the data and compute burden behind that supervision

## Why This Design

Saved `ours-eig` runs already contain most of the needed signal:

- `graph.json`
  - final graph state
  - nodes with `created_at`
  - edges with `created_at`
  - actions with `timestamp`
  - round summaries
  - trace metadata:
    - `agent_traces`
    - `final_synthesis_trace`
    - `utility_controller_overrides`
- `summary.json`
  - clean per-run aggregates
  - local evaluation
  - sometimes benchmark-native evaluation
- batch summaries
  - coarse aggregate token and call counts

The exporter should therefore use `graph.json` as the primary source and
`summary.json` as the clean per-run label source.

## Scope

### In Scope

- scan saved run directories
- export one run-level manifest row per run
- export one transition example per selected EIG action
- reconstruct a timestamp-sliced pre-action graph snapshot
- aggregate token / call / cost / runtime indicators into a dataset profile
- preserve missingness explicitly when a run lacks traces or native scores

### Out Of Scope

- training the critic
- generating new trajectories
- candidate-level ranking export when no candidate traces exist
- exact pricing lookup from provider APIs

## Output Layout

Exporter root:

`outputs/graph_critic_datasets/<dataset_name>/`

Files:

- `run_manifest.jsonl`
  - one row per discovered run
- `trajectory_examples.jsonl`
  - one row per selected EIG action
- `dataset_profile.json`
  - aggregate counts, coverage, and overhead indicators
- `state_snapshots/`
  - one JSON file per exported transition state
- `README.md`
  - schema and caveats

## Discovery Rules

The exporter accepts one or more input roots.

For each input root:

- recursively scan for `summary.json`
- keep only directories that also contain `graph.json`
- do not assume the run came from a batch directory

The exporter should write a manifest row for every discovered run, including
non-EIG baselines, but only export transition examples for runs with graph
actions.

## Run Manifest Schema

Each run row should contain:

- `run_dir`
- `benchmark`
- `instance_name`
- `baseline_name`
- `topic`
- `is_eig_run`
- `has_graph`
- `has_summary`
- `executed_round_count`
- `action_count`
- `node_count`
- `edge_count`
- `action_source_counts`
- `stopped_early`
- `matured_at_round`
- `final_local_overall`
- `final_local_alignment`
- `final_native_average`
- `trace_llm_call_count`
- `trace_prompt_tokens`
- `trace_completion_tokens`
- `trace_total_tokens`
- `estimated_cost`
- `wall_clock_seconds`
- `has_agent_traces`
- `has_final_synthesis_trace`
- `has_override_trace`
- `has_native_eval`
- `has_local_eval`

## Transition Example Schema

Each transition row should contain:

- `run_dir`
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
  - relative path to a JSON snapshot file
- `before_state_node_count`
- `before_state_edge_count`
- `before_state_contradiction_count`
- `before_state_support_edge_count`
- `previous_round_summary`
- `after_round_summary`
- `final_return_local`
- `final_return_native`
- `trace_prompt_tokens_step`
- `trace_completion_tokens_step`
- `trace_total_tokens_step`
- `llm_override_applied`
- `llm_proposed_kind`
- `llm_predicted_gain`
- `override_predicted_gain`
- `commit_target`

Notes:

- `after_round_summary` is the round-level summary for the action's round
- `previous_round_summary` is the last completed round summary before the action
- `commit_target` may be `null` in the first pass when hindsight labels are not
  yet constructed

## State Reconstruction

The exporter should reconstruct a pre-action graph snapshot for each selected
action.

### Available Signals

- nodes have `created_at`
- edges have `created_at`
- actions have `timestamp`
- contradiction edges have final `resolved`
- repair timing can be inferred from later `propose_repair` actions that target
  the contradiction target node

### Reconstruction Rule

For action timestamp `t`:

- include nodes with `created_at <= t`
- include edges with `created_at <= t`
- for contradiction edges:
  - if the edge is unresolved in the final graph, keep it unresolved
  - if the edge is resolved in the final graph, infer `resolved_at` from the
    earliest later `propose_repair` action whose target matches the edge target
  - mark the edge unresolved when `t < resolved_at`
  - mark it resolved when `t >= resolved_at`

This is a timestamp-based reconstruction, not a full replay engine. The spec
should call it an approximate but structurally faithful state snapshot.

## Overhead Indicators

The exporter must record later paper-facing profiling signals.

### Dataset Scale

- discovered run count
- usable run count
- usable EIG run count
- transition example count
- average actions per usable EIG run
- benchmark split counts
- baseline split counts

### Trace Coverage

- fraction with `agent_traces`
- fraction with `final_synthesis_trace`
- fraction with `utility_controller_overrides`
- fraction with local evaluation
- fraction with benchmark-native evaluation

### Token Usage

Aggregate from:

- `metadata.agent_traces[*].raw_response.usage`
- `metadata.final_synthesis_trace.raw_response.usage`

Record:

- prompt tokens
- completion tokens
- total tokens
- mean tokens per run
- mean tokens per transition example

### Cost

If optional pricing arguments are provided to the script, compute:

- estimated prompt cost
- estimated completion cost
- estimated total cost

If pricing is absent:

- keep `estimated_cost = null`
- record token counts anyway

### Runtime

Estimate wall-clock span from the earliest and latest available timestamps among:

- `agent_traces[*].raw_response.created`
- `final_synthesis_trace.raw_response.created`
- `actions[*].timestamp`

If fewer than two timestamps exist:

- keep `wall_clock_seconds = null`

## Script Interface

Add:

`scripts/export_graph_critic_dataset.py`

Recommended arguments:

- `--input-root` repeated, or one-or-more values
- `--output-dir`
- `--dataset-name`
- `--baseline` optional filter
- `--benchmark` optional filter
- `--prompt-price-per-1m-tokens` optional
- `--completion-price-per-1m-tokens` optional
- `--limit-runs` optional

## Code Layout

Add one library module:

- `src/idea_graph/trajectory_dataset.py`

Responsibilities:

- run discovery
- artifact loading
- token / cost / runtime extraction
- state reconstruction
- manifest row construction
- transition row construction
- dataset profile aggregation

The script should stay thin and delegate all logic to the library module.

## Testing Plan

Use TDD.

### Test File

- `tests/test_trajectory_dataset.py`

### Required Tests

1. discovers valid run directories with both `summary.json` and `graph.json`
2. extracts token usage and llm call counts from trace metadata
3. reconstructs a pre-action state snapshot from node / edge / action timestamps
4. infers contradiction resolution timing from later repair actions
5. exports run manifest rows for both EIG and non-EIG runs
6. exports transition rows only for runs with actions
7. aggregates dataset profile counts and overhead indicators correctly
8. computes estimated cost when pricing arguments are supplied

### Verification

Run:

`python -m pytest tests/test_trajectory_dataset.py -q`

then:

`python scripts/export_graph_critic_dataset.py --input-root <pilot-root> --output-dir <tmp-output> --dataset-name smoke`

The smoke run should finish and produce the expected files.
