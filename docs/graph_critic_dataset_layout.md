# Graph Critic Dataset Layout

This note describes the local organization of
`outputs/graph_critic_datasets/`.

The `outputs/` tree is gitignored, so this file is the tracked reference for
where the current local datasets live.

## Current Layout

### `outputs/graph_critic_datasets/01_active_text_critic`

Active artifacts for the current text-critic line:

- `current_benchmarked_ours_eig_full_g1_commit_enriched`
- `current_benchmarked_ours_eig_full_g2_commit_enriched`
- `current_benchmarked_ours_eig_full_g25_commit_enriched`
- `current_benchmarked_ours_eig_full_g35_partitions`
- `development_pool_v1_critic_train_qwen_v1_g1`

Use this folder when the task is:

- text-critic warm start
- online text-critic adaptation
- development-pool v1 replay/export checks

### `outputs/graph_critic_datasets/02_active_graph_critic`

Active artifacts for the current graph-critic and dataset-expansion line:

- `development_pool_v3_candidate_pool_v1`
- `development_pool_v3_combined_g1`
- `development_pool_v3_combined_g2`
- `development_pool_v3_combined_g2_partitions`
- `development_pool_v3_combined_g25`
- `development_pool_v3_combined_readiness`
- `development_pool_v2_candidate_pool_v1`
- `development_pool_v2_combined_g1`
- `development_pool_v2_combined_g2`
- `development_pool_v2_combined_g2_partitions`
- `development_pool_v2_combined_g25`
- `development_pool_v2_combined_readiness`
- `paper_eval_candidate_pool_v1`

Use this folder when the task is:

- graph-critic offline training
- development-pool v3 analysis
- leakage-safe split inspection
- readiness reporting
- future paper-eval planning

The v2 roots are retained for provenance and older comparisons. New graph
critic work should start from v3 unless a task explicitly asks to reproduce an
older v2 result.

### `outputs/graph_critic_datasets/03_archive`

Historical or intermediate artifacts kept for provenance:

- `current_benchmarked_ours_eig_full_g1`
- `current_benchmarked_ours_eig_full_g2`
- `current_benchmarked_ours_eig_full_g25`
- `development_pool_v2_expansion_only_g1`
- `smoke_r009_safe_grounding`
- `smoke_r009_safe_grounding_g2`
- `smoke_r009_safe_grounding_g2_cli_validation_check`

These are useful for:

- reproducing earlier stages
- checking old smoke runs
- debugging export history

They should not be used as the default training root for new critic work.

## Practical Rule

If you are starting new critic training now:

1. For text critic, start in
   `outputs/graph_critic_datasets/01_active_text_critic`
2. For graph critic, start in
   `outputs/graph_critic_datasets/02_active_graph_critic`
3. Only inspect `03_archive` when you need historical provenance

## Current Recommended Roots

- Text-critic candidate dataset:
  `outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g25_commit_enriched`
- Text-critic partition manifest:
  `outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g35_partitions/partition_manifest.jsonl`
- Graph-critic candidate dataset:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g25`
- Graph-critic partition manifest:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g2_partitions/partition_manifest.jsonl`
- Graph-critic snapshot root:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g1`
- Graph-critic readiness report:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_readiness/training_readiness_report.md`
- Latest broad online graph-controller gate:
  `outputs/m2_graph_critic_online_scaleup_v2_merged118`
