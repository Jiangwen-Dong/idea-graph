# Critic Pools

This note is the canonical human-readable description of graph-critic data
pool usage.

Dataset layout guide:

- `docs/graph_critic_dataset_layout.md`

## Current Frozen Development Pool

Pool name:

- `development_pool_v1`

Source artifacts:

- partition manifest:
  `outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g35_partitions/partition_manifest.jsonl`
- split registry:
  `outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g35_partitions/split_registry.jsonl`

Allowed use:

- `critic_train`
  - offline warm start
  - online adaptation
  - development-only controller analysis
- `critic_dev`
  - checkpoint selection
  - threshold calibration
  - development-only analysis

Not allowed:

- final paper benchmark reporting for the learned controller

## Current Development Expansion Pool

Pool name:

- `development_pool_v2_candidate_pool_v1`

Source artifacts:

- candidate list:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_candidate_pool_v1/candidate_instances.json`
- partition manifest:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_candidate_pool_v1/partition_manifest.jsonl`
- split registry:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_candidate_pool_v1/split_registry.jsonl`

Status:

- candidate pool materialized
- partition and split registry materialized
- API-backed train/dev episode collections completed
- expanded G1/G2/G2.5 artifacts materialized
- explicit split overrides written so `development_pool_v1` assignments stay
  frozen while new v2 `critic_train` / `critic_dev` rows map to
  train / validation

Role counts:

- `critic_train`: `8`
- `critic_dev`: `4`

Allowed use:

- `critic_train`
  - development-only offline/online critic training expansion
- `critic_dev`
  - development-only checkpoint and threshold validation expansion

Not allowed:

- final paper benchmark reporting for the learned controller

## First Candidate Final-Evaluation Pool

Pool name:

- `paper_eval_candidate_pool_v1`

Source artifact:

- `outputs/graph_critic_datasets/02_active_graph_critic/paper_eval_candidate_pool_v1/candidate_instances.json`

Status:

- proposed only
- not yet generated
- not yet evaluated

Allowed use:

- planning the future untouched final benchmark packet

Not allowed:

- critic training
- critic development tuning
- online adaptation

## Practical Rule

If a benchmark instance is listed inside `development_pool_v1`, it must not be
used as final frozen paper evidence for the learned-controller line.

If a benchmark instance is listed inside `development_pool_v2_candidate_pool_v1`,
it must also remain development-only.

If a benchmark instance is listed inside `paper_eval_candidate_pool_v1`, it
must remain untouched until the critic is frozen.

## Current Collection Root

The active collection helper for real train-group episode generation is:

- `scripts/collect_critic_train_episodes.py`

Current verified artifact roots:

- dry-run manifest smoke:
  `outputs/graph_critic_online_episodes/development_pool_v1_critic_train_manifest_smoke`
- deterministic execute smoke:
  `outputs/graph_critic_online_episodes/development_pool_v1_critic_train_execute_smoke_det`
- first real openai-compatible train-group collection:
  `outputs/graph_critic_online_episodes/development_pool_v1_critic_train_qwen_v1`
- development-pool v2 train collection:
  `outputs/graph_critic_online_episodes/development_pool_v2_critic_train_qwen_v1`
- development-pool v2 dev collection:
  `outputs/graph_critic_online_episodes/development_pool_v2_critic_dev_qwen_v1`
- expanded combined G1:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g1`
- expanded combined G2:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g2`
- expanded combined G2.5:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g25`
- expanded readiness report:
  `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_readiness/training_readiness_report.md`

Practical rule:

- new training episode collection must target only rows selected from a
  development pool with `partition_role=critic_train`
- development-only dev collection may target `partition_role=critic_dev` with
  `required_usage=development_analysis`
- collected runs should live under a dedicated
  `outputs/graph_critic_online_episodes/<collection>/runs` root
- replay buffers derived from those runs should remain under the same
  collection root so provenance stays explicit
- future `paper_eval` rows must never be generated through this helper

Verified expansion collection roots:

- train collection:
  `outputs/graph_critic_online_episodes/development_pool_v2_critic_train_qwen_v1`
  - selected groups: `8`
  - completed groups: `8`
  - traced tokens: `1,623,590`
- dev collection:
  `outputs/graph_critic_online_episodes/development_pool_v2_critic_dev_qwen_v1`
  - selected groups: `4`
  - completed groups: `4`
  - traced tokens: `777,552`
