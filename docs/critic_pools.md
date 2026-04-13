# Critic Pools

This note is the canonical human-readable description of graph-critic data
pool usage.

## Current Frozen Development Pool

Pool name:

- `development_pool_v1`

Source artifacts:

- partition manifest:
  `outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions/partition_manifest.jsonl`
- split registry:
  `outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions/split_registry.jsonl`

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

## First Candidate Final-Evaluation Pool

Pool name:

- `paper_eval_candidate_pool_v1`

Source artifact:

- `outputs/graph_critic_datasets/paper_eval_candidate_pool_v1/candidate_instances.json`

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

Practical rule:

- new episode collection must target only rows selected from
  `development_pool_v1` with `partition_role=critic_train`
- collected runs should live under a dedicated
  `outputs/graph_critic_online_episodes/<collection>/runs` root
- replay buffers derived from those runs should remain under the same
  collection root so provenance stays explicit
- `critic_dev` and future `paper_eval` rows must never be generated through
  this helper
