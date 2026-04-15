# Paper Eval Candidate Pool V2 Freeze Builder

## Current Planning Context

This builder is downstream of the broad development-only online gate, not a
parallel replacement for it.

Current untouched paper-eval proposal size is still too small:

- `AI_Idea_Bench_2025`: `6`
- `LiveIdeaBench`: `4`

So `paper_eval_candidate_pool_v2` is the first real frozen paper-eval pool.

Target:

- minimum:
  - `64` `AI_Idea_Bench_2025`
  - `48` `LiveIdeaBench`
- preferred:
  - `96` `AI_Idea_Bench_2025`
  - `64` `LiveIdeaBench`

Execution rule:

- code for the builder can land now
- the final frozen pool should be materialized only after the broad-gate
  freeze memo says `go`

## Goals

- Freeze a new candidate pool that is entirely disjoint from all current development pools and from the previously proposed paper-eval slate.
- Support AI_Idea_Bench_2025 and LiveIdeaBench as the two untouched benchmarks, meeting explicit `--target-aiib`/`--target-live` counts.
- Emit the same artifacts called out in the execution plan: `candidate_instances.json`, `README.md`, and `pool_stats.json`, all under `outputs/graph_critic_datasets/02_active_graph_critic/paper_eval_candidate_pool_v2` (or another CLI-specified root).
- Surface enough metadata so downstream scripts, memos, and humans can verify the freeze decision (counts, blocked groups, README story, CLI output).

## Inputs

- **AIIB metadata:** default `data/benchmarks/ai_idea_bench_2025/target_paper_data.json`. Each row has an `index` used to build `instance_name = ai-idea-bench-2025-{index}` plus a `summary` (we will describe the topic in `notes` if available).
- **LiveIdeaBench CSV:** default `data/benchmarks/liveideabench/liveideabench_hf.csv`; each row exposes a `keywords` field and row index to form `instance_name = liveideabench-{keyword}-{row_index}`.
- **Blocked artifacts:** defaults should block the full current development-only history plus the earlier proposed paper-eval slate. The default blocked set should therefore include:
  - `outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g35_partitions/partition_manifest.jsonl`
  - `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_candidate_pool_v1/candidate_instances.json`
  - `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/candidate_instances.json`
  - `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g2_partitions/partition_manifest.jsonl`
  - `outputs/graph_critic_datasets/02_active_graph_critic/paper_eval_candidate_pool_v1/candidate_instances.json`
  The CLI will allow repeating `--blocked-candidate-file` and `--blocked-split-registry` flags so additional manifests can be specified later.

## Selection algorithm

1. **Gather blocked group IDs.** Reuse `make_group_id` from `idea_graph.critic_pool_expansion`. Parse each supplied split-registry/manifest line and each candidate row (benchmark + instance name or explicit `group_id`) to build a set of blocked IDs. The default blocked set must cover all current development pools plus `paper_eval_candidate_pool_v1`.
2. **Iterate available rows deterministically.**
   * AIIB rows are iterated in the JSON array order (which roughly follows the `index` field). For each row, compute the target instance name, skip it if its group ID is blocked, and stop once `target_aiib` candidates are collected.
   * LiveIdeaBench rows are iterated in CSV order; form the instance name from the raw keyword plus row index, skip blocked group IDs, and stop when `target_live` rows are collected.
3. **Fail fast** if either benchmark cannot supply enough new, unblocked rows to meet its target (raise `ValueError` with the missing count).
4. **Enrich notes.** Each row records `status = "frozen"`, `intended_role = "paper_eval"`, and a short `notes` paragraph such as:
   * For AIIB: `f"Frozen AIIB candidate; topic is {summary.get('revised_topic') or summary.get('topic', '<unknown>')}."`
   * For LiveIdeaBench: `f"Frozen LiveIdeaBench candidate; keyword is {keyword}."`

Rows keep the same schema as v1, and the combined list is sorted by `(benchmark, instance_name)` before being written.

## CLI contract

- Required flags: `--output-root` (the directory where the three artifacts will be created), `--target-aiib`, and `--target-live` (both positive integers).
- Optional overrides with defaults:
  * `--aiib-metadata` (default `data/benchmarks/ai_idea_bench_2025/target_paper_data.json`)
  * `--live-csv` (default `data/benchmarks/liveideabench/liveideabench_hf.csv`)
  * `--blocked-candidate-file` (repeatable; default bucket includes v2/v3 development candidate pools plus `paper_eval_candidate_pool_v1`)
  * `--blocked-split-registry` (repeatable; default bucket includes `development_pool_v1` and `development_pool_v3_combined_g2_partitions`)
- The script will:
  * Build candidate rows as described and write them to `candidate_instances.json`.
  * Write a README that summarizes the pool status, references the freeze decision (frozen vs proposed), enumerates the benchmarks, and lists the blocked sources that were used for the disjointness guarantee.
  * Write `pool_stats.json` containing per-benchmark counts (`target`, `selected`, `blocked`) and overall totals plus a `generated_at` timestamp.
  * Print a short summary (counts, blocked IDs, output root) for human visibility, similar to other `scripts/build_*` utilities.

## Testing

- `tests/test_paper_eval_freeze_pool.py` will:
  * Use temporary AIIB metadata (a small JSON list) and LiveIdeaBench CSV (3–4 rows) written into `tmp_path` to keep tests hermetic.
  * Create fake blocked candidate JSON files and split registry files to ensure the builder skips those entries and still meets targets.
  * Include a regression that passes multiple blocked sources at once and verifies the selected rows are disjoint from every blocked group id.
  * Exercise the helper functions (blocked ID aggregation + row selection) directly and assert the returned rows have the right counts, names, and notes, and no overlap with blocked IDs.
  * Run the CLI (`python scripts/build_paper_eval_freeze_pool.py ...`) to confirm the three artifacts exist, contain the expected number of entries, and that the printed summary mentions the counts.
  * Assert a `ValueError`/non-zero exit when not enough new rows remain to satisfy `--target-aiib` or `--target-live`.

## Outputs expectations

- `candidate_instances.json`: deterministic sort order, one entry per selected instance, fully populated schema.
- `README.md`: describes the pool (`paper_eval_candidate_pool_v2`), states “Status: frozen,” and explains the zero-overlap policy along with the targeted counts.
- `pool_stats.json`: example structure:

```json
{
  "generated_at": "2026-04-14T12:00:00Z",
  "benchmarks": {
    "AI_Idea_Bench_2025": {"target": 64, "selected": 64, "blocked": 28},
    "liveideabench": {"target": 48, "selected": 48, "blocked": 14}
  },
  "total": {"target": 112, "selected": 112, "blocked": 42}
}
```

## Risk mitigations

- Reading large metadata assets might fail on Windows encoding; always use `encoding="utf-8"` when opening files.
- Blocked files may list duplicate group IDs; de-duplicate cautiously so the stats still reflect the true count.
- The README and stats should explicitly mention the blocked sources used for easier auditing.
- Because this pool is the first real frozen paper-eval set, the emitted README should also state that materialization happened only after the broad-gate freeze decision.
