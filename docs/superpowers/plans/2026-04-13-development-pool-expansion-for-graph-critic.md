# Development Pool Expansion For Graph Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the learned-controller development dataset with more leakage-safe benchmark-instance groups while keeping `development_pool_v1` frozen and preserving a clean train/dev story for later graph-critic work.

**Architecture:** Reuse the current split-registry and dataset-export stack, but make two targeted infrastructure upgrades before collecting more data: generalize episode collection beyond `critic_train` only, and let the G2 builder honor explicit frozen split assignments instead of reshuffling groups. Then materialize a clearly named `development_pool_v2_candidate_pool_v1`, collect fresh runs for both train and dev groups, rebuild the combined G1/G2/G2.5 datasets, and write a new readiness report that explicitly tracks sample count, token usage, and estimated cost.

**Tech Stack:** Python 3.10+, existing `idea_graph` dataset/export scripts, JSONL manifests, OpenAI-compatible backend, `pytest`

---

## Scope

This slice should:

- keep `development_pool_v1` untouched
- add more **unique** groups rather than more reruns of dominant groups
- support explicit `critic_train` and `critic_dev` expansion rows
- rebuild a clean combined dataset for later text-critic and graph-critic work
- record expansion overhead with sample, token, and cost indicators

This slice should **not**:

- reuse `paper_eval_candidate_pool_v1` for training
- reshuffle the old `development_pool_v1` assignments
- claim paper-ready learned-controller results from the expanded development pool

## File Map

### New Files

- Create: `src/idea_graph/critic_pool_expansion.py`
  - validates candidate pools against blocked groups and materializes partition rows
- Create: `scripts/build_critic_expansion_pool.py`
  - CLI for writing expansion partition and registry artifacts
- Create: `tests/test_critic_pool_expansion.py`
- Create: `outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/candidate_instances.json`
- Create: `outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/README.md`
- Create: `outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/partition_manifest.jsonl`
- Create: `outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/split_registry.jsonl`
- Create: `outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/split_registry_stats.json`
- Create: `outputs/graph_critic_datasets/development_pool_v2_combined_g1`
- Create: `outputs/graph_critic_datasets/development_pool_v2_combined_g2`
- Create: `outputs/graph_critic_datasets/development_pool_v2_combined_g25`
- Create: `outputs/graph_critic_datasets/development_pool_v2_combined_readiness/training_readiness_report.md`

### Files To Modify

- Modify: `src/idea_graph/critic_episode_collection.py`
  - generalize selection beyond `critic_train`
- Modify: `scripts/collect_critic_train_episodes.py`
  - add explicit `--partition-role` / `--required-usage` arguments while preserving the current default behavior
- Modify: `tests/test_critic_episode_collection.py`
- Modify: `src/idea_graph/critic_dataset.py`
  - support explicit split overrides for frozen combined datasets
- Modify: `scripts/build_graph_critic_dataset.py`
  - add `--split-overrides` input
- Modify: `tests/test_candidate_slate_dataset.py`
  - if needed for combined-dataset invariants
- Modify: `docs/critic_pools.md`
- Modify: `docs/eig_graph_critic_plan.md`
- Modify: `docs/experiment_execution_log.md`

## Task 1: Generalize Episode Collection To Explicit Partition Roles

**Files:**
- Modify: `src/idea_graph/critic_episode_collection.py`
- Modify: `scripts/collect_critic_train_episodes.py`
- Modify: `tests/test_critic_episode_collection.py`

- [ ] **Step 1.1: Write failing tests for role-aware selection**

Add to `tests/test_critic_episode_collection.py`:

```python
def test_select_pool_rows_can_return_critic_dev_rows() -> None:
    rows = [
        {
            "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-25",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "ai-idea-bench-2025-25",
            "pool_name": "development_pool_v2_candidate_pool_v1",
            "partition_role": "critic_dev",
            "allowed_usages": ["select_checkpoint", "development_analysis"],
        },
        {
            "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-30",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "ai-idea-bench-2025-30",
            "pool_name": "development_pool_v2_candidate_pool_v1",
            "partition_role": "critic_train",
            "allowed_usages": ["train_offline_critic", "train_online_critic", "development_analysis"],
        },
    ]
    selected = select_pool_rows(
        rows,
        pool_name="development_pool_v2_candidate_pool_v1",
        partition_role="critic_dev",
        required_usage="development_analysis",
    )
    assert [row["group_id"] for row in selected] == [
        "AI_Idea_Bench_2025::ai-idea-bench-2025-25"
    ]


def test_select_pool_rows_rejects_missing_requested_group() -> None:
    rows = [
        {
            "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-30",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "ai-idea-bench-2025-30",
            "pool_name": "development_pool_v2_candidate_pool_v1",
            "partition_role": "critic_train",
            "allowed_usages": ["train_offline_critic", "train_online_critic", "development_analysis"],
        }
    ]
    with pytest.raises(ValueError, match="Requested group_ids were not found"):
        select_pool_rows(
            rows,
            pool_name="development_pool_v2_candidate_pool_v1",
            partition_role="critic_dev",
            group_ids=["AI_Idea_Bench_2025::ai-idea-bench-2025-25"],
            required_usage="development_analysis",
        )
```

- [ ] **Step 1.2: Run the collection tests to verify failure**

Run:
`python -m pytest tests/test_critic_episode_collection.py -q`

Expected:
- failure because only `select_critic_train_rows(...)` exists today

- [ ] **Step 1.3: Generalize the selector in the library**

Modify `src/idea_graph/critic_episode_collection.py`:

```python
def select_pool_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    pool_name: str,
    partition_role: str,
    group_ids: Sequence[str] | None = None,
    limit: int | None = None,
    required_usage: str = "",
) -> list[dict[str, Any]]:
    ...
```

Rules:

- `pool_name` must match exactly
- `partition_role` must match exactly
- `required_usage`, when non-empty, must appear in `allowed_usages`
- ordering should remain `(benchmark, instance_name, group_id)`
- keep `select_critic_train_rows(...)` as a thin backwards-compatible wrapper

- [ ] **Step 1.4: Add CLI flags without breaking current commands**

Modify `scripts/collect_critic_train_episodes.py`:

```python
parser.add_argument(
    "--partition-role",
    type=str,
    default="critic_train",
    help="Partition role to select from the registry, such as critic_train or critic_dev.",
)
parser.add_argument(
    "--required-usage",
    type=str,
    default="train_online_critic",
    help="Allowed-usage tag required for selection. Use development_analysis for dev-only collection.",
)
```

Then switch:

```python
selected_rows = select_pool_rows(
    registry_rows,
    pool_name=args.pool_name,
    partition_role=args.partition_role,
    group_ids=args.group_id,
    limit=args.limit,
    required_usage=args.required_usage,
)
```

- [ ] **Step 1.5: Re-run the collection tests**

Run:
`python -m pytest tests/test_critic_episode_collection.py -q`

Expected:
- tests pass
- old `critic_train` CLI flow still works by default

## Task 2: Add Explicit Split Overrides To The G2 Builder

**Files:**
- Modify: `src/idea_graph/critic_dataset.py`
- Modify: `scripts/build_graph_critic_dataset.py`
- Modify: `tests/test_candidate_slate_dataset.py`

- [ ] **Step 2.1: Write failing tests for fixed split assignment**

Add a new test file section, preferably in `tests/test_candidate_slate_dataset.py` or a new `tests/test_critic_dataset.py` if that already exists elsewhere:

```python
def test_build_graph_critic_dataset_honors_split_overrides(tmp_path: Path) -> None:
    g1_dir = make_small_g1_dataset(tmp_path)
    overrides_path = tmp_path / "split_overrides.jsonl"
    overrides_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-13",
                        "benchmark": "AI_Idea_Bench_2025",
                        "instance_name": "ai-idea-bench-2025-13",
                        "split": "train",
                    }
                ),
                json.dumps(
                    {
                        "group_id": "AI_Idea_Bench_2025::ai-idea-bench-2025-9849",
                        "benchmark": "AI_Idea_Bench_2025",
                        "instance_name": "ai-idea-bench-2025-9849",
                        "split": "validation",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = build_graph_critic_dataset(
        g1_dataset_dir=g1_dir,
        output_dir=tmp_path,
        dataset_name="combined_g2",
        split_overrides_path=overrides_path,
    )
    split_rows = load_jsonl(result.dataset_dir / "split_manifest.jsonl")
    by_group = {row["group_id"]: row["split"] for row in split_rows}
    assert by_group["AI_Idea_Bench_2025::ai-idea-bench-2025-13"] == "train"
    assert by_group["AI_Idea_Bench_2025::ai-idea-bench-2025-9849"] == "validation"
```

- [ ] **Step 2.2: Run the dataset-builder tests to verify failure**

Run:
`python -m pytest tests/test_candidate_slate_dataset.py -q`

Expected:
- failure because `split_overrides_path` is not implemented

- [ ] **Step 2.3: Implement split override loading**

Modify `src/idea_graph/critic_dataset.py`:

```python
def load_split_override_rows(path: Path) -> list[dict[str, Any]]:
    return _load_jsonl(path)


def assign_group_splits(
    group_rows: Sequence[Mapping[str, Any]],
    validation_fraction: float = 0.2,
    split_override_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    ...
```

Rules:

- if a group appears in `split_override_rows`, use that split exactly
- only allowed splits: `train`, `validation`
- raise if override rows mention unknown groups
- groups not covered by overrides keep the current deterministic per-benchmark rule

- [ ] **Step 2.4: Expose the CLI flag**

Modify `scripts/build_graph_critic_dataset.py`:

```python
parser.add_argument(
    "--split-overrides",
    type=Path,
    default=None,
    help="Optional JSONL file with explicit group_id -> split assignments for frozen combined datasets.",
)
```

Then pass:

```python
result = build_graph_critic_dataset(
    g1_dataset_dir=args.g1_dataset_dir,
    output_dir=args.output_dir,
    dataset_name=args.dataset_name,
    validation_fraction=args.validation_fraction,
    split_overrides_path=args.split_overrides,
)
```

- [ ] **Step 2.5: Re-run the dataset-builder tests**

Run:
`python -m pytest tests/test_candidate_slate_dataset.py -q`

Expected:
- explicit split overrides work
- old CLI behavior remains unchanged when the flag is absent

## Task 3: Materialize The New Candidate Pool Cleanly

**Files:**
- Create: `src/idea_graph/critic_pool_expansion.py`
- Create: `scripts/build_critic_expansion_pool.py`
- Create: `tests/test_critic_pool_expansion.py`
- Create: `outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/candidate_instances.json`
- Create: `outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/README.md`

- [ ] **Step 3.1: Write failing tests for overlap blocking**

Add `tests/test_critic_pool_expansion.py`:

```python
def test_build_expansion_partition_rows_rejects_overlap_with_blocked_groups() -> None:
    candidates = [
        {
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "ai-idea-bench-2025-13",
            "partition_role": "critic_train",
        }
    ]
    blocked = {"AI_Idea_Bench_2025::ai-idea-bench-2025-13"}
    with pytest.raises(ValueError, match="blocked overlap"):
        build_expansion_partition_rows(
            candidates,
            blocked_group_ids=blocked,
        )


def test_build_expansion_partition_rows_preserves_requested_roles() -> None:
    candidates = [
        {
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "ai-idea-bench-2025-25",
            "partition_role": "critic_train",
        },
        {
            "benchmark": "liveideabench",
            "instance_name": "liveideabench-galaxies-163",
            "partition_role": "critic_dev",
        },
    ]
    rows = build_expansion_partition_rows(candidates, blocked_group_ids=set())
    by_group = {row["group_id"]: row["partition_role"] for row in rows}
    assert by_group["AI_Idea_Bench_2025::ai-idea-bench-2025-25"] == "critic_train"
    assert by_group["liveideabench::liveideabench-galaxies-163"] == "critic_dev"
```

- [ ] **Step 3.2: Run the expansion-pool tests to verify failure**

Run:
`python -m pytest tests/test_critic_pool_expansion.py -q`

Expected:
- import failure because the new helper module does not exist yet

- [ ] **Step 3.3: Implement the pool builder**

Create `src/idea_graph/critic_pool_expansion.py`:

```python
def make_group_id(benchmark: str, instance_name: str) -> str:
    return f"{benchmark}::{instance_name}"


def build_expansion_partition_rows(
    candidates: Sequence[Mapping[str, Any]],
    *,
    blocked_group_ids: set[str],
) -> list[dict[str, object]]:
    ...
```

Rules:

- allowed `partition_role` values: `critic_train`, `critic_dev`
- emit `source_split = "train"` for `critic_train`
- emit `source_split = "validation"` for `critic_dev`
- reject any candidate whose `group_id` is already in:
  - current `development_pool_v1`
  - `paper_eval_candidate_pool_v1`
- reject duplicate candidates

- [ ] **Step 3.4: Add the CLI to write partition and registry artifacts**

Create `scripts/build_critic_expansion_pool.py`:

```python
python scripts/build_critic_expansion_pool.py ^
  --candidate-file outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/candidate_instances.json ^
  --blocked-split-registry outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions/split_registry.jsonl ^
  --blocked-candidate-file outputs/graph_critic_datasets/paper_eval_candidate_pool_v1/candidate_instances.json ^
  --pool-name development_pool_v2_candidate_pool_v1 ^
  --output-dir outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1
```

The script should write:

- `partition_manifest.jsonl`
- `split_registry.jsonl`
- `split_registry_stats.json`

- [ ] **Step 3.5: Write the exact first-pass candidate pool**

Create `outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/candidate_instances.json`:

```json
[
  {"benchmark": "AI_Idea_Bench_2025", "instance_name": "ai-idea-bench-2025-25", "partition_role": "critic_train", "notes": "Untouched AIIB expansion candidate."},
  {"benchmark": "AI_Idea_Bench_2025", "instance_name": "ai-idea-bench-2025-30", "partition_role": "critic_train", "notes": "Untouched AIIB expansion candidate."},
  {"benchmark": "AI_Idea_Bench_2025", "instance_name": "ai-idea-bench-2025-36", "partition_role": "critic_train", "notes": "Untouched AIIB expansion candidate."},
  {"benchmark": "AI_Idea_Bench_2025", "instance_name": "ai-idea-bench-2025-82", "partition_role": "critic_train", "notes": "Untouched AIIB expansion candidate."},
  {"benchmark": "AI_Idea_Bench_2025", "instance_name": "ai-idea-bench-2025-87", "partition_role": "critic_train", "notes": "Untouched AIIB expansion candidate."},
  {"benchmark": "AI_Idea_Bench_2025", "instance_name": "ai-idea-bench-2025-95", "partition_role": "critic_train", "notes": "Untouched AIIB expansion candidate."},
  {"benchmark": "AI_Idea_Bench_2025", "instance_name": "ai-idea-bench-2025-110", "partition_role": "critic_dev", "notes": "Held-out AIIB dev expansion candidate."},
  {"benchmark": "AI_Idea_Bench_2025", "instance_name": "ai-idea-bench-2025-125", "partition_role": "critic_dev", "notes": "Held-out AIIB dev expansion candidate."},
  {"benchmark": "liveideabench", "instance_name": "liveideabench-hurricanes-118", "partition_role": "critic_train", "notes": "Untouched LiveIdeaBench expansion candidate."},
  {"benchmark": "liveideabench", "instance_name": "liveideabench-phycology-140", "partition_role": "critic_train", "notes": "Untouched LiveIdeaBench expansion candidate."},
  {"benchmark": "liveideabench", "instance_name": "liveideabench-galaxies-163", "partition_role": "critic_dev", "notes": "Held-out LiveIdeaBench dev expansion candidate."},
  {"benchmark": "liveideabench", "instance_name": "liveideabench-global positioning system-191", "partition_role": "critic_dev", "notes": "Held-out LiveIdeaBench dev expansion candidate."}
]
```

Create `outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/README.md` describing:

- this pool is development-only
- it is disjoint from `development_pool_v1`
- it is disjoint from `paper_eval_candidate_pool_v1`
- exact intended role counts:
  - `critic_train = 8`
  - `critic_dev = 4`

- [ ] **Step 3.6: Run the new builder and verify artifacts**

Run:

```powershell
python scripts/build_critic_expansion_pool.py `
  --candidate-file outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/candidate_instances.json `
  --blocked-split-registry outputs/graph_critic_datasets/current_benchmarked_ours_eig_full_g35_partitions/split_registry.jsonl `
  --blocked-candidate-file outputs/graph_critic_datasets/paper_eval_candidate_pool_v1/candidate_instances.json `
  --pool-name development_pool_v2_candidate_pool_v1 `
  --output-dir outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1
```

Expected:
- overlap-free pool artifacts are written
- the new registry has `12` rows
- role counts are `critic_train = 8`, `critic_dev = 4`

## Task 4: Collect Fresh Expansion Episodes For Both Train And Dev

**Files:**
- Modify: `docs/experiment_execution_log.md`
- Modify: `docs/critic_pools.md`

- [ ] **Step 4.1: Dry-run the new train collection manifest**

Run:

```powershell
python scripts/collect_critic_train_episodes.py `
  --split-registry outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/split_registry.jsonl `
  --output-dir outputs/graph_critic_online_episodes `
  --collection-name development_pool_v2_critic_train_qwen_v1 `
  --pool-name development_pool_v2_candidate_pool_v1 `
  --partition-role critic_train `
  --required-usage train_online_critic
```

Expected:
- dry-run manifest only
- `selected_group_count = 8`

- [ ] **Step 4.2: Dry-run the new dev collection manifest**

Run:

```powershell
python scripts/collect_critic_train_episodes.py `
  --split-registry outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/split_registry.jsonl `
  --output-dir outputs/graph_critic_online_episodes `
  --collection-name development_pool_v2_critic_dev_qwen_v1 `
  --pool-name development_pool_v2_candidate_pool_v1 `
  --partition-role critic_dev `
  --required-usage development_analysis
```

Expected:
- dry-run manifest only
- `selected_group_count = 4`

- [ ] **Step 4.3: Execute the train collection**

Run:

```powershell
python scripts/collect_critic_train_episodes.py `
  --split-registry outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/split_registry.jsonl `
  --output-dir outputs/graph_critic_online_episodes `
  --collection-name development_pool_v2_critic_train_qwen_v1 `
  --pool-name development_pool_v2_candidate_pool_v1 `
  --partition-role critic_train `
  --required-usage train_online_critic `
  --baseline ours-eig `
  --max-rounds 5 `
  --agent-backend openai-compatible `
  --llm-config configs/openai_compatible.example.json `
  --native-eval `
  --execute `
  --skip-existing
```

- [ ] **Step 4.4: Execute the dev collection**

Run:

```powershell
python scripts/collect_critic_train_episodes.py `
  --split-registry outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/split_registry.jsonl `
  --output-dir outputs/graph_critic_online_episodes `
  --collection-name development_pool_v2_critic_dev_qwen_v1 `
  --pool-name development_pool_v2_candidate_pool_v1 `
  --partition-role critic_dev `
  --required-usage development_analysis `
  --baseline ours-eig `
  --max-rounds 5 `
  --agent-backend openai-compatible `
  --llm-config configs/openai_compatible.example.json `
  --native-eval `
  --execute `
  --skip-existing
```

- [ ] **Step 4.5: Record the raw overhead indicators**

Collect from:

- `outputs/graph_critic_online_episodes/development_pool_v2_critic_train_qwen_v1/collection_summary.json`
- `outputs/graph_critic_online_episodes/development_pool_v2_critic_dev_qwen_v1/collection_summary.json`

Record in `docs/experiment_execution_log.md`:

- selected group count
- completed group count
- total trace tokens
- estimated total cost
- failures, if any

## Task 5: Rebuild The Combined G1/G2/G2.5 Stack And Write A New Readiness Report

**Files:**
- Modify: `docs/eig_graph_critic_plan.md`
- Modify: `docs/experiment_execution_log.md`
- Modify: `docs/critic_pools.md`

- [ ] **Step 5.1: Export the fresh expansion runs into G1**

Run:

```powershell
python scripts/export_graph_critic_dataset.py `
  --input-root outputs/graph_critic_online_episodes/development_pool_v2_critic_train_qwen_v1/runs `
  --input-root outputs/graph_critic_online_episodes/development_pool_v2_critic_dev_qwen_v1/runs `
  --output-dir outputs/graph_critic_datasets `
  --dataset-name development_pool_v2_expansion_only_g1 `
  --baseline ours-eig
```

Expected:
- `outputs/graph_critic_datasets/development_pool_v2_expansion_only_g1`

- [ ] **Step 5.2: Export the full combined G1 dataset**

Run:

```powershell
python scripts/export_graph_critic_dataset.py `
  --input-root outputs `
  --input-root outputs/graph_critic_online_episodes/development_pool_v2_critic_train_qwen_v1/runs `
  --input-root outputs/graph_critic_online_episodes/development_pool_v2_critic_dev_qwen_v1/runs `
  --output-dir outputs/graph_critic_datasets `
  --dataset-name development_pool_v2_combined_g1 `
  --baseline ours-eig
```

Expected:
- combined G1 contains both the old benchmarked runs and the new expansion runs

- [ ] **Step 5.3: Write the exact split overrides for the combined dataset**

Create:
`outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/group_split_overrides.jsonl`

Contents:

```json
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-13","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-13","split":"train"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-15","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-15","split":"train"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-18","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-18","split":"train"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-21","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-21","split":"train"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-3883","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-3883","split":"train"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-7909","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-7909","split":"train"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-9849","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-9849","split":"validation"}
{"group_id":"liveideabench::liveideabench-earthquakes-70","benchmark":"liveideabench","instance_name":"liveideabench-earthquakes-70","split":"train"}
{"group_id":"liveideabench::liveideabench-meteorology-0","benchmark":"liveideabench","instance_name":"liveideabench-meteorology-0","split":"train"}
{"group_id":"liveideabench::liveideabench-periodic table-23","benchmark":"liveideabench","instance_name":"liveideabench-periodic table-23","split":"train"}
{"group_id":"liveideabench::liveideabench-weather forecasting-47","benchmark":"liveideabench","instance_name":"liveideabench-weather forecasting-47","split":"validation"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-25","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-25","split":"train"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-30","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-30","split":"train"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-36","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-36","split":"train"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-82","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-82","split":"train"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-87","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-87","split":"train"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-95","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-95","split":"train"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-110","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-110","split":"validation"}
{"group_id":"AI_Idea_Bench_2025::ai-idea-bench-2025-125","benchmark":"AI_Idea_Bench_2025","instance_name":"ai-idea-bench-2025-125","split":"validation"}
{"group_id":"liveideabench::liveideabench-hurricanes-118","benchmark":"liveideabench","instance_name":"liveideabench-hurricanes-118","split":"train"}
{"group_id":"liveideabench::liveideabench-phycology-140","benchmark":"liveideabench","instance_name":"liveideabench-phycology-140","split":"train"}
{"group_id":"liveideabench::liveideabench-galaxies-163","benchmark":"liveideabench","instance_name":"liveideabench-galaxies-163","split":"validation"}
{"group_id":"liveideabench::liveideabench-global positioning system-191","benchmark":"liveideabench","instance_name":"liveideabench-global positioning system-191","split":"validation"}
```

This preserves:

- old frozen development assignments
- new expansion groups with `critic_train = 8`
- new expansion groups with `critic_dev = 4`

Total expected `critic_dev` groups after expansion: `6`

- [ ] **Step 5.4: Build the combined G2 dataset**

Run:

```powershell
python scripts/build_graph_critic_dataset.py `
  --g1-dataset-dir outputs/graph_critic_datasets/development_pool_v2_combined_g1 `
  --output-dir outputs/graph_critic_datasets `
  --dataset-name development_pool_v2_combined_g2 `
  --split-overrides outputs/graph_critic_datasets/development_pool_v2_candidate_pool_v1/group_split_overrides.jsonl
```

- [ ] **Step 5.5: Build the combined G2.5 candidate-slate dataset**

Run:

```powershell
python scripts/build_graph_critic_candidate_dataset.py `
  --g1-dataset-dir outputs/graph_critic_datasets/development_pool_v2_combined_g1 `
  --g2-dataset-dir outputs/graph_critic_datasets/development_pool_v2_combined_g2 `
  --output-dir outputs/graph_critic_datasets `
  --dataset-name development_pool_v2_combined_g25
```

- [ ] **Step 5.6: Write the readiness report**

Create:
`outputs/graph_critic_datasets/development_pool_v2_combined_readiness/training_readiness_report.md`

The report must include:

- run count
- unique group count
- `critic_train` group count
- `critic_dev` group count
- transition count
- candidate row count
- per-benchmark group counts
- total trace tokens
- mean tokens per run
- estimated total cost
- duplicate burden
- a short judgment:
  - whether the dataset is still only enough for text critic
  - or now enough for the first offline graph critic v1

- [ ] **Step 5.7: Run the verification packet**

Run:

```powershell
python -m pytest tests/test_critic_episode_collection.py tests/test_critic_pool_expansion.py tests/test_candidate_slate_dataset.py -q
```

Expected:
- role-aware collection passes
- overlap-safe pool building passes
- fixed split overrides pass

## Final Verification Checklist

- [ ] `development_pool_v1` remains frozen and untouched
- [ ] expansion pool is disjoint from both `development_pool_v1` and `paper_eval_candidate_pool_v1`
- [ ] collection supports both `critic_train` and `critic_dev`
- [ ] combined G2 honors explicit split overrides
- [ ] expansion overhead is recorded with group count, token count, and estimated cost
- [ ] combined readiness report clearly states whether the dataset is sufficient for graph-critic offline v1
