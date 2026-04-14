# Graph Critic Stage A Scale-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the final development-only Stage A graph-critic training pool on top of a shadow-commit-observable runtime, then export the enlarged `G1` / `G2` / `G2.5` datasets and a new readiness report before any larger controller packet.

**Architecture:** Reuse the existing graph-critic data stack rather than inventing a new one. First port the already validated shadow-commit observability patch from the isolated `g62` worktree into `main`, so the next traced runs preserve richer stop diagnostics. Then materialize a larger, leak-safe `development_pool_v3_candidate_pool_v1`, collect train/dev runs, rebuild the combined `G1` / `G2` / `G2.5` datasets with frozen split overrides, and record the resulting scale, token overhead, and commit coverage in a new readiness report.

**Tech Stack:** Python 3.10+, existing `idea_graph` runtime/export scripts, JSON/JSONL manifests, OpenAI-compatible backend, `pytest`, git worktrees

---

## Scope

This slice should:

- keep all existing development pools and paper-eval pools untouched
- enlarge development-only group coverage from `23` groups to about `59`
- preserve strict train/dev separation at the benchmark-instance group level
- carry forward controller observability including shadow-commit traces
- rebuild a clean active graph-critic dataset root for later offline training
- record run count, group count, token usage, and commit-positive growth

This slice should **not**:

- tune on any future `paper_eval` instance
- launch the `24`-case diagnosis packet yet
- change the final paper-eval method definition
- introduce true online calibration

## File Map

### Files To Modify

- Modify: `src/idea_graph/engine.py`
  - port shadow-commit observability from the validated `g62` worktree patch
- Modify: `src/idea_graph/io.py`
  - surface `shadow_commit_log` in saved run summaries
- Modify: `tests/test_engine.py`
  - verify controller traces and summary payloads expose shadow-commit fields
- Modify: `docs/critic_pools.md`
  - register the new `development_pool_v3` artifacts as the current active expansion pool
- Modify: `docs/eig_graph_critic_plan.md`
  - record that Stage A v3 supersedes the smaller v2 expansion as the active scale-up dataset
- Modify: `docs/experiment_execution_log.md`
  - record the Stage A commands, outputs, counts, and readiness judgment

### Files To Create

- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/candidate_instances.json`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/README.md`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/partition_manifest.jsonl`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/split_registry.jsonl`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/split_registry_stats.json`
- Create: `outputs/graph_critic_online_episodes/development_pool_v3_critic_train_qwen_v1/`
- Create: `outputs/graph_critic_online_episodes/development_pool_v3_critic_dev_qwen_v1/`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_expansion_only_g1/`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g1/`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g2/`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g25/`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g2_partitions/partition_manifest.jsonl`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/group_split_overrides.jsonl`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_readiness/training_readiness_report.md`

### Existing Scripts Reused As-Is

- Reuse: `scripts/build_critic_expansion_pool.py`
- Reuse: `scripts/collect_critic_train_episodes.py`
- Reuse: `scripts/export_graph_critic_dataset.py`
- Reuse: `scripts/build_graph_critic_dataset.py`
- Reuse: `scripts/build_critic_partition_manifest.py`
- Reuse: `scripts/build_graph_critic_candidate_dataset.py`

## Task 1: Land Shadow-Commit Observability On `main`

**Files:**
- Modify: `src/idea_graph/engine.py`
- Modify: `src/idea_graph/io.py`
- Modify: `tests/test_engine.py`

- [ ] **Step 1.1: Verify the donor patch exists in the `g62` worktree**

Run:

```powershell
git -C .worktrees/g62-graph-critic-dev-packet show --stat --oneline d3f6439
```

Expected:

- commit `d3f6439` exists
- touched files are exactly:
  - `src/idea_graph/engine.py`
  - `src/idea_graph/io.py`
  - `tests/test_engine.py`

- [ ] **Step 1.2: Cherry-pick the validated shadow-commit patch**

Run:

```powershell
git cherry-pick d3f6439
```

Expected:

- cherry-pick succeeds cleanly on the current execution branch
- no unrelated files are modified by hand

- [ ] **Step 1.3: Run the targeted engine verification**

Run:

```powershell
python -m pytest tests/test_engine.py -q
```

Expected:

- pass
- engine summaries now expose `shadow_commit_log`

## Task 2: Materialize The `development_pool_v3_candidate_pool_v1` Expansion Pool

**Files:**
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/candidate_instances.json`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/README.md`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/partition_manifest.jsonl`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/split_registry.jsonl`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/split_registry_stats.json`

- [ ] **Step 2.1: Create the exact candidate list**

Run:

```powershell
@'
import json
from pathlib import Path

output_path = Path("outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/candidate_instances.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

train_aiib = [10, 27, 44, 61, 78, 97, 114, 131, 148, 165, 182, 199, 216, 233, 250, 267, 284, 301]
dev_aiib = [318, 335, 352, 369, 386, 403]
train_live = [
    "liveideabench-weather forecasting-50",
    "liveideabench-tsunamis-400",
    "liveideabench-endocrinology-900",
    "liveideabench-pathology-1400",
    "liveideabench-gis-2100",
    "liveideabench-global warming-2800",
    "liveideabench-vaccines-3500",
    "liveideabench-metabolism-4300",
    "liveideabench-good manufacturing practices-5200",
]
dev_live = [
    "liveideabench-epigenetics-6100",
    "liveideabench-gerontology-7000",
    "liveideabench-string theory-7900",
]

rows = []
for index in train_aiib:
    rows.append(
        {
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": f"ai-idea-bench-2025-{index}",
            "partition_role": "critic_train",
            "notes": "Stage A v3 AIIB train candidate.",
        }
    )
for index in dev_aiib:
    rows.append(
        {
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": f"ai-idea-bench-2025-{index}",
            "partition_role": "critic_dev",
            "notes": "Stage A v3 AIIB dev candidate.",
        }
    )
for name in train_live:
    rows.append(
        {
            "benchmark": "liveideabench",
            "instance_name": name,
            "partition_role": "critic_train",
            "notes": "Stage A v3 LiveIdeaBench train candidate.",
        }
    )
for name in dev_live:
    rows.append(
        {
            "benchmark": "liveideabench",
            "instance_name": name,
            "partition_role": "critic_dev",
            "notes": "Stage A v3 LiveIdeaBench dev candidate.",
        }
    )

output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(output_path)
print(f"rows={len(rows)}")
print(f"critic_train={sum(1 for row in rows if row['partition_role'] == 'critic_train')}")
print(f"critic_dev={sum(1 for row in rows if row['partition_role'] == 'critic_dev')}")
'@ | python -
```

Expected:

- file exists
- `rows=36`
- `critic_train=27`
- `critic_dev=9`

- [ ] **Step 2.2: Add the pool README**

Create `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/README.md` with:

```markdown
# Development Pool V3 Candidate Pool

This pool is the final Stage A development-only graph-critic expansion pool.

- benchmark groups: `36`
- `critic_train`: `27`
- `critic_dev`: `9`
- target mix:
  - `AI_Idea_Bench_2025`: `24`
  - `LiveIdeaBench`: `12`

Practical rules:

- These instances are development-only.
- They must not be reused for future frozen `paper_eval` benchmarking.
- This pool supersedes `development_pool_v2_candidate_pool_v1` as the active scale-up pool for Stage A.
```

- [ ] **Step 2.3: Build the overlap-safe partition and split registry**

Run:

```powershell
python scripts/build_critic_expansion_pool.py `
  --candidate-file outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/candidate_instances.json `
  --blocked-split-registry outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g35_partitions/split_registry.jsonl `
  --blocked-split-registry outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_candidate_pool_v1/split_registry.jsonl `
  --blocked-candidate-file outputs/graph_critic_datasets/02_active_graph_critic/paper_eval_candidate_pool_v1/candidate_instances.json `
  --pool-name development_pool_v3_candidate_pool_v1 `
  --output-dir outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1
```

Expected:

- `partition_manifest.jsonl` exists
- `split_registry.jsonl` exists
- `split_registry_stats.json` exists
- no overlap error is raised

- [ ] **Step 2.4: Verify the partition counts**

Run:

```powershell
@'
import json
from collections import Counter
from pathlib import Path
path = Path("outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/partition_manifest.jsonl")
counts = Counter()
bench = Counter()
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    counts[row["partition_role"]] += 1
    bench[row["benchmark"]] += 1
print(dict(counts))
print(dict(bench))
'@ | python -
```

Expected:

- `{'critic_train': 27, 'critic_dev': 9}`
- `{'AI_Idea_Bench_2025': 24, 'liveideabench': 12}`

## Task 3: Run Collection Smokes On The New Pool

**Files:**
- Create: `outputs/graph_critic_online_episodes/development_pool_v3_critic_train_manifest_smoke/`
- Create: `outputs/graph_critic_online_episodes/development_pool_v3_critic_dev_manifest_smoke/`
- Create: `outputs/graph_critic_online_episodes/development_pool_v3_critic_train_execute_smoke_det/`
- Create: `outputs/graph_critic_online_episodes/development_pool_v3_critic_dev_execute_smoke_det/`

- [ ] **Step 3.1: Dry-run the train manifest**

Run:

```powershell
python scripts/collect_critic_train_episodes.py `
  --split-registry outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/split_registry.jsonl `
  --output-dir outputs/graph_critic_online_episodes `
  --collection-name development_pool_v3_critic_train_manifest_smoke `
  --pool-name development_pool_v3_candidate_pool_v1 `
  --partition-role critic_train `
  --required-usage train_online_critic
```

Expected:

- dry-run summary written
- selected group count is `27`

- [ ] **Step 3.2: Dry-run the dev manifest**

Run:

```powershell
python scripts/collect_critic_train_episodes.py `
  --split-registry outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/split_registry.jsonl `
  --output-dir outputs/graph_critic_online_episodes `
  --collection-name development_pool_v3_critic_dev_manifest_smoke `
  --pool-name development_pool_v3_candidate_pool_v1 `
  --partition-role critic_dev `
  --required-usage development_analysis
```

Expected:

- dry-run summary written
- selected group count is `9`

- [ ] **Step 3.3: Execute one deterministic train smoke**

Run:

```powershell
python scripts/collect_critic_train_episodes.py `
  --split-registry outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/split_registry.jsonl `
  --output-dir outputs/graph_critic_online_episodes `
  --collection-name development_pool_v3_critic_train_execute_smoke_det `
  --pool-name development_pool_v3_candidate_pool_v1 `
  --partition-role critic_train `
  --required-usage train_online_critic `
  --limit 1 `
  --execute `
  --skip-existing `
  --agent-backend deterministic
```

Expected:

- exactly one train run is materialized

- [ ] **Step 3.4: Execute one deterministic dev smoke**

Run:

```powershell
python scripts/collect_critic_train_episodes.py `
  --split-registry outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/split_registry.jsonl `
  --output-dir outputs/graph_critic_online_episodes `
  --collection-name development_pool_v3_critic_dev_execute_smoke_det `
  --pool-name development_pool_v3_candidate_pool_v1 `
  --partition-role critic_dev `
  --required-usage development_analysis `
  --limit 1 `
  --execute `
  --skip-existing `
  --agent-backend deterministic
```

Expected:

- exactly one dev run is materialized

## Task 4: Collect The Full API-Backed Stage A Episodes

**Files:**
- Create: `outputs/graph_critic_online_episodes/development_pool_v3_critic_train_qwen_v1/`
- Create: `outputs/graph_critic_online_episodes/development_pool_v3_critic_dev_qwen_v1/`

- [ ] **Step 4.1: Launch the train collection**

Run:

```powershell
python scripts/collect_critic_train_episodes.py `
  --split-registry outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/split_registry.jsonl `
  --output-dir outputs/graph_critic_online_episodes `
  --collection-name development_pool_v3_critic_train_qwen_v1 `
  --pool-name development_pool_v3_candidate_pool_v1 `
  --partition-role critic_train `
  --required-usage train_online_critic `
  --baseline ours-eig `
  --max-rounds 5 `
  --native-eval `
  --agent-backend openai-compatible `
  --llm-config configs/openai_compatible.example.json `
  --execute `
  --skip-existing
```

Expected:

- train collection completes across the `27` train groups

- [ ] **Step 4.2: Launch the dev collection**

Run:

```powershell
python scripts/collect_critic_train_episodes.py `
  --split-registry outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/split_registry.jsonl `
  --output-dir outputs/graph_critic_online_episodes `
  --collection-name development_pool_v3_critic_dev_qwen_v1 `
  --pool-name development_pool_v3_candidate_pool_v1 `
  --partition-role critic_dev `
  --required-usage development_analysis `
  --baseline ours-eig `
  --max-rounds 5 `
  --native-eval `
  --agent-backend openai-compatible `
  --llm-config configs/openai_compatible.example.json `
  --execute `
  --skip-existing
```

Expected:

- dev collection completes across the `9` dev groups

- [ ] **Step 4.3: Record collection summaries**

Run:

```powershell
Get-Content outputs/graph_critic_online_episodes/development_pool_v3_critic_train_qwen_v1/collection_summary.json
Get-Content outputs/graph_critic_online_episodes/development_pool_v3_critic_dev_qwen_v1/collection_summary.json
```

Expected:

- both summaries show completed collections
- token counts are non-zero

## Task 5: Export `G1` / `G2` / `G2.5` For The New Active Stage A Dataset

**Files:**
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_expansion_only_g1/`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g1/`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/group_split_overrides.jsonl`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g2/`
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g25/`

- [ ] **Step 5.1: Export the expansion-only G1 dataset**

Run:

```powershell
python scripts/export_graph_critic_dataset.py `
  --input-root outputs/graph_critic_online_episodes/development_pool_v3_critic_train_qwen_v1/runs `
  --input-root outputs/graph_critic_online_episodes/development_pool_v3_critic_dev_qwen_v1/runs `
  --output-dir outputs/graph_critic_datasets/02_active_graph_critic `
  --dataset-name development_pool_v3_expansion_only_g1 `
  --baseline ours-eig
```

Expected:

- `development_pool_v3_expansion_only_g1` exists

- [ ] **Step 5.2: Export the combined G1 dataset**

Run:

```powershell
python scripts/export_graph_critic_dataset.py `
  --input-root outputs `
  --input-root outputs/graph_critic_online_episodes/development_pool_v3_critic_train_qwen_v1/runs `
  --input-root outputs/graph_critic_online_episodes/development_pool_v3_critic_dev_qwen_v1/runs `
  --output-dir outputs/graph_critic_datasets/02_active_graph_critic `
  --dataset-name development_pool_v3_combined_g1 `
  --baseline ours-eig
```

Expected:

- `development_pool_v3_combined_g1` exists
- dataset includes old benchmarked runs plus new v3 expansion runs

- [ ] **Step 5.3: Write the combined split overrides**

Run:

```powershell
@'
import json
from pathlib import Path

old_partition_path = Path("outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_combined_g2_partitions/partition_manifest.jsonl")
new_partition_path = Path("outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/partition_manifest.jsonl")
output_path = Path("outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/group_split_overrides.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

rows = []
for path in [old_partition_path, new_partition_path]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = row.get("split")
        if split is None:
            partition_role = row["partition_role"]
            split = "train" if partition_role == "critic_train" else "validation"
        rows.append(
            {
                "group_id": row["group_id"],
                "benchmark": row["benchmark"],
                "instance_name": row["instance_name"],
                "split": split,
            }
        )

dedup = {}
for row in rows:
    dedup[row["group_id"]] = row

ordered = [dedup[key] for key in sorted(dedup)]
output_path.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in ordered),
    encoding="utf-8",
)
print(output_path)
print(f"rows={len(ordered)}")
'@ | python -
```

Expected:

- split override file exists
- row count equals old frozen groups plus `36` new v3 groups

- [ ] **Step 5.4: Build the combined G2 dataset**

Run:

```powershell
python scripts/build_graph_critic_dataset.py `
  --g1-dataset-dir outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g1 `
  --output-dir outputs/graph_critic_datasets/02_active_graph_critic `
  --dataset-name development_pool_v3_combined_g2 `
  --split-overrides outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/group_split_overrides.jsonl
```

Expected:

- combined G2 exists
- train/dev split matches the override file exactly

- [ ] **Step 5.5: Build the combined partition manifest dataset**

Run:

```powershell
python scripts/build_critic_partition_manifest.py `
  --g2-dataset-dir outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g2 `
  --output-dir outputs/graph_critic_datasets/02_active_graph_critic `
  --dataset-name development_pool_v3_combined_g2_partitions
```

Expected:

- `development_pool_v3_combined_g2_partitions/partition_manifest.jsonl` exists
- partition roles are deterministic and group-level

- [ ] **Step 5.6: Build the combined G2.5 candidate-slate dataset**

Run:

```powershell
python scripts/build_graph_critic_candidate_dataset.py `
  --g1-dataset-dir outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g1 `
  --g2-dataset-dir outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g2 `
  --output-dir outputs/graph_critic_datasets/02_active_graph_critic `
  --dataset-name development_pool_v3_combined_g25
```

Expected:

- combined G2.5 exists
- commit candidate rows are present for every state

## Task 6: Write The Readiness Report And Update The Docs

**Files:**
- Create: `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_readiness/training_readiness_report.md`
- Modify: `docs/critic_pools.md`
- Modify: `docs/eig_graph_critic_plan.md`
- Modify: `docs/experiment_execution_log.md`

- [ ] **Step 6.1: Generate the readiness report**

Run:

```powershell
@'
import json
from pathlib import Path

g1_profile = json.loads(Path("outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g1/dataset_profile.json").read_text(encoding="utf-8"))
g2_stats = json.loads(Path("outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g2/dataset_stats.json").read_text(encoding="utf-8"))
g25_stats = json.loads(Path("outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g25/dataset_stats.json").read_text(encoding="utf-8"))

output_dir = Path("outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_readiness")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "training_readiness_report.md"

text = f'''# Development Pool V3 Graph-Critic Training-Readiness Report

## Scope

This report summarizes the final Stage A development-only graph-critic expansion.

Artifacts:

- `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1`
- `outputs/graph_critic_online_episodes/development_pool_v3_critic_train_qwen_v1`
- `outputs/graph_critic_online_episodes/development_pool_v3_critic_dev_qwen_v1`
- `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g1`
- `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g2`
- `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g25`

## Combined G1

- runs: `{g1_profile["run_count"]}`
- transitions: `{g1_profile["transition_count"]}`
- terminal commit states: `{g1_profile["terminal_state_count"]}`
- prompt tokens: `{g1_profile["token_usage"]["prompt_tokens"]}`
- completion tokens: `{g1_profile["token_usage"]["completion_tokens"]}`
- total tokens: `{g1_profile["token_usage"]["total_tokens"]}`
- mean tokens per run: `{g1_profile["token_usage"]["mean_tokens_per_run"]:.2f}`
- mean tokens per transition: `{g1_profile["token_usage"]["mean_tokens_per_transition"]:.2f}`

## Combined G2

- groups: `{g2_stats["group_count"]}`
- train groups: `{g2_stats["train_group_count"]}`
- validation groups: `{g2_stats["validation_group_count"]}`
- transitions: `{g2_stats["transition_count"]}`
- train transitions: `{g2_stats["train_transition_count"]}`
- validation transitions: `{g2_stats["validation_transition_count"]}`

## Combined G2.5

- states: `{g25_stats["state_count"]}`
- candidate rows: `{g25_stats["candidate_count"]}`
- commit candidates: `{g25_stats["commit_count"]}`
- commit-positive states: `{g25_stats["commit_positive_count"]}`
- validation states: `{g25_stats["split_state_counts"]["validation"]}`
- validation candidate rows: `{g25_stats["split_candidate_counts"]["validation"]}`

## Judgment

This Stage A dataset is development-only and remains excluded from final paper-eval benchmarking.

It is now intended to be the active dataset for:

- refreshed offline text-critic training
- refreshed offline graph-critic training
- the next medium controller diagnosis packet

It should be treated as sufficient for the next offline freeze gate only if:

- the graph critic still beats or matches the text scorer on the frozen validation groups
- commit-positive supervision is materially larger than the prior v2 pool
- controller traces remain leak-safe and auditable
'''

output_path.write_text(text, encoding="utf-8")
print(output_path)
'@ | python -
```

Expected:

- readiness report exists

- [ ] **Step 6.2: Update the three tracking docs**

Update:

- `docs/critic_pools.md`
- `docs/eig_graph_critic_plan.md`
- `docs/experiment_execution_log.md`

Required content:

- register `development_pool_v3_candidate_pool_v1` as the active Stage A expansion pool
- register the `development_pool_v3_*` collection and dataset roots
- record the new group counts, runs, and readiness judgment

- [ ] **Step 6.3: Run the verification packet**

Run:

```powershell
python -m pytest tests/test_engine.py tests/test_critic_pool_expansion.py tests/test_critic_episode_collection.py tests/test_trajectory_dataset.py tests/test_candidate_slate_dataset.py -q
```

Expected:

- all targeted verification passes

## Final Verification Checklist

- [ ] shadow-commit observability is now on the active execution branch
- [ ] `development_pool_v3_candidate_pool_v1` is disjoint from the old development and paper-eval pools
- [ ] Stage A expands to `36` new groups with `27` train and `9` dev
- [ ] combined split overrides preserve the old frozen groups and add the new v3 groups deterministically
- [ ] the active combined dataset now uses the `development_pool_v3_*` artifact family
- [ ] the readiness report records run count, group count, token usage, and commit-positive growth
