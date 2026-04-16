# Parallel Runtime Heuristic Collection And Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `parallel_graph_v2` usable as a smooth heuristic teacher, collect enough protocol-matched replay for the two-head critic, then run clean heuristic, learned critic, and calibrated learned critic ablations.

**Architecture:** Keep the heuristic `parallel_graph_v2` controller as the first reliable teacher and paper ablation. Add only the missing plumbing needed for large-scale collection, fixed train/dev split overrides, learned two-head runtime control, and frozen development-set commit calibration. The final paper stack compares the same external benchmark I/O under three internal controllers: heuristic, learned two-head critic, and calibrated learned two-head critic.

**Tech Stack:** Python 3, pytest, existing benchmark loaders, JSONL replay exports, `scripts/collect_critic_train_episodes.py`, `scripts/export_graph_critic_dataset.py`, `scripts/build_parallel_two_head_dataset.py`, PyTorch two-head critic training, OpenAI-compatible Qwen collection where configured.

---

## File Structure

- Modify: `src/idea_graph/critic_episode_collection.py`
  Purpose: pass `runtime_protocol` and optional maturity-stop behavior into generated `run_pipeline.py` commands.
- Modify: `scripts/collect_critic_train_episodes.py`
  Purpose: expose `--runtime-protocol` and `--disable-maturity-stop` for heuristic teacher collection packets.
- Modify: `tests/test_critic_episode_collection.py`
  Purpose: prevent silent fallback to `sequential_v1` during large `parallel_graph_v2` replay collection.
- Create: `src/idea_graph/critic_split_overrides.py`
  Purpose: convert a split registry into `train`/`validation` split overrides for two-head dataset building.
- Create: `scripts/build_critic_split_overrides.py`
  Purpose: write `split_overrides.jsonl` from a split registry with `critic_train -> train` and `critic_dev -> validation`.
- Create: `tests/test_critic_split_overrides.py`
  Purpose: verify fixed train/dev mapping and ensure `paper_eval` is excluded from training split overrides.
- Create: `src/idea_graph/relation_graph_two_head_runtime_critic.py`
  Purpose: load a trained two-head critic and score parallel role-local edit slates plus post-round commit states.
- Modify: `src/idea_graph/parallel_runtime.py`
  Purpose: stop ignoring `runtime_controller`; support shadow logging, learned edit selection, and calibrated commit decisions.
- Modify: `src/idea_graph/baselines.py`
  Purpose: add explicit paper-facing ablation specs for heuristic parallel, learned two-head, and calibrated two-head controllers.
- Modify: `tests/test_parallel_runtime.py`
  Purpose: lock that heuristic mode remains unchanged and learned mode can override selected role decisions without changing replay schema.
- Modify: `tests/test_benchmark_mode_and_baselines.py`
  Purpose: lock new baseline names and controller metadata.
- Create: `src/idea_graph/relation_graph_two_head_calibration.py`
  Purpose: fit and apply frozen commit-head calibration on `critic_dev` rows.
- Create: `scripts/calibrate_relation_graph_two_head_commit.py`
  Purpose: produce a calibration artifact from a trained two-head model and held-out dev commit rows.
- Create: `tests/test_relation_graph_two_head_calibration.py`
  Purpose: verify threshold selection, calibration artifact schema, and guardrails for missing positive/negative dev labels.
- Output: `outputs/graph_critic_online_episodes/parallel_v2_heuristic_teacher_qwen_v1/`
  Purpose: train/dev heuristic teacher episode collection root.
- Output: `outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_g1/`
  Purpose: exported replay with edit and post-round commit examples.
- Output: `outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_two_head/`
  Purpose: fixed-split two-head dataset.
- Output: `outputs/graph_critic_models/parallel_v2_two_head_qwen_v1/`
  Purpose: trained learned critic checkpoint and calibration artifacts.
- Output: `outputs/paper_eval_parallel_v2_ablation_qwen_v1/`
  Purpose: final ablation runs and summary artifacts.

## Acceptance Rules

- Heuristic collection is acceptable if it is smooth, replay-complete, and reasonable; it does not need to force positive `skip` labels.
- `skip` must remain present as a candidate in every active role-local slate and must be audited, but `selected_skip_count == 0` is not a blocking failure by itself.
- Large training curation should collect all available `critic_train` and `critic_dev` groups in the chosen split registry unless a small `--limit` smoke is explicitly being run.
- Paper evaluation must use frozen `paper_eval` groups that were not used for training, checkpoint selection, or calibration.
- Learned live commit should not be enabled before frozen dev calibration exists and has both positive and negative commit labels.

## Task 1: Prevent Silent Sequential Collection

**Files:**
- Modify: `src/idea_graph/critic_episode_collection.py`
- Modify: `scripts/collect_critic_train_episodes.py`
- Test: `tests/test_critic_episode_collection.py`

- [ ] **Step 1: Write the failing command test**

Add this test to `tests/test_critic_episode_collection.py`:

```python
def test_build_episode_launch_manifest_passes_parallel_runtime_protocol(self) -> None:
    manifest = build_episode_launch_manifest(
        [self.registry_rows[0]],
        baseline_name="ours-eig",
        max_rounds=5,
        native_eval=False,
        runs_dir=self.tmp_dir / "runs",
        agent_backend="deterministic",
        runtime_protocol="parallel_graph_v2",
    )

    command = manifest[0]["command"]
    self.assertIn("--runtime-protocol", command)
    self.assertIn("parallel_graph_v2", command)
    self.assertEqual(manifest[0]["runtime_protocol"], "parallel_graph_v2")
```

- [ ] **Step 2: Run the targeted test to verify RED**

Run:

```powershell
python -m pytest tests/test_critic_episode_collection.py::CriticEpisodeCollectionTests::test_build_episode_launch_manifest_passes_parallel_runtime_protocol -q
```

Expected: fail because `build_episode_launch_manifest(...)` does not accept `runtime_protocol` yet.

- [ ] **Step 3: Add runtime protocol to collection command construction**

In `src/idea_graph/critic_episode_collection.py`, update signatures:

```python
def build_run_pipeline_command(
    manifest_row: Mapping[str, Any],
    *,
    runs_dir: Path,
    python_executable: str | None = None,
    llm_config_path: Path | None = None,
    benchmark_root: Path | None = None,
    agent_backend: str = "openai-compatible",
    runtime_protocol: str = "parallel_graph_v2",
    disable_maturity_stop: bool = False,
) -> list[str]:
```

Add to the command list after `--max-rounds`:

```python
        "--runtime-protocol",
        str(runtime_protocol),
```

Append maturity stop only when requested:

```python
    if bool(disable_maturity_stop):
        command.append("--disable-maturity-stop")
```

Update `build_episode_launch_manifest(...)` to accept the same two keyword arguments, store them on each `manifest_row`, and pass them into `build_run_pipeline_command(...)`.

- [ ] **Step 4: Add CLI flags**

In `scripts/collect_critic_train_episodes.py`, add:

```python
    parser.add_argument(
        "--runtime-protocol",
        choices=["sequential_v1", "parallel_graph_v2"],
        default="parallel_graph_v2",
        help="Runtime protocol passed through to run_pipeline.py.",
    )
    parser.add_argument(
        "--disable-maturity-stop",
        action="store_true",
        help="Keep collected episodes running to --max-rounds even if maturity is reached.",
    )
```

Pass both values into `build_episode_launch_manifest(...)` and include both fields in `collection_config`.

- [ ] **Step 5: Verify GREEN**

Run:

```powershell
python -m pytest tests/test_critic_episode_collection.py -q
```

Expected: all critic episode collection tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/idea_graph/critic_episode_collection.py scripts/collect_critic_train_episodes.py tests/test_critic_episode_collection.py
git commit -m "feat: collect critic episodes with runtime protocol"
git push
```

## Task 2: Build Fixed Split Overrides From Split Registry

**Files:**
- Create: `src/idea_graph/critic_split_overrides.py`
- Create: `scripts/build_critic_split_overrides.py`
- Test: `tests/test_critic_split_overrides.py`

- [ ] **Step 1: Write the failing split-override tests**

Create `tests/test_critic_split_overrides.py`:

```python
from __future__ import annotations

import unittest

from idea_graph.critic_split_overrides import build_split_override_rows


class CriticSplitOverrideTests(unittest.TestCase):
    def test_build_split_override_rows_maps_train_and_dev(self) -> None:
        rows = build_split_override_rows(
            [
                {"group_id": "g-train", "partition_role": "critic_train"},
                {"group_id": "g-dev", "partition_role": "critic_dev"},
            ]
        )

        self.assertEqual(
            rows,
            [
                {"group_id": "g-dev", "split": "validation"},
                {"group_id": "g-train", "split": "train"},
            ],
        )

    def test_build_split_override_rows_excludes_paper_eval(self) -> None:
        rows = build_split_override_rows(
            [
                {"group_id": "g-paper", "partition_role": "paper_eval"},
                {"group_id": "g-train", "partition_role": "critic_train"},
            ]
        )

        self.assertEqual(rows, [{"group_id": "g-train", "split": "train"}])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the targeted test to verify RED**

```powershell
python -m pytest tests/test_critic_split_overrides.py -q
```

Expected: fail because `idea_graph.critic_split_overrides` does not exist.

- [ ] **Step 3: Implement split override builder**

Create `src/idea_graph/critic_split_overrides.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fs_utils import read_text_file, write_text_file


_ROLE_TO_SPLIT = {
    "critic_train": "train",
    "critic_dev": "validation",
}


def load_split_registry_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_index, raw_line in enumerate(read_text_file(path).splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} line {line_index} must contain a JSON object.")
        rows.append(dict(payload))
    return rows


def build_split_override_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    overrides: dict[str, str] = {}
    for row in rows:
        group_id = str(row.get("group_id", "")).strip()
        if not group_id:
            raise ValueError("split registry row is missing required group_id.")
        partition_role = str(row.get("partition_role", "")).strip()
        split = _ROLE_TO_SPLIT.get(partition_role)
        if split is None:
            continue
        existing = overrides.get(group_id)
        if existing is not None and existing != split:
            raise ValueError(f"Conflicting split override for group_id '{group_id}'.")
        overrides[group_id] = split
    return [
        {"group_id": group_id, "split": split}
        for group_id, split in sorted(overrides.items())
    ]


def write_split_override_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    write_text_file(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
    )
```

- [ ] **Step 4: Add CLI wrapper**

Create `scripts/build_critic_split_overrides.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from idea_graph.critic_split_overrides import (
    build_split_override_rows,
    load_split_registry_rows,
    write_split_override_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/validation split overrides from a split registry.")
    parser.add_argument("--split-registry", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    rows = load_split_registry_rows(args.split_registry)
    overrides = build_split_override_rows(rows)
    write_split_override_rows(args.output_path, overrides)
    print(f"Wrote {len(overrides)} split overrides to {args.output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify GREEN**

```powershell
python -m pytest tests/test_critic_split_overrides.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add src/idea_graph/critic_split_overrides.py scripts/build_critic_split_overrides.py tests/test_critic_split_overrides.py
git commit -m "feat: build critic split overrides"
git push
```

## Task 3: Heuristic Pipeline Acceptance Gate

**Files:**
- Reuse: `scripts/run_pipeline.py`
- Reuse: `scripts/export_graph_critic_dataset.py`
- Reuse: `scripts/build_parallel_two_head_dataset.py`
- Reuse: `scripts/train_relation_graph_two_head_critic.py`
- Output: `outputs/parallel-runtime-acceptance/`

- [ ] **Step 1: Run focused tests before acceptance smokes**

```powershell
python -m pytest tests/test_parallel_runtime.py tests/test_parallel_replay.py tests/test_engine.py tests/test_trajectory_dataset.py tests/test_candidate_slate_dataset.py tests/test_relation_graph_two_head_data.py tests/test_relation_graph_two_head_model.py tests/test_relation_graph_two_head_train.py tests/test_critic_episode_collection.py tests/test_critic_split_overrides.py -q
```

Expected: all pass.

- [ ] **Step 2: Run local deterministic acceptance smokes**

```powershell
python scripts/run_pipeline.py --input outputs/parallel-runtime-smokes/curation-inputs/parallel-curation-a.json --output-dir outputs/parallel-runtime-acceptance/local --baseline ours-eig --agent-backend deterministic --runtime-protocol parallel_graph_v2 --max-rounds 5
python scripts/run_pipeline.py --input outputs/parallel-runtime-smokes/curation-inputs/parallel-curation-b.json --output-dir outputs/parallel-runtime-acceptance/local --baseline ours-eig --agent-backend deterministic --runtime-protocol parallel_graph_v2 --max-rounds 5
python scripts/run_pipeline.py --input outputs/parallel-runtime-smokes/curation-inputs/parallel-curation-c.json --output-dir outputs/parallel-runtime-acceptance/local --baseline ours-eig --agent-backend deterministic --runtime-protocol parallel_graph_v2 --max-rounds 5
```

Expected: all complete, all write `summary.json` and `graph.json`, and every `graph.metadata.parallel_round_traces[*]` has `selected_role_decisions`, `edit_patches`, `materialized_graph_actions`, and `post_round_commit`.

- [ ] **Step 3: Export and build an acceptance dataset**

```powershell
python scripts/export_graph_critic_dataset.py --input-root outputs/parallel-runtime-acceptance/local --output-dir outputs/graph_critic_datasets --dataset-name parallel_v2_acceptance_local_g1 --baseline ours-eig
python scripts/build_parallel_two_head_dataset.py --g1-dataset-dir outputs/graph_critic_datasets/parallel_v2_acceptance_local_g1 --output-dir outputs/graph_critic_datasets --dataset-name parallel_v2_acceptance_local_two_head
```

Expected:

- `post_round_commit_examples.jsonl` exists.
- `parallel_label_quality.json` has non-zero edit states and non-zero post-round commit states.
- `dataset_stats.json` has non-zero edit and commit rows.

- [ ] **Step 4: Train one acceptance epoch**

```powershell
python scripts/train_relation_graph_two_head_critic.py --dataset-dir outputs/graph_critic_datasets/parallel_v2_acceptance_local_two_head --g1-dataset-dir outputs/graph_critic_datasets/parallel_v2_acceptance_local_g1 --output-dir outputs/graph_critic_models/parallel_v2_acceptance_local_model --text-backend hash --embedding-dim 64 --hidden-dim 64 --batch-size 4 --epochs 1 --lr 1e-3
```

Expected: `model.pt`, `edit_metrics.json`, `commit_metrics.json`, `metadata.json`, and `training_config.json` exist.

## Task 4: Build The Frozen Parallel V2 Split Registry

**Files:**
- Create: `outputs/graph_critic_datasets/parallel_v2_split_registry/candidate_instances.json`
- Reuse: `scripts/build_critic_expansion_pool.py`
- Reuse: `scripts/build_critic_split_registry.py`
- Output: `outputs/graph_critic_datasets/parallel_v2_split_registry/partition_manifest.jsonl`
- Output: `outputs/graph_critic_datasets/parallel_v2_split_registry/split_registry.jsonl`

- [ ] **Step 1: Create the candidate instance file**

Create `outputs/graph_critic_datasets/parallel_v2_split_registry/candidate_instances.json` with this schema:

```json
[
  {
    "benchmark": "AI_Idea_Bench_2025",
    "instance_name": "ai-idea-bench-2025-13",
    "partition_role": "critic_train"
  },
  {
    "benchmark": "AI_Idea_Bench_2025",
    "instance_name": "ai-idea-bench-2025-9849",
    "partition_role": "critic_dev"
  },
  {
    "benchmark": "liveideabench",
    "instance_name": "liveideabench-meteorology-0",
    "partition_role": "critic_train"
  }
]
```

Target scale:

- use all safe development groups available in the current benchmark cache
- prefer at least `300` `critic_train` groups if available
- prefer at least `100` `critic_dev` groups if available
- balance AI Idea Bench and LiveIdeaBench when possible
- exclude all frozen `paper_eval` groups

This file is intentionally an output artifact, not a tracked source file.

- [ ] **Step 2: Build partition and split-registry artifacts**

```powershell
python scripts/build_critic_expansion_pool.py --candidate-file outputs/graph_critic_datasets/parallel_v2_split_registry/candidate_instances.json --pool-name development_pool_v1 --output-dir outputs/graph_critic_datasets/parallel_v2_split_registry
python scripts/build_critic_split_registry.py --partition-manifest outputs/graph_critic_datasets/parallel_v2_split_registry/partition_manifest.jsonl --pool-name development_pool_v1 --output-dir outputs/graph_critic_datasets/parallel_v2_split_registry
```

Expected:

- `partition_manifest.jsonl` exists
- `split_registry.jsonl` exists
- `split_registry_stats.json` exists
- roles include both `critic_train` and `critic_dev`

- [ ] **Step 3: Audit registry counts**

```powershell
Get-Content outputs/graph_critic_datasets/parallel_v2_split_registry/split_registry_stats.json
```

Acceptance:

- `critic_train` count is non-zero
- `critic_dev` count is non-zero
- no `paper_eval` row appears in this development registry
- no group overlaps with any known paper-eval freeze pool

## Task 5: Large Heuristic Teacher Collection

**Files:**
- Reuse: `scripts/collect_critic_train_episodes.py`
- Reuse: `scripts/export_graph_critic_dataset.py`
- Reuse: `scripts/build_critic_split_overrides.py`
- Reuse: `scripts/build_parallel_two_head_dataset.py`
- Input: a frozen split registry with `critic_train` and `critic_dev` rows
- Output: `outputs/graph_critic_online_episodes/parallel_v2_heuristic_teacher_qwen_v1/`

- [ ] **Step 1: Confirm provider without exposing secrets**

Use an environment variable for the API key and do not write it into any tracked file.

```powershell
python scripts/check_openai_compatible.py --llm-config configs/openai_compatible.example.json --llm-api-key-env DASHSCOPE_API_KEY
```

Expected: the endpoint responds successfully.

- [ ] **Step 2: Dry-run the train collection manifest**

```powershell
python scripts/collect_critic_train_episodes.py --split-registry outputs/graph_critic_datasets/parallel_v2_split_registry/split_registry.jsonl --output-dir outputs/graph_critic_online_episodes/parallel_v2_heuristic_teacher_qwen_v1 --collection-name train --pool-name development_pool_v1 --partition-role critic_train --required-usage train_online_critic --baseline ours-eig --max-rounds 5 --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --runtime-protocol parallel_graph_v2
```

Expected: `launch_manifest.jsonl`, `collection_config.json`, and `collection_summary.json` are written under the `train` collection directory with `runtime_protocol == "parallel_graph_v2"`.

- [ ] **Step 3: Execute the train collection**

```powershell
python scripts/collect_critic_train_episodes.py --split-registry outputs/graph_critic_datasets/parallel_v2_split_registry/split_registry.jsonl --output-dir outputs/graph_critic_online_episodes/parallel_v2_heuristic_teacher_qwen_v1 --collection-name train --pool-name development_pool_v1 --partition-role critic_train --required-usage train_online_critic --baseline ours-eig --max-rounds 5 --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --runtime-protocol parallel_graph_v2 --execute --skip-existing
```

Expected: no more than 5 percent failed runs. Failed runs must be inspected before export.

- [ ] **Step 4: Dry-run and execute the dev collection**

```powershell
python scripts/collect_critic_train_episodes.py --split-registry outputs/graph_critic_datasets/parallel_v2_split_registry/split_registry.jsonl --output-dir outputs/graph_critic_online_episodes/parallel_v2_heuristic_teacher_qwen_v1 --collection-name dev --pool-name development_pool_v1 --partition-role critic_dev --required-usage calibrate_threshold --baseline ours-eig --max-rounds 5 --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --runtime-protocol parallel_graph_v2 --execute --skip-existing
```

Expected: dev collection writes complete run artifacts and uses only `critic_dev` groups.

- [ ] **Step 5: Export the combined heuristic replay**

```powershell
python scripts/export_graph_critic_dataset.py --input-root outputs/graph_critic_online_episodes/parallel_v2_heuristic_teacher_qwen_v1 --output-dir outputs/graph_critic_datasets --dataset-name parallel_v2_heuristic_teacher_qwen_v1_g1 --baseline ours-eig
```

Expected: export scans both `train/runs` and `dev/runs`.

- [ ] **Step 6: Build fixed split overrides**

```powershell
python scripts/build_critic_split_overrides.py --split-registry outputs/graph_critic_datasets/parallel_v2_split_registry/split_registry.jsonl --output-path outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_g1/split_overrides.jsonl
```

Expected: overrides contain `critic_train -> train` and `critic_dev -> validation`, with no `paper_eval` rows.

- [ ] **Step 7: Build the two-head dataset with fixed splits**

```powershell
python scripts/build_parallel_two_head_dataset.py --g1-dataset-dir outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_g1 --output-dir outputs/graph_critic_datasets --dataset-name parallel_v2_heuristic_teacher_qwen_v1_two_head --split-overrides-path outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_g1/split_overrides.jsonl
```

Expected: `dataset_stats.json` shows both `train` and `validation` splits.

- [ ] **Step 8: Audit label quality**

Inspect:

```powershell
Get-Content outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_g1/parallel_label_quality.json
Get-Content outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_g1/parallel_edit_profile.json
Get-Content outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_two_head/dataset_stats.json
```

Acceptance:

- `edit_state_count > 0`
- `edit_candidate_count > edit_state_count`
- `post_round_commit_state_count > 0`
- `post_round_commit_positive_count > 0`
- `post_round_continue_count > post_round_commit_positive_count`
- `skip_candidate_count == edit_state_count`
- `selected_skip_count` is audited but not required to be positive

## Task 6: Train And Select The Two-Head Critic

**Files:**
- Reuse: `scripts/train_relation_graph_two_head_critic.py`
- Output: `outputs/graph_critic_models/parallel_v2_two_head_qwen_v1/`

- [ ] **Step 1: Run a short train smoke on the large dataset**

```powershell
python scripts/train_relation_graph_two_head_critic.py --dataset-dir outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_two_head --g1-dataset-dir outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_g1 --output-dir outputs/graph_critic_models/parallel_v2_two_head_qwen_v1/smoke --text-backend hash --embedding-dim 64 --hidden-dim 64 --batch-size 8 --epochs 1 --lr 1e-3
```

Expected: writes model and metrics.

- [ ] **Step 2: Train the main hash-backend model**

```powershell
python scripts/train_relation_graph_two_head_critic.py --dataset-dir outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_two_head --g1-dataset-dir outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_g1 --output-dir outputs/graph_critic_models/parallel_v2_two_head_qwen_v1/hash_main --text-backend hash --embedding-dim 128 --hidden-dim 128 --batch-size 16 --epochs 10 --lr 5e-4
```

Expected: validation edit MRR and commit accuracy are recorded.

- [ ] **Step 3: Optionally train a sentence-transformer model if local environment supports it**

```powershell
python scripts/train_relation_graph_two_head_critic.py --dataset-dir outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_two_head --g1-dataset-dir outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_g1 --output-dir outputs/graph_critic_models/parallel_v2_two_head_qwen_v1/st_main --text-backend sentence-transformer --text-model-name sentence-transformers/all-MiniLM-L6-v2 --hidden-dim 128 --batch-size 16 --epochs 10 --lr 5e-4
```

Expected: if dependencies or model download are unavailable, do not block the hash-backend path.

## Task 7: Calibrate The Commit Head

**Files:**
- Create: `src/idea_graph/relation_graph_two_head_calibration.py`
- Create: `scripts/calibrate_relation_graph_two_head_commit.py`
- Test: `tests/test_relation_graph_two_head_calibration.py`
- Output: `outputs/graph_critic_models/parallel_v2_two_head_qwen_v1/hash_main/commit_calibration.json`

- [ ] **Step 1: Write failing calibration tests**

`tests/test_relation_graph_two_head_calibration.py` should verify:

- threshold search chooses a threshold that maximizes balanced accuracy on dev logits and labels
- calibration rejects all-positive or all-negative dev labels
- artifact has `schema_version`, `threshold`, `metric_name`, `positive_count`, `negative_count`, and `source_model_dir`

- [ ] **Step 2: Implement threshold calibration**

`src/idea_graph/relation_graph_two_head_calibration.py` should expose:

```python
def fit_commit_threshold(logits: list[float], labels: list[int]) -> dict[str, object]:
    ...


def apply_commit_threshold(logit: float, calibration: Mapping[str, object]) -> bool:
    ...
```

Use balanced accuracy as the default objective so rare positive commit labels do not collapse into an always-continue policy.

- [ ] **Step 3: Implement calibration CLI**

`scripts/calibrate_relation_graph_two_head_commit.py` should:

- load the trained model
- score validation commit examples from the fixed-split two-head dataset
- fit threshold
- write `commit_calibration.json`

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/test_relation_graph_two_head_calibration.py -q
```

Expected: pass.

- [ ] **Step 5: Run calibration**

```powershell
python scripts/calibrate_relation_graph_two_head_commit.py --model-dir outputs/graph_critic_models/parallel_v2_two_head_qwen_v1/hash_main --dataset-dir outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_two_head --g1-dataset-dir outputs/graph_critic_datasets/parallel_v2_heuristic_teacher_qwen_v1_g1 --output-path outputs/graph_critic_models/parallel_v2_two_head_qwen_v1/hash_main/commit_calibration.json
```

Expected: calibration artifact is written only if dev labels include both commit and continue labels.

- [ ] **Step 6: Commit**

```powershell
git add src/idea_graph/relation_graph_two_head_calibration.py scripts/calibrate_relation_graph_two_head_commit.py tests/test_relation_graph_two_head_calibration.py
git commit -m "feat: calibrate two-head commit critic"
git push
```

## Task 8: Wire Learned Two-Head Critic Into `parallel_graph_v2`

**Files:**
- Create: `src/idea_graph/relation_graph_two_head_runtime_critic.py`
- Modify: `src/idea_graph/parallel_runtime.py`
- Modify: `src/idea_graph/baselines.py`
- Test: `tests/test_parallel_runtime.py`
- Test: `tests/test_benchmark_mode_and_baselines.py`
- Test: `tests/test_relation_graph_two_head_model.py`

- [ ] **Step 1: Write failing runtime integration tests**

Add tests that prove:

- heuristic mode still ignores missing runtime controller and produces the same replay fields
- shadow mode logs model edit scores and commit probability but does not change selected decisions
- learned edit mode can replace one role's `selected_role_decision`
- calibrated commit mode uses `commit_calibration.json` and records both raw and calibrated commit signals

- [ ] **Step 2: Implement two-head runtime loader**

Create `src/idea_graph/relation_graph_two_head_runtime_critic.py` with a loader that reads:

- `model.pt`
- `training_config.json`
- `metadata.json`
- optional `commit_calibration.json`

It should provide:

```python
class LoadedRelationGraphTwoHeadRuntimeCritic:
    def score_edit_slate(self, graph, *, round_name: str, role: str, candidates: list[dict[str, object]]) -> list[float]:
        ...

    def score_commit_state(self, graph, *, round_name: str) -> dict[str, object]:
        ...
```

- [ ] **Step 3: Use runtime controller in parallel edit selection**

In `src/idea_graph/parallel_runtime.py`, remove the current discard:

```python
del runtime_controller, runtime_controller_metadata, progress_callback
```

Then support controller modes from metadata:

- `parallel_two_head_shadow`: score and log, but keep heuristic decision
- `parallel_two_head_edit`: use edit head for role-local selected decision, keep heuristic maturity stop
- `parallel_two_head_calibrated`: use edit head and calibrated commit decision

All modes must keep the replay schema:

- `selected_role_decisions`
- `edit_patches`
- `materialized_graph_actions`
- `post_round_commit`
- `post_round_commit_rows`

- [ ] **Step 4: Add explicit ablation baselines**

In `src/idea_graph/baselines.py`, add baseline specs:

- `ours-eig-parallel-heuristic`
- `ours-eig-parallel-two-head`
- `ours-eig-parallel-two-head-calibrated`

Each should preserve external benchmark I/O and set metadata defaults rather than changing benchmark packet schemas.

- [ ] **Step 5: Verify targeted tests**

```powershell
python -m pytest tests/test_parallel_runtime.py tests/test_benchmark_mode_and_baselines.py tests/test_relation_graph_two_head_model.py -q
```

Expected: pass.

- [ ] **Step 6: Run full focused suite**

```powershell
python -m pytest tests/test_parallel_runtime.py tests/test_parallel_replay.py tests/test_engine.py tests/test_trajectory_dataset.py tests/test_candidate_slate_dataset.py tests/test_relation_graph_two_head_data.py tests/test_relation_graph_two_head_model.py tests/test_relation_graph_two_head_train.py tests/test_critic_episode_collection.py tests/test_critic_split_overrides.py tests/test_relation_graph_two_head_calibration.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```powershell
git add src/idea_graph/relation_graph_two_head_runtime_critic.py src/idea_graph/parallel_runtime.py src/idea_graph/baselines.py tests/test_parallel_runtime.py tests/test_benchmark_mode_and_baselines.py tests/test_relation_graph_two_head_model.py
git commit -m "feat: wire two-head critic into parallel runtime"
git push
```

## Task 9: Paper Ablation Evaluation

**Files:**
- Reuse: `scripts/run_controller_eval_packet.py`
- Reuse: `scripts/summarize_controller_eval_packet.py`
- Output: `outputs/paper_eval_parallel_v2_ablation_qwen_v1/`

- [ ] **Step 1: Run a small ablation smoke before full paper eval**

Run the same small held-out packet for:

- `ours-eig-parallel-heuristic`
- `ours-eig-parallel-two-head`
- `ours-eig-parallel-two-head-calibrated`

Expected: all three complete with the same external benchmark I/O and write comparable summaries.

- [ ] **Step 2: Run full paper eval only after smoke passes**

Full paper eval must use frozen `paper_eval` groups only. Do not reuse `critic_train` or `critic_dev`.

- [ ] **Step 3: Summarize ablation results**

Report:

- final benchmark quality
- wall-clock runtime
- token cost
- executed rounds
- action count
- support coverage
- commit timing
- premature stop rate
- late stop rate
- edit-head agreement with heuristic teacher
- calibrated-vs-raw commit behavior

- [ ] **Step 4: Freeze artifacts**

Copy the exact model, calibration artifact, split registry, dataset stats, and evaluation summary into the final paper artifact directory.

## Progress Checkpoint (2026-04-16 Evening)

- Completed:
  - Fixed worktree-aware benchmark-root defaults for `run_pipeline.py`, `collect_critic_train_episodes.py`, `run_quality_batch.py`, and `run_controller_eval_packet.py`.
  - Added regression coverage in `tests/test_script_default_paths.py`; focused parser/benchmark-facing suite passed with `38 passed`.
  - Fixed Windows long-path artifact writing in `write_run_artifacts(...)` via shared filesystem helper usage.
  - Added regression coverage in `tests/test_io.py`; focused suite passed with `7 passed` across `test_run_quality_batch.py`, `test_script_default_paths.py`, and `test_io.py`.
  - Regenerated dry-run train/dev collection manifests under `outputs/graph_critic_online_episodes/parallel_v2_heuristic_teacher_qwen_v1/`; commands now resolve `--benchmark-root` to the shared repo cache rather than the worktree path.
  - Verified real OpenAI-compatible `parallel_graph_v2` execution with `scripts/run_pipeline.py` on AI Idea Bench 2025 index `13`.
  - Verified real collection-wrapper execution with `scripts/collect_critic_train_episodes.py --execute --limit 1` for both `critic_train` and `critic_dev`.

- Current quality checkpoint:
  - Previous small comparison batch:
    - `outputs/quality_batches/20260416-165345-p2v2-qwen-2x2/`
  - Improved heuristic-teacher rerun with native evaluation:
    - `outputs/quality_batches/20260416-192152-p2v2-anchor-v11-2x2/`
  - Combined 2x2 aggregate on the improved rerun:
    - `ours-eig` (`parallel_graph_v2`): overall `5.88`, alignment `4.37`, expert `7.40`, graph `7.82`, calls `6.0`, tokens `14131`, rounds `4.50`, actions `13.00`, native `7.61`
    - `self-refine`: overall `5.78`, alignment `4.25`, expert `7.30`, calls `3.0`, tokens `6056`, native `8.04`
    - `direct`: overall `5.39`, alignment `3.55`, expert `7.23`, calls `1.0`, tokens `1713`, native `8.04`
    - `ai-researcher`: overall `4.37`, alignment `1.80`, expert `6.93`, calls `6.0`, tokens `11304`, native `8.11`
  - Delta versus the previous teacher smoke for `ours-eig`:
    - overall `+0.00`, alignment `+0.07`, expert `-0.07`, graph `-0.99`
    - tokens `-251`, rounds `-0.25`, actions `-10.75`
  - Per-case `ours-eig` highlights:
    - `AI_Idea_Bench_2025/13`: stop reason improved from `max_rounds_reached` to `mature_at_Round3`, alignment `+0.33`, actions `25 -> 9`
    - `AI_Idea_Bench_2025/15`: quality remained close while actions dropped `25 -> 17`
    - `liveideabench/0` and `liveideabench/23`: overall quality stayed essentially flat while actions dropped from `20/25` to `13/13`
  - Improved rerun per-benchmark means:
    - AI Idea Bench 2025: `ours-eig` `4.66`, `direct` `4.59`, `self-refine` `4.57`, `ai-researcher` `4.42`
    - LiveIdeaBench: `ours-eig` `7.11`, `self-refine` `6.99`, `direct` `6.20`, `ai-researcher` `4.32`

- Interpretation:
  - The improved parallel heuristic remains acceptable as the first teacher for critic-label curation because it is still protocol-matched, replay-complete, and ranked first on the refreshed combined slice while using materially fewer materialized graph actions.
  - The main gain from the anchor-and-activation upgrade is efficiency plus a cleaner AI Idea Bench stop pattern, not a dramatic jump in headline quality. This is the right tradeoff for supervision harvesting, where we care about action labels and commit timing consistency as much as raw benchmark score.
  - The margin over `self-refine` remains small overall and especially narrow on AI Idea Bench 2025, so this should still be framed as a reasonable bootstrap teacher rather than a final strong paper-claim result.
  - Because the collection artifacts retain round counts, action counts, stop reasons, selected role decisions, and protocol stamps, later filtering and reweighting can preferentially keep higher-quality supervision rows without changing the replay schema.

- Collection status:
  - Foreground execution is verified for both:
    - `outputs/graph_critic_online_episodes/parallel_v2_heuristic_teacher_qwen_v1_smoke_exec/train`
    - `outputs/graph_critic_online_episodes/parallel_v2_heuristic_teacher_qwen_v1_smoke_exec/dev`
  - Runtime settings:
    - baseline `ours-eig`
    - backend `openai-compatible`
    - runtime protocol `parallel_graph_v2`
  - Important operational note:
    - direct foreground execution inherits `DASHSCOPE_API_KEY` correctly and completes;
    - the ad hoc background launcher attempt from this session dropped the API-key environment in child processes, so full unattended collection should be relaunched only after using a launcher pattern that preserves the environment end-to-end.

## Plan Self-Review

- Spec coverage:
  - smooth heuristic `parallel_graph_v2`: Tasks 1, 3, and 4
  - heuristic as teacher for critic training: Task 4
  - enough samples with fixed train/dev: Tasks 2, 4, and 5
  - two-head learned critic training: Task 6
  - commit calibration: Task 7
  - heuristic, learned critic, calibrated critic ablations: Tasks 8 and 9
- Important gap captured:
  - current collection tooling defaults to `sequential_v1` unless we pass `--runtime-protocol`; Task 1 closes this.
  - current parallel runtime ignores `runtime_controller`; Task 7 closes this before learned-critic ablations.
- Placeholder scan:
  - no unresolved placeholder text remains.
- Type consistency:
  - runtime mode names are consistently `parallel_two_head_shadow`, `parallel_two_head_edit`, and `parallel_two_head_calibrated`.
  - dataset split labels are consistently `train` and `validation`.
