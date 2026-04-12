# Graph Critic Dataset Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `G2` critic-ready dataset layer with leakage-safe benchmark-instance splits and separate `weak_local` versus `native` label namespaces.

**Architecture:** Keep `G1` as the raw trajectory export layer, but extend its run manifest so it carries the full local/native label payloads needed by `G2`. Then add a new `critic_dataset` module that consumes only `G1` dataset directories, constructs deterministic group-level splits, packages label namespaces and normalized scalar targets, and writes critic-ready JSONL/statistics artifacts.

**Tech Stack:** Python 3, stdlib `json`/`pathlib`, existing repo file IO helpers, `unittest`, `pytest`

---

## File Structure

### Upstream G1 Extension

- Modify: `src/idea_graph/trajectory_dataset.py`
- Modify: `tests/test_trajectory_dataset.py`

Reason:

- current `run_manifest.jsonl` exposes only scalar local/native summaries
- approved `G2` expects full `weak_local.category_scores` and `native.metrics`
- adding them once at `G1` keeps `G2` honest and avoids rescanning raw run
  directories

### New G2 Layer

- Create: `src/idea_graph/critic_dataset.py`
- Create: `scripts/build_graph_critic_dataset.py`
- Create: `tests/test_critic_dataset.py`

### Active Records

- Modify: `docs/paper_experiment_tracker.md`
- Modify: `docs/experiment_execution_log.md`

---

### Task 1: Extend G1 run-manifest labels so G2 can stay self-contained

**Files:**
- Modify: `tests/test_trajectory_dataset.py`
- Modify: `src/idea_graph/trajectory_dataset.py`

- [ ] **Step 1: Add failing tests for richer manifest labels**

Append tests like:

```python
    def test_build_run_manifest_row_preserves_full_local_and_native_label_payloads(self) -> None:
        summary_payload = {
            **self._summary_payload(),
            "idea_evaluation": {
                "overall_score": 6.4,
                "category_scores": {
                    "benchmark_alignment": 4.8,
                    "expert_style_quality": 7.1,
                    "graph_process": 7.5,
                },
            },
            "benchmark_native_evaluation": {
                "benchmark": "AI_Idea_Bench_2025",
                "metrics": [
                    {"key": "i2i_motivation", "score": 4.0, "max_score": 5.0, "available": True},
                    {"key": "fps", "score": 4.0, "max_score": 5.0, "available": True},
                ],
                "summary": {"available_average_normalized_10": 6.9},
            },
        }
        row = build_run_manifest_row(self.tmp_dir / "eig_run", summary_payload, self._graph_payload())
        self.assertEqual(row["local_category_scores"]["graph_process"], 7.5)
        self.assertEqual(row["native_metric_map"]["i2i_motivation"]["score"], 4.0)
        self.assertEqual(row["native_metric_map"]["fps"]["max_score"], 5.0)
```

- [ ] **Step 2: Run the focused G1 tests to verify the new failures**

Run: `python -m pytest tests/test_trajectory_dataset.py -q`
Expected: FAIL because `local_category_scores` and `native_metric_map` are not yet exported by `build_run_manifest_row`

- [ ] **Step 3: Implement the minimal manifest enrichment**

Add logic in `src/idea_graph/trajectory_dataset.py` near `build_run_manifest_row`:

```python
    local_category_scores = dict(category_scores)
    native_metric_map = {}
    metrics_payload = native_evaluation.get("metrics", [])
    if isinstance(metrics_payload, list):
        for item in metrics_payload:
            if not isinstance(item, Mapping):
                continue
            key = str(item.get("key", "")).strip()
            if not key:
                continue
            native_metric_map[key] = dict(item)
```

and include:

```python
        "local_category_scores": local_category_scores,
        "native_metric_map": native_metric_map,
```

inside the manifest row.

- [ ] **Step 4: Re-run the G1 tests**

Run: `python -m pytest tests/test_trajectory_dataset.py -q`
Expected: PASS

### Task 2: Add failing tests for G2 split assignment and label packaging

**Files:**
- Create: `tests/test_critic_dataset.py`

- [ ] **Step 1: Write the failing G2 test file**

Create `tests/test_critic_dataset.py` with cases like:

```python
from idea_graph.critic_dataset import (
    assign_group_splits,
    build_critic_dataset_rows,
    build_dataset_stats,
    build_group_manifest,
    build_label_schema,
    load_g1_dataset,
)
```

and tests:

```python
class CriticDatasetTests(unittest.TestCase):
    def test_assign_group_splits_is_deterministic(self) -> None:
        manifest_rows, transition_rows = load_g1_dataset(self.dataset_dir)
        group_rows = build_group_manifest(manifest_rows, transition_rows)
        first = assign_group_splits(group_rows, validation_fraction=0.2)
        second = assign_group_splits(group_rows, validation_fraction=0.2)
        self.assertEqual(first, second)

    def test_all_rows_from_same_group_share_same_split(self) -> None:
        manifest_rows, transition_rows = load_g1_dataset(self.dataset_dir)
        split_rows = assign_group_splits(build_group_manifest(manifest_rows, transition_rows))
        critic_rows = build_critic_dataset_rows(manifest_rows, transition_rows, split_rows)
        split_by_group = {}
        for row in critic_rows:
            split_by_group.setdefault(row["group_id"], set()).add(row["split"])
        self.assertEqual(split_by_group["AI_Idea_Bench_2025::aiib-13"], {"train"})

    def test_duplicate_runs_are_preserved_and_indexed(self) -> None:
        manifest_rows, transition_rows = load_g1_dataset(self.dataset_dir)
        split_rows = assign_group_splits(build_group_manifest(manifest_rows, transition_rows))
        critic_rows = build_critic_dataset_rows(manifest_rows, transition_rows, split_rows)
        duplicate_rows = [row for row in critic_rows if row["group_id"] == "AI_Idea_Bench_2025::aiib-13"]
        self.assertEqual({row["group_run_count"] for row in duplicate_rows}, {2})
        self.assertEqual({row["group_run_index"] for row in duplicate_rows}, {0, 1})

    def test_build_critic_dataset_rows_separates_weak_and_native_labels(self) -> None:
        manifest_rows, transition_rows = load_g1_dataset(self.dataset_dir)
        split_rows = assign_group_splits(build_group_manifest(manifest_rows, transition_rows))
        critic_rows = build_critic_dataset_rows(manifest_rows, transition_rows, split_rows)
        first_row = critic_rows[0]
        self.assertTrue(first_row["weak_local"]["available"])
        self.assertEqual(first_row["weak_local"]["category_scores"]["graph_process"], 8.0)
        self.assertEqual(first_row["native"]["metrics"]["fps"]["score"], 4.0)
        self.assertAlmostEqual(first_row["targets"]["weak_value_01"], 0.6)

    def test_missing_native_labels_remain_explicitly_null(self) -> None:
        manifest_rows, transition_rows = load_g1_dataset(self.dataset_dir)
        split_rows = assign_group_splits(build_group_manifest(manifest_rows, transition_rows))
        critic_rows = build_critic_dataset_rows(manifest_rows, transition_rows, split_rows)
        live_row = next(row for row in critic_rows if row["group_id"] == "liveideabench::meteorology-0")
        self.assertFalse(live_row["native"]["available"])
        self.assertIsNone(live_row["native"]["average_10"])
        self.assertFalse(live_row["label_availability"]["has_native_average"])

    def test_build_dataset_stats_matches_row_counts(self) -> None:
        manifest_rows, transition_rows = load_g1_dataset(self.dataset_dir)
        split_rows = assign_group_splits(build_group_manifest(manifest_rows, transition_rows))
        critic_rows = build_critic_dataset_rows(manifest_rows, transition_rows, split_rows)
        stats = build_dataset_stats(critic_rows, split_rows)
        self.assertEqual(stats["group_count"], 2)
        self.assertEqual(stats["transition_count"], len(critic_rows))
        self.assertEqual(stats["train_group_count"] + stats["validation_group_count"], 2)
```

- [ ] **Step 2: Use G1-style fixture payloads rather than raw run folders**

Inside the test file, create a temporary `g1_dataset/` directory with:

```python
manifest_lines = [
    json.dumps(
        {
            "run_dir": "C:/tmp/run_a",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "aiib-13",
            "baseline_name": "ours-eig",
            "topic": "3D language field modeling",
            "local_category_scores": {"benchmark_alignment": 4.5, "graph_process": 8.0},
            "final_local_overall": 6.0,
            "final_local_alignment": 4.5,
            "native_metric_map": {"fps": {"score": 4.0, "max_score": 5.0, "available": True}},
            "final_native_average": 8.0,
            "has_local_eval": True,
            "has_native_eval": True,
        }
    ),
    json.dumps(
        {
            "run_dir": "C:/tmp/run_b",
            "benchmark": "AI_Idea_Bench_2025",
            "instance_name": "aiib-13",
            "baseline_name": "ours-eig",
            "topic": "3D language field modeling",
            "local_category_scores": {"benchmark_alignment": 5.0, "graph_process": 8.2},
            "final_local_overall": 6.4,
            "final_local_alignment": 5.0,
            "native_metric_map": {"fps": {"score": 4.5, "max_score": 5.0, "available": True}},
            "final_native_average": 8.4,
            "has_local_eval": True,
            "has_native_eval": True,
        }
    ),
    json.dumps(
        {
            "run_dir": "C:/tmp/run_c",
            "benchmark": "liveideabench",
            "instance_name": "meteorology-0",
            "baseline_name": "ours-eig",
            "topic": "meteorology",
            "local_category_scores": {"benchmark_alignment": 3.2, "graph_process": 7.6},
            "final_local_overall": 5.2,
            "final_local_alignment": 3.2,
            "native_metric_map": {},
            "final_native_average": None,
            "has_local_eval": True,
            "has_native_eval": False,
        }
    ),
]
transition_lines = [
    json.dumps({"run_dir": "C:/tmp/run_a", "benchmark": "AI_Idea_Bench_2025", "instance_name": "aiib-13", "step_index": 0}),
    json.dumps({"run_dir": "C:/tmp/run_a", "benchmark": "AI_Idea_Bench_2025", "instance_name": "aiib-13", "step_index": 1}),
    json.dumps({"run_dir": "C:/tmp/run_b", "benchmark": "AI_Idea_Bench_2025", "instance_name": "aiib-13", "step_index": 0}),
    json.dumps({"run_dir": "C:/tmp/run_c", "benchmark": "liveideabench", "instance_name": "meteorology-0", "step_index": 0}),
]
write_text_file(dataset_dir / "run_manifest.jsonl", "\n".join(manifest_lines) + "\n")
write_text_file(dataset_dir / "trajectory_examples.jsonl", "\n".join(transition_lines) + "\n")
write_text_file(dataset_dir / "dataset_profile.json", json.dumps({"run_count": 2}))
```

The manifest fixture should include:

```python
{
    "run_dir": "C:/tmp/run_a",
    "benchmark": "AI_Idea_Bench_2025",
    "instance_name": "aiib-13",
    "baseline_name": "ours-eig",
    "local_category_scores": {"benchmark_alignment": 4.5, "graph_process": 8.0},
    "final_local_overall": 6.0,
    "final_local_alignment": 4.5,
    "native_metric_map": {"fps": {"score": 4.0, "max_score": 5.0, "available": True}},
    "final_native_average": 8.0,
    "has_local_eval": True,
    "has_native_eval": True,
}
```

- [ ] **Step 3: Run the new G2 tests to verify they fail**

Run: `python -m pytest tests/test_critic_dataset.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'idea_graph.critic_dataset'`

### Task 3: Implement the G2 dataset-construction library

**Files:**
- Create: `src/idea_graph/critic_dataset.py`

- [ ] **Step 1: Implement G1 dataset loading helpers**

Add:

```python
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Mapping

from .fs_utils import read_text_file, write_text_file


def load_g1_dataset(dataset_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows = _load_jsonl(dataset_dir / "run_manifest.jsonl")
    transition_rows = _load_jsonl(dataset_dir / "trajectory_examples.jsonl")
    return manifest_rows, transition_rows
```

Use exact JSONL loading:

```python
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in read_text_file(path).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} contains a non-object JSONL row.")
        rows.append(payload)
    return rows
```

- [ ] **Step 2: Implement group-manifest construction**

Add functions:

```python
def make_group_id(row: Mapping[str, Any]) -> str:
    return f"{row['benchmark']}::{row['instance_name']}"


def build_group_manifest(
    manifest_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    transition_count_by_group = {}
    for row in transition_rows:
        group_id = make_group_id(row)
        transition_count_by_group[group_id] = transition_count_by_group.get(group_id, 0) + 1
    run_rows_by_group = {}
    for row in manifest_rows:
        group_id = make_group_id(row)
        run_rows_by_group.setdefault(group_id, []).append(row)
    group_rows = []
    for group_id, rows in sorted(run_rows_by_group.items()):
        weak_values = [float(row["final_local_overall"]) / 10.0 for row in rows if row.get("final_local_overall") is not None]
        native_values = [float(row["final_native_average"]) / 10.0 for row in rows if row.get("final_native_average") is not None]
        exemplar = rows[0]
        group_rows.append(
            {
                "group_id": group_id,
                "benchmark": exemplar["benchmark"],
                "instance_name": exemplar["instance_name"],
                "run_count": len(rows),
                "transition_count": transition_count_by_group.get(group_id, 0),
                "has_any_weak_local": any(bool(row.get("has_local_eval", False)) for row in rows),
                "has_any_native": any(bool(row.get("has_native_eval", False)) for row in rows),
                "mean_weak_value_01": sum(weak_values) / len(weak_values) if weak_values else None,
                "mean_native_value_01": sum(native_values) / len(native_values) if native_values else None,
            }
        )
    return group_rows
```

Each group row should compute:

```python
{
    "group_id": "AI_Idea_Bench_2025::aiib-13",
    "benchmark": "AI_Idea_Bench_2025",
    "instance_name": "aiib-13",
    "run_count": 2,
    "transition_count": 3,
    "has_any_weak_local": True,
    "has_any_native": True,
    "mean_weak_value_01": 0.62,
    "mean_native_value_01": 0.82,
}
```

- [ ] **Step 3: Implement deterministic group-level split assignment**

Add:

```python
def assign_group_splits(
    group_rows: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.2,
) -> list[dict[str, Any]]:
    output_rows = []
    benchmark_to_groups = {}
    for row in group_rows:
        benchmark_to_groups.setdefault(row["benchmark"], []).append(row)
    for benchmark, rows in benchmark_to_groups.items():
        ordered = sorted(rows, key=lambda item: item["group_id"])
        validation_count = max(1, round(len(ordered) * validation_fraction)) if len(ordered) >= 3 else 0
        for index, row in enumerate(ordered):
            split = "validation" if index >= len(ordered) - validation_count else "train"
            output_rows.append({**row, "split": split})
    return sorted(output_rows, key=lambda item: item["group_id"])
```

Use benchmark-local deterministic assignment:

```python
benchmark_to_groups = {}
for row in group_rows:
    benchmark_to_groups.setdefault(row["benchmark"], []).append(row)

for benchmark, rows in benchmark_to_groups.items():
    ordered = sorted(rows, key=lambda item: item["group_id"])
    validation_count = max(1, round(len(ordered) * validation_fraction)) if len(ordered) >= 3 else 0
```

Assign:

```python
split = "validation" if index >= len(ordered) - validation_count else "train"
```

- [ ] **Step 4: Implement label packaging and normalized targets**

Add:

```python
def build_critic_dataset_rows(
    manifest_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    split_by_group = {row["group_id"]: row for row in split_rows}
    manifest_by_run = {row["run_dir"]: row for row in manifest_rows}
    run_rows_by_group = {}
    for row in manifest_rows:
        run_rows_by_group.setdefault(make_group_id(row), []).append(row["run_dir"])
    for run_dirs in run_rows_by_group.values():
        run_dirs.sort()
    critic_rows = []
    for transition_row in transition_rows:
        manifest_row = manifest_by_run[transition_row["run_dir"]]
        group_id = make_group_id(manifest_row)
        split_row = split_by_group[group_id]
        ordered_run_dirs = run_rows_by_group[group_id]
        critic_rows.append(
            {
                **transition_row,
                "group_id": group_id,
                "split": split_row["split"],
                "group_run_count": len(ordered_run_dirs),
                "group_run_index": ordered_run_dirs.index(manifest_row["run_dir"]),
                "weak_local": weak_local,
                "native": native,
                "label_availability": label_availability,
                "targets": targets,
            }
        )
    return critic_rows
```

For each transition row, attach:

```python
weak_local = {
    "available": bool(manifest_row.get("has_local_eval", False)),
    "overall_10": manifest_row.get("final_local_overall"),
    "overall_01": _normalize_10_to_01(manifest_row.get("final_local_overall")),
    "benchmark_alignment_10": manifest_row.get("final_local_alignment"),
    "benchmark_alignment_01": _normalize_10_to_01(manifest_row.get("final_local_alignment")),
    "category_scores": dict(manifest_row.get("local_category_scores", {}) or {}),
}
native = {
    "available": bool(manifest_row.get("has_native_eval", False)),
    "benchmark": manifest_row.get("benchmark"),
    "average_10": manifest_row.get("final_native_average"),
    "average_01": _normalize_10_to_01(manifest_row.get("final_native_average")),
    "metrics": dict(manifest_row.get("native_metric_map", {}) or {}),
}
label_availability = {
    "has_weak_local": weak_local["available"],
    "has_native": native["available"],
    "has_native_average": native["average_10"] is not None,
}
targets = {
    "weak_value_01": weak_local["overall_01"],
    "native_value_01": native["average_01"],
}
```

Also compute:

```python
group_run_count
group_run_index
group_id
split
```

- [ ] **Step 5: Implement schema and dataset statistics writers**

Add:

```python
def build_label_schema() -> dict[str, Any]:
    return {
        "weak_local": {
            "overall_10": "float|null",
            "overall_01": "float|null",
            "benchmark_alignment_10": "float|null",
            "benchmark_alignment_01": "float|null",
            "category_scores": "dict[str, float]",
        },
        "native": {
            "benchmark": "str|null",
            "average_10": "float|null",
            "average_01": "float|null",
            "metrics": "dict[str, dict[str, object]]",
        },
        "targets": {
            "weak_value_01": "float|null",
            "native_value_01": "float|null",
        },
    }


def build_dataset_stats(
    critic_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    group_count = len(split_rows)
    train_group_count = sum(1 for row in split_rows if row["split"] == "train")
    validation_group_count = sum(1 for row in split_rows if row["split"] == "validation")
    train_transition_count = sum(1 for row in critic_rows if row["split"] == "train")
    validation_transition_count = sum(1 for row in critic_rows if row["split"] == "validation")
    benchmark_group_counts = {}
    benchmark_transition_counts = {}
    for row in split_rows:
        benchmark_group_counts[row["benchmark"]] = benchmark_group_counts.get(row["benchmark"], 0) + 1
    for row in critic_rows:
        benchmark_transition_counts[row["benchmark"]] = benchmark_transition_counts.get(row["benchmark"], 0) + 1
    return {
        "run_count": len({row["run_dir"] for row in critic_rows}),
        "transition_count": len(critic_rows),
        "group_count": group_count,
        "train_group_count": train_group_count,
        "validation_group_count": validation_group_count,
        "train_transition_count": train_transition_count,
        "validation_transition_count": validation_transition_count,
        "benchmark_group_counts": benchmark_group_counts,
        "benchmark_transition_counts": benchmark_transition_counts,
        "label_coverage": {
            "weak_local_fraction": sum(1 for row in critic_rows if row["label_availability"]["has_weak_local"]) / len(critic_rows),
            "native_fraction": sum(1 for row in critic_rows if row["label_availability"]["has_native"]) / len(critic_rows),
            "native_average_fraction": sum(1 for row in critic_rows if row["label_availability"]["has_native_average"]) / len(critic_rows),
        },
        "duplicate_burden": {
            "mean_runs_per_group": sum(row["run_count"] for row in split_rows) / len(split_rows),
            "max_runs_per_group": max(row["run_count"] for row in split_rows),
        },
    }
```

Statistics must include:

```python
{
    "run_count": 3,
    "transition_count": 4,
    "group_count": 2,
    "train_group_count": 2,
    "validation_group_count": 0,
    "train_transition_count": 4,
    "validation_transition_count": 0,
    "benchmark_group_counts": {"AI_Idea_Bench_2025": 1, "liveideabench": 1},
    "benchmark_transition_counts": {"AI_Idea_Bench_2025": 3, "liveideabench": 1},
    "label_coverage": {"weak_local_fraction": 1.0, "native_fraction": 0.75, "native_average_fraction": 0.75},
    "duplicate_burden": {"mean_runs_per_group": 1.5, "max_runs_per_group": 2},
}
```

- [ ] **Step 6: Add the top-level build function**

Add:

```python
@dataclass(frozen=True)
class CriticDatasetBuildResult:
    dataset_dir: Path
    group_count: int
    transition_count: int


def build_graph_critic_dataset(
    *,
    g1_dataset_dir: Path,
    output_dir: Path,
    dataset_name: str,
    validation_fraction: float = 0.2,
) -> CriticDatasetBuildResult:
    manifest_rows, transition_rows = load_g1_dataset(g1_dataset_dir)
    group_rows = build_group_manifest(manifest_rows, transition_rows)
    split_rows = assign_group_splits(group_rows, validation_fraction=validation_fraction)
    critic_rows = build_critic_dataset_rows(manifest_rows, transition_rows, split_rows)
    stats = build_dataset_stats(critic_rows, split_rows)
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_text_file(dataset_dir / "critic_dataset.jsonl", _jsonl_lines(critic_rows))
    write_text_file(dataset_dir / "split_manifest.jsonl", _jsonl_lines(split_rows))
    write_text_file(dataset_dir / "label_schema.json", json.dumps(build_label_schema(), indent=2))
    write_text_file(dataset_dir / "dataset_stats.json", json.dumps(stats, indent=2))
    write_text_file(dataset_dir / "README.md", _readme_text())
    return CriticDatasetBuildResult(dataset_dir=dataset_dir, group_count=len(split_rows), transition_count=len(critic_rows))
```

Write:

```python
write_text_file(dataset_dir / "critic_dataset.jsonl", _jsonl_lines(critic_rows))
write_text_file(dataset_dir / "split_manifest.jsonl", _jsonl_lines(split_rows))
write_text_file(dataset_dir / "label_schema.json", json.dumps(build_label_schema(), indent=2))
write_text_file(dataset_dir / "dataset_stats.json", json.dumps(stats, indent=2))
write_text_file(dataset_dir / "README.md", _readme_text())
```

- [ ] **Step 7: Run the G2 tests**

Run: `python -m pytest tests/test_critic_dataset.py -q`
Expected: PASS

### Task 4: Add the thin G2 builder CLI and run smoke checks

**Files:**
- Create: `scripts/build_graph_critic_dataset.py`

- [ ] **Step 1: Implement CLI argument parsing**

Add:

```python
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

with parser:

```python
parser.add_argument("--g1-dataset-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "graph_critic_datasets")
parser.add_argument("--dataset-name", type=str, required=True)
parser.add_argument("--validation-fraction", type=float, default=0.2)
```

- [ ] **Step 2: Call the library builder and print the result**

```python
from idea_graph.critic_dataset import build_graph_critic_dataset

result = build_graph_critic_dataset(
    g1_dataset_dir=args.g1_dataset_dir,
    output_dir=args.output_dir,
    dataset_name=args.dataset_name,
    validation_fraction=args.validation_fraction,
)
print(f"Dataset directory: {result.dataset_dir}")
print(f"Group count: {result.group_count}")
print(f"Transition count: {result.transition_count}")
```

- [ ] **Step 3: Re-run both focused test files**

Run: `python -m pytest tests/test_trajectory_dataset.py tests/test_critic_dataset.py -q`
Expected: PASS

- [ ] **Step 4: Refresh the small G1 smoke dataset after the manifest-enrichment patch**

Run: `python scripts/export_graph_critic_dataset.py --input-root outputs/m2_aiib_r009_diagnosis_safe_grounding --output-dir outputs/graph_critic_datasets --dataset-name smoke_r009_safe_grounding --limit-runs 4`
Expected: exit code `0` and refreshed `run_manifest.jsonl` containing `local_category_scores` and `native_metric_map`

- [ ] **Step 5: Run the first G2 smoke build**

Run: `python scripts/build_graph_critic_dataset.py --g1-dataset-dir outputs/graph_critic_datasets/smoke_r009_safe_grounding --output-dir outputs/graph_critic_datasets --dataset-name smoke_r009_safe_grounding_g2`
Expected: exit code `0` and files:

- `outputs/graph_critic_datasets/smoke_r009_safe_grounding_g2/critic_dataset.jsonl`
- `outputs/graph_critic_datasets/smoke_r009_safe_grounding_g2/split_manifest.jsonl`
- `outputs/graph_critic_datasets/smoke_r009_safe_grounding_g2/label_schema.json`
- `outputs/graph_critic_datasets/smoke_r009_safe_grounding_g2/dataset_stats.json`
- `outputs/graph_critic_datasets/smoke_r009_safe_grounding_g2/README.md`

### Task 5: Record G2 build status in active docs

**Files:**
- Modify: `docs/paper_experiment_tracker.md`
- Modify: `docs/experiment_execution_log.md`

- [ ] **Step 1: Update the tracker milestone rows**

Add concrete progress for:

```md
| `G002` | `G1` | trajectory export | saved EIG runs | small `M1` + `R009` pilots | MUST | DONE | enriched manifest now carries full local/native label payloads for `G2` |
| `G003` | `G2` | critic dataset construction | exported trajectories | train/validation by benchmark instance | MUST | IN PROGRESS | split-ready dataset built from `smoke_r009_safe_grounding` and ready for `G3` |
```

- [ ] **Step 2: Record the first G2 smoke profile**

Add lines like:

```md
- unique groups: `4`
- train groups: `3`
- validation groups: `1`
- transitions: `100`
- weak-label coverage: `1.0`
- native-label coverage: `1.0`
```

- [ ] **Step 3: Run final focused verification**

Run: `python -m pytest tests/test_trajectory_dataset.py tests/test_critic_dataset.py -q`
Expected: PASS

Run: `git status --short`
Expected: only the intended G1/G2 code, tests, scripts, and doc files are changed, with unrelated pre-existing work preserved
