# paper-eval freeze pool builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic paper_eval_candidate_pool_v2 builder and its tests so the frozen candidate list can be created with zero overlap from existing pools.

**Architecture:** Read AIIB JSON + LiveIdeaBench CSV, filter out blocked group ids gathered from the full development-only history plus prior paper-eval candidate files, and emit the new candidate list plus README/stats via a CLI wrapper script. The code can land before the online broad gate finishes, but the final pool must only be materialized after `outputs/m2_graph_critic_online_scaleup_v2/freeze_decision.md` says `go`.

**Tech Stack:** Python CLI via `argparse`, JSON/CSV parsing in `src/idea_graph/paper_eval_pool.py`, and `pytest` for TDD verification.

---

### Task 1: Candidate-selection logic and tests

**Files:**
- Create: `src/idea_graph/paper_eval_pool.py`
- Modify: `tests/test_paper_eval_freeze_pool.py`

Blocked-source defaults this builder must honor:

- `outputs/graph_critic_datasets/01_active_text_critic/current_benchmarked_ours_eig_full_g35_partitions/partition_manifest.jsonl`
- `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v2_candidate_pool_v1/candidate_instances.json`
- `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_candidate_pool_v1/candidate_instances.json`
- `outputs/graph_critic_datasets/02_active_graph_critic/development_pool_v3_combined_g2_partitions/partition_manifest.jsonl`
- `outputs/graph_critic_datasets/02_active_graph_critic/paper_eval_candidate_pool_v1/candidate_instances.json`

- [ ] **Step 1: Write the failing tests**

```python
def test_selects_aiib_and_live_candidates(tmp_path):
    aiib_json = tmp_path / "aiib.json"
    aiib_json.write_text(json.dumps([
        {"index": 1, "summary": {"topic": "Topic A"}},
        {"index": 2, "summary": {"topic": "Topic B"}},
    ]))

    live_csv = tmp_path / "live.csv"
    live_csv.write_text("keywords,idea\nalpha,idea1\nbeta,idea2\n")

    blocked = tmp_path / "blocked.json"
    blocked.write_text(json.dumps([
        {"benchmark": "AI_Idea_Bench_2025", "instance_name": "ai-idea-bench-2025-1"}
    ]))

    rows = select_paper_eval_candidates(
        aiib_metadata=aiib_json,
        live_csv=live_csv,
        target_aiib=1,
        target_live=1,
        blocked_candidate_files=[blocked],
        blocked_split_registries=[],
    )
    assert len(rows) == 2
    assert rows[0]["instance_name"] == "ai-idea-bench-2025-2"
    assert any(row["benchmark"] == "liveideabench" for row in rows)


def test_selection_errors_when_insufficient_candidates(tmp_path):
    aiib_json = tmp_path / "aiib.json"
    aiib_json.write_text(json.dumps([{"index": 5, "summary": {"topic": "Topic"}}]))
    with pytest.raises(ValueError):
        select_paper_eval_candidates(
            aiib_metadata=aiib_json,
            live_csv=tmp_path / "live.csv",
            target_aiib=2,
            target_live=0,
        )
```

These tests fail because `select_paper_eval_candidates` does not exist yet.

- [ ] **Step 2: Run the tests and verify they fail**

```bash
python -m pytest tests/test_paper_eval_freeze_pool.py::test_selects_aiib_and_live_candidates -q
```

Expected: FAIL with `ImportError` or `NameError` because `select_paper_eval_candidates` is undefined.

- [ ] **Step 3: Implement the minimal selection module**

```python
def select_paper_eval_candidates(...):
    blocked = _load_blocked_group_ids(...)
    aiib = _aiib_candidate_rows(...)
    live = _live_candidate_rows(...)
    return sorted(aiib + live, key=lambda row: (row["benchmark"], row["instance_name"]))
```

Implement helpers to read AIIB JSON rows, parse the LiveIdeaBench CSV, format `instance_name`, and join the blocked group ids via `make_group_id` from `idea_graph.critic_pool_expansion`. Each helper raises `ValueError` when it cannot hit its target, and every candidate has `status="frozen"` and `intended_role="paper_eval"`.

The helper layer must make the blocked-source defaults explicit so future pool
freezes do not silently forget older development-only pools.

- [ ] **Step 4: Run the test suite to make sure the new logic passes**

```bash
python -m pytest tests/test_paper_eval_freeze_pool.py::test_selects_aiib_and_live_candidates -q
```

Expected: PASS.

- [ ] **Step 5: Commit the selection logic**

```bash
git add src/idea_graph/paper_eval_pool.py tests/test_paper_eval_freeze_pool.py
git commit -m "feat: add paper eval selection helpers"
```

### Task 2: CLI builder + README/pool stats

**Files:**
- Create: `scripts/build_paper_eval_freeze_pool.py`
- Modify: `tests/test_paper_eval_freeze_pool.py`

- [ ] **Step 1: Write CLI-focused failing tests**

```python
def test_cli_writes_artifacts(tmp_path):
    output_root = tmp_path / "pool"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_paper_eval_freeze_pool.py",
            "--output-root",
            str(output_root),
            "--target-aiib",
            "1",
            "--target-live",
            "1",
            "--aiib-metadata",
            str(aiib_path),
            "--live-csv",
            str(live_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0  # CLI not implemented yet
```

Add another assertion in this test (after CLI works) to read `README.md` and `pool_stats.json` once the script exists, and assert the printed summary mentions candidate counts.

- [ ] **Step 2: Run the CLI test and make sure it fails**

```bash
python -m pytest tests/test_paper_eval_freeze_pool.py::test_cli_writes_artifacts -q
```

Expected: FAIL because `scripts/build_paper_eval_freeze_pool.py` is missing or argparse not set up.

- [ ] **Step 3: Implement the CLI script**

```python
def main():
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()
    rows = select_paper_eval_candidates(...)
    write_json(args.output_root / "candidate_instances.json", rows)
    write_text_file(args.output_root / "README.md", build_readme(args, rows))
    write_text_file(args.output_root / "pool_stats.json", json.dumps(build_stats(rows, args), indent=2))
    print(...)

if __name__ == "__main__":
    main()
```

Reuse `select_paper_eval_candidates` from Task 1; build helper functions to render the README (including status, selection policy, and references to blocked files) and to calculate the `pool_stats` structure described in the spec. The CLI should also print counts for debugging.

The README must state:

- this is `paper_eval_candidate_pool_v2`
- status is `frozen`
- zero overlap with all current development pools
- it was materialized only after the broad-gate freeze decision

- [ ] **Step 4: Rerun the CLI test to ensure it now passes**

```bash
python -m pytest tests/test_paper_eval_freeze_pool.py::test_cli_writes_artifacts -q
```

Expected: PASS.

- [ ] **Step 5: Commit the CLI builder**

```bash
git add scripts/build_paper_eval_freeze_pool.py tests/test_paper_eval_freeze_pool.py
git commit -m "feat: add paper eval freeze pool builder CLI"
```
