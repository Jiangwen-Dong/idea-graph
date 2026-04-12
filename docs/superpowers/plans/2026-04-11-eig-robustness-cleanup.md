# EIG Robustness Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ours-eig` more benchmark-faithful and robust on `AI_Idea_Bench_2025` by fixing benchmark-safe grounding, improving the transparent utility and maturity controller, and tightening mature-subgraph synthesis before the full `R009` run.

**Architecture:** Keep the controller heuristic and interpretable. First remove any benchmark-fidelity risk in grounding, then revise graph scoring and mature-subgraph selection so the chosen chain better matches paper-ready scientific structure, and finally make synthesis slot-aware so strong graph states become stronger final proposals.

**Tech Stack:** Python, pytest, existing `idea_graph` controller/synthesis pipeline, saved `AI_Idea_Bench_2025` smoke artifacts for diagnosis.

---

### Task 1: Make Benchmark-Mode Grounding Strictly Safe

**Files:**
- Modify: `src/idea_graph/literature_grounding.py`
- Modify: `src/idea_graph/agent_backend.py`
- Test: `tests/test_literature_grounding.py`
- Test: `tests/test_agent_backend.py`

- [ ] **Step 1: Write the failing literature-grounding tests**

Add tests that enforce two invariants:

```python
def test_safe_grounding_does_not_recover_hidden_target_benchmark_fields():
    safe_metadata = {
        "benchmark": "AI_Idea_Bench_2025",
        "paper_grounding": {
            "reference_paper_snippets": [
                {
                    "resolved_title": "SeeClick",
                    "abstract": "Visual GUI grounding for cross-platform interaction.",
                    "method": "Use screenshot-grounded interaction instead of structured text.",
                    "evaluation": "Evaluate on OSWorld and report success rate and error rate.",
                }
            ]
        },
        "benchmark_input_packet": {
            "benchmark": "AI_Idea_Bench_2025",
            "topic": "The topic of this paper is improving GUI grounding and OOD generalization for GUI agents.",
            "reference_packet": [
                {"title": "SeeClick", "snippet": "Visual grounding for GUI agents."}
            ],
        },
    }

    grounding = build_literature_grounding(literature=["SeeClick"], metadata=safe_metadata)

    assert grounding.target_paper == ""
    assert all("os atlas" not in item.casefold() for item in grounding.design_highlights)
    assert all("13 million" not in item.casefold() for item in grounding.dataset_items)


def test_safe_grounding_filters_noisy_reference_fragments_as_fake_datasets_or_metrics():
    safe_metadata = {
        "benchmark": "AI_Idea_Bench_2025",
        "paper_grounding": {
            "reference_paper_snippets": [
                {
                    "resolved_title": "SeeClick",
                    "abstract": "This innovative approach bypasses structured text and adapts to GUI platforms.",
                    "method": "This innovative approach bypasses structured text and adapts to GUI platforms.",
                    "evaluation": "This innovative approach bypasses structured text and adapts to GUI platforms.",
                }
            ]
        },
    }

    grounding = build_literature_grounding(literature=["SeeClick"], metadata=safe_metadata)

    assert grounding.dataset_items == []
    assert grounding.metric_items == []
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run:

```powershell
python -m pytest tests/test_literature_grounding.py tests/test_agent_backend.py -k "safe_grounding or noisy_reference" -v
```

Expected:

- at least one failure showing the current safe path still produces noisy
  dataset or metric items or allows unsafe recovery behavior

- [ ] **Step 3: Tighten `build_literature_grounding(...)` for prompt-safe metadata**

Implement the minimal logic needed to distinguish:

- structured benchmark metadata allowed only in internal evaluation or
  development contexts
- prompt-safe snippet-derived metadata allowed in generation

Required code direction:

```python
def _is_prompt_safe_metadata(metadata: dict[str, Any]) -> bool:
    return not any(
        key in metadata
        for key in ("raw_record", "target_paper", "method_summary", "motivation")
    )


def _datasets_text(metadata: dict[str, Any]) -> str:
    if not _is_prompt_safe_metadata(metadata):
        ...
    return _reference_snippet_signal_text(
        metadata,
        preferred_fields=("evaluation",),
    )
```

And add stronger filters so generic method or abstract fragments cannot become
fake datasets or metrics.

- [ ] **Step 4: Make postprocess grounding safe in benchmark mode**

Update `_postprocess_grounding(...)` so benchmark-mode generation never falls
back to hidden-target metadata when the stored grounding is absent or empty.

Target behavior:

```python
def _postprocess_grounding(graph: IdeaGraph) -> LiteratureGrounding:
    stored = graph.metadata.get("literature_grounding", {})
    if isinstance(stored, dict) and stored:
        ...
    if graph.metadata.get("benchmark_mode"):
        safe_metadata = _prompt_safe_metadata(graph.metadata)
        return build_literature_grounding(literature=graph.literature, metadata=safe_metadata)
    return build_literature_grounding(literature=graph.literature, metadata=graph.metadata)
```

- [ ] **Step 5: Re-run the targeted tests**

Run:

```powershell
python -m pytest tests/test_literature_grounding.py tests/test_agent_backend.py -k "safe_grounding or noisy_reference" -v
```

Expected:

- PASS

- [ ] **Step 6: Commit**

```powershell
git add src/idea_graph/literature_grounding.py src/idea_graph/agent_backend.py tests/test_literature_grounding.py tests/test_agent_backend.py
git commit -m "fix: make benchmark grounding safe and less noisy"
```

### Task 2: Revise Utility To Reward Benchmark-Facing Scientific Structure

**Files:**
- Modify: `src/idea_graph/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing utility tests**

Add tests that check:

```python
def test_utility_prefers_specific_method_and_evaluation_over_generic_chain():
    ...
    assert specific_breakdown.total > generic_breakdown.total


def test_utility_penalizes_reference_copy_collapse():
    ...
    assert copied_breakdown.total < original_breakdown.total
```

Use small synthetic graphs where one branch has:

- concrete method text
- concrete eval plan with datasets/metrics

and another branch has:

- generic method text
- generic evaluation text

- [ ] **Step 2: Run the targeted utility tests and confirm failure**

Run:

```powershell
python -m pytest tests/test_engine.py -k "utility_prefers_specific or utility_penalizes_reference_copy" -v
```

Expected:

- FAIL because the current utility is mostly graph-internal

- [ ] **Step 3: Add transparent benchmark-facing utility terms**

Extend `utility_breakdown(...)` with transparent additional factors such as:

```python
specificity = _benchmark_specificity_score(active_nodes)
experiment_alignment = _experiment_method_alignment_score(active_nodes, relevant_edges)
reference_copy_penalty = _reference_copy_penalty(graph, active_nodes)
role_balance = _role_slot_balance_score(active_nodes)
```

Blend them conservatively into the final total rather than replacing the
existing factors.

- [ ] **Step 4: Surface the new utility factors in saved outputs**

Ensure the expanded breakdown is still serialized in run summaries and graph
artifacts, so diagnosis remains possible after reruns.

- [ ] **Step 5: Re-run the targeted engine tests**

Run:

```powershell
python -m pytest tests/test_engine.py -k "utility_prefers_specific or utility_penalizes_reference_copy" -v
```

Expected:

- PASS

- [ ] **Step 6: Commit**

```powershell
git add src/idea_graph/engine.py tests/test_engine.py
git commit -m "feat: add benchmark-aware utility signals for eig"
```

### Task 3: Improve Mature-Subgraph And Maturity Selection

**Files:**
- Modify: `src/idea_graph/claim_chain.py`
- Modify: `src/idea_graph/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write failing claim-chain tests**

Add tests for:

```python
def test_claim_chain_prefers_concrete_method_over_generic_hypothesis_when_available():
    ...
    assert selected["slots"]["mechanism"] == concrete_method.id


def test_claim_chain_prefers_evaluation_node_with_visible_benchmark_anchors():
    ...
    assert selected["slots"]["evaluation"] == grounded_eval.id
```

- [ ] **Step 2: Run the targeted claim-chain tests and confirm failure**

Run:

```powershell
python -m pytest tests/test_engine.py -k "claim_chain_prefers_concrete_method or claim_chain_prefers_evaluation_node" -v
```

Expected:

- FAIL because the current slot selection is not specific enough

- [ ] **Step 3: Add role-aware and specificity-aware slot scoring**

Revise `select_claim_chain(...)` and its node scoring so:

- mechanism slots prefer `Method` over `Hypothesis` when specificity is higher
- evaluation slots prefer visible datasets, metrics, baselines, or ablations
- the selected slots align with the intended functional roles when possible

- [ ] **Step 4: Tighten maturity to reflect synthesis readiness**

Update `_compute_maturity_snapshot(...)` so maturity depends not only on slot
coverage and support, but also on:

- concrete mechanism quality
- concrete evaluation quality
- absence of generic selected slots

- [ ] **Step 5: Re-run the targeted maturity tests**

Run:

```powershell
python -m pytest tests/test_engine.py -k "claim_chain_prefers_concrete_method or claim_chain_prefers_evaluation_node" -v
```

Expected:

- PASS

- [ ] **Step 6: Commit**

```powershell
git add src/idea_graph/claim_chain.py src/idea_graph/engine.py tests/test_engine.py
git commit -m "feat: make mature subgraph selection more benchmark-ready"
```

### Task 4: Make Final Synthesis Slot-Wise And Less Stitched

**Files:**
- Modify: `src/idea_graph/agent_backend.py`
- Test: `tests/test_agent_backend.py`

- [ ] **Step 1: Write failing synthesis tests**

Add tests for:

```python
def test_synthesis_payload_exposes_role_aligned_slot_summary():
    payload = json.loads(_synthesis_user_prompt(graph, subgraph))
    assert "slot_summary" in payload
    assert "problem" in payload["slot_summary"]
    assert "mechanism" in payload["slot_summary"]
    assert "evaluation" in payload["slot_summary"]


def test_postprocess_final_proposal_rewrites_fragmentary_evaluation_into_one_coherent_plan():
    proposal = FinalProposal(
        ...,
        evaluation="Evaluate on OSWorld. Report success rate. Compare against SeeClick."
    )
    cleaned = _postprocess_final_proposal(graph, proposal)
    assert "OSWorld" in cleaned.evaluation
    assert "success rate" in cleaned.evaluation
    assert cleaned.evaluation.count("Evaluate on") <= 1
```

- [ ] **Step 2: Run the targeted synthesis tests and confirm failure**

Run:

```powershell
python -m pytest tests/test_agent_backend.py -k "slot_summary or coherent_plan" -v
```

Expected:

- FAIL because the current synthesis payload is still mostly node-bag based

- [ ] **Step 3: Add a structured slot summary to synthesis payloads**

Extend `_synthesis_user_prompt(...)` to include a compact slot summary derived
from the selected claim chain.

Suggested shape:

```python
"slot_summary": {
    "problem": {...},
    "gap": {...},
    "mechanism": {...},
    "evaluation": {...},
    "caveat": {...},
}
```

- [ ] **Step 4: Tighten postprocess cleanup of evaluation prose**

Revise `_postprocess_final_proposal(...)` so benchmark-mode evaluation text is
assembled as one coherent paragraph instead of as stitched sentence fragments.

- [ ] **Step 5: Re-run the targeted synthesis tests**

Run:

```powershell
python -m pytest tests/test_agent_backend.py -k "slot_summary or coherent_plan" -v
```

Expected:

- PASS

- [ ] **Step 6: Commit**

```powershell
git add src/idea_graph/agent_backend.py tests/test_agent_backend.py
git commit -m "feat: make eig synthesis slot-aware and less stitched"
```

### Task 5: Verify End-To-End And Prepare The Next Diagnosis Packet

**Files:**
- Modify: `docs/experiment_execution_log.md`
- Modify: `docs/paper_experiment_tracker.md`
- Modify: `docs/archive/r009_aiib_launch_plan_pre_critic.md`

- [ ] **Step 1: Run the focused regression suite**

Run:

```powershell
python -m pytest tests/test_literature_grounding.py tests/test_agent_backend.py tests/test_engine.py -v
```

Expected:

- PASS

- [ ] **Step 2: Run the narrow AIIB diagnosis rerun**

Run the agreed small EIG-focused check first on the existing smoke subset:

```powershell
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 3883 --baseline ours-eig --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --output-dir outputs/m2_aiib_r009_diagnosis --max-rounds 5
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 13 --baseline ours-eig --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --output-dir outputs/m2_aiib_r009_diagnosis --max-rounds 5
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 7909 --baseline ours-eig --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --output-dir outputs/m2_aiib_r009_diagnosis --max-rounds 5
python scripts/run_pipeline.py --benchmark ai_idea_bench_2025 --benchmark-index 9849 --baseline ours-eig --agent-backend openai-compatible --llm-config configs/openai_compatible.example.json --output-dir outputs/m2_aiib_r009_diagnosis --max-rounds 5
```

- [ ] **Step 3: Run native evaluation on the reruns**

Run:

```powershell
Get-ChildItem 'outputs/m2_aiib_r009_diagnosis' -Directory | ForEach-Object {
  python scripts/evaluate_run.py --run-dir $_.FullName --native-eval --llm-config configs/openai_compatible.example.json
}
```

- [ ] **Step 4: Update the active docs with diagnosis results**

Record:

- whether fairness was preserved
- whether `3883` improved
- whether the revised utility and maturity signals are more predictive
- whether the full `R009` slice should now launch

- [ ] **Step 5: Commit**

```powershell
git add docs/experiment_execution_log.md docs/paper_experiment_tracker.md docs/archive/r009_aiib_launch_plan_pre_critic.md
git commit -m "docs: record eig robustness cleanup verification"
```

## Self-Review

- Spec coverage:
  - benchmark-safe grounding: covered by Task 1
  - benchmark-aware utility: covered by Task 2
  - mature-subgraph and maturity cleanup: covered by Task 3
  - slot-wise synthesis cleanup: covered by Task 4
  - diagnosis rerun and documentation: covered by Task 5
- Placeholder scan:
  - no `TBD`, `TODO`, or unresolved task labels remain
- Type consistency:
  - file targets, test files, and function names match the current codebase

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-11-eig-robustness-cleanup.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
