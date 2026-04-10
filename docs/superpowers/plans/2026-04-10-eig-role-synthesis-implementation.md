# EIG Role And Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign `ours-eig` role contracts, maturity gating, and final synthesis around a validated claim chain while keeping baseline methods unchanged and preserving robustness on weak-context `LiveIdeaBench`.

**Architecture:** Keep the existing internal role identifiers for compatibility, but remap their prompt contracts to the new scientific functions. Add a focused claim-chain utility module that scores coverage for both rich-context and weak-context benchmarks, then wire engine maturity and final-subgraph selection through that module so synthesis consumes one validated chain instead of a loose connected subgraph.

**Tech Stack:** Python, pytest, existing `idea_graph` engine/backend modules, deterministic graph utilities, OpenAI-compatible LLM prompts

---

## File Structure

- Create: `src/idea_graph/claim_chain.py`
  - Responsibility: claim-slot coverage, weak-context-safe chain selection, synthesis-ready chain metadata
- Modify: `src/idea_graph/agent_backend.py`
  - Responsibility: role contract wording, prompt payloads, synthesis prompt/user payload wiring
- Modify: `src/idea_graph/engine.py`
  - Responsibility: maturity gating and final subgraph selection through claim-chain utilities
- Modify: `src/idea_graph/schema.py`
  - Responsibility: optional role display mapping or helper constants, without breaking internal role IDs
- Test: `tests/test_claim_chain.py`
  - Responsibility: direct unit tests for coverage and chain selection on AI Idea Bench and LiveIdeaBench style inputs
- Modify: `tests/test_agent_backend.py`
  - Responsibility: prompt and synthesis payload regression tests for the new role contracts and weak-context behavior
- Modify: `docs/paper_experiment_tracker.md`
  - Responsibility: record this implementation stage and the first regeneration checkpoint

### Task 1: Add Claim-Chain Utilities And Weak-Context Coverage Tests

**Files:**
- Create: `src/idea_graph/claim_chain.py`
- Create: `tests/test_claim_chain.py`
- Modify: `src/idea_graph/engine.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_select_claim_chain_prefers_complete_scientific_path():
    graph = build_small_graph_with_problem_gap_method_eval_risk()
    chain = select_claim_chain(graph)
    assert chain is not None
    assert chain["slots"]["problem"]
    assert chain["slots"]["gap"]
    assert chain["slots"]["mechanism"]
    assert chain["slots"]["evaluation"]
    assert chain["slots"]["caveat"]


def test_claim_chain_handles_liveideabench_weak_context_without_fake_literature_gap():
    graph = build_liveideabench_style_graph()
    chain = select_claim_chain(graph)
    assert chain is not None
    assert chain["weak_context_mode"] is True
    assert chain["slots"]["problem"]
    assert chain["slots"]["mechanism"]
    assert chain["slots"]["evaluation"]
    assert chain["coverage"]["is_synthesis_ready"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_claim_chain.py -v`  
Expected: FAIL with import or missing-function errors for `select_claim_chain` and coverage helpers

- [ ] **Step 3: Write the minimal claim-chain implementation**

```python
from dataclasses import dataclass


CLAIM_CHAIN_SLOT_TYPES = {
    "problem": ("Problem",),
    "gap": ("NoveltyClaim", "EvidenceNeed", "Problem"),
    "mechanism": ("Hypothesis", "Method"),
    "evaluation": ("EvalPlan",),
    "caveat": ("Risk", "Assumption", "Repair"),
}


def select_claim_chain(graph):
    weak_context_mode = bool(graph.metadata.get("benchmark_input_packet", {}).get("keyword"))
    # Choose one best node per slot, but allow weak-context graphs to satisfy
    # the gap slot from benchmark/task framing when explicit literature is sparse.
    ...
```

- [ ] **Step 4: Wire engine fallback selection through the new utility**

```python
from .claim_chain import select_claim_chain


def select_final_subgraph(graph):
    chain = select_claim_chain(graph)
    if chain is not None and chain["coverage"]["is_synthesis_ready"]:
        return chain["subgraph"]
    ...
```

- [ ] **Step 5: Run focused tests to verify they pass**

Run: `pytest tests/test_claim_chain.py -v`  
Expected: PASS for both new coverage tests

- [ ] **Step 6: Run adjacent engine regressions**

Run: `pytest tests/test_agent_backend.py -q`  
Expected: existing prompt/synthesis tests still pass

### Task 2: Redesign `ours-eig` Role Contracts In Prompt Space

**Files:**
- Modify: `src/idea_graph/agent_backend.py`
- Modify: `src/idea_graph/schema.py`
- Modify: `tests/test_agent_backend.py`

- [ ] **Step 1: Write the failing prompt-contract tests**

```python
def test_seed_prompt_uses_functional_role_contract_language():
    prompt = _seed_system_prompt("ImpactReframer")
    assert "TaskFramer" in prompt
    assert "exact benchmark task" in prompt


def test_action_prompt_requests_task_gap_mechanism_evaluation_repairs():
    prompt = _action_system_prompt(graph, "NoveltyExaminer", "Round2", allowed_actions)
    assert "LiteratureGrounder" in prompt
    assert "visible reference-based gap" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_backend.py -k "functional_role_contract or task_gap_mechanism" -v`  
Expected: FAIL because the current prompts still expose the old role semantics only

- [ ] **Step 3: Implement role-contract remapping without changing internal role IDs**

```python
ROLE_DISPLAY_NAMES = {
    "ImpactReframer": "TaskFramer",
    "NoveltyExaminer": "LiteratureGrounder",
    "MechanismProposer": "MethodArchitect",
    "EvaluationDesigner": "ExperimentDesigner",
    "FeasibilityCritic": "SkepticRepairer",
}
```

```python
ROLE_GUIDANCE = {
    "ImpactReframer": "TaskFramer: own the exact benchmark task, failure mode, and why it matters.",
    ...
}
```

- [ ] **Step 4: Add weak-context guardrails to the prompt payload**

```python
payload["weak_context_guidance"] = {
    "goal": "Stay keyword-faithful without inventing external literature.",
    "allow_gap_from_task_framing": True,
    "require_mechanism_and_evaluation_specificity": True,
}
```

- [ ] **Step 5: Run focused prompt tests to verify they pass**

Run: `pytest tests/test_agent_backend.py -k "functional_role_contract or weak_context" -v`  
Expected: PASS

- [ ] **Step 6: Run the full backend test file**

Run: `pytest tests/test_agent_backend.py -q`  
Expected: PASS with no regressions in existing salvage or synthesis checks

### Task 3: Tighten Maturity And Synthesis Around The Claim Chain

**Files:**
- Modify: `src/idea_graph/engine.py`
- Modify: `src/idea_graph/agent_backend.py`
- Modify: `tests/test_claim_chain.py`
- Modify: `tests/test_agent_backend.py`

- [ ] **Step 1: Write the failing maturity and synthesis tests**

```python
def test_maturity_requires_complete_claim_chain():
    graph = build_graph_missing_evaluation_slot()
    snapshot = maturity_snapshot(graph)
    assert snapshot.is_mature is False


def test_synthesis_payload_exposes_selected_claim_chain():
    payload = json.loads(_synthesis_user_prompt(graph, subgraph))
    assert "selected_claim_chain" in payload
    assert payload["selected_claim_chain"]["coverage"]["is_synthesis_ready"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_claim_chain.py tests/test_agent_backend.py -k "claim_chain or maturity_requires_complete_claim_chain or selected_claim_chain" -v`  
Expected: FAIL because maturity does not yet require chain coverage and synthesis does not yet expose the selected chain

- [ ] **Step 3: Implement maturity gating through claim-chain coverage**

```python
chain = select_claim_chain(graph)
has_complete_claim_chain = bool(chain and chain["coverage"]["is_synthesis_ready"])
is_mature = structural_ready and has_complete_claim_chain
```

- [ ] **Step 4: Implement synthesis from the selected claim chain**

```python
payload["selected_claim_chain"] = chain
payload["writing_target"]["must_follow_claim_chain"] = True
```

```python
if chain is not None:
    selected_node_ids = chain["subgraph"]["node_ids"]
```

- [ ] **Step 5: Preserve weak-context robustness for `LiveIdeaBench`**

```python
if weak_context_mode:
    # Do not require a named literature limitation if the packet is keyword-only.
    # Instead require a task-specific bottleneck plus one explicit mechanism and evaluation anchor.
    ...
```

- [ ] **Step 6: Run targeted tests to verify they pass**

Run: `pytest tests/test_claim_chain.py tests/test_agent_backend.py -k "claim_chain or selected_claim_chain or maturity_requires_complete_claim_chain" -v`  
Expected: PASS

- [ ] **Step 7: Run the full test suite**

Run: `pytest -q`  
Expected: PASS with the full suite green

### Task 4: Record Progress And Prepare The First Regeneration Checkpoint

**Files:**
- Modify: `docs/paper_experiment_tracker.md`

- [ ] **Step 1: Add a tracker note for this redesign stage**

```markdown
## 2026-04-10: EIG Role/Synthesis Revision

- implemented claim-chain coverage for `ours-eig`
- preserved baseline prompt families
- added weak-context-safe maturity and synthesis behavior for `LiveIdeaBench`
```

- [ ] **Step 2: Verify the tracker edit**

Run: `rg -n "EIG Role/Synthesis Revision" docs/paper_experiment_tracker.md`  
Expected: one matching entry

- [ ] **Step 3: Prepare the first small regeneration command**

Run:
`python scripts/run_quality_batch.py --llm-config configs\\openai_compatible.example.json --external-baseline-config configs\\external_baselines.qwen.json --ai-indices 13 15 --live-row-indices 0 23 --baselines direct self-refine ai-researcher ours-eig --batch-name m1-role-synthesis-check`

Expected: command is recorded, but generation should be launched only after the implementation tests are green

## Self-Review

- Spec coverage:
  - functional roles for `ours-eig`: covered by Task 2
  - claim-chain completeness: covered by Tasks 1 and 3
  - stronger maturity logic: covered by Task 3
  - synthesis from validated chain: covered by Task 3
  - weak-context `LiveIdeaBench` robustness: covered by Tasks 1, 2, and 3
  - baseline stability: preserved by architecture and task boundaries
- Placeholder scan:
  - no `TODO` or deferred implementation markers are left in the plan body
- Type consistency:
  - `select_claim_chain(...)` is introduced before being referenced by engine and synthesis tasks
