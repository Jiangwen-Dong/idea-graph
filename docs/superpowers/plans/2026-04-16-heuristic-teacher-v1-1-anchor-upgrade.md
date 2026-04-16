# Heuristic Teacher V1.1 Anchor Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the `parallel_graph_v2` heuristic teacher before full harvest by making commit decisions and synthesis more benchmark-anchor aware while reducing low-value parallel role calls.

**Architecture:** Keep the external benchmark I/O, replay schema, selected role decision labels, and post-round commit labels unchanged. Add a hybrid visible-anchor specificity score for benchmark maturity, a cheap role activation policy, and final synthesis postprocessing that repairs generic benchmark methods using only visible packet/grounding anchors.

**Tech Stack:** Python 3, pytest, existing `idea_graph.engine`, `idea_graph.role_activation`, `idea_graph.agent_backend`, and current benchmark artifacts.

---

## File Structure

- Modify: `tests/test_engine.py`
  Purpose: lock that AI Idea Bench graphs can mature from visible reference/method anchors even when dataset/metric metadata is sparse.
- Create: `tests/test_role_activation.py`
  Purpose: lock cheap role activation behavior and prevent all-five-role activation from remaining the only path.
- Modify: `tests/test_agent_backend.py`
  Purpose: lock benchmark final-proposal repair for generic methods using visible anchor terms.
- Modify: `src/idea_graph/engine.py`
  Purpose: replace dataset-only benchmark specificity with hybrid topic/reference/design/evaluation anchor specificity.
- Modify: `src/idea_graph/role_activation.py`
  Purpose: activate only roles with useful work, with a safety floor for early rounds and weak graphs.
- Modify: `src/idea_graph/agent_backend.py`
  Purpose: pass missing anchor information into synthesis and repair generic benchmark methods post hoc without hidden target fields.

## Tasks

- [ ] **Task 1: Add failing anchor-specific maturity test**
  Add a test where an AI Idea Bench graph has no dataset or metric metadata but contains visible reference anchors such as `Gaussian Splatting`, `LERF`, `CLIP`, and `hierarchical semantics`; after enough utility history, `maturity_snapshot(...).is_mature` should be true.

- [ ] **Task 2: Implement hybrid benchmark specificity**
  In `engine.py`, score visible benchmark anchors from topic, reference titles, reference snippets, grounding design highlights, dataset items, and metric items. Use this score in `utility_breakdown` and maturity gates.

- [ ] **Task 3: Add failing role activation tests**
  Add tests proving a mature graph does not activate all five roles, while an incomplete graph still activates the missing role owners.

- [ ] **Task 4: Implement heuristic role activation**
  In `role_activation.py`, activate roles from graph deficiencies: missing task framing, missing mechanism, weak grounding, weak evaluation, unresolved contradictions, and open risks. Keep at least two roles active and all five roles active during the first round.

- [ ] **Task 5: Add failing benchmark synthesis repair test**
  Add a test where AI Idea Bench synthesis returns a generic method even though visible design anchors are present; postprocessing should inject concrete visible anchors and not use hidden target-paper metadata.

- [ ] **Task 6: Implement synthesis anchor repair**
  In `agent_backend.py`, compute missing visible anchor terms and repair generic benchmark `method`, `hypothesis`, and `evaluation` fields using only literature grounding and benchmark packet metadata.

- [ ] **Task 7: Verify and commit**
  Run focused tests, then the broader parser/runtime suite. Commit with `feat: improve parallel heuristic teacher anchors` and push.

