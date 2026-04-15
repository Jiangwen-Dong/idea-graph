# LiveIdeaBench Transfer And Calibration Design

**Date:** 2026-04-15  
**Scope:** diagnose the graph-controller transfer gap after the completed
59-group broad gate and define the next safe improvement step.

## Current Evidence

The broad online graph-controller gate is complete:

- merged root:
  `outputs/m2_graph_critic_online_scaleup_v2_merged118`
- paired groups:
  `59`
- held-out `critic_dev` delta:
  `+0.0192`
- pooled delta:
  `-0.1156`
- decision:
  `NO-GO` for paper-eval promotion right now

`LiveIdeaBench` is not a total failure, but it is the transfer risk:

- all `20` `LiveIdeaBench` groups:
  - `ours-eig`: `7.4530`
  - `ours-eig-graph-critic`: `7.4840`
  - delta: `+0.0310`
- held-out `LiveIdeaBench critic_dev` (`6` groups):
  - `ours-eig`: `7.4633`
  - `ours-eig-graph-critic`: `7.3600`
  - delta: `-0.1033`

The graph-controller variant is therefore competitive overall, but not yet
stable enough on the clean held-out transfer slice.

## Important Runtime Fact

The current `ours-eig-graph-critic` runtime keeps learned live `commit`
disabled:

- `runtime_controller_use_commit = False`

Therefore, the completed broad gate did **not** test a learned optimal-stopping
controller. It tested graph-critic edit reranking under the existing heuristic
maturity stop.

This matters for calibration:

- calibrating live `commit` now would be premature
- the immediate calibration target should be weak-context edit override policy
  and the maturity interaction after graph-critic overrides

## Diagnosis

The likely transfer gap is the interaction of three factors:

1. `LiveIdeaBench` provides keyword-only weak context, so graph states can look
   structurally mature before the proposal is semantically specific enough.
2. The graph critic can choose a locally plausible edit that changes the graph
   trajectory without adding enough new grounding signal.
3. The unchanged heuristic maturity stop can then stop earlier than is helpful.

Observed stop pattern:

- `ours-eig` on all `LiveIdeaBench` groups:
  - `mature_at_Round3 = 10`
  - `mature_at_Round4 = 8`
  - `mature_at_Round5 = 1`
  - `max_rounds_reached = 1`
- `ours-eig-graph-critic` on all `LiveIdeaBench` groups:
  - `mature_at_Round3 = 13`
  - `mature_at_Round4 = 5`
  - `mature_at_Round5 = 2`

This suggests that the next patch should reduce premature weak-context
maturity, not simply increase critic authority.

## Approaches

### Option A: Immediate Live Commit Calibration

Enable learned `commit` and tune `tau_commit` / `gamma_commit` on the
development pool.

Pros:

- closest to the long-term controller story
- directly addresses optimal stopping

Cons:

- the current online runs did not exercise live `commit`
- premature stopping is already the concern
- a weakly calibrated commit head could worsen transfer

Decision:

- do not use this as the next step

### Option B: Recommended Weak-Context Controller Calibration

Keep live `commit` disabled and calibrate only the edit override policy plus
the maturity interaction under weak-context inputs.

Candidate patch shape:

- when context is keyword-only or reference-light, require stronger evidence or
  support gain before accepting a critic override that can make the graph
  mature
- add a one-round maturity confirmation rule after critic-selected overrides in
  weak-context mode
- keep override thresholds frozen from development data, not tuned on final
  paper-eval cases
- keep shadow-commit logging active for later calibration analysis

Pros:

- targets the observed failure mode
- low risk relative to enabling live commit
- preserves a clean paper story: calibration controls uncertainty under weak
  context rather than tuning to a benchmark name

Cons:

- still relies on heuristic maturity for stopping
- may reduce critic authority if too conservative

Decision:

- recommended next implementation

### Option C: More Robustness Data Before Any Patch

Collect additional `LiveIdeaBench` development rows and retrain before changing
runtime policy.

Pros:

- may improve transfer without extra hand policy
- useful if current held-out slice is too noisy

Cons:

- broad gate already includes `20` `LiveIdeaBench` groups
- current symptom is policy/materialization, not only offline ranking
- delays the next actionable diagnosis

Decision:

- use only if Option B still fails

## Recommended Next Experiment

Implement Option B as a small, auditable runtime patch, then run:

1. `LiveIdeaBench` transfer packet:
   - all `20` `LiveIdeaBench` groups from the broad gate, if budget allows
   - at minimum the `6` held-out `critic_dev` rows plus the worst negative
     train-side rows
2. Held-out `critic_dev` sanity:
   - all `12` current held-out development groups
3. Compare three systems:
   - `ours-eig`
   - current `ours-eig-graph-critic`
   - patched `ours-eig-graph-critic`

Promotion signal:

- `LiveIdeaBench critic_dev` should be non-negative or clearly improved
- pooled `LiveIdeaBench` should not become worse
- stop behavior should shift away from systematic earlier `Round3` maturity
- controller override rate should remain non-zero so the critic is not silently
  disabled

## Paper Framing

If this succeeds, frame it as:

> Development-only weak-context calibration for safe graph-controller
> materialization.

Do not frame it as:

> Per-benchmark tuning or online test-time adaptation.

The calibration must be frozen before any untouched `paper_eval` launch.
