# Paper Experiment Plan

**Problem**: benchmark-faithful scientific ideation with literature-grounded or benchmark-conditioned inputs  
**Method Thesis**: `Evolving Idea Graphs (EIG)` improve scientific ideation quality by maintaining an explicit idea graph and learning a graph critic that selects useful edits or commits adaptively, instead of collapsing reasoning into a single draft too early.  
**Date**: 2026-04-12

## Current Method Track

The forward method track is `EIG with a learned graph critic`.

The pre-critic heuristic controller remains important, but it is no longer the
target final method. It should be used as:

- a prototype baseline for ablation
- a source of saved graph-evolution trajectories
- a diagnosis tool for identifying useful graph features and failure cases

The full `R009` larger AIIB launch is paused until the graph-critic data and
controller plan are implemented.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1. `EIG` improves final idea quality under benchmark-faithful evaluation. | This is the main paper claim. A NeurIPS reviewer must see better benchmark outcomes, not only cleaner internal graphs. | `EIG` beats or clearly matches the strongest exact baseline on benchmark-native automatic metrics on `AI_Idea_Bench_2025`, remains competitive on `LiveIdeaBench`, and improves blind human ratings on a balanced subset. | B1, B2, B3 |
| C2. The graph mechanism is causal rather than decorative. | If gains come only from extra rounds or prompt polish, the paper contribution weakens. | Graph-critic variants should outperform text-only or heuristic controllers, and graph-process shifts should align with final quality. | B4, B5 |
| C3. Learned graph-state control improves over hand-designed maturity rules. | Reviewers may view heuristic maturity thresholds as engineered and brittle. | A graph critic should improve edit selection and commit decisions over the heuristic controller, especially under few-shot calibration and held-out benchmark cases. | B4, B5 |

## Anti-Claims To Rule Out

- The gain comes only from using more rounds or more tokens.
- The gain comes only from a stronger final writing prompt.
- The graph looks cleaner internally, but final ideas do not improve.
- Improvements come from benchmark leakage rather than better collaboration.
- The method only works on `AI_Idea_Bench_2025` and fails under the lighter `LiveIdeaBench` setting.

## Paper Storyline

- Main paper must prove:
  - `EIG` improves scientific ideation quality on the primary benchmark `AI_Idea_Bench_2025`.
  - the quality trend transfers, at least partially, to `LiveIdeaBench`.
  - the graph-specific mechanisms matter beyond plain multi-round generation.
- Appendix can support:
  - graph-process diagnostics
  - fallback and controller reliability analysis
  - round-wise maturity trajectories
  - cost and runtime analysis
  - failure-case studies
- Experiments intentionally cut from the main story:
  - proxy baselines as headline evidence
  - repository-local merged heuristic `overall_score` as the primary paper metric
  - matched-budget framing as a central claim

## Benchmark And Baseline Policy

- Primary benchmark: `AI_Idea_Bench_2025`
- Secondary benchmark: `LiveIdeaBench`
- Main baseline families:
  - `direct`
  - `self-refine`
  - `ai-researcher`
  - `virsci` only if benchmark-faithful integration becomes runnable
  - `ours-eig`
- Supporting exact baseline:
  - `scipip`, if the official pipeline remains stable under the shared benchmark-mode packet
- Development-only systems:
  - any `*-proxy` baseline
  - local deterministic evaluator as a debugging layer

## Experiment Blocks

### Block 1: Primary Automatic Evaluation on `AI_Idea_Bench_2025`

- Claim tested:
  - `C1`
- Why this block exists:
  - this is the benchmark that best matches the paper's scientific-ideation setting
- Dataset / split / task:
  - `AI_Idea_Bench_2025` benchmark-mode packet
  - use only benchmark-allowed topic and inspiration context during generation
  - run a staged slice:
    - smoke slice: `4` instances for infrastructure verification
    - core paper slice: at least `24` instances, stratified across subareas
    - extension slice: expand to `48` or more if the trend is stable and cost allows
- Compared systems:
  - `direct`
  - `self-refine`
  - `ai-researcher`
  - `ours-eig`
  - `virsci`, only if the benchmark-faithful runner passes `M0`
- Metrics:
  - benchmark-native `I2T`
  - benchmark-native `I2I`
  - benchmark-native `IMCQ`
  - benchmark-native `NA`
  - benchmark-native `FA`
  - benchmark-native `FPS`
  - `IC` on matched system pools once the core slice is complete
- Setup details:
  - same backbone family across systems where controllable
  - same benchmark packet
  - same structured output schema
  - separate cost reporting
- Success criterion:
  - `ours-eig` is best or statistically tied for best on the mean automatic profile, with visible gains on alignment-sensitive metrics rather than only on fluency-like dimensions
- Failure interpretation:
  - if `ours-eig` is only better on internal graph metrics, the paper claim is not yet supported
- Table / figure target:
  - main paper automatic-results table, benchmark subsection
- Priority:
  - `MUST-RUN`

### Block 2: Secondary Automatic Evaluation on `LiveIdeaBench`

- Claim tested:
  - `C1`
- Why this block exists:
  - checks whether the method survives a lighter-context benchmark instead of overfitting to the target-paper reconstruction setting
- Dataset / split / task:
  - `LiveIdeaBench` keyword-conditioned packet
  - staged slice:
    - smoke slice: `4` rows
    - core paper slice: at least `24` rows across distinct domains
    - extension slice: expand to `48` or more if the primary benchmark trend holds
- Compared systems:
  - same main comparison set as Block 1, except any baseline that cannot consume the shared packet cleanly should be removed explicitly rather than approximated silently
- Metrics:
  - benchmark-native `Originality`
  - benchmark-native `Feasibility`
  - benchmark-native `Fluency`
  - benchmark-native `Flexibility`
  - benchmark-native `Clarity`
  - benchmark-native `Average`
- Setup details:
  - same shared output contract and backbone family
  - no access to held-out benchmark annotations
- Success criterion:
  - `ours-eig` remains competitive and does not collapse on alignment or clarity under the lighter-context setting
- Failure interpretation:
  - if `ours-eig` only works on the literature-grounded benchmark, the cross-benchmark claim must be softened
- Table / figure target:
  - main paper automatic-results table or a paired cross-benchmark table
- Priority:
  - `MUST-RUN`

### Block 3: Shared Human Blind Review

- Claim tested:
  - `C1`
- Why this block exists:
  - benchmark-native automatic metrics alone are not enough for a scientific-ideation paper
- Dataset / split / task:
  - balanced subset from both benchmarks after automatic runs are complete
  - recommended paper subset:
    - `8` to `12` instances from `AI_Idea_Bench_2025`
    - `8` to `12` instances from `LiveIdeaBench`
- Compared systems:
  - `direct`
  - `self-refine`
  - `ai-researcher`
  - `ours-eig`
  - include `virsci` only if it is benchmark-faithful and already part of the main automatic table
- Metrics:
  - `Novelty`
  - `Significance`
  - `Feasibility`
  - `Clarity`
  - `Context Adherence`
  - `Overall`
- Setup details:
  - blind system labels
  - same output formatting
  - `3` reviewers per idea when feasible
- Success criterion:
  - `ours-eig` should improve `Overall` and at least one mechanism-relevant dimension such as `Novelty` or `Context Adherence`
- Failure interpretation:
  - if humans do not prefer `ours-eig`, the automatic gains may reflect benchmark-specific optimization rather than better ideation
- Table / figure target:
  - main paper human-evaluation table
- Priority:
  - `MUST-RUN`

### Block 4: EIG Ablation Study

- Claim tested:
  - `C2`
  - `C3`
- Why this block exists:
  - isolates whether the graph mechanisms matter
- Dataset / split / task:
  - run on a smaller but balanced subset after the main system is stable
  - recommended minimum: `6` AI Idea Bench cases plus `6` LiveIdeaBench cases
- Compared systems:
  - `ours-eig-heuristic`
  - `ours-eig-critic-text`
  - `ours-eig-critic-graph`
  - `ours-eig-critic-no-commit`
  - `ours-eig-critic-calibrated`
  - optional: `ours-eig` with flat final synthesis instead of graph-conditioned synthesis
- Metrics:
  - same benchmark-native automatic metrics as the benchmark being tested
  - supplementary graph metrics:
    - `Evidence coverage`
    - `Contradiction resolution`
    - `Claim-chain completeness`
    - `Commit round`
    - `Action diversity`
    - critic-vs-heuristic action agreement
    - premature-commit and late-commit rates
- Setup details:
  - keep prompts and backbone fixed
  - change one mechanism at a time
- Success criterion:
  - at least one core removal should reduce final quality, and the graph-process shifts should be directionally consistent with the design claim
- Failure interpretation:
  - if all ablations tie the full method, the method section likely overstates the contribution of the graph controller
- Table / figure target:
  - main paper ablation table or appendix if space is tight
- Priority:
  - `MUST-RUN`

### Block 5: Process, Reliability, and Cost Analysis

- Claim tested:
  - `C2`, indirectly
- Why this block exists:
  - validates that the graph controller behaves coherently and documents the real coordination cost
- Dataset / split / task:
  - reuse the main automatic batch
- Compared systems:
  - focus on `ours-eig`
  - compare cost against `direct`, `self-refine`, and `ai-researcher`
- Metrics:
- `Evidence coverage`
- `Contradiction resolution`
- `Claim-chain completeness`
- `Commit round`
- `Action diversity`
- graph-critic action agreement
- commit-vs-continue calibration
- fallback rate
- controller override rate
- API calls
  - token cost
  - wall-clock runtime
- Setup details:
  - report separately from main quality
  - do not merge process metrics into a headline score
- Success criterion:
  - process metrics should explain quality gains, not substitute for them
- Failure interpretation:
  - if process metrics improve without final quality gains, the controller is internally tidy but externally ineffective
- Table / figure target:
  - appendix tables plus one compact cost summary in the main paper if needed
- Priority:
  - `MUST-RUN` for appendix, `NICE-TO-HAVE` for main-text placement

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| `M0` | protocol and infrastructure sanity | `1` AI Idea Bench case and `1` LiveIdeaBench case for `ours-eig`; `1` AI Idea Bench case for `ai-researcher`; `virsci` feasibility audit | benchmark packet, output schema, and benchmark-native scorer all run end to end | low to medium API cost | exact baseline wrappers may fail on benchmark-mode inputs |
| `M1` | exact baseline smoke test | `direct`, `self-refine`, `ai-researcher`, `ours-eig` on `4 + 4` cross-benchmark slice | all main systems must finish cleanly before scaling | medium API cost | unstable wrappers or scorer failures |
| `G0` | graph-critic planning and data audit | no new expensive generation | active docs updated; saved trajectory sources identified | low | old heuristic records may be incomplete for training |
| `G1` | trajectory export | export states/actions/scores from saved runs | exported examples include graph state, action, final score, and commit labels where available | low | saved artifacts may lack alternative candidates |
| `G2` | critic dataset construction | train/validation splits from saved runs | splits are by benchmark instance, not by state, to avoid leakage | low to medium | labels may be noisy |
| `G3` | text critic baseline | train flattened-state critic | text critic trains and validates better than random action scoring | medium | weak labels may be insufficient |
| `G4` | graph critic | train graph-structured critic | graph critic beats text critic or reveals why graph structure needs more data | medium | graph encoder may not help at small data scale |
| `G5` | critic-controlled pilot | 4-case AIIB gate with critic controller | critic controller is stable before larger generation | medium API cost | premature commit or over-editing |
| `M2` | core automatic batch | main comparison set on at least `24 + 24` cross-benchmark slice | run only after `G5` stabilizes or explicitly fall back to the heuristic controller | high API cost | benchmark alignment may remain weak and weak-context rows may show higher variance |
| `M3` | baseline decision gate | either add `virsci` if benchmark-faithful or formally demote it from the main paper table | no silent proxy substitution | low engineering cost, potentially high integration cost | upstream system lacks fixed-topic entrypoint |
| `M4` | human evaluation packet | prepare anonymized outputs from the strongest systems on a balanced subset | only proceed after automatic outputs are clean and non-generic | medium human effort | rating burden and reviewer consistency |
| `M5` | ablation and appendix analyses | EIG ablations plus process, reliability, and cost analyses | keep only ablations that change reviewer belief | medium API cost | too many variants can dilute the story |
| `M6` | paper artifact refresh | regenerate tables, figures, and experiment text | final numbers must all trace back to saved run artifacts | low engineering cost | stale paper tables or mixed protocols |

## Must-Run vs Nice-to-Have

### Must-Run

- `M0` protocol sanity
- `M1` exact baseline smoke test
- `M2` core automatic batch on both benchmarks
- `M4` human blind-review subset
- `M5` core EIG ablations

### Nice-to-Have

- expansion from `24 + 24` to `48 + 48`
- exact `virsci` integration, if upstream can be patched cleanly
- exact `scipip` as an additional appendix baseline
- broader failure-case writeups and round-trajectory figures

## Compute and Data Budget

- Total estimated GPU-hours:
  - not the dominant budget item; the main cost is API generation and judge calls
- Data preparation needs:
  - verified benchmark packets for `AI_Idea_Bench_2025` and `LiveIdeaBench`
  - stable saved-output format for blind review export
- Human evaluation needs:
  - `3` reviewers per idea when feasible
  - balanced subset from both benchmarks
- Biggest bottleneck:
  - exact baseline stability plus benchmark-native judge cost

## Risks and Mitigations

- Risk:
  - `virsci` exact integration remains impossible in benchmark mode
- Mitigation:
  - formally exclude it from the main table and document the reason

- Risk:
  - `ours-eig` remains coherent but too generic at final synthesis time
- Mitigation:
  - improve benchmark-grounded synthesis prompts before scaling beyond `M1`

- Risk:
  - benchmark-native evaluation is expensive or partially unavailable
- Mitigation:
  - prioritize `AI_Idea_Bench_2025` first, cache artifacts carefully, and mark unavailable metrics honestly

- Risk:
  - human evaluation becomes the schedule bottleneck
- Mitigation:
  - prepare the packet immediately after `M2` and keep the reviewed subset compact

## Pre-Critic Pilot Review

- `M0` is complete.
- The refreshed small-`M1` packet is complete and should be treated as the current
  reference development batch:
  - `outputs/quality_batches/20260411-000159-refreshed-m1-mini-synthesis-cleanup-v2-native`
- The current evidence supports the following interpretation:
  - `ours-eig` is clearly strongest on `AI_Idea_Bench_2025`
  - `ours-eig` is competitive on `LiveIdeaBench`, but weak-context meteorology
    still shows some stochastic variance
  - `self-refine` remains the strongest single-agent baseline and should stay as
    the main non-graph comparison point
  - `ai-researcher` is useful as an exact baseline, but is not currently the
    strongest comparison system and should not determine controller design

## Immediate Next Step

Pause the larger `M2/R009` launch and start the graph-critic track.

1. `G0`:
   - finish documentation cleanup around the graph-critic method direction
   - mark pre-critic `R009` as paused, not cancelled
2. `G1`:
   - implement trajectory export from saved heuristic EIG runs
   - include graph states, actions, final scores, and available judge metadata
3. `G2`:
   - construct train/validation splits by benchmark instance
   - separate weak local labels from benchmark-native labels
4. `G3/G4`:
   - train a text critic first, then the graph critic
5. `G5`:
   - run the same 4-case AIIB gate before spending on the full `24`-case batch

## First Concrete Run Queue

1. Freeze current pre-critic outputs as pilot data.
2. Export trajectories from the small `M1` and `R009` pilot artifacts.
3. Train and evaluate a text-only critic as the first learned-control sanity
   check.
4. Train and evaluate the graph critic.
5. Rerun the 4-case AIIB gate with critic-controlled edit/commit decisions.

## Final Checklist

- [x] Main paper tables are specified
- [x] Novelty is isolated with ablations
- [x] Process metrics are separated from headline quality
- [x] Human evaluation is part of the required evidence
- [x] Nice-to-have runs are separated from must-run runs
