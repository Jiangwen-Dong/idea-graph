# April 03 Ablation Pilot

This note records the first API-backed ablation pilot after stabilizing the
batch runner and paper-artifact pipeline.

## Batch

- Batch summary:
  `outputs/quality_batches/20260403-132901-april03-ablation-pilot/batch_summary.json`
- Targets:
  `AI_Idea_Bench_2025`: `13`, `18`
  `LiveIdeaBench`: `0`, `96`
- Methods:
  `ours-delayed-consensus`
  `ours-early-consensus`
  `ours-no-maturity-stop`
  `ours-no-coverage-safeguard`
  `ours-no-reference-grounding`
- Backbone:
  `qwen3-8b`
- Scoring:
  repository-local deterministic evaluator for the full pilot
  plus targeted benchmark-native judge checks on selected AI Idea Bench runs

## Local Pilot Summary

| Method | Overall | Alignment | Graph | Mean Tokens |
| --- | ---: | ---: | ---: | ---: |
| `ours-delayed-consensus` | `6.492` | `2.618` | `8.907` | `140114` |
| `ours-early-consensus` | `5.125` | `1.953` | `4.915` | `76711` |
| `ours-no-maturity-stop` | `6.520` | `2.835` | `8.995` | `168559` |
| `ours-no-coverage-safeguard` | `6.258` | `2.842` | `7.980` | `127300` |
| `ours-no-reference-grounding` | `6.547` | `2.627` | `8.932` | `113375` |

## Targeted Native AI-Idea-Bench Check

To resolve the suspicious local result for `ours-no-reference-grounding`, we ran
benchmark-native scoring on the AI Idea Bench cases `13` and `18`.

| Method | AI-13 Native | AI-18 Native | Mean Native |
| --- | ---: | ---: | ---: |
| `ours-delayed-consensus` | `8.29` | `5.71` | `7.00` |
| `ours-no-maturity-stop` | `8.86` | `5.71` | `7.29` |
| `ours-no-reference-grounding` | `5.43` | `7.14` | `6.29` |

## Main Takeaways

1. `ours-early-consensus` is clearly weaker than delayed consensus on both local
   quality and graph-process scores. This is the cleanest current evidence for
   the delayed-consensus mechanism.
2. `ours-no-maturity-stop` is not a collapse case. On this tiny pilot it is
   slightly stronger than delayed consensus on local scores and slightly better
   on the targeted AI native check, but it also uses substantially more tokens.
   The honest paper story is therefore that maturity stop is primarily a
   cost-control and stabilization mechanism, not yet a universal quality booster.
3. `ours-no-reference-grounding` looks deceptively strong under the local proxy,
   but the targeted AI native check lowers its mean native score below
   delayed consensus. This suggests that stronger benchmark-native or human
   evaluation is necessary before drawing claims about literature grounding.
4. `ours-no-coverage-safeguard` is consistently worse than delayed consensus,
   especially on graph-process quality. This matches the earlier hard-case
   failure analysis and supports keeping the completeness safeguard.

## Recommended Paper Positioning

1. Keep the main mechanism claim centered on delayed consensus vs early
   synthesis.
2. Present maturity stop as a cost-quality tradeoff knob unless a larger batch
   shows a clearer quality gain.
3. Do not claim that reference grounding is unnecessary. Instead, report that
   the local proxy is noisy on this ablation and that targeted native scoring
   still favors grounded DCIG on average for the checked AI cases.
