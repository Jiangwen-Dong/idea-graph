# Evaluation

This repository now includes a local deterministic evaluator for generated
research ideas.

## Design Principle

The rubric combines two common perspectives from recent idea-generation work:

- `expert_style_quality`
  Inspired by human-review style rubrics used in recent ICLR-era work on LLM
  ideation, especially novelty, significance/excitement, feasibility, and
  effectiveness.
- `benchmark_alignment`
  Inspired by AI Idea Bench 2025, which evaluates agreement with held-out
  ground-truth papers together with judgment against general reference
  material.
- `graph_process`
  A process-oriented view that measures whether the multi-agent collaboration
  graph actually matured before synthesis.

## Metrics

All available metrics are reported on a `0-10` scale.

- `Novelty`
  Penalizes high lexical similarity to nearby reference papers and explicit
  `overlaps_prior` edges.
- `Significance`
  Rewards a clear important problem and a non-empty expected-contribution
  section.
- `Feasibility`
  Uses contradiction resolution, support coverage, repair balance, and the
  specificity of the experiment plan.
- `Effectiveness And Testability`
  Checks whether the idea forms a coherent problem-hypothesis-method-evaluation
  chain with concrete experimental hooks.
- `Clarity And Coherence`
  Rewards complete, sufficiently detailed, and non-redundant proposal sections.
- `Literature Grounding`
  Measures graph evidence use, literature-aware existing-method discussion, and
  grounded experiment design.
- `Topic Alignment`
  Measures whether the proposal stays aligned with the benchmark topic and its
  keywords.
- `Ground-Truth Concordance`
  When target-paper metadata is available, compares the generated idea against
  held-out motivation and method summaries.
- `Experiment Alignment`
  Checks whether the proposal uses benchmark datasets and metrics when they are
  available.
- `Graph Maturity`
  Uses support coverage, contradiction repair, completeness, utility, and
  action diversity from the collaboration process.

## Outputs

Each pipeline run now writes:

- `evaluation.json`
- `evaluation.md`

The run `summary.json` also includes an `idea_evaluation` field.

## Re-evaluate An Existing Run

```bash
python scripts/evaluate_run.py --run-dir outputs/<timestamp>-<instance>
```

This regenerates `evaluation.json` and `evaluation.md` from the saved
`graph.json`.
