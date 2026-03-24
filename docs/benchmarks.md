# Benchmarks

This project currently supports two benchmark integrations.

## AI_Idea_Bench_2025

Source:

- GitHub: `yansheng-qiu/AI_Idea_Bench_2025`
- Hugging Face dataset: `yanshengqiu/AI_Idea_Bench_2025`

Local integration:

- metadata file: `target_paper_data.json`
- optional large archive: `Idea_bench_data.zip`
- loader module: `src/idea_graph/benchmarks/ai_idea_bench_2025.py`

Conversion strategy:

- topic comes from the benchmark topic or revised topic
- literature comes from benchmark reference titles and paper paths
- target-paper-side metadata is preserved in artifacts for inspection and evaluation
- generation prompts now use a prompt-safe benchmark view that keeps reference papers but hides oracle fields such as the target paper title, gold motivation, and gold method summary

## liveideabench

Source:

- Hugging Face dataset: `6cf/liveideabench`

Local integration:

- benchmark file: `liveideabench_hf.csv`
- loader module: `src/idea_graph/benchmarks/liveideabench.py`

Conversion strategy:

- topic comes from the benchmark keyword
- benchmark idea text is retained as metadata, not used as literature
- literature is currently a lightweight prompt scaffold because the benchmark does not ship retrieved papers

## Important Difference

`AI_Idea_Bench_2025` contains explicit paper context, so its generated seed graphs can be grounded in benchmark references.

`liveideabench` is keyword-and-score oriented, so it needs an extra retrieval step in future work to provide comparable literature grounding.
