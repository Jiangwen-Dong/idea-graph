from __future__ import annotations


def append_parallel_round_trace(metadata: dict[str, object], payload: dict[str, object]) -> None:
    traces = metadata.setdefault("parallel_round_traces", [])
    if isinstance(traces, list):
        traces.append(dict(payload))
