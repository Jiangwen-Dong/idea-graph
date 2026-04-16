from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy


def collect_parallel_role_decisions(graph, round_name, collaboration_backend, roles):
    snapshot = deepcopy(graph)

    def _run(role: str):
        return role, collaboration_backend.choose_action(snapshot, round_name, role)

    with ThreadPoolExecutor(max_workers=max(1, len(roles))) as pool:
        return list(pool.map(_run, roles))
