from __future__ import annotations

from .models import ParallelRoleRoundResult
from .parallel_role_executor import collect_parallel_role_decisions
from .role_activation import active_roles_for_round


def execute_parallel_role_round(
    graph,
    *,
    round_name,
    collaboration_backend,
    runtime_controller,
    runtime_controller_metadata,
    progress_callback,
):
    del runtime_controller, runtime_controller_metadata, progress_callback
    roles = active_roles_for_round(graph, round_name)
    return collect_parallel_role_decisions(graph, round_name, collaboration_backend, roles)


__all__ = ["ParallelRoleRoundResult", "execute_parallel_role_round"]
