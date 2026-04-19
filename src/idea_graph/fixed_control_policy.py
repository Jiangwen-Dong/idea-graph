from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

CandidateRow = Mapping[str, object]
RoundRoleKey = tuple[str, str]


@dataclass(frozen=True)
class FixedControlPolicy:
    ordered_kind_priors: Mapping[RoundRoleKey, Sequence[str]] = field(default_factory=dict)

    def choose(
        self,
        *,
        round_name: str,
        role: str,
        candidate_specs: Sequence[CandidateRow] | None = None,
        candidates: Sequence[CandidateRow] | None = None,
    ) -> dict[str, object]:
        slate = candidate_specs if candidate_specs is not None else candidates
        if slate is None:
            raise ValueError("candidate_specs or candidates must be provided.")
        candidates = slate

        if not candidates:
            raise ValueError("candidates must not be empty.")

        ordered_kinds = tuple(self.ordered_kind_priors.get((round_name, role), ()))

        for preferred_kind in ordered_kinds:
            for row in candidates:
                if str(row.get("kind", "")).strip() == preferred_kind:
                    return dict(row)

        for row in candidates:
            if str(row.get("kind", "")).strip() == "skip":
                return dict(row)

        return dict(candidates[0])


def load_fixed_control_policy(path: str | Path) -> FixedControlPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Fixed policy JSON must be an object.")

    normalized: dict[RoundRoleKey, tuple[str, ...]] = {}
    for round_name, roles_payload in payload.items():
        if not isinstance(roles_payload, Mapping):
            raise ValueError("Each round entry must be an object of role -> ordered kinds.")
        for role, ordered_kinds in roles_payload.items():
            if not isinstance(ordered_kinds, Sequence) or isinstance(ordered_kinds, (str, bytes)):
                raise ValueError("Each role entry must be a list of candidate kinds.")
            normalized[(str(round_name), str(role))] = tuple(str(kind) for kind in ordered_kinds)
    return FixedControlPolicy(ordered_kind_priors=normalized)
