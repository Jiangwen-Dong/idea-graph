from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

CandidateRow = Mapping[str, object]


@dataclass(frozen=True)
class FixedControlPolicy:
    ordered_kind_priors: Mapping[str, Sequence[str]] = field(default_factory=dict)

    @staticmethod
    def _utility_candidates(candidates: Sequence[CandidateRow]) -> list[dict[str, object]]:
        filtered: list[dict[str, object]] = []
        for row in candidates:
            candidate_source = str(row.get("candidate_source", "")).strip()
            if candidate_source.startswith("utility_"):
                filtered.append(dict(row))
        return filtered

    @staticmethod
    def _skip_candidate(candidates: Sequence[CandidateRow]) -> dict[str, object] | None:
        for row in candidates:
            if str(row.get("kind", "")).strip() == "skip":
                return dict(row)
        return None

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

        utility_candidates = self._utility_candidates(candidates)
        ordered_kinds = tuple(self.ordered_kind_priors.get(round_name, ()))

        for preferred_kind in ordered_kinds:
            for row in utility_candidates:
                if str(row.get("kind", "")).strip() == preferred_kind:
                    return dict(row)

        if utility_candidates:
            return dict(utility_candidates[0])

        skip_candidate = self._skip_candidate(candidates)
        if skip_candidate is not None:
            return skip_candidate

        return dict(candidates[0])


def load_fixed_control_policy(path: str | Path) -> FixedControlPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Fixed policy JSON must be an object.")

    normalized: dict[str, tuple[str, ...]] = {}
    for round_name, ordered_kinds in payload.items():
        if not isinstance(ordered_kinds, Sequence) or isinstance(ordered_kinds, (str, bytes)):
            raise ValueError("Each round entry must be a list of ordered utility kinds.")
        normalized[str(round_name)] = tuple(str(kind) for kind in ordered_kinds)
    return FixedControlPolicy(ordered_kind_priors=normalized)
