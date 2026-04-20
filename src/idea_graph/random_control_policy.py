from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

CandidateRow = Mapping[str, object]


@dataclass
class RandomControlPolicy:
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def choose(
        self,
        *,
        round_name: str,
        role: str,
        candidate_specs: Sequence[CandidateRow] | None = None,
        candidates: Sequence[CandidateRow] | None = None,
    ) -> dict[str, object]:
        del round_name
        del role

        slate = candidate_specs if candidate_specs is not None else candidates
        if slate is None:
            raise ValueError("candidate_specs or candidates must be provided.")
        candidates = slate

        if not candidates:
            raise ValueError("candidates must not be empty.")

        utility_candidates = [
            dict(row)
            for row in candidates
            if str(row.get("candidate_source", "")).strip().startswith("utility_")
        ]
        if utility_candidates:
            return dict(self._rng.choice(utility_candidates))

        for row in candidates:
            if str(row.get("kind", "")).strip() == "skip":
                return dict(row)

        return dict(candidates[0])
