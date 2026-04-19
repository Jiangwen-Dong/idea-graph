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
        return dict(self._rng.choice(list(candidates)))
