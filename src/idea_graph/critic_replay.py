from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

TRAIN_PARTITION_ROLE = "critic_train"
REPLAY_REQUIRED_FIELDS = (
    "state_id",
    "candidate_id",
    "state_text",
    "candidate_text",
    "is_logged_selected",
    "is_commit",
    "is_commit_positive_state",
    "targets",
)


@dataclass(frozen=True)
class OnlineCriticEpisode:
    episode_id: str
    group_id: str
    partition_role: str
    benchmark: str
    run_dir: str
    final_return: float | None
    final_native_return: float | None
    transitions: tuple[Mapping[str, Any], ...]


def _validate_transition_row(row: Mapping[str, Any]) -> None:
    for field_name in REPLAY_REQUIRED_FIELDS:
        if field_name not in row:
            raise ValueError(f"Replay transition row is missing required field '{field_name}'.")


def episode_to_training_rows(episode: OnlineCriticEpisode) -> list[dict[str, Any]]:
    if episode.partition_role != TRAIN_PARTITION_ROLE:
        raise ValueError("Replay accepts only critic_train episodes.")
    rows: list[dict[str, Any]] = []
    for transition in episode.transitions:
        _validate_transition_row(transition)
        row = dict(transition)
        row["episode_id"] = episode.episode_id
        row["group_id"] = episode.group_id
        row["partition_role"] = episode.partition_role
        row["benchmark"] = episode.benchmark
        row["run_dir"] = episode.run_dir
        row["final_return"] = episode.final_return
        row["final_native_return"] = episode.final_native_return
        row["source"] = "online"
        rows.append(row)
    return rows


class CriticReplayBuffer:
    def __init__(self) -> None:
        self._episodes: list[OnlineCriticEpisode] = []

    def append_episode(self, episode: OnlineCriticEpisode) -> None:
        if episode.partition_role != TRAIN_PARTITION_ROLE:
            raise ValueError("Replay accepts only critic_train episodes.")
        for transition in episode.transitions:
            _validate_transition_row(transition)
        self._episodes.append(episode)

    def training_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for episode in self._episodes:
            rows.extend(episode_to_training_rows(episode))
        return rows


def _filter_train_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("partition_role", "")).strip() != TRAIN_PARTITION_ROLE:
            continue
        filtered.append(dict(row))
    return filtered


def _sample_rows(
    rows: Sequence[dict[str, Any]],
    *,
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return list(rows)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected_indices = sorted(indices[:count])
    return [dict(rows[index]) for index in selected_indices]


def build_mixed_training_rows(
    offline_rows: Sequence[Mapping[str, Any]],
    online_rows: Sequence[Mapping[str, Any]],
    *,
    offline_fraction: float,
    max_examples: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    if not 0.0 <= offline_fraction <= 1.0:
        raise ValueError("offline_fraction must be between 0.0 and 1.0.")
    if max_examples <= 0:
        raise ValueError("max_examples must be positive.")

    offline_train = _filter_train_rows(offline_rows)
    online_train = _filter_train_rows(online_rows)
    rng = random.Random(random_seed)

    offline_target = min(len(offline_train), int(round(max_examples * offline_fraction)))
    online_target = min(len(online_train), max_examples - offline_target)

    remaining = max_examples - offline_target - online_target
    offline_remaining = len(offline_train) - offline_target
    take_offline_extra = min(max(offline_remaining, 0), remaining)
    offline_target += take_offline_extra
    remaining -= take_offline_extra

    online_remaining = len(online_train) - online_target
    take_online_extra = min(max(online_remaining, 0), remaining)
    online_target += take_online_extra

    sampled_offline = _sample_rows(offline_train, count=offline_target, rng=rng)
    sampled_online = _sample_rows(online_train, count=online_target, rng=rng)
    return sampled_offline + sampled_online
