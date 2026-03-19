from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional


NUM_PERIODS = 8
LINEUP_SIZE = 5
SUPPORTED_PLAYER_COUNTS = tuple(range(5, 13))


SNAKE_SLOT_PERIODS: Dict[int, Dict[int, List[int]]] = {
    5: {
        1: [1, 2, 3, 4, 5, 6, 7, 8],
        2: [1, 2, 3, 4, 5, 6, 7, 8],
        3: [1, 2, 3, 4, 5, 6, 7, 8],
        4: [1, 2, 3, 4, 5, 6, 7, 8],
        5: [1, 2, 3, 4, 5, 6, 7, 8],
    },
    6: {
        1: [1, 2, 3, 4, 5, 7, 8],
        2: [1, 2, 3, 4, 6, 7, 8],
        3: [1, 2, 3, 5, 6, 7, 8],
        4: [1, 2, 4, 5, 6, 7, 8],
        5: [1, 3, 4, 5, 6, 7],
        6: [2, 3, 4, 5, 6, 8],
    },
    7: {
        1: [1, 2, 3, 5, 6, 8],
        2: [1, 2, 4, 5, 6, 8],
        3: [1, 2, 4, 5, 7, 8],
        4: [1, 3, 4, 5, 7, 8],
        5: [1, 3, 4, 6, 7, 8],
        6: [2, 3, 4, 6, 7],
        7: [2, 3, 5, 6, 7],
    },
    8: {
        1: [1, 2, 4, 5, 7],
        2: [1, 2, 4, 6, 7],
        3: [1, 3, 4, 6, 7],
        4: [1, 3, 4, 6, 8],
        5: [1, 3, 5, 6, 8],
        6: [2, 3, 5, 6, 8],
        7: [2, 3, 5, 7, 8],
        8: [2, 4, 5, 7, 8],
    },
    9: {
        1: [1, 2, 4, 6, 8],
        2: [1, 3, 4, 6, 8],
        3: [1, 3, 5, 6, 8],
        4: [1, 3, 5, 7, 8],
        5: [1, 3, 5, 7],
        6: [2, 3, 5, 7],
        7: [2, 4, 5, 7],
        8: [2, 4, 6, 7],
        9: [2, 4, 6, 8],
    },
    10: {
        1: [1, 4, 6, 8],
        2: [1, 3, 6, 8],
        3: [1, 3, 5, 8],
        4: [1, 3, 5, 7],
        5: [1, 3, 5, 7],
        6: [2, 3, 5, 7],
        7: [2, 4, 5, 7],
        8: [2, 4, 6, 7],
        9: [2, 4, 6, 8],
        10: [2, 4, 6, 8],
    },
    11: {
        1: [1, 3, 5, 7],
        2: [1, 3, 5, 7],
        3: [1, 3, 5, 8],
        4: [1, 3, 6, 8],
        5: [1, 4, 6, 8],
        6: [2, 4, 6, 8],
        7: [2, 4, 6, 8],
        8: [2, 4, 6],
        9: [2, 4, 7],
        10: [2, 5, 7],
        11: [3, 5, 7],
    },
    12: {
        1: [1, 3, 5, 8],
        2: [1, 3, 6, 8],
        3: [1, 3, 6, 8],
        4: [1, 4, 6, 8],
        5: [1, 4, 6],
        6: [2, 4, 6],
        7: [2, 4, 7],
        8: [2, 4, 7],
        9: [2, 5, 7],
        10: [2, 5, 7],
        11: [3, 5, 7],
        12: [3, 5, 8],
    },
}


@dataclass(frozen=True)
class SnakeTemplate:
    num_players: int
    num_periods: int
    lineup_size: int
    slot_to_periods: Dict[int, List[int]]
    period_to_slots: Dict[int, List[int]]
    turns_per_slot: Dict[int, int]
    slot_patterns: Dict[int, List[int]]


def _build_period_to_slots(slot_to_periods: Dict[int, List[int]]) -> Dict[int, List[int]]:
    period_to_slots = {period: [] for period in range(1, NUM_PERIODS + 1)}
    for slot, periods in slot_to_periods.items():
        for period in periods:
            period_to_slots[period].append(slot)
    for period in period_to_slots:
        period_to_slots[period] = sorted(period_to_slots[period])
    return period_to_slots


def _build_slot_patterns(slot_to_periods: Dict[int, List[int]]) -> Dict[int, List[int]]:
    patterns: Dict[int, List[int]] = {}
    for slot, periods in slot_to_periods.items():
        patterns[slot] = [
            1 if period in periods else 0
            for period in range(1, NUM_PERIODS + 1)
        ]
    return patterns


def validate_snake_template(template: SnakeTemplate) -> None:
    expected_total_appearances = template.num_periods * template.lineup_size
    actual_total_appearances = 0

    expected_slots = set(range(1, template.num_players + 1))
    actual_slots = set(template.slot_to_periods.keys())

    if actual_slots != expected_slots:
        raise ValueError(
            f"Slot mismatch. Expected {sorted(expected_slots)}, got {sorted(actual_slots)}"
        )

    for slot, periods in template.slot_to_periods.items():
        if not periods:
            raise ValueError(f"Slot {slot} has no assigned periods")
        if sorted(periods) != periods:
            raise ValueError(f"Slot {slot} periods are not sorted: {periods}")
        if len(periods) != len(set(periods)):
            raise ValueError(f"Slot {slot} has duplicate periods: {periods}")
        for period in periods:
            if period < 1 or period > template.num_periods:
                raise ValueError(f"Slot {slot} has invalid period {period}")
        actual_total_appearances += len(periods)

    for period, slots in template.period_to_slots.items():
        if len(slots) != template.lineup_size:
            raise ValueError(
                f"Period {period} has {len(slots)} players, expected {template.lineup_size}"
            )
        if len(slots) != len(set(slots)):
            raise ValueError(f"Period {period} contains duplicate slots: {slots}")

    if actual_total_appearances != expected_total_appearances:
        raise ValueError(
            f"Total appearances mismatch. Expected {expected_total_appearances}, got {actual_total_appearances}"
        )


def get_snake_template(num_players: int) -> SnakeTemplate:
    if num_players not in SNAKE_SLOT_PERIODS:
        raise ValueError(
            f"Unsupported player count: {num_players}. Supported values: {SUPPORTED_PLAYER_COUNTS}"
        )

    slot_to_periods = {
        slot: list(periods)
        for slot, periods in SNAKE_SLOT_PERIODS[num_players].items()
    }
    period_to_slots = _build_period_to_slots(slot_to_periods)
    turns_per_slot = {slot: len(periods) for slot, periods in slot_to_periods.items()}
    slot_patterns = _build_slot_patterns(slot_to_periods)

    template = SnakeTemplate(
        num_players=num_players,
        num_periods=NUM_PERIODS,
        lineup_size=LINEUP_SIZE,
        slot_to_periods=slot_to_periods,
        period_to_slots=period_to_slots,
        turns_per_slot=turns_per_slot,
        slot_patterns=slot_patterns,
    )
    validate_snake_template(template)
    return template


def compute_slot_overlap(template: SnakeTemplate) -> Dict[tuple[int, int], int]:
    overlap: Dict[tuple[int, int], int] = {}
    for slot_a in template.slot_to_periods:
        periods_a = set(template.slot_to_periods[slot_a])
        for slot_b in template.slot_to_periods:
            periods_b = set(template.slot_to_periods[slot_b])
            overlap[(slot_a, slot_b)] = len(periods_a & periods_b)
    return overlap


def get_turn_buckets(template: SnakeTemplate) -> Dict[int, List[int]]:
    buckets: Dict[int, List[int]] = {}
    for slot, turns in template.turns_per_slot.items():
        buckets.setdefault(turns, []).append(slot)
    for turns in buckets:
        buckets[turns] = sorted(buckets[turns])
    return dict(sorted(buckets.items(), reverse=True))


def get_turn_distribution(num_players: int) -> Dict[int, int]:
    template = get_snake_template(num_players)
    distribution: Dict[int, int] = {}
    for turns in template.turns_per_slot.values():
        distribution[turns] = distribution.get(turns, 0) + 1
    return dict(sorted(distribution.items(), reverse=True))


def derive_default_turn_targets(
    player_ids: List[int],
    priority_scores: Dict[int, float],
    num_players: int,
) -> Dict[int, int]:
    template = get_snake_template(num_players)
    slot_turns_sorted = sorted(template.turns_per_slot.values(), reverse=True)

    ranked_players = sorted(
        player_ids,
        key=lambda pid: (-priority_scores.get(pid, 0.0), pid),
    )

    return {
        pid: slot_turns_sorted[idx]
        for idx, pid in enumerate(ranked_players)
    }


def validate_turn_override(
    override_targets: Dict[int, int],
    num_players: int,
    expected_player_ids: Optional[Iterable[int]] = None,
) -> None:
    template = get_snake_template(num_players)
    expected_turns = sorted(template.turns_per_slot.values())
    actual_turns = sorted(override_targets.values())

    if actual_turns != expected_turns:
        raise ValueError(
            f"Invalid turn override distribution. Expected {expected_turns}, got {actual_turns}"
        )

    if expected_player_ids is not None:
        expected_ids = sorted(expected_player_ids)
        actual_ids = sorted(override_targets.keys())
        if actual_ids != expected_ids:
            raise ValueError(
                f"Turn override player IDs mismatch. Expected {expected_ids}, got {actual_ids}"
            )


def build_turn_override_template(
    player_ids: List[int],
    priority_scores: Dict[int, float],
    num_players: int,
) -> Dict[int, int]:
    targets = derive_default_turn_targets(
        player_ids=player_ids,
        priority_scores=priority_scores,
        num_players=num_players,
    )
    validate_turn_override(targets, num_players, expected_player_ids=player_ids)
    return targets
