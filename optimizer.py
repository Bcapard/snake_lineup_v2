from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from snake_rules import get_snake_template


PLAYERS_ON_COURT_DEFAULT = 5
NUM_PERIODS_DEFAULT = 8
DEFAULT_TIME_LIMIT_SECONDS = 10
DEFAULT_SCORE_SCALE = 1000


@dataclass(frozen=True)
class OptimizerDiagnostics:
    period_scores: Dict[int, float]
    average_period_score: float
    period_score_gap: float
    total_deviation_from_average: float
    solver_status: str


def _seed_metadata(slot: int, num_players: int) -> dict:
    chunk = ((slot - 1) // PLAYERS_ON_COURT_DEFAULT) + 1
    position_in_chunk = ((slot - 1) % PLAYERS_ON_COURT_DEFAULT) + 1

    chunk_start = (chunk - 1) * PLAYERS_ON_COURT_DEFAULT + 1
    chunk_end = min(chunk_start + PLAYERS_ON_COURT_DEFAULT - 1, num_players)
    chunk_size = chunk_end - chunk_start + 1

    position_for_sort = (
        position_in_chunk
        if (chunk % 2) == 1
        else chunk_size - position_in_chunk + 1
    )

    return {
        "chunk": chunk,
        "position_in_chunk": position_in_chunk,
        "chunk_size": chunk_size,
        "position_for_sort": position_for_sort,
    }


def _solver_status_name(status: int) -> str:
    mapping = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    return mapping.get(status, f"STATUS_{status}")


def _validate_comp_df(comp_df: pd.DataFrame) -> pd.DataFrame:
    required = ["player_id", "name", "jersey", "composite"]
    missing = [c for c in required if c not in comp_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in comp_df: {missing}")

    df = comp_df.copy()
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["jersey"] = pd.to_numeric(df["jersey"], errors="coerce").astype("Int64")
    df["name"] = df["name"].astype(str).str.strip()
    df["composite"] = pd.to_numeric(df["composite"], errors="coerce")

    if df["player_id"].isna().any():
        raise ValueError("Some player_id values are missing or invalid.")
    if df["name"].eq("").any():
        raise ValueError("Some player names are empty.")
    if df["composite"].isna().any():
        raise ValueError("Some composite scores are missing or invalid.")
    if df["player_id"].duplicated().any():
        raise ValueError("Duplicate player_id values detected.")

    return df.reset_index(drop=True)


def build_optimized_official_snake_schedule(
    comp_df: pd.DataFrame,
    time_limit_seconds: int = DEFAULT_TIME_LIMIT_SECONDS,
    score_scale: int = DEFAULT_SCORE_SCALE,
) -> Tuple[pd.DataFrame, pd.DataFrame, OptimizerDiagnostics]:
    """
    Assign players to official snake slots to balance period strength.

    Returns:
        seeded_view: player-to-slot assignment table
        schedule_df: long schedule by period/position
        diagnostics: balance metrics
    """
    df = _validate_comp_df(comp_df)
    num_players = len(df)
    template = get_snake_template(num_players)

    # Stable sort only as a display fallback, not as the assignment logic
    df = df.sort_values(["composite", "player_id"], ascending=[False, True]).reset_index(drop=True)
    df["initial_rank"] = np.arange(1, len(df) + 1, dtype=int)

    players = df.to_dict("records")
    player_ids = [int(r["player_id"]) for r in players]
    slots = list(range(1, num_players + 1))
    periods = list(range(1, NUM_PERIODS_DEFAULT + 1))

    score_map = {
        int(r["player_id"]): int(round(float(r["composite"]) * score_scale))
        for r in players
    }
    raw_score_map = {
        int(r["player_id"]): float(r["composite"])
        for r in players
    }

    model = cp_model.CpModel()

    # x[(player_id, slot)] = 1 if player assigned to slot
    x = {}
    for pid in player_ids:
        for slot in slots:
            x[(pid, slot)] = model.NewBoolVar(f"x_p{pid}_s{slot}")

    # Each player gets exactly one slot
    for pid in player_ids:
        model.Add(sum(x[(pid, slot)] for slot in slots) == 1)

    # Each slot gets exactly one player
    for slot in slots:
        model.Add(sum(x[(pid, slot)] for pid in player_ids) == 1)

    total_score_upper = sum(score_map.values())

    period_score_vars = {}
    for period in periods:
        active_slots = template.period_to_slots[period]
        period_score_vars[period] = model.NewIntVar(0, total_score_upper, f"period_score_{period}")
        model.Add(
            period_score_vars[period]
            == sum(
                score_map[pid] * x[(pid, slot)]
                for pid in player_ids
                for slot in active_slots
            )
        )

    max_period_score = model.NewIntVar(0, total_score_upper, "max_period_score")
    min_period_score = model.NewIntVar(0, total_score_upper, "min_period_score")

    for period in periods:
        model.Add(max_period_score >= period_score_vars[period])
        model.Add(min_period_score <= period_score_vars[period])

    spread_var = model.NewIntVar(0, total_score_upper, "period_score_spread")
    model.Add(spread_var == max_period_score - min_period_score)

    total_period_score = model.NewIntVar(0, total_score_upper * NUM_PERIODS_DEFAULT, "total_period_score")
    model.Add(total_period_score == sum(period_score_vars[p] for p in periods))

    deviation_vars = {}
    deviation_upper = total_score_upper * NUM_PERIODS_DEFAULT
    for period in periods:
        deviation_vars[period] = model.NewIntVar(0, deviation_upper, f"deviation_{period}")
        lhs = NUM_PERIODS_DEFAULT * period_score_vars[period] - total_period_score
        model.Add(deviation_vars[period] >= lhs)
        model.Add(deviation_vars[period] >= -lhs)

    # Stage 1: minimize gap between strongest and weakest period
    model.Minimize(spread_var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 8
    status_1 = solver.Solve(model)

    if status_1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError(f"No feasible optimization result: {_solver_status_name(status_1)}")

    best_spread = solver.Value(spread_var)

    # Stage 2: among best-gap solutions, minimize total deviation
    model.Add(spread_var == best_spread)
    model.Minimize(sum(deviation_vars[p] for p in periods))

    solver2 = cp_model.CpSolver()
    solver2.parameters.max_time_in_seconds = time_limit_seconds
    solver2.parameters.num_search_workers = 8
    status_2 = solver2.Solve(model)

    if status_2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError(f"Second optimization stage failed: {_solver_status_name(status_2)}")

    slot_to_player = {}
    player_to_slot = {}

    for slot in slots:
        assigned_pid = None
        for pid in player_ids:
            if solver2.Value(x[(pid, slot)]) == 1:
                assigned_pid = pid
                break
        if assigned_pid is None:
            raise ValueError(f"No player assigned to slot {slot}")
        slot_to_player[slot] = assigned_pid
        player_to_slot[assigned_pid] = slot

    rec_map = {int(r["player_id"]): dict(r) for r in players}

    seeded_rows = []
    for slot in slots:
        pid = slot_to_player[slot]
        rec = dict(rec_map[pid])
        rec["seed_order"] = slot
        rec["slot"] = slot
        rec["turns"] = template.turns_per_slot[slot]
        rec.update(_seed_metadata(slot, num_players))
        seeded_rows.append(rec)

    seeded_view = pd.DataFrame(seeded_rows).sort_values("seed_order").reset_index(drop=True)

    schedule_rows = []
    for period, active_slots in template.period_to_slots.items():
        for pos, slot in enumerate(active_slots, start=1):
            pid = slot_to_player[slot]
            rec = rec_map[pid]
            schedule_rows.append(
                {
                    "period": period,
                    "pos": pos,
                    "player_id": int(rec["player_id"]),
                    "name": str(rec["name"]),
                    "jersey": int(rec["jersey"]) if not pd.isna(rec["jersey"]) else None,
                    "seed_order": int(slot),
                    "slot": int(slot),
                    "turns": int(template.turns_per_slot[slot]),
                    "chunk": int(_seed_metadata(slot, num_players)["chunk"]),
                    "position_in_chunk": int(_seed_metadata(slot, num_players)["position_in_chunk"]),
                    "composite": float(rec["composite"]),
                }
            )

    schedule_df = pd.DataFrame(schedule_rows)

    period_scores = {}
    for period in periods:
        active_slots = template.period_to_slots[period]
        score = sum(raw_score_map[slot_to_player[slot]] for slot in active_slots)
        period_scores[period] = round(score, 4)

    average_period_score = round(sum(period_scores.values()) / NUM_PERIODS_DEFAULT, 4)
    period_score_gap = round(max(period_scores.values()) - min(period_scores.values()), 4)
    total_deviation = round(
        sum(abs(s - average_period_score) for s in period_scores.values()),
        4,
    )

    diagnostics = OptimizerDiagnostics(
        period_scores=period_scores,
        average_period_score=average_period_score,
        period_score_gap=period_score_gap,
        total_deviation_from_average=total_deviation,
        solver_status=_solver_status_name(status_2),
    )

    seeded_cols = [
        "seed_order",
        "player_id",
        "name",
        "jersey",
        "composite",
        "slot",
        "turns",
        "chunk",
        "position_in_chunk",
        "chunk_size",
        "position_for_sort",
        "initial_rank",
    ]
    for col in seeded_cols:
        if col not in seeded_view.columns:
            seeded_view[col] = None
    seeded_view = seeded_view[seeded_cols]

    return seeded_view, schedule_df, diagnostics
