from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from snake_rules import (
    get_snake_template,
    compute_slot_overlap,
    validate_turn_override,
    build_turn_override_template,
)


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

    period_attack_scores: Dict[int, float]
    average_period_attack_score: float
    period_attack_score_gap: float
    total_attack_deviation_from_average: float

    player_turn_targets: Dict[int, int]
    player_assigned_turns: Dict[int, int]
    total_turn_mismatch: int

    top_scorers_per_period: Dict[int, int]
    total_top_scorer_excess: int

    total_pair_overlap_penalty: int
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


def _validate_player_df(player_df: pd.DataFrame) -> pd.DataFrame:
    required = ["player_id", "name", "jersey", "composite"]
    missing = [c for c in required if c not in player_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in player_df: {missing}")

    df = player_df.copy()

    for col in ["attack_score", "space_score", "extra_turn_priority"]:
        if col not in df.columns:
            df[col] = df["composite"]

    if "is_top_scorer" not in df.columns:
        df["is_top_scorer"] = False

    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["jersey"] = pd.to_numeric(df["jersey"], errors="coerce").astype("Int64")
    df["name"] = df["name"].astype(str).str.strip()

    numeric_cols = ["composite", "attack_score", "space_score", "extra_turn_priority"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_top_scorer"] = df["is_top_scorer"].fillna(False).astype(bool)

    if df["player_id"].isna().any():
        raise ValueError("Some player_id values are missing or invalid.")
    if df["name"].eq("").any():
        raise ValueError("Some player names are empty.")
    if df["player_id"].duplicated().any():
        raise ValueError("Duplicate player_id values detected.")

    for col in numeric_cols:
        if df[col].isna().any():
            raise ValueError(f"Some {col} values are missing or invalid.")

    return df.reset_index(drop=True)


def _build_period_total_vars(
    model: cp_model.CpModel,
    periods: List[int],
    player_ids: List[int],
    x: Dict[Tuple[int, int], cp_model.IntVar],
    period_to_slots: Dict[int, List[int]],
    score_map: Dict[int, int],
    total_score_upper: int,
    prefix: str,
) -> Dict[int, cp_model.IntVar]:
    period_vars: Dict[int, cp_model.IntVar] = {}
    for period in periods:
        active_slots = period_to_slots[period]
        period_vars[period] = model.NewIntVar(0, total_score_upper, f"{prefix}_{period}")
        model.Add(
            period_vars[period]
            == sum(
                score_map[pid] * x[(pid, slot)]
                for pid in player_ids
                for slot in active_slots
            )
        )
    return period_vars


def _build_deviation_vars(
    model: cp_model.CpModel,
    periods: List[int],
    period_vars: Dict[int, cp_model.IntVar],
    total_score_upper: int,
    prefix: str,
) -> Tuple[cp_model.IntVar, Dict[int, cp_model.IntVar]]:
    total_var = model.NewIntVar(
        0,
        total_score_upper * NUM_PERIODS_DEFAULT,
        f"total_{prefix}",
    )
    model.Add(total_var == sum(period_vars[p] for p in periods))

    deviation_vars = {}
    deviation_upper = total_score_upper * NUM_PERIODS_DEFAULT
    for period in periods:
        deviation_vars[period] = model.NewIntVar(0, deviation_upper, f"{prefix}_deviation_{period}")
        lhs = NUM_PERIODS_DEFAULT * period_vars[period] - total_var
        model.Add(deviation_vars[period] >= lhs)
        model.Add(deviation_vars[period] >= -lhs)

    return total_var, deviation_vars


def build_optimized_official_snake_schedule(
    comp_df: pd.DataFrame,
    time_limit_seconds: int = DEFAULT_TIME_LIMIT_SECONDS,
    score_scale: int = DEFAULT_SCORE_SCALE,
    manual_turn_targets: Optional[Dict[int, int]] = None,
    pair_split_prefs: Optional[List[Tuple[str, str]]] = None,
    max_top_scorers_per_period: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, OptimizerDiagnostics]:
    """
    Assign players to official snake slots while preserving snake legality and optimizing:
    1. turn-target fit
    2. composite balance
    3. attack balance
    4. scorer clustering
    5. split-pair separation
    """
    df = _validate_player_df(comp_df)
    num_players = len(df)
    template = get_snake_template(num_players)

    df = df.sort_values(["composite", "player_id"], ascending=[False, True]).reset_index(drop=True)
    df["initial_rank"] = np.arange(1, len(df) + 1, dtype=int)

    players = df.to_dict("records")
    player_ids = [int(r["player_id"]) for r in players]
    slots = list(range(1, num_players + 1))
    periods = list(range(1, NUM_PERIODS_DEFAULT + 1))

    if manual_turn_targets is not None:
        manual_turn_targets = {int(k): int(v) for k, v in manual_turn_targets.items()}
        validate_turn_override(
            manual_turn_targets,
            num_players=num_players,
            expected_player_ids=player_ids,
        )
        turn_targets = manual_turn_targets
    else:
        priority_scores = {
            int(r["player_id"]): float(r["extra_turn_priority"])
            for r in players
        }
        turn_targets = build_turn_override_template(
            player_ids=player_ids,
            priority_scores=priority_scores,
            num_players=num_players,
        )

    composite_score_map = {
        int(r["player_id"]): int(round(float(r["composite"]) * score_scale))
        for r in players
    }
    raw_composite_map = {
        int(r["player_id"]): float(r["composite"])
        for r in players
    }

    attack_score_map = {
        int(r["player_id"]): int(round(float(r["attack_score"]) * score_scale))
        for r in players
    }
    raw_attack_map = {
        int(r["player_id"]): float(r["attack_score"])
        for r in players
    }

    top_scorer_map = {
        int(r["player_id"]): 1 if bool(r["is_top_scorer"]) else 0
        for r in players
    }

    model = cp_model.CpModel()

    x = {}
    for pid in player_ids:
        for slot in slots:
            x[(pid, slot)] = model.NewBoolVar(f"x_p{pid}_s{slot}")

    for pid in player_ids:
        model.Add(sum(x[(pid, slot)] for slot in slots) == 1)

    for slot in slots:
        model.Add(sum(x[(pid, slot)] for pid in player_ids) == 1)

    # Stage 0 metric: turn mismatch
    turn_mismatch_terms = []
    assigned_turns_expr = {}
    for pid in player_ids:
        assigned_turns_expr[pid] = sum(
            template.turns_per_slot[slot] * x[(pid, slot)]
            for slot in slots
        )
        mismatch = model.NewIntVar(0, NUM_PERIODS_DEFAULT, f"turn_mismatch_{pid}")
        model.Add(mismatch >= assigned_turns_expr[pid] - turn_targets[pid])
        model.Add(mismatch >= turn_targets[pid] - assigned_turns_expr[pid])
        turn_mismatch_terms.append(mismatch)

    total_turn_mismatch = model.NewIntVar(0, NUM_PERIODS_DEFAULT * num_players, "total_turn_mismatch")
    model.Add(total_turn_mismatch == sum(turn_mismatch_terms))

    composite_upper = sum(composite_score_map.values())
    attack_upper = sum(attack_score_map.values())

    period_composite_vars = _build_period_total_vars(
        model=model,
        periods=periods,
        player_ids=player_ids,
        x=x,
        period_to_slots=template.period_to_slots,
        score_map=composite_score_map,
        total_score_upper=composite_upper,
        prefix="period_composite",
    )

    period_attack_vars = _build_period_total_vars(
        model=model,
        periods=periods,
        player_ids=player_ids,
        x=x,
        period_to_slots=template.period_to_slots,
        score_map=attack_score_map,
        total_score_upper=attack_upper,
        prefix="period_attack",
    )

    max_period_composite = model.NewIntVar(0, composite_upper, "max_period_composite")
    min_period_composite = model.NewIntVar(0, composite_upper, "min_period_composite")
    for period in periods:
        model.Add(max_period_composite >= period_composite_vars[period])
        model.Add(min_period_composite <= period_composite_vars[period])

    composite_spread_var = model.NewIntVar(0, composite_upper, "composite_spread")
    model.Add(composite_spread_var == max_period_composite - min_period_composite)

    _, composite_deviation_vars = _build_deviation_vars(
        model=model,
        periods=periods,
        period_vars=period_composite_vars,
        total_score_upper=composite_upper,
        prefix="composite",
    )

    max_period_attack = model.NewIntVar(0, attack_upper, "max_period_attack")
    min_period_attack = model.NewIntVar(0, attack_upper, "min_period_attack")
    for period in periods:
        model.Add(max_period_attack >= period_attack_vars[period])
        model.Add(min_period_attack <= period_attack_vars[period])

    attack_spread_var = model.NewIntVar(0, attack_upper, "attack_spread")
    model.Add(attack_spread_var == max_period_attack - min_period_attack)

    _, attack_deviation_vars = _build_deviation_vars(
        model=model,
        periods=periods,
        period_vars=period_attack_vars,
        total_score_upper=attack_upper,
        prefix="attack",
    )

    top_scorer_count_vars = {}
    top_scorer_excess_vars = {}
    for period in periods:
        count_var = model.NewIntVar(0, PLAYERS_ON_COURT_DEFAULT, f"top_scorer_count_{period}")
        model.Add(
            count_var
            == sum(
                top_scorer_map[pid] * x[(pid, slot)]
                for pid in player_ids
                for slot in template.period_to_slots[period]
            )
        )
        top_scorer_count_vars[period] = count_var

        excess_var = model.NewIntVar(0, PLAYERS_ON_COURT_DEFAULT, f"top_scorer_excess_{period}")
        model.Add(excess_var >= count_var - max_top_scorers_per_period)
        model.Add(excess_var >= 0)
        top_scorer_excess_vars[period] = excess_var

    total_top_scorer_excess = model.NewIntVar(
        0, PLAYERS_ON_COURT_DEFAULT * NUM_PERIODS_DEFAULT, "total_top_scorer_excess"
    )
    model.Add(total_top_scorer_excess == sum(top_scorer_excess_vars[p] for p in periods))

    pair_overlap_terms = []
    slot_overlap = compute_slot_overlap(template)

    if pair_split_prefs:
        name_to_pid = {
            str(r["name"]): int(r["player_id"])
            for r in players
        }

        for left_name, right_name in pair_split_prefs:
            if left_name not in name_to_pid or right_name not in name_to_pid:
                continue

            left_pid = name_to_pid[left_name]
            right_pid = name_to_pid[right_name]

            for left_slot in slots:
                for right_slot in slots:
                    overlap = slot_overlap[(left_slot, right_slot)]
                    if overlap <= 0:
                        continue

                    both_assigned = model.NewBoolVar(
                        f"pair_{left_pid}_{right_pid}_s{left_slot}_{right_slot}"
                    )
                    model.AddBoolAnd([x[(left_pid, left_slot)], x[(right_pid, right_slot)]]).OnlyEnforceIf(both_assigned)
                    model.AddBoolOr([x[(left_pid, left_slot)].Not(), x[(right_pid, right_slot)].Not()]).OnlyEnforceIf(both_assigned.Not())

                    weighted_overlap = model.NewIntVar(0, NUM_PERIODS_DEFAULT, f"weighted_overlap_{left_pid}_{right_pid}_{left_slot}_{right_slot}")
                    model.Add(weighted_overlap == overlap).OnlyEnforceIf(both_assigned)
                    model.Add(weighted_overlap == 0).OnlyEnforceIf(both_assigned.Not())
                    pair_overlap_terms.append(weighted_overlap)

    total_pair_overlap_penalty = model.NewIntVar(
        0,
        max(1, len(pair_overlap_terms)) * NUM_PERIODS_DEFAULT,
        "total_pair_overlap_penalty",
    )
    if pair_overlap_terms:
        model.Add(total_pair_overlap_penalty == sum(pair_overlap_terms))
    else:
        model.Add(total_pair_overlap_penalty == 0)

    def _solve_stage(model_obj, objective_var_or_expr, time_limit):
        model_obj.Minimize(objective_var_or_expr)
        solver_local = cp_model.CpSolver()
        solver_local.parameters.max_time_in_seconds = time_limit
        solver_local.parameters.num_search_workers = 8
        status_local = solver_local.Solve(model_obj)
        if status_local not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise ValueError(f"No feasible optimization result: {_solver_status_name(status_local)}")
        return solver_local, status_local

    # Stage 1: turn target fit
    solver1, status1 = _solve_stage(model, total_turn_mismatch, time_limit_seconds)
    best_turn_mismatch = solver1.Value(total_turn_mismatch)
    model.Add(total_turn_mismatch == best_turn_mismatch)

    # Stage 2: composite spread
    solver2, status2 = _solve_stage(model, composite_spread_var, time_limit_seconds)
    best_composite_spread = solver2.Value(composite_spread_var)
    model.Add(composite_spread_var == best_composite_spread)

    # Stage 3: composite deviation
    composite_dev_sum = sum(composite_deviation_vars[p] for p in periods)
    solver3, status3 = _solve_stage(model, composite_dev_sum, time_limit_seconds)
    best_composite_dev = sum(solver3.Value(composite_deviation_vars[p]) for p in periods)
    model.Add(composite_dev_sum == best_composite_dev)

    # Stage 4: attack spread
    solver4, status4 = _solve_stage(model, attack_spread_var, time_limit_seconds)
    best_attack_spread = solver4.Value(attack_spread_var)
    model.Add(attack_spread_var == best_attack_spread)

    # Stage 5: attack deviation
    attack_dev_sum = sum(attack_deviation_vars[p] for p in periods)
    solver5, status5 = _solve_stage(model, attack_dev_sum, time_limit_seconds)
    best_attack_dev = sum(solver5.Value(attack_deviation_vars[p]) for p in periods)
    model.Add(attack_dev_sum == best_attack_dev)

    # Stage 6: top scorer clustering
    solver6, status6 = _solve_stage(model, total_top_scorer_excess, time_limit_seconds)
    best_top_excess = solver6.Value(total_top_scorer_excess)
    model.Add(total_top_scorer_excess == best_top_excess)

    # Stage 7: split pair overlap
    solver7, status7 = _solve_stage(model, total_pair_overlap_penalty, time_limit_seconds)

    slot_to_player = {}
    player_to_slot = {}
    for slot in slots:
        assigned_pid = None
        for pid in player_ids:
            if solver7.Value(x[(pid, slot)]) == 1:
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
        rec["target_turns"] = turn_targets[pid]
        rec["turn_mismatch"] = abs(template.turns_per_slot[slot] - turn_targets[pid])
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
                    "target_turns": int(turn_targets[pid]),
                    "chunk": int(_seed_metadata(slot, num_players)["chunk"]),
                    "position_in_chunk": int(_seed_metadata(slot, num_players)["position_in_chunk"]),
                    "composite": float(rec["composite"]),
                    "attack_score": float(rec["attack_score"]),
                    "space_score": float(rec["space_score"]),
                    "is_top_scorer": bool(rec["is_top_scorer"]),
                }
            )

    schedule_df = pd.DataFrame(schedule_rows)

    period_scores = {}
    for period in periods:
        active_slots = template.period_to_slots[period]
        score = sum(raw_composite_map[slot_to_player[slot]] for slot in active_slots)
        period_scores[period] = round(score, 4)

    average_period_score = round(sum(period_scores.values()) / NUM_PERIODS_DEFAULT, 4)
    period_score_gap = round(max(period_scores.values()) - min(period_scores.values()), 4)
    total_deviation = round(sum(abs(s - average_period_score) for s in period_scores.values()), 4)

    period_attack_scores = {}
    for period in periods:
        active_slots = template.period_to_slots[period]
        score = sum(raw_attack_map[slot_to_player[slot]] for slot in active_slots)
        period_attack_scores[period] = round(score, 4)

    average_period_attack_score = round(sum(period_attack_scores.values()) / NUM_PERIODS_DEFAULT, 4)
    period_attack_score_gap = round(max(period_attack_scores.values()) - min(period_attack_scores.values()), 4)
    total_attack_deviation = round(
        sum(abs(s - average_period_attack_score) for s in period_attack_scores.values()), 4
    )

    player_assigned_turns = {
        pid: int(template.turns_per_slot[player_to_slot[pid]])
        for pid in player_ids
    }

    top_scorers_per_period = {
        period: int(
            sum(
                top_scorer_map[slot_to_player[slot]]
                for slot in template.period_to_slots[period]
            )
        )
        for period in periods
    }

    diagnostics = OptimizerDiagnostics(
        period_scores=period_scores,
        average_period_score=average_period_score,
        period_score_gap=period_score_gap,
        total_deviation_from_average=total_deviation,

        period_attack_scores=period_attack_scores,
        average_period_attack_score=average_period_attack_score,
        period_attack_score_gap=period_attack_score_gap,
        total_attack_deviation_from_average=total_attack_deviation,

        player_turn_targets={int(k): int(v) for k, v in turn_targets.items()},
        player_assigned_turns=player_assigned_turns,
        total_turn_mismatch=int(solver7.Value(total_turn_mismatch)),

        top_scorers_per_period=top_scorers_per_period,
        total_top_scorer_excess=int(solver7.Value(total_top_scorer_excess)),

        total_pair_overlap_penalty=int(solver7.Value(total_pair_overlap_penalty)),
        solver_status=_solver_status_name(status7),
    )

    seeded_cols = [
        "seed_order",
        "player_id",
        "name",
        "jersey",
        "composite",
        "attack_score",
        "space_score",
        "extra_turn_priority",
        "is_top_scorer",
        "slot",
        "turns",
        "target_turns",
        "turn_mismatch",
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
