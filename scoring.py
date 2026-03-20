from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


SKILL_COLS = ["Scoring", "Defense", "BallHandling", "Height", "Hustle"]

# Derived-metric formulas used by the optimizer layer.
# Composite still comes from the user-selected dashboard weights.
ATTACK_WEIGHTS = {
    "Scoring": 0.55,
    "BallHandling": 0.20,
    "Hustle": 0.15,
    "Height": 0.10,
}

SPACE_WEIGHTS = {
    "Scoring": 0.60,
    "BallHandling": 0.30,
    "Hustle": 0.10,
}

BALL_SECURITY_WEIGHTS = {
    "BallHandling": 0.65,
    "Hustle": 0.35,
}

EXTRA_TURN_PRIORITY_WEIGHTS = {
    "composite": 0.50,
    "attack_score": 0.30,
    "ball_security_score": 0.20,
}


def normalize_weights(weights_df: pd.DataFrame) -> Dict[str, float]:
    if weights_df is None or weights_df.empty:
        return {k: 1.0 / len(SKILL_COLS) for k in SKILL_COLS}

    df = weights_df.copy()
    df["Metric"] = df["Metric"].astype(str).str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)

    weight_map = {k: 0.0 for k in SKILL_COLS}
    for _, row in df.iterrows():
        metric = row["Metric"]
        if metric in weight_map:
            weight_map[metric] = max(float(row["Weight"]), 0.0)

    total = sum(weight_map.values())
    if total <= 0:
        return {k: 1.0 / len(SKILL_COLS) for k in SKILL_COLS}

    return {k: v / total for k, v in weight_map.items()}


def _prepare_players_df(players_df: pd.DataFrame) -> pd.DataFrame:
    if players_df is None or players_df.empty:
        if players_df is None:
            return pd.DataFrame(columns=SKILL_COLS)
        return players_df.copy()

    out = players_df.copy()
    for col in SKILL_COLS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


def _weighted_score(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    score = np.zeros(len(df), dtype=float)
    for col, weight in weights.items():
        if col not in df.columns:
            continue
        score += df[col].astype(float).values * float(weight)
    return score


def compute_composites(players_df: pd.DataFrame, weights_df: pd.DataFrame | None) -> pd.DataFrame:
    if players_df is None or players_df.empty:
        if players_df is None:
            return pd.DataFrame(columns=["composite"])
        out = players_df.copy()
        out["composite"] = []
        return out

    out = _prepare_players_df(players_df)
    norm_weights = normalize_weights(weights_df)

    composite = np.zeros(len(out), dtype=float)
    for col in SKILL_COLS:
        composite += out[col].astype(float).values * norm_weights[col]

    out["composite"] = np.round(composite, 4)
    return out


def compute_optimizer_metrics(
    players_df: pd.DataFrame,
    weights_df: pd.DataFrame | None,
    top_scorers_n: int = 4,
) -> pd.DataFrame:
    """
    Builds the richer metric set required by the optimizer.

    Derived outputs:
    - composite
    - attack_score
    - space_score
    - ball_security_score
    - extra_turn_priority
    - is_top_scorer

    Notes:
    - top scorers are derived from the attending group only
    - if manual_target_turns already exists in players_df, it is preserved
    """
    if players_df is None or players_df.empty:
        out = compute_composites(players_df, weights_df)
        for col in [
            "attack_score",
            "space_score",
            "ball_security_score",
            "extra_turn_priority",
            "is_top_scorer",
        ]:
            out[col] = []
        return out

    out = compute_composites(players_df, weights_df)

    out["attack_score"] = np.round(_weighted_score(out, ATTACK_WEIGHTS), 4)
    out["space_score"] = np.round(_weighted_score(out, SPACE_WEIGHTS), 4)
    out["ball_security_score"] = np.round(_weighted_score(out, BALL_SECURITY_WEIGHTS), 4)

    out["extra_turn_priority"] = np.round(
        EXTRA_TURN_PRIORITY_WEIGHTS["composite"] * out["composite"].astype(float).values
        + EXTRA_TURN_PRIORITY_WEIGHTS["attack_score"] * out["attack_score"].astype(float).values
        + EXTRA_TURN_PRIORITY_WEIGHTS["ball_security_score"] * out["ball_security_score"].astype(float).values,
        4,
    )

    out["is_top_scorer"] = False
    if len(out) > 0:
        top_n = max(1, min(int(top_scorers_n), len(out)))
        top_idx = (
            out.sort_values(
                ["space_score", "Scoring", "BallHandling", "composite"],
                ascending=[False, False, False, False],
            )
            .head(top_n)
            .index
        )
        out.loc[top_idx, "is_top_scorer"] = True

    if "manual_target_turns" in out.columns:
        out["manual_target_turns"] = pd.to_numeric(
            out["manual_target_turns"], errors="coerce"
        ).astype("Int64")

    return out
