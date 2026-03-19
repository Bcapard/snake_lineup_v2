from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


SKILL_COLS = ["Scoring", "Defense", "BallHandling", "Height", "Hustle"]


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


def compute_composites(players_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    if players_df is None or players_df.empty:
        if players_df is None:
            return pd.DataFrame(columns=["composite"])
        out = players_df.copy()
        out["composite"] = []
        return out

    out = players_df.copy()

    for col in SKILL_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    norm_weights = normalize_weights(weights_df)

    composite = np.zeros(len(out), dtype=float)
    for col in SKILL_COLS:
        composite += out[col].astype(float).values * norm_weights[col]

    out["composite"] = np.round(composite, 4)
    return out
