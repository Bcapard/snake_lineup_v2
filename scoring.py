import pandas as pd

DEFAULT_WEIGHTS = {
    "Scoring": 20,
    "Defense": 20,
    "BallHandling": 20,
    "Height": 20,
    "Hustle": 20,
}


def compute_composites(players_df: pd.DataFrame, weights_df: pd.DataFrame | None = None) -> pd.DataFrame:
    weights = DEFAULT_WEIGHTS.copy()

    if weights_df is not None:
        for _, row in weights_df.iterrows():
            weights[row["skill"]] = row["weight"]

    df = players_df.copy()

    total_weight = sum(weights.values())

    df["composite"] = (
        df["Scoring"] * weights["Scoring"]
        + df["Defense"] * weights["Defense"]
        + df["BallHandling"] * weights["BallHandling"]
        + df["Height"] * weights["Height"]
        + df["Hustle"] * weights["Hustle"]
    ) / total_weight

    return df


def compute_optimizer_metrics(players_df: pd.DataFrame, weights_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = compute_composites(players_df, weights_df)

    # Attack / scoring ability
    df["attack_score"] = (
        0.55 * df["Scoring"]
        + 0.20 * df["BallHandling"]
        + 0.15 * df["Hustle"]
        + 0.10 * df["Height"]
    )

    # Spacing / gravity
    df["space_score"] = (
        0.60 * df["Scoring"]
        + 0.30 * df["BallHandling"]
        + 0.10 * df["Hustle"]
    )

    # Ball security / reliability
    df["ball_security_score"] = (
        0.65 * df["BallHandling"]
        + 0.35 * df["Hustle"]
    )

    # Priority for extra turns
    df["extra_turn_priority"] = (
        0.50 * df["composite"]
        + 0.30 * df["attack_score"]
        + 0.20 * df["ball_security_score"]
    )

    # Identify top scorers dynamically (top 3 by space_score)
    df = df.sort_values("space_score", ascending=False)
    df["is_top_scorer"] = False
    df.loc[df.head(min(3, len(df))).index, "is_top_scorer"] = True

    return df
