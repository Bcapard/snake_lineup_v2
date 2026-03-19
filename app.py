from __future__ import annotations

import os
import json, base64, io
from pathlib import Path

from dash import Dash, html, dcc, Input, Output, State, dash_table, no_update, ctx
import pandas as pd
import numpy as np

from snake_rules import (
    SUPPORTED_PLAYER_COUNTS,
    get_turn_distribution,
    build_turn_override_template,
    validate_turn_override,
)
from scoring import compute_optimizer_metrics
from optimizer import build_optimized_official_snake_schedule

# =========================================
# CONFIG & CONSTANTS
# =========================================
BASE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("DATA_DIR", BASE / "data"))
DATA_ROOT.mkdir(parents=True, exist_ok=True)

PERSIST_PLAYERS = DATA_ROOT / "players_db.json"
PERSIST_WEIGHTS = DATA_ROOT / "weights_db.json"
EXPORT_LINEUPS = DATA_ROOT / "lineups_export.csv"
EXPORT_SEEDING = DATA_ROOT / "seeding_export.csv"
EXPORT_WIDE = DATA_ROOT / "lineups_wide_export.csv"

PLAYER_COLUMNS = [
    "player_id", "name", "jersey",
    "Scoring", "Defense", "BallHandling", "Height", "Hustle"
]
SKILL_COLS = ["Scoring", "Defense", "BallHandling", "Height", "Hustle"]

WEIGHTS_COLUMNS = ["Metric", "Weight"]
WEIGHTS_ALLOWED_METRICS = SKILL_COLS
WEIGHTS_TARGET_SUM = 100

NUM_PERIODS_DEFAULT = 8
PLAYERS_ON_COURT_DEFAULT = 5
MAX_ATTENDING = 12

USE_SERVER_PERSIST = os.getenv("USE_SERVER_PERSIST", "0") == "1"

# =========================================
# HELPERS: Shared
# =========================================
def _normalize_headers(cols):
    norm_map = {
        "player_id": "player_id", "id": "player_id", "pid": "player_id",
        "name": "name", "player_name": "name", "full_name": "name",
        "jersey": "jersey", "jersey_number": "jersey", "number": "jersey",
        "scoring": "Scoring", "defense": "Defense", "ballhandling": "BallHandling",
        "height": "Height", "hustle": "Hustle",
        "metric": "Metric", "weight": "Weight"
    }
    out = []
    for c in cols:
        key = str(c).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        mapped = None
        for k, v in norm_map.items():
            if key == k.replace("_", ""):
                mapped = v
                break
        out.append(mapped if mapped else str(c))
    return out


def _upload_to_df(contents, filename):
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(decoded))
    if filename.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(decoded))
    raise ValueError("Unsupported file format. Please upload CSV or XLSX.")


def _turn_distribution_text(num_players: int) -> str:
    dist = get_turn_distribution(num_players)
    parts = [f"{count} player(s) x {turns} turns" for turns, count in dist.items()]
    return ", ".join(parts)


# =========================================
# PLAYERS
# =========================================
def players_load():
    if PERSIST_PLAYERS.exists():
        with open(PERSIST_PLAYERS, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    return None


def players_save(df: pd.DataFrame):
    if not USE_SERVER_PERSIST:
        return
    records = df[PLAYER_COLUMNS].replace({np.nan: None}).to_dict(orient="records")
    PERSIST_PLAYERS.parent.mkdir(parents=True, exist_ok=True)
    with open(PERSIST_PLAYERS, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def players_parse_uploaded(contents, filename):
    df = _upload_to_df(contents, filename)
    df.columns = _normalize_headers(df.columns)
    for col in PLAYER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[[*PLAYER_COLUMNS]]
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["jersey"] = pd.to_numeric(df["jersey"], errors="coerce").astype("Int64")
    df["name"] = df["name"].astype(str).str.strip()
    for c in SKILL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def players_validate(df: pd.DataFrame):
    issues = []
    missing = [c for c in PLAYER_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    if df["player_id"].isna().any():
        issues.append("Some player_id values are missing or invalid.")
    if df["name"].str.strip().eq("").any():
        issues.append("Some names are empty.")
    if df["jersey"].isna().any():
        issues.append("Some jersey numbers are missing or invalid.")
    dup_pid = df["player_id"][df["player_id"].duplicated(keep=False)]
    if dup_pid.notna().any():
        issues.append("Duplicate player_id detected.")
    dup_j = df["jersey"][df["jersey"].duplicated(keep=False)]
    if dup_j.notna().any():
        issues.append("Duplicate jersey numbers detected.")
    for c in SKILL_COLS:
        bad = df[c].isna() | ~df[c].between(1, 5)
        if bad.any():
            issues.append(f"{c}: {int(bad.sum())} value(s) missing or out of 1–5 range.")
    return issues


def players_empty_table():
    row = {c: None for c in PLAYER_COLUMNS}
    row["name"] = ""
    return pd.DataFrame([row])


# =========================================
# WEIGHTS
# =========================================
def weights_load():
    if PERSIST_WEIGHTS.exists():
        with open(PERSIST_WEIGHTS, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    return None


def weights_save(df: pd.DataFrame):
    if not USE_SERVER_PERSIST:
        return
    records = df.replace({np.nan: None}).to_dict(orient="records")
    PERSIST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    with open(PERSIST_WEIGHTS, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def weights_default():
    per = WEIGHTS_TARGET_SUM / len(WEIGHTS_ALLOWED_METRICS)
    return pd.DataFrame({"Metric": WEIGHTS_ALLOWED_METRICS, "Weight": [per] * len(WEIGHTS_ALLOWED_METRICS)})


def weights_empty_table():
    return weights_default().copy()


def weights_parse_uploaded(contents, filename):
    df = _upload_to_df(contents, filename)
    df.columns = _normalize_headers(df.columns)
    cols_lower = [c.lower() for c in df.columns]
    tidy = ("metric" in cols_lower) and ("weight" in cols_lower)
    if tidy:
        df = df[[col for col in df.columns if col in ["Metric", "Weight"]]]
        df["Metric"] = df["Metric"].astype(str).str.strip()
        df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    else:
        keep = [c for c in df.columns if c in WEIGHTS_ALLOWED_METRICS]
        if not keep:
            df = weights_default()
        else:
            melted = df[keep].iloc[:1].melt(var_name="Metric", value_name="Weight")
            melted["Weight"] = pd.to_numeric(melted["Weight"], errors="coerce")
            df = melted
    return df


def weights_validate(df: pd.DataFrame):
    issues = []
    missing = [c for c in WEIGHTS_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    df["Metric"] = df["Metric"].astype(str).str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    bad_metric = ~df["Metric"].isin(WEIGHTS_ALLOWED_METRICS)
    if bad_metric.any():
        bad_list = df.loc[bad_metric, "Metric"].unique().tolist()
        issues.append(f"Unknown metrics: {bad_list}. Allowed: {WEIGHTS_ALLOWED_METRICS}")
    dup_m = df["Metric"][df["Metric"].duplicated(keep=False)]
    if dup_m.any():
        issues.append("Duplicate Metric rows detected.")
    if df["Weight"].isna().any():
        issues.append("Some Weight values are missing or invalid.")
    if (df["Weight"] < 0).any():
        issues.append("Negative Weight values are not allowed.")
    wsum = float(df["Weight"].sum()) if df["Weight"].notna().all() else np.nan
    if not np.isnan(wsum) and abs(wsum - WEIGHTS_TARGET_SUM) > 1e-6:
        issues.append(f"Weights must sum to {WEIGHTS_TARGET_SUM}. Current sum = {wsum:g}.")
    return issues


# =========================================
# LINEUP DISPLAY HELPERS
# =========================================
def schedule_to_wide(schedule_df: pd.DataFrame, seeded: pd.DataFrame) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame()

    # Fixed left-to-right column order
    ordered_names = seeded.sort_values("seed_order")["name"].astype(str).tolist()
    periods = sorted(schedule_df["period"].unique())

    wide = pd.DataFrame({"period": periods})
    for name in ordered_names:
        wide[name] = ""

    prev_names = set()
    prev_start_idx = None

    for p in periods:
        subset = schedule_df[schedule_df["period"] == p].copy()
        current_names = set(subset["name"].astype(str).tolist())

        if not current_names:
            prev_names = set()
            prev_start_idx = None
            continue

        # Players in the current lineup who were NOT in the previous lineup
        incoming_names = current_names - prev_names

        # Start at the first incoming player from the left
        incoming_indices = [
            i for i, name in enumerate(ordered_names)
            if name in incoming_names
        ]

        if incoming_indices:
            start_idx = incoming_indices[0]
        else:
            # Fallback when no one new enters:
            # keep the previous start if possible, otherwise use the leftmost active player
            active_indices = [
                i for i, name in enumerate(ordered_names)
                if name in current_names
            ]

            if prev_start_idx is not None:
                rotated_indices = list(range(prev_start_idx, len(ordered_names))) + list(range(0, prev_start_idx))
                start_idx = next((i for i in rotated_indices if i in active_indices), active_indices[0])
            else:
                start_idx = active_indices[0]

        # Circular scan from start_idx, assign 1..5 only to active players
        rotated_names = ordered_names[start_idx:] + ordered_names[:start_idx]
        rank = 1
        for name in rotated_names:
            if name in current_names:
                wide.loc[wide["period"] == p, name] = str(rank)
                rank += 1

        prev_names = current_names
        prev_start_idx = start_idx

    return wide


def schedule_to_names(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame(columns=["period", "players"])
    rows = []
    for p in sorted(schedule_df["period"].unique()):
        names = (
            schedule_df[schedule_df["period"] == p]
            .sort_values("pos")["name"]
            .tolist()
        )
        rows.append({"period": p, "players": ", ".join(names)})
    return pd.DataFrame(rows)


def build_period_rating_summary(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame(
            columns=[
                "period", "total_rating", "avg_player_rating",
                "total_attack", "avg_attack", "top_scorers", "players"
            ]
        )

    summary = (
        schedule_df.groupby("period", as_index=False)
        .agg(
            total_rating=("composite", "sum"),
            avg_player_rating=("composite", "mean"),
            total_attack=("attack_score", "sum"),
            avg_attack=("attack_score", "mean"),
            top_scorers=("is_top_scorer", "sum"),
            players=("name", lambda s: ", ".join(s.tolist()))
        )
        .sort_values("period")
        .reset_index(drop=True)
    )
    summary["total_rating"] = summary["total_rating"].round(2)
    summary["avg_player_rating"] = summary["avg_player_rating"].round(2)
    summary["total_attack"] = summary["total_attack"].round(2)
    summary["avg_attack"] = summary["avg_attack"].round(2)
    summary["top_scorers"] = summary["top_scorers"].astype(int)
    return summary


def build_player_rating_summary(seeded_view: pd.DataFrame) -> pd.DataFrame:
    if seeded_view is None or seeded_view.empty:
        return pd.DataFrame(
            columns=[
                "initial_rank", "seed_order", "player_id", "name", "jersey",
                "overall_rating", "attack_score", "extra_turn_priority",
                "slot", "turns", "target_turns", "turn_mismatch", "is_top_scorer"
            ]
        )

    out = seeded_view.copy()
    out = out.rename(columns={"composite": "overall_rating"})
    keep_cols = [
        "initial_rank", "seed_order", "player_id", "name", "jersey",
        "overall_rating", "attack_score", "extra_turn_priority",
        "slot", "turns", "target_turns", "turn_mismatch", "is_top_scorer"
    ]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = None
    out = out[keep_cols].copy()
    out["overall_rating"] = pd.to_numeric(out["overall_rating"], errors="coerce").round(2)
    out["attack_score"] = pd.to_numeric(out["attack_score"], errors="coerce").round(2)
    out["extra_turn_priority"] = pd.to_numeric(out["extra_turn_priority"], errors="coerce").round(2)
    return out.sort_values(["seed_order", "player_id"]).reset_index(drop=True)


def build_turn_override_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    player_ids = metrics_df["player_id"].astype(int).tolist()
    priority_scores = dict(zip(metrics_df["player_id"].astype(int), metrics_df["extra_turn_priority"].astype(float)))
    default_targets = build_turn_override_template(
        player_ids=player_ids,
        priority_scores=priority_scores,
        num_players=len(metrics_df),
    )

    out = metrics_df[["player_id", "name", "jersey", "composite", "attack_score", "extra_turn_priority"]].copy()
    out["default_turns"] = out["player_id"].astype(int).map(default_targets)
    out["target_turns"] = out["default_turns"]
    out["manual_override"] = False
    out["composite"] = pd.to_numeric(out["composite"], errors="coerce").round(2)
    out["attack_score"] = pd.to_numeric(out["attack_score"], errors="coerce").round(2)
    out["extra_turn_priority"] = pd.to_numeric(out["extra_turn_priority"], errors="coerce").round(2)
    return out


# =========================================
# APP (Tabs 1–3)
# =========================================
app = Dash(__name__, title="U10 Lineup", suppress_callback_exceptions=True)
server = app.server

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <meta name="apple-mobile-web-app-capable" content="yes"/>
    <meta name="apple-mobile-web-app-status-bar-style" content="default"/>
    <meta name="theme-color" content="#0b5ed7"/>
    <link rel="manifest" href="/assets/manifest.json"/>
    <link rel="apple-touch-icon" href="/assets/icon-192.png"/>
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      <script>
        if ('serviceWorker' in navigator) {
          navigator.serviceWorker.register('/assets/sw.js');
        }
      </script>
      {%renderer%}
    </footer>
  </body>
</html>
"""

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px"},
    children=[
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "12px",
                "marginBottom": "12px",
            },
            children=[
                html.Img(
                    src="/assets/app-icon.png?v=1",
                    alt="App icon",
                    draggable="false",
                    style={"width": "36px", "height": "36px", "objectFit": "contain"},
                ),
                html.H2("Snake Lineup Generator", style={"margin": 0}),
            ],
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-players",
            className="u10-tabs",
            children=[
                dcc.Tab(
                    label="1) Players — Upload / Edit / Save",
                    value="tab-players",
                    className="u10-tabs",
                    children=[
                        dcc.Store(id="players-store", storage_type="local"),
                        dcc.Store(id="players-pending-upload"),
                        html.Div(
                            style={"marginTop": "12px", "padding": "16px", "border": "1px solid #333", "borderRadius": "12px"},
                            children=[
                                html.H4("Step 1 — Upload Players (CSV/XLSX)"),
                                html.P("If already saved, you can skip this step and click 'Load Saved'."),
                                dcc.Upload(
                                    id="players-uploader",
                                    children=html.Div(["Drag & Drop or ", html.U("Select a file")]),
                                    style={
                                        "width": "100%", "height": "80px", "lineHeight": "80px",
                                        "borderWidth": "1px", "borderStyle": "dashed",
                                        "borderRadius": "8px", "textAlign": "center"
                                    },
                                    multiple=False,
                                ),
                                html.Div(id="players-upload-filename", style={"marginTop": "6px", "fontStyle": "italic"}),
                            ],
                        ),
                        html.Div(style={"height": "14px"}),
                        html.Div(
                            style={"padding": "16px", "border": "1px solid #333", "borderRadius": "12px"},
                            children=[
                                html.H4("Step 2 — Edit Players"),
                                dash_table.DataTable(
                                    id="players-table",
                                    data=[],
                                    columns=[
                                        {"name": "player_id", "id": "player_id", "type": "numeric"},
                                        {"name": "name", "id": "name", "type": "text"},
                                        {"name": "jersey", "id": "jersey", "type": "numeric"},
                                        {"name": "Scoring", "id": "Scoring", "type": "numeric"},
                                        {"name": "Defense", "id": "Defense", "type": "numeric"},
                                        {"name": "BallHandling", "id": "BallHandling", "type": "numeric"},
                                        {"name": "Height", "id": "Height", "type": "numeric"},
                                        {"name": "Hustle", "id": "Hustle", "type": "numeric"},
                                    ],
                                    editable=True,
                                    row_deletable=True,
                                    page_size=15,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"minWidth": 110, "maxWidth": 180, "whiteSpace": "normal"},
                                ),
                                html.Div(
                                    style={"marginTop": "10px", "display": "flex", "gap": "8px", "flexWrap": "wrap"},
                                    children=[
                                        html.Button("Add Row", id="players-add-row", n_clicks=0),
                                        html.Button("Save (Persist)", id="players-save", n_clicks=0, style={"background": "#0E2B5C", "color": "white"}),
                                        html.Button("Load Saved", id="players-load-saved", n_clicks=0),
                                    ],
                                ),
                                html.Div(id="players-validation", style={"marginTop": "12px", "color": "#b00020"}),
                                html.Div(id="players-save-status", style={"marginTop": "6px", "color": "#088a2a"}),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label="2) Weights — Upload / Edit / Save",
                    value="tab-weights",
                    className="u10-tabs",
                    children=[
                        dcc.Store(id="weights-store", storage_type="local"),
                        html.Div(
                            style={"marginTop": "12px", "padding": "16px", "border": "1px solid #333", "borderRadius": "12px"},
                            children=[
                                html.H4("Weights — Upload (CSV/XLSX)"),
                                html.P(f"Expected tidy format: columns {WEIGHTS_COLUMNS} — Metric in {WEIGHTS_ALLOWED_METRICS}, weights sum to {WEIGHTS_TARGET_SUM}."),
                                dcc.Upload(
                                    id="weights-uploader",
                                    children=html.Div(["Drag & Drop or ", html.U("Select a file")]),
                                    style={
                                        "width": "100%", "height": "80px", "lineHeight": "80px",
                                        "borderWidth": "1px", "borderStyle": "dashed",
                                        "borderRadius": "8px", "textAlign": "center"
                                    },
                                    multiple=False,
                                ),
                                html.Div(id="weights-upload-filename", style={"marginTop": "6px", "fontStyle": "italic"}),
                            ],
                        ),
                        html.Div(style={"height": "14px"}),
                        html.Div(
                            style={"padding": "16px", "border": "1px solid #333", "borderRadius": "12px"},
                            children=[
                                html.H4("Edit Weights"),
                                dash_table.DataTable(
                                    id="weights-table",
                                    data=[],
                                    columns=[
                                        {"name": "Metric", "id": "Metric", "type": "text", "presentation": "dropdown"},
                                        {"name": f"Weight (sum to {WEIGHTS_TARGET_SUM})", "id": "Weight", "type": "numeric"},
                                    ],
                                    editable=True,
                                    row_deletable=False,
                                    dropdown={"Metric": {"options": [{"label": m, "value": m} for m in WEIGHTS_ALLOWED_METRICS]}},
                                    page_size=10,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"minWidth": 120, "maxWidth": 220, "whiteSpace": "normal"},
                                ),
                                html.Div(
                                    style={"marginTop": "10px", "display": "flex", "gap": "8px", "flexWrap": "wrap"},
                                    children=[
                                        html.Button("Reset to Even", id="weights-reset", n_clicks=0),
                                        html.Button("Save (Persist)", id="weights-save", n_clicks=0, style={"background": "#0E2B5C", "color": "white"}),
                                        html.Button("Load Saved", id="weights-load-saved", n_clicks=0),
                                    ],
                                ),
                                html.Div(id="weights-msg", style={"marginTop": "10px", "color": "#088a2a"}),
                                html.Div(id="weights-err", style={"marginTop": "6px", "color": "#b00020"}),
                                html.Div(id="weights-sum", style={"marginTop": "6px", "fontStyle": "italic"}),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label="3) Lineup — Snake Generator",
                    value="tab-snake",
                    className="u10-tabs",
                    children=[
                        dcc.Store(id="snake-attending-prev", data=[]),
                        dcc.Store(id="turn-override-store"),

                        html.Div(
                            style={"marginTop": "12px", "padding": "16px", "border": "1px solid #333", "borderRadius": "12px"},
                            children=[
                                html.H4("Inputs"),
                                html.Div(
                                    style={"padding": "12px", "border": "1px solid #999", "borderRadius": "10px", "marginBottom": "12px"},
                                    children=[
                                        html.Div("Official mode", style={"fontWeight": 700, "marginBottom": "6px"}),
                                        html.Div(f"{PLAYERS_ON_COURT_DEFAULT} players on court, {NUM_PERIODS_DEFAULT} periods"),
                                        html.Div(f"Supported attendance: {SUPPORTED_PLAYER_COUNTS}", style={"marginTop": "4px", "fontStyle": "italic"}),
                                        html.Div(id="snake-turn-distribution", style={"marginTop": "6px", "fontStyle": "italic"}),
                                    ],
                                ),

                                html.Label(f"Select attending players (max {MAX_ATTENDING})"),
                                html.Div(
                                    style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "alignItems": "center", "padding": "10px"},
                                    children=[
                                        html.Button("Select All", id="snake-select-all", n_clicks=0),
                                        html.Button("Clear", id="snake-clear", n_clicks=0),
                                        html.Div(id="snake-count", style={"marginLeft": "auto", "fontWeight": 600}),
                                    ],
                                ),

                                dcc.Checklist(
                                    id="snake-attending-list",
                                    options=[],
                                    value=[],
                                    labelStyle={
                                        "display": "inline-flex",
                                        "alignItems": "center",
                                        "gap": "10px",
                                        "padding": "10px 12px",
                                        "margin": "6px",
                                        "border": "1px solid #999",
                                        "borderRadius": "10px",
                                        "minWidth": "140px",
                                        "userSelect": "none",
                                        "fontSize": "16px"
                                    },
                                    inputStyle={"width": "20px", "height": "20px"},
                                    style={"marginTop": "10px", "display": "flex", "flexWrap": "wrap"}
                                ),

                                html.Div(id="snake-picker-err", style={"marginTop": "8px", "color": "#b00020"}),

                                html.Div(style={"height": "14px"}),
                                html.Div(
                                    style={"padding": "16px", "border": "1px solid #999", "borderRadius": "10px"},
                                    children=[
                                        html.H4("Turn targets"),
                                        html.Div(
                                            "Default turn targets are derived from the selected players' skill profile. "
                                            "You can override them manually, but the overall turn distribution must still match the official snake template.",
                                            style={"marginBottom": "10px"},
                                        ),
                                        dash_table.DataTable(
                                            id="turn-override-table",
                                            data=[],
                                            columns=[
                                                {"name": "player_id", "id": "player_id"},
                                                {"name": "name", "id": "name"},
                                                {"name": "jersey", "id": "jersey"},
                                                {"name": "composite", "id": "composite"},
                                                {"name": "attack_score", "id": "attack_score"},
                                                {"name": "priority", "id": "extra_turn_priority"},
                                                {"name": "default_turns", "id": "default_turns"},
                                                {"name": "target_turns", "id": "target_turns", "type": "numeric"},
                                                {"name": "manual_override", "id": "manual_override"},
                                            ],
                                            editable=True,
                                            row_deletable=False,
                                            page_size=20,
                                            style_table={"overflowX": "auto"},
                                            style_cell={"minWidth": 90, "whiteSpace": "normal"},
                                        ),
                                        html.Div(id="turn-override-msg", style={"marginTop": "8px", "color": "#088a2a"}),
                                        html.Div(id="turn-override-err", style={"marginTop": "6px", "color": "#b00020"}),
                                    ],
                                ),

                                html.Br(),
                                html.Button("Generate Lineups", id="snake-generate", n_clicks=0, style={"background": "#0E2B5C", "color": "white"}),
                                html.Div(id="snake-err", style={"marginTop": "10px", "color": "#b00020"}),
                                html.Div(id="snake-msg", style={"marginTop": "6px", "color": "#088a2a"}),
                            ],
                        ),

                        html.Div(style={"height": "14px"}),
                        html.Div(
                            style={"padding": "16px", "border": "1px solid #333", "borderRadius": "12px"},
                            children=[
                                html.H4("Lineup Rating by Period"),
                                dash_table.DataTable(
                                    id="snake-period-ratings",
                                    data=[],
                                    columns=[
                                        {"name": "period", "id": "period"},
                                        {"name": "total_rating", "id": "total_rating"},
                                        {"name": "avg_player_rating", "id": "avg_player_rating"},
                                        {"name": "total_attack", "id": "total_attack"},
                                        {"name": "avg_attack", "id": "avg_attack"},
                                        {"name": "top_scorers", "id": "top_scorers"},
                                        {"name": "players", "id": "players"},
                                    ],
                                    page_size=20,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"minWidth": 110, "whiteSpace": "normal"},
                                ),
                            ],
                        ),

                        html.Div(style={"height": "14px"}),
                        html.Div(
                            style={"padding": "16px", "border": "1px solid #333", "borderRadius": "12px"},
                            children=[
                                html.H4("Player Rating Overview"),
                                dash_table.DataTable(
                                    id="snake-player-ratings",
                                    data=[],
                                    columns=[
                                        {"name": "initial_rank", "id": "initial_rank"},
                                        {"name": "seed_order", "id": "seed_order"},
                                        {"name": "player_id", "id": "player_id"},
                                        {"name": "name", "id": "name"},
                                        {"name": "jersey", "id": "jersey"},
                                        {"name": "overall_rating", "id": "overall_rating"},
                                        {"name": "attack_score", "id": "attack_score"},
                                        {"name": "extra_turn_priority", "id": "extra_turn_priority"},
                                        {"name": "slot", "id": "slot"},
                                        {"name": "turns", "id": "turns"},
                                        {"name": "target_turns", "id": "target_turns"},
                                        {"name": "turn_mismatch", "id": "turn_mismatch"},
                                        {"name": "is_top_scorer", "id": "is_top_scorer"},
                                    ],
                                    page_size=20,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"minWidth": 100, "whiteSpace": "normal"},
                                ),
                            ],
                        ),

                        html.Div(style={"height": "14px"}),
                        html.Div(
                            style={"padding": "16px", "border": "1px solid #333", "borderRadius": "12px"},
                            children=[
                                html.H4("Lineups by Period (wide)"),
                                dash_table.DataTable(
                                    id="snake-lineups-wide",
                                    data=[],
                                    columns=[{"name": "period", "id": "period"}],
                                    page_size=20,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"minWidth": 80, "whiteSpace": "normal"},
                                ),
                            ],
                        ),
                        html.Div(style={"height": "14px"}),
                        html.Div(
                            style={"padding": "16px", "border": "1px solid #333", "borderRadius": "12px"},
                            children=[
                                html.H4("Lineups by Period (names)"),
                                dash_table.DataTable(
                                    id="snake-lineups-names",
                                    data=[],
                                    columns=[{"name": "period", "id": "period"}, {"name": "players", "id": "players"}],
                                    page_size=20,
                                    style_table={"overflowX": "auto"},
                                    style_cell={"minWidth": 120, "whiteSpace": "normal"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
    ],
)

# ---------- TAB 1 CALLBACKS ----------
@app.callback(
    Output("players-upload-filename", "children"),
    Output("players-pending-upload", "data"),
    Input("players-uploader", "contents"),
    State("players-uploader", "filename"),
    prevent_initial_call=True
)
def players_handle_upload(contents, filename):
    if contents is None:
        return no_update, no_update
    try:
        df = players_parse_uploaded(contents, filename)
        return f"Uploaded: {filename}", df.replace({np.nan: None}).to_dict(orient="records")
    except Exception as e:
        return f"Upload error: {e}", None


@app.callback(
    Output("players-table", "data"),
    Output("players-store", "data", allow_duplicate=True),
    Input("players-store", "data"),
    Input("players-pending-upload", "data"),
    Input("players-load-saved", "n_clicks"),
    prevent_initial_call="initial_duplicate",
)
def players_seed_table(store_data, pending, n_load):
    trig = ctx.triggered_id
    if trig == "players-pending-upload" and pending:
        return pending, pending
    if trig == "players-load-saved" and n_load:
        if store_data:
            return store_data, store_data
        df = players_load()
        if df is not None:
            data = df.replace({np.nan: None}).to_dict(orient="records")
            return data, data
        data = players_empty_table().replace({np.nan: None}).to_dict(orient="records")
        return data, data
    if store_data:
        return store_data, store_data
    df0 = players_load()
    if df0 is not None:
        data = df0.replace({np.nan: None}).to_dict(orient="records")
        return data, data
    data = players_empty_table().replace({np.nan: None}).to_dict(orient="records")
    return data, data


@app.callback(
    Output("players-table", "data", allow_duplicate=True),
    Input("players-add-row", "n_clicks"),
    State("players-table", "data"),
    prevent_initial_call=True
)
def players_add_row(n, rows):
    if not n:
        return no_update
    new = {c: None for c in PLAYER_COLUMNS}
    new["name"] = ""
    return rows + [new]


@app.callback(
    Output("players-validation", "children"),
    Output("players-save-status", "children"),
    Output("players-store", "data", allow_duplicate=True),
    Input("players-save", "n_clicks"),
    State("players-table", "data"),
    prevent_initial_call=True
)
def players_save_cb(n, rows):
    if not n:
        return no_update, no_update, no_update
    df = pd.DataFrame(rows)
    for col in PLAYER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["jersey"] = pd.to_numeric(df["jersey"], errors="coerce").astype("Int64")
    df["name"] = df["name"].astype(str).str.strip()
    for c in SKILL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    issues = players_validate(df)
    if issues:
        return [html.Ul([html.Li(i) for i in issues])], "", no_update
    players_save(df)
    clean_records = df.replace({np.nan: None}).to_dict(orient="records")
    return "", "Saved! (This browser)", clean_records


# ---------- TAB 2 CALLBACKS ----------
@app.callback(
    Output("weights-store", "data"),
    Output("weights-upload-filename", "children"),
    Output("weights-msg", "children"),
    Output("weights-err", "children"),
    Output("weights-sum", "children"),
    Output("weights-table", "data"),
    Input("weights-store", "data"),
    Input("weights-uploader", "contents"),
    Input("weights-save", "n_clicks"),
    Input("weights-load-saved", "n_clicks"),
    Input("weights-reset", "n_clicks"),
    State("weights-uploader", "filename"),
    State("weights-table", "data"),
    prevent_initial_call=False,
)
def weights_master_cb(store_data, contents, n_save, n_load, n_reset, filename, table_rows):
    trig = ctx.triggered_id
    stored_df = pd.DataFrame(store_data) if store_data not in (None, []) else None
    loaded_df = weights_load() if stored_df is None else None
    cur_df = stored_df if stored_df is not None else (loaded_df if loaded_df is not None else weights_empty_table())

    out_file = no_update
    out_msg = ""
    out_err = ""
    out_sum = ""
    out_table = cur_df.replace({np.nan: None}).to_dict(orient="records")

    try:
        if trig == "weights-uploader" and contents:
            df_up = weights_parse_uploaded(contents, filename)
            issues = weights_validate(df_up)
            if issues:
                out_err = html.Ul([html.Li(i) for i in issues])
                out_msg = ""
            else:
                cur_df = df_up
                out_table = cur_df.replace({np.nan: None}).to_dict(orient="records")
                out_msg = "Weights uploaded."
                out_err = ""
            out_file = f"Uploaded: {filename}"
            out_sum = f"Current sum: {float(cur_df['Weight'].sum()):g} / {WEIGHTS_TARGET_SUM}"

        elif trig == "weights-save" and n_save:
            df_cur = pd.DataFrame(table_rows) if table_rows else cur_df
            issues = weights_validate(df_cur)
            if issues:
                out_err = html.Ul([html.Li(i) for i in issues])
                out_msg = ""
            else:
                cur_df = df_cur
                out_table = cur_df.replace({np.nan: None}).to_dict(orient="records")
                weights_save(cur_df)
                out_msg = "Saved! (This browser)"
                out_err = ""
            out_sum = f"Current sum: {float(cur_df['Weight'].sum()):g} / {WEIGHTS_TARGET_SUM}"

        elif trig == "weights-load-saved" and n_load:
            if store_data:
                cur_df = pd.DataFrame(store_data)
                out_msg = "Loaded weights from this browser."
            else:
                df_s = weights_load()
                if df_s is None:
                    cur_df = weights_empty_table()
                    out_msg = "No server copy. Loaded defaults."
                else:
                    cur_df = df_s
                    out_msg = "Loaded server copy."
            out_err = ""
            out_table = cur_df.replace({np.nan: None}).to_dict(orient="records")
            out_sum = f"Current sum: {float(cur_df['Weight'].sum()):g} / {WEIGHTS_TARGET_SUM}"

        elif trig == "weights-reset" and n_reset:
            cur_df = weights_empty_table()
            out_table = cur_df.replace({np.nan: None}).to_dict(orient="records")
            out_msg = "Reset to even weights."
            out_err = ""
            out_sum = f"Current sum: {float(cur_df['Weight'].sum()):g} / {WEIGHTS_TARGET_SUM}"

        else:
            if store_data:
                cur_df = pd.DataFrame(store_data)
                out_msg = "Loaded weights from this browser."
            else:
                df0 = weights_load()
                cur_df = df0 if df0 is not None else weights_empty_table()
                out_msg = "Loaded server copy." if df0 is not None else "Initialized with default weights."
            out_err = ""
            out_table = cur_df.replace({np.nan: None}).to_dict(orient="records")
            out_sum = f"Current sum: {float(cur_df['Weight'].sum()):g} / {WEIGHTS_TARGET_SUM}"

    except Exception as e:
        out_err = f"Error: {e}"

    out_store = cur_df.replace({np.nan: None}).to_dict(orient="records")
    return out_store, out_file, out_msg, out_err, out_sum, out_table


# ---------- TAB 3: ATTENDANCE PICKER CALLBACKS ----------
@app.callback(
    Output("snake-attending-list", "options"),
    Output("snake-attending-list", "value"),
    Output("snake-attending-prev", "data"),
    Output("snake-count", "children"),
    Input("tabs", "value"),
    Input("players-store", "data"),
    State("snake-attending-list", "value"),
    prevent_initial_call=False
)
def snake_seed_attending(tab, store_players, current_val):
    if tab != "tab-snake":
        return no_update, no_update, no_update, no_update

    if store_players:
        players_df = pd.DataFrame(store_players)
    else:
        players_df = players_load()
        if players_df is None or players_df.empty:
            return [], [], [], "0 selected"

    if players_df is None or players_df.empty:
        return [], [], [], "0 selected"

    def label_for(r):
        j = f"#{int(r['jersey'])}" if not pd.isna(r["jersey"]) else ""
        return f"{r['name']} {j}".strip()

    options = [{"label": label_for(r), "value": int(r["player_id"])} for _, r in players_df.iterrows()]
    all_ids = [int(r["player_id"]) for _, r in players_df.iterrows()]

    default_vals = all_ids[:MAX_ATTENDING]
    val = current_val if current_val else default_vals
    val = val[:MAX_ATTENDING]

    count_text = f"{len(val)} selected (max {MAX_ATTENDING})"
    return options, val, val, count_text


@app.callback(
    Output("snake-attending-list", "value", allow_duplicate=True),
    Output("snake-attending-prev", "data", allow_duplicate=True),
    Output("snake-picker-err", "children"),
    Output("snake-count", "children", allow_duplicate=True),
    Input("snake-select-all", "n_clicks"),
    Input("snake-clear", "n_clicks"),
    Input("snake-attending-list", "value"),
    State("snake-attending-list", "options"),
    State("snake-attending-prev", "data"),
    prevent_initial_call=True
)
def snake_picker(n_all, n_clear, new_val, options, prev_val):
    trig = ctx.triggered_id
    options = options or []
    prev_val = prev_val or []

    err = ""
    count_text = f"{len(new_val or [])} selected (max {MAX_ATTENDING})"

    if trig == "snake-select-all" and n_all:
        all_ids = [o["value"] for o in options]
        val = all_ids[:MAX_ATTENDING]
        count_text = f"{len(val)} selected (max {MAX_ATTENDING})"
        return val, val, err, count_text

    if trig == "snake-clear" and n_clear:
        return [], [], err, f"0 selected (max {MAX_ATTENDING})"

    if trig == "snake-attending-list":
        val = new_val or []
        if len(val) > MAX_ATTENDING:
            err = f"Max {MAX_ATTENDING} players. Extra selections ignored."
            count_text = f"{len(prev_val)} selected (max {MAX_ATTENDING})"
            return prev_val, prev_val, err, count_text
        count_text = f"{len(val)} selected (max {MAX_ATTENDING})"
        return val, val, "", count_text

    return new_val or [], new_val or [], err, count_text


@app.callback(
    Output("turn-override-table", "data"),
    Output("turn-override-msg", "children"),
    Output("turn-override-err", "children"),
    Output("snake-turn-distribution", "children"),
    Output("turn-override-store", "data"),
    Input("snake-attending-list", "value"),
    State("players-store", "data"),
    State("weights-store", "data"),
    prevent_initial_call=False
)
def seed_turn_override_table(attending_ids, store_players, store_weights):
    if not attending_ids:
        return [], "", "", "", None

    players = pd.DataFrame(store_players) if store_players else players_load()
    weights = pd.DataFrame(store_weights) if store_weights else weights_load()

    if players is None or players.empty:
        return [], "", "No saved players found (Tab 1).", "", None

    if weights is None or weights.empty:
        weights = weights_empty_table()

    players_att = players[players["player_id"].isin(attending_ids)].copy()
    if len(players_att) not in SUPPORTED_PLAYER_COUNTS:
        return [], "", (
            f"Official snake templates support {SUPPORTED_PLAYER_COUNTS}. "
            f"Current selection: {len(players_att)} players."
        ), "", None

    wissues = weights_validate(weights.copy())
    if wissues:
        return [], "", f"Weights invalid: {'; '.join(wissues)}", "", None

    metrics_df = compute_optimizer_metrics(players_att, weights)
    override_df = build_turn_override_rows(metrics_df)

    dist_text = f"Official turn distribution for {len(players_att)} players: {_turn_distribution_text(len(players_att))}"
    msg = "Default turn targets generated from the selected players' derived priority."

    data = override_df.replace({np.nan: None}).to_dict(orient="records")
    return data, msg, "", dist_text, data


# ---------- TAB 3 GENERATE CALLBACK ----------
@app.callback(
    Output("snake-period-ratings", "data"),
    Output("snake-period-ratings", "columns"),
    Output("snake-player-ratings", "data"),
    Output("snake-player-ratings", "columns"),
    Output("snake-lineups-wide", "data"),
    Output("snake-lineups-wide", "columns"),
    Output("snake-lineups-names", "data"),
    Output("snake-lineups-names", "columns"),
    Output("snake-msg", "children"),
    Output("snake-err", "children"),
    Input("snake-generate", "n_clicks"),
    State("snake-attending-list", "value"),
    State("players-store", "data"),
    State("weights-store", "data"),
    State("turn-override-table", "data"),
    prevent_initial_call=True
)
def snake_generate(n, attending_ids, store_players, store_weights, turn_override_rows):
    if not n:
        return (
            no_update, no_update,
            no_update, no_update,
            no_update, no_update,
            no_update, no_update,
            no_update, no_update
        )

    players = pd.DataFrame(store_players) if store_players else players_load()
    weights = pd.DataFrame(store_weights) if store_weights else weights_load()

    empty_cols = [{"name": "period", "id": "period"}]
    names_cols = [{"name": "period", "id": "period"}, {"name": "players", "id": "players"}]
    period_rating_cols = [
        {"name": "period", "id": "period"},
        {"name": "total_rating", "id": "total_rating"},
        {"name": "avg_player_rating", "id": "avg_player_rating"},
        {"name": "total_attack", "id": "total_attack"},
        {"name": "avg_attack", "id": "avg_attack"},
        {"name": "top_scorers", "id": "top_scorers"},
        {"name": "players", "id": "players"},
    ]
    player_rating_cols = [
        {"name": "initial_rank", "id": "initial_rank"},
        {"name": "seed_order", "id": "seed_order"},
        {"name": "player_id", "id": "player_id"},
        {"name": "name", "id": "name"},
        {"name": "jersey", "id": "jersey"},
        {"name": "overall_rating", "id": "overall_rating"},
        {"name": "attack_score", "id": "attack_score"},
        {"name": "extra_turn_priority", "id": "extra_turn_priority"},
        {"name": "slot", "id": "slot"},
        {"name": "turns", "id": "turns"},
        {"name": "target_turns", "id": "target_turns"},
        {"name": "turn_mismatch", "id": "turn_mismatch"},
        {"name": "is_top_scorer", "id": "is_top_scorer"},
    ]

    if players is None or players.empty:
        return [], period_rating_cols, [], player_rating_cols, [], empty_cols, [], names_cols, "", "No saved players found (Tab 1)."

    if weights is None or weights.empty:
        weights = weights_empty_table()

    if not attending_ids:
        return [], period_rating_cols, [], player_rating_cols, [], empty_cols, [], names_cols, "", "No attending players selected."

    if len(attending_ids) > MAX_ATTENDING:
        attending_ids = attending_ids[:MAX_ATTENDING]

    players_att = players[players["player_id"].isin(attending_ids)].copy()

    if len(players_att) not in SUPPORTED_PLAYER_COUNTS:
        return [], period_rating_cols, [], player_rating_cols, [], empty_cols, [], names_cols, "", (
            f"Official snake templates support {SUPPORTED_PLAYER_COUNTS}. Current selection: {len(players_att)} players."
        )

    wissues = weights_validate(weights.copy())
    if wissues:
        return [], period_rating_cols, [], player_rating_cols, [], empty_cols, [], names_cols, "", f"Weights invalid: {'; '.join(wissues)}"

    metrics_df = compute_optimizer_metrics(players_att, weights)

    manual_turn_targets = None
    if turn_override_rows:
        override_df = pd.DataFrame(turn_override_rows).copy()
        if not override_df.empty:
            for col in ["player_id", "target_turns", "default_turns"]:
                if col in override_df.columns:
                    override_df[col] = pd.to_numeric(override_df[col], errors="coerce")

            if override_df["player_id"].isna().any():
                return [], period_rating_cols, [], player_rating_cols, [], empty_cols, [], names_cols, "", "Turn override table contains invalid player_id values."

            if override_df["target_turns"].isna().any():
                return [], period_rating_cols, [], player_rating_cols, [], empty_cols, [], names_cols, "", "Turn override table contains invalid target_turns values."

            manual_turn_targets = {
                int(row["player_id"]): int(row["target_turns"])
                for _, row in override_df.iterrows()
            }

            try:
                validate_turn_override(
                    manual_turn_targets,
                    num_players=len(metrics_df),
                    expected_player_ids=metrics_df["player_id"].astype(int).tolist(),
                )
            except Exception as e:
                return [], period_rating_cols, [], player_rating_cols, [], empty_cols, [], names_cols, "", f"Turn override invalid: {e}"

    pair_split = []
    names_present = set(metrics_df["name"].astype(str))
    if {"Jens", "Pepijn"}.issubset(names_present):
        pair_split.append(("Jens", "Pepijn"))

    try:
        seeded_view, schedule_df, diagnostics = build_optimized_official_snake_schedule(
            metrics_df,
            manual_turn_targets=manual_turn_targets,
            pair_split_prefs=pair_split,
        )
    except Exception as e:
        return [], period_rating_cols, [], player_rating_cols, [], empty_cols, [], names_cols, "", f"Schedule build error: {e}"

    period_rating_df = build_period_rating_summary(schedule_df)
    player_rating_df = build_player_rating_summary(seeded_view)
    wide_df = schedule_to_wide(schedule_df, seeded_view)
    names_df = schedule_to_names(schedule_df)

    try:
        schedule_df.to_csv(EXPORT_LINEUPS, index=False)
        seeded_view.to_csv(EXPORT_SEEDING, index=False)
        if not wide_df.empty:
            wide_df.to_csv(EXPORT_WIDE, index=False)

        msg_parts = [
            f"Generated optimized official {len(players_att)}-player snake lineup.",
            f"Composite gap={diagnostics.period_score_gap:g}",
            f"Composite avg={diagnostics.average_period_score:g}",
            f"Attack gap={diagnostics.period_attack_score_gap:g}",
            f"Turn mismatch={diagnostics.total_turn_mismatch:g}",
            f"Top-scorer excess={diagnostics.total_top_scorer_excess:g}",
        ]
        if pair_split:
            msg_parts.append(f"Split-pair overlap penalty={diagnostics.total_pair_overlap_penalty:g}")
        msg = " ".join(msg_parts)
    except Exception as e:
        msg = f"Generated optimized lineup. Export failed: {e}"

    period_rating_data = period_rating_df.to_dict(orient="records")
    player_rating_data = player_rating_df.to_dict(orient="records")

    if wide_df.empty:
        wide_cols = [{"name": "period", "id": "period"}]
        wide_data = []
    else:
        wide_cols = [{"name": c, "id": c} for c in wide_df.columns]
        wide_data = wide_df.to_dict(orient="records")

    if names_df.empty:
        n_cols = [{"name": "period", "id": "period"}, {"name": "players", "id": "players"}]
        names_data = []
    else:
        n_cols = [{"name": c, "id": c} for c in names_df.columns]
        names_data = names_df.to_dict(orient="records")

    return (
        period_rating_data,
        period_rating_cols,
        player_rating_data,
        player_rating_cols,
        wide_data,
        wide_cols,
        names_data,
        n_cols,
        msg,
        ""
    )


# ---------- HEALTH ENDPOINT (for hosts) ----------
@server.get("/health")
def health():
    return "ok", 200


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    PERSIST_PLAYERS.parent.mkdir(parents=True, exist_ok=True)
    PERSIST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8052"))
    app.run(debug=False, host=host, port=port)
