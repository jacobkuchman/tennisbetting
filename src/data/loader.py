from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "match_date",
    "tournament",
    "tour",
    "surface",
    "round",
    "player_1",
    "player_2",
    "odds_p1",
    "odds_p2",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common tennis schemas into a stable internal schema."""
    data = df.copy()
    data.columns = [c.strip().lower() for c in data.columns]

    col_map = {
        "date": "match_date",
        "winner": "winner_name",
        "loser": "loser_name",
        "winner_name": "winner_name",
        "loser_name": "loser_name",
        "w_odds": "odds_p1",
        "l_odds": "odds_p2",
        "b365w": "odds_p1",
        "b365l": "odds_p2",
    }
    data = data.rename(columns={k: v for k, v in col_map.items() if k in data.columns})

    for col in REQUIRED_COLUMNS:
        if col not in data.columns:
            data[col] = pd.NA

    # Optional defaults
    if "indoor" not in data.columns:
        data["indoor"] = pd.NA

    data["match_date"] = pd.to_datetime(data["match_date"], errors="coerce")
    data = data.dropna(subset=["match_date", "player_1", "player_2"]).copy()
    data = data.sort_values("match_date").reset_index(drop=True)

    if "p1_win" not in data.columns:
        data["p1_win"] = pd.NA

    if "winner_name" in data.columns:
        missing_target = data["p1_win"].isna()
        data.loc[missing_target, "p1_win"] = (
            data.loc[missing_target, "winner_name"] == data.loc[missing_target, "player_1"]
        ).astype(float)

    if "odds_p1" in data.columns:
        data["odds_p1"] = pd.to_numeric(data["odds_p1"], errors="coerce")
    if "odds_p2" in data.columns:
        data["odds_p2"] = pd.to_numeric(data["odds_p2"], errors="coerce")

    return data


def validate_time_order(df: pd.DataFrame) -> None:
    if not df["match_date"].is_monotonic_increasing:
        raise ValueError("Data must be sorted by match_date ascending.")
