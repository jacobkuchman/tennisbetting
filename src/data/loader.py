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
]


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_match_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [c.strip().lower() for c in data.columns]

    col_map = {
        "date": "match_date",
        "winner": "winner_name",
        "loser": "loser_name",
        "w_odds": "odds_p1",
        "l_odds": "odds_p2",
    }
    data = data.rename(columns={k: v for k, v in col_map.items() if k in data.columns})

    for col in REQUIRED_COLUMNS:
        if col not in data.columns:
            data[col] = pd.NA

    data["match_date"] = pd.to_datetime(data["match_date"], errors="coerce")
    data = data.sort_values("match_date").reset_index(drop=True)

    if "winner_name" in data.columns and "player_1" in data.columns:
        data["p1_win"] = (data["winner_name"] == data["player_1"]).astype("float")

    return data


def validate_time_order(df: pd.DataFrame) -> None:
    if not df["match_date"].is_monotonic_increasing:
        raise ValueError("Data must be sorted by match_date ascending.")
