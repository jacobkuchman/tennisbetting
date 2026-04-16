from __future__ import annotations

"""
Example transformation helper for Phase 1 moneyline data.

This script demonstrates how to map typical raw exports into repo schema:
- Tennis Abstract historical match results -> historical_results.csv
- The Odds API historical/upcoming odds -> *_moneyline_odds.csv
- Upcoming match list -> upcoming_matches.csv

Adjust column mappings to your exact provider export fields.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


def transform_tennis_abstract_results(input_csv: str, output_csv: str) -> None:
    raw = pd.read_csv(input_csv)

    # Example mapping. Update keys for your exact downloaded file.
    col_map = {
        "Date": "match_date",
        "Tour": "tour",
        "Tournament": "tournament",
        "Surface": "surface",
        "Round": "round",
        "Winner": "winner_name",
        "Loser": "loser_name",
    }
    data = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

    required = ["match_date", "tour", "tournament", "surface", "round", "winner_name", "loser_name"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Tennis Abstract transform missing columns after mapping: {missing}")

    out = data[required].copy()
    out.to_csv(output_csv, index=False)


def transform_odds_api_moneyline(input_csv: str, output_csv: str) -> None:
    raw = pd.read_csv(input_csv)

    # Example mapping for flattened Odds API exports.
    col_map = {
        "commence_time": "match_date",
        "sport_title": "tour",
        "event_name": "tournament",
        "home_team": "player_1",
        "away_team": "player_2",
        "home_price": "odds_p1",
        "away_price": "odds_p2",
    }
    data = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

    required = ["match_date", "tour", "tournament", "player_1", "player_2", "odds_p1", "odds_p2"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Odds API transform missing columns after mapping: {missing}")

    out = data[required].copy()
    out.to_csv(output_csv, index=False)


def transform_upcoming_matches(input_csv: str, output_csv: str) -> None:
    raw = pd.read_csv(input_csv)

    col_map = {
        "match_date": "match_date",
        "tour": "tour",
        "tournament": "tournament",
        "surface": "surface",
        "round": "round",
        "player_1": "player_1",
        "player_2": "player_2",
    }
    data = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

    required = ["match_date", "tour", "tournament", "surface", "round", "player_1", "player_2"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Upcoming matches transform missing columns after mapping: {missing}")

    out = data[required].copy()
    out.to_csv(output_csv, index=False)


def main() -> None:
    print("Transformation helper loaded.")
    print("Edit mappings in scripts/transform_raw_to_schema.py for your raw provider columns.")
    print("Then call the functions from a Python shell or notebook.")


if __name__ == "__main__":
    main()
