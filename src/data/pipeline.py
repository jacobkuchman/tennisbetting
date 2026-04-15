from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import load_csv, normalize_match_data, validate_time_order
from src.data.real_ingestion import (
    OddsSchema,
    RealSchema,
    load_historical_results,
    load_moneyline_odds,
    merge_results_with_odds,
    merge_upcoming_with_odds,
)
from src.features.build import add_prematch_features
from src.features.elo import compute_elo_features


def build_modeling_dataset(
    input_csv: str | Path,
    output_csv: str | Path,
    recent_windows: list[int] | None = None,
) -> pd.DataFrame:
    raw = load_csv(input_csv)
    normalized = normalize_match_data(raw)
    validate_time_order(normalized)

    with_elo = compute_elo_features(normalized)
    with_features = add_prematch_features(with_elo, recent_windows=recent_windows)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with_features.to_csv(output_csv, index=False)
    return with_features


def build_real_moneyline_datasets(
    historical_results_path: str | Path,
    historical_odds_path: str | Path,
    upcoming_matches_path: str | Path,
    upcoming_odds_path: str | Path,
    output_historical_merged_csv: str | Path,
    output_upcoming_merged_csv: str | Path,
    recent_windows: list[int] | None = None,
    historical_schema: RealSchema | None = None,
    odds_schema: OddsSchema | None = None,
    upcoming_match_schema: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    historical_schema = historical_schema or RealSchema(
        match_date="match_date",
        tour="tour",
        tournament="tournament",
        surface="surface",
        round="round",
        winner_name="winner_name",
        loser_name="loser_name",
    )
    odds_schema = odds_schema or OddsSchema(
        odds_date="match_date",
        tour="tour",
        tournament="tournament",
        player_1="player_1",
        player_2="player_2",
        odds_p1="odds_p1",
        odds_p2="odds_p2",
    )
    upcoming_match_schema = upcoming_match_schema or {
        "match_date": "match_date",
        "tour": "tour",
        "tournament": "tournament",
        "surface": "surface",
        "round": "round",
        "player_1": "player_1",
        "player_2": "player_2",
    }

    results = load_historical_results(historical_results_path, schema=historical_schema)
    hist_odds = load_moneyline_odds(historical_odds_path, schema=odds_schema, is_upcoming=False)
    historical_merged = merge_results_with_odds(results, hist_odds)

    upcoming_raw = load_csv(upcoming_matches_path).rename(columns=upcoming_match_schema)
    required_upcoming_cols = ["match_date", "tour", "tournament", "surface", "round", "player_1", "player_2"]
    miss = [c for c in required_upcoming_cols if c not in upcoming_raw.columns]
    if miss:
        raise ValueError(f"Upcoming matches missing required columns: {miss}")
    upcoming_raw = upcoming_raw[required_upcoming_cols].copy()
    upcoming_raw["match_date"] = pd.to_datetime(upcoming_raw["match_date"], errors="coerce")
    upcoming_raw = upcoming_raw.dropna(subset=["match_date", "player_1", "player_2"])

    upcoming_odds = load_moneyline_odds(upcoming_odds_path, schema=odds_schema, is_upcoming=True)
    upcoming_merged = merge_upcoming_with_odds(upcoming_raw, upcoming_odds)

    Path(output_historical_merged_csv).parent.mkdir(parents=True, exist_ok=True)
    historical_merged.to_csv(output_historical_merged_csv, index=False)
    upcoming_merged.to_csv(output_upcoming_merged_csv, index=False)

    model_ready = build_modeling_dataset(
        input_csv=output_historical_merged_csv,
        output_csv=Path(output_historical_merged_csv).with_name("model_dataset.csv"),
        recent_windows=recent_windows,
    )
    return historical_merged, upcoming_merged, model_ready
