import pandas as pd

from src.data.real_ingestion import (
    OddsSchema,
    RealSchema,
    merge_results_with_odds,
    normalize_player_name,
    normalize_tournament_name,
)


def test_name_normalization_handles_accents_and_punctuation():
    assert normalize_player_name("J. Mónaco") == "j monaco"
    assert normalize_player_name("Rafael Nadal!") == "rafael nadal"


def test_tournament_normalization_reduces_aliases():
    assert normalize_tournament_name("ATP Masters 1000 Rome") == "masters rome"


def test_merge_results_odds_with_normalized_names():
    results = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2024-01-01"]),
            "tour": ["ATP"],
            "tournament": ["ATP Masters 1000 Rome"],
            "tournament_norm": ["masters rome"],
            "surface": ["Clay"],
            "round": ["R32"],
            "winner_name": ["Rafael Nadal"],
            "loser_name": ["Novak Djokovic"],
            "winner_norm": ["rafael nadal"],
            "loser_norm": ["novak djokovic"],
            "player_1_norm": ["novak djokovic"],
            "player_2_norm": ["rafael nadal"],
            "p1_win": [0.0],
        }
    )
    odds = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2024-01-01"]),
            "tour": ["ATP"],
            "tournament_norm": ["masters rome"],
            "player_1_norm": ["novak djokovic"],
            "player_2_norm": ["rafael nadal"],
            "odds_p1": [1.91],
            "odds_p2": [1.91],
        }
    )

    merged = merge_results_with_odds(results, odds)
    assert merged.loc[0, "odds_p1"] == 1.91
    assert merged.loc[0, "player_1"] == "novak djokovic"


def test_schema_dataclasses_construct():
    _ = RealSchema("d", "t", "tr", "s", "r", "w", "l")
    _ = OddsSchema("d", "t", "tr", "p1", "p2", "o1", "o2")


def test_merge_results_allows_unmatched_odds_rows():
    results = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2024-01-01"]),
            "tour": ["ATP"],
            "tournament": ["Rome"],
            "tournament_norm": ["rome"],
            "surface": ["Clay"],
            "round": ["R32"],
            "winner_name": ["A"],
            "loser_name": ["B"],
            "winner_norm": ["a"],
            "loser_norm": ["b"],
            "player_1_norm": ["a"],
            "player_2_norm": ["b"],
            "p1_win": [1.0],
        }
    )
    odds = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2024-01-01"]),
            "tour": ["ATP"],
            "tournament_norm": ["rome"],
            "player_1_norm": ["x"],
            "player_2_norm": ["y"],
            "odds_p1": [1.9],
            "odds_p2": [1.9],
        }
    )
    merged = merge_results_with_odds(results, odds)
    assert merged.loc[0, "odds_matched"] == False
    assert pd.isna(merged.loc[0, "odds_p1"])
