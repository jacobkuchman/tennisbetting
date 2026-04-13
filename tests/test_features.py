import pandas as pd

from src.features.build import add_features_with_history, add_prematch_features
from src.features.elo import compute_elo_features


def test_elo_no_leakage_shape():
    df = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "player_1": ["A", "A"],
            "player_2": ["B", "C"],
            "surface": ["Hard", "Hard"],
            "p1_win": [1, 0],
        }
    )
    out = compute_elo_features(df)
    assert "elo_diff" in out.columns
    assert out.loc[0, "elo_p1"] == 1500


def test_recent_form_feature_exists():
    df = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "player_1": ["A", "A", "A"],
            "player_2": ["B", "C", "D"],
            "surface": ["Hard", "Hard", "Clay"],
            "tour": ["ATP", "ATP", "ATP"],
            "p1_win": [1, 0, 1],
        }
    )
    out = add_prematch_features(df, recent_windows=[2])
    assert "recent_win_rate_diff_2" in out.columns


def test_target_features_use_history_only():
    hist = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "player_1": ["A", "A"],
            "player_2": ["B", "C"],
            "surface": ["Hard", "Hard"],
            "tour": ["ATP", "ATP"],
            "p1_win": [1, 1],
        }
    )
    upcoming = pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2024-01-03"]),
            "player_1": ["A"],
            "player_2": ["D"],
            "surface": ["Hard"],
            "tour": ["ATP"],
        }
    )
    out = add_features_with_history(hist, upcoming, recent_windows=[2])
    assert len(out) == 1
    assert out.loc[0, "recent_win_rate_diff_2"] > 0
