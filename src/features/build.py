from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _safe_get(row: pd.Series, key: str, default: float = np.nan) -> float:
    val = row.get(key, default)
    return default if pd.isna(val) else float(val)


def add_prematch_features(df: pd.DataFrame, recent_windows: list[int] | None = None) -> pd.DataFrame:
    recent_windows = recent_windows or [5, 10]
    data = df.copy().sort_values("match_date").reset_index(drop=True)

    win_hist = defaultdict(list)
    surface_win_hist = defaultdict(list)
    matches_played = defaultdict(list)
    long_match_hist = defaultdict(list)
    h2h = defaultdict(float)

    out_rows: list[dict] = []

    for _, row in data.iterrows():
        p1 = row["player_1"]
        p2 = row["player_2"]
        surface = row.get("surface", "unknown")
        tour = row.get("tour", "UNK")

        feat: dict = {}
        feat["ranking_gap"] = _safe_get(row, "rank_p2") - _safe_get(row, "rank_p1")
        feat["elo_diff"] = _safe_get(row, "elo_diff", 0.0)
        feat["surface_elo_diff"] = _safe_get(row, "surface_elo_diff", 0.0)

        for w in recent_windows:
            p1_recent = win_hist[p1][-w:]
            p2_recent = win_hist[p2][-w:]
            feat[f"recent_win_rate_diff_{w}"] = (
                (np.mean(p1_recent) if p1_recent else 0.5)
                - (np.mean(p2_recent) if p2_recent else 0.5)
            )

            p1_surface_recent = surface_win_hist[(p1, surface)][-w:]
            p2_surface_recent = surface_win_hist[(p2, surface)][-w:]
            feat[f"surface_recent_win_rate_diff_{w}"] = (
                (np.mean(p1_surface_recent) if p1_surface_recent else 0.5)
                - (np.mean(p2_surface_recent) if p2_surface_recent else 0.5)
            )

        feat["fatigue_matches_7d_diff"] = sum(matches_played[p1][-7:]) - sum(matches_played[p2][-7:])
        feat["fatigue_long_matches_14d_diff"] = sum(long_match_hist[p1][-14:]) - sum(long_match_hist[p2][-14:])

        feat["h2h_weighted_diff"] = h2h[(p1, p2)] - h2h[(p2, p1)]
        feat["is_atp"] = 1 if str(tour).upper() == "ATP" else 0

        feat["tournament_tier"] = _safe_get(row, "tournament_tier", 0)

        feat["serve_points_won_diff"] = _safe_get(row, "p1_serve_points_won_pct") - _safe_get(row, "p2_serve_points_won_pct")
        feat["return_points_won_diff"] = _safe_get(row, "p1_return_points_won_pct") - _safe_get(row, "p2_return_points_won_pct")
        feat["hold_pct_diff"] = _safe_get(row, "p1_hold_pct") - _safe_get(row, "p2_hold_pct")
        feat["break_pct_diff"] = _safe_get(row, "p1_break_pct") - _safe_get(row, "p2_break_pct")

        out_rows.append(feat)

        if pd.notna(row.get("p1_win")):
            p1_win = float(row["p1_win"])
            win_hist[p1].append(p1_win)
            win_hist[p2].append(1 - p1_win)
            surface_win_hist[(p1, surface)].append(p1_win)
            surface_win_hist[(p2, surface)].append(1 - p1_win)
            h2h[(p1, p2)] += p1_win * 0.5
            h2h[(p2, p1)] += (1 - p1_win) * 0.5

        total_games = _safe_get(row, "total_games", 0)
        long_flag = 1 if total_games >= 30 else 0
        matches_played[p1].append(1)
        matches_played[p2].append(1)
        long_match_hist[p1].append(long_flag)
        long_match_hist[p2].append(long_flag)

    features = pd.DataFrame(out_rows)
    return pd.concat([data.reset_index(drop=True), features], axis=1)


def get_feature_columns(df: pd.DataFrame, target_col: str = "p1_win") -> list[str]:
    excluded = {
        target_col,
        "match_date",
        "player_1",
        "player_2",
        "winner_name",
        "loser_name",
        "tournament",
        "surface",
        "round",
        "tour",
    }
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
