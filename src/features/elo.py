from __future__ import annotations

from collections import defaultdict

import pandas as pd


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1 + 10 ** ((rating_b - rating_a) / 400))


def compute_elo_features(df: pd.DataFrame, k_factor: float = 24.0) -> pd.DataFrame:
    data = df.copy().sort_values("match_date").reset_index(drop=True)

    overall = defaultdict(lambda: 1500.0)
    by_surface = defaultdict(lambda: 1500.0)

    elo_p1_list = []
    elo_p2_list = []
    s_elo_p1_list = []
    s_elo_p2_list = []

    for _, row in data.iterrows():
        p1, p2 = row["player_1"], row["player_2"]
        surface = row.get("surface", "unknown")

        e1 = overall[p1]
        e2 = overall[p2]
        se1 = by_surface[(p1, surface)]
        se2 = by_surface[(p2, surface)]

        elo_p1_list.append(e1)
        elo_p2_list.append(e2)
        s_elo_p1_list.append(se1)
        s_elo_p2_list.append(se2)

        if pd.notna(row.get("p1_win")):
            p1_win = float(row["p1_win"])
            exp1 = expected_score(e1, e2)
            exp1_s = expected_score(se1, se2)
            overall[p1] = e1 + k_factor * (p1_win - exp1)
            overall[p2] = e2 + k_factor * ((1 - p1_win) - (1 - exp1))
            by_surface[(p1, surface)] = se1 + k_factor * (p1_win - exp1_s)
            by_surface[(p2, surface)] = se2 + k_factor * ((1 - p1_win) - (1 - exp1_s))

    data["elo_p1"] = elo_p1_list
    data["elo_p2"] = elo_p2_list
    data["elo_diff"] = data["elo_p1"] - data["elo_p2"]
    data["surface_elo_p1"] = s_elo_p1_list
    data["surface_elo_p2"] = s_elo_p2_list
    data["surface_elo_diff"] = data["surface_elo_p1"] - data["surface_elo_p2"]
    return data


def compute_elo_with_history(historical_df: pd.DataFrame, target_df: pd.DataFrame, k_factor: float = 24.0) -> pd.DataFrame:
    """Compute ELO for target rows using only historical row outcomes."""
    hist = historical_df.copy()
    tgt = target_df.copy()
    hist["_is_target"] = 0
    tgt["_is_target"] = 1
    if "p1_win" not in tgt.columns:
        tgt["p1_win"] = pd.NA
    tgt["p1_win"] = pd.NA

    combo = pd.concat([hist, tgt], ignore_index=True, sort=False).sort_values("match_date").reset_index(drop=True)
    combo = compute_elo_features(combo, k_factor=k_factor)
    return combo[combo["_is_target"] == 1].drop(columns=["_is_target"]).reset_index(drop=True)
