from __future__ import annotations

import pandas as pd

from src.pricing.odds import decimal_to_american, edge_vs_market, expected_value, remove_vig_two_way, safe_implied_prob_to_decimal


def price_match_winner_market(df: pd.DataFrame, model_prob_col: str = "model_prob_p1") -> pd.DataFrame:
    out = df.copy()
    out[["no_vig_prob_p1", "no_vig_prob_p2"]] = out.apply(
        lambda r: pd.Series(remove_vig_two_way(r["odds_p1"], r["odds_p2"])), axis=1
    )
    out["edge"] = out.apply(lambda r: edge_vs_market(r[model_prob_col], r["no_vig_prob_p1"]), axis=1)
    out["ev"] = out.apply(lambda r: expected_value(r[model_prob_col], r["odds_p1"]), axis=1)
    out["fair_odds_decimal_p1"] = out[model_prob_col].apply(safe_implied_prob_to_decimal)
    out["fair_odds_american_p1"] = out["fair_odds_decimal_p1"].apply(decimal_to_american)
    return out
