from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features.build import add_prematch_features
from src.features.elo import compute_elo_features
from src.pricing.bankroll import recommend_stake
from src.pricing.odds import edge_vs_market, expected_value, implied_prob_to_decimal, remove_vig_two_way
from src.utils.config import load_config


def main(config_path: str = "config/example_config.yaml"):
    cfg = load_config(config_path)
    upcoming = pd.read_csv(cfg["paths"]["upcoming_matches"], parse_dates=["match_date"])

    upcoming = compute_elo_features(upcoming)
    upcoming = add_prematch_features(upcoming, cfg["features"]["recent_windows"])

    artifacts = joblib.load(Path(cfg["paths"]["model_output"]) / "match_winner_logreg.joblib")
    probs = artifacts.pipeline.predict_proba(upcoming[artifacts.feature_columns])[:, 1]
    upcoming["model_prob_p1"] = probs

    upcoming[["no_vig_prob_p1", "no_vig_prob_p2"]] = upcoming.apply(
        lambda r: pd.Series(remove_vig_two_way(r["odds_p1"], r["odds_p2"])), axis=1
    )
    upcoming["edge"] = upcoming.apply(lambda r: edge_vs_market(r["model_prob_p1"], r["no_vig_prob_p1"]), axis=1)
    upcoming["ev"] = upcoming.apply(lambda r: expected_value(r["model_prob_p1"], r["odds_p1"]), axis=1)
    upcoming["fair_odds_decimal_p1"] = upcoming["model_prob_p1"].apply(implied_prob_to_decimal)

    bankroll = cfg["bankroll"]["starting_bankroll"]
    upcoming["recommended_stake"] = upcoming.apply(
        lambda r: recommend_stake(
            bankroll=bankroll,
            prob=r["model_prob_p1"],
            decimal_odds=r["odds_p1"],
            mode=cfg["bankroll"]["staking_mode"],
            flat_stake=cfg["bankroll"]["flat_stake"],
            max_stake_pct=cfg["bankroll"]["max_stake_pct"],
        ),
        axis=1,
    )

    picks = upcoming[upcoming["edge"] >= cfg["pricing"]["default_min_edge"]].sort_values("edge", ascending=False)

    out_dir = Path(cfg["paths"]["picks_output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "daily_picks.csv"
    picks.to_csv(out_csv, index=False)

    cols = [
        "match_date",
        "tour",
        "player_1",
        "player_2",
        "odds_p1",
        "no_vig_prob_p1",
        "model_prob_p1",
        "fair_odds_decimal_p1",
        "edge",
        "ev",
        "recommended_stake",
    ]
    print(picks[cols].to_string(index=False))
    print(f"Saved picks to {out_csv}")


if __name__ == "__main__":
    main()
