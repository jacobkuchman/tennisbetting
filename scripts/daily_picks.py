from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data.loader import normalize_match_data
from src.features.build import add_features_with_history
from src.features.elo import compute_elo_features, compute_elo_with_history
from src.pricing.bankroll import recommend_stake
from src.pricing.markets import price_match_winner_market
from src.utils.config import load_config


def main(config_path: str = "config/example_config.yaml"):
    cfg = load_config(config_path)

    hist = pd.read_csv(cfg["paths"]["historical_matches"], parse_dates=["match_date"])
    upcoming = pd.read_csv(cfg["paths"]["upcoming_matches"], parse_dates=["match_date"])
    hist = normalize_match_data(hist)
    upcoming = normalize_match_data(upcoming)

    upcoming_elo = compute_elo_with_history(hist, upcoming)
    upcoming_feat = add_features_with_history(
        historical_df=compute_elo_features(hist),
        target_df=upcoming_elo,
        recent_windows=cfg["features"]["recent_windows"],
    )

    model_path = Path(cfg["paths"]["model_output"]) / "match_winner_logreg.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model missing. Run: python scripts/train_match_winner.py")
    artifacts = joblib.load(model_path)

    probs = artifacts.pipeline.predict_proba(upcoming_feat[artifacts.feature_columns])[:, 1]
    upcoming_feat["model_prob_p1"] = probs

    priced = price_match_winner_market(upcoming_feat, model_prob_col="model_prob_p1")

    bankroll = cfg["bankroll"]["starting_bankroll"]
    priced["recommended_stake"] = priced.apply(
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

    picks = priced[priced["edge"] >= cfg["pricing"]["default_min_edge"]].sort_values("edge", ascending=False)

    out_dir = Path(cfg["paths"]["picks_output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "daily_picks.csv"
    picks.to_csv(out_csv, index=False)

    show_cols = [
        "match_date",
        "tour",
        "player_1",
        "player_2",
        "odds_p1",
        "no_vig_prob_p1",
        "model_prob_p1",
        "fair_odds_decimal_p1",
        "ev",
        "edge",
        "recommended_stake",
    ]
    print(picks[show_cols].to_string(index=False))
    print(f"Saved picks to {out_csv}")


if __name__ == "__main__":
    main()
