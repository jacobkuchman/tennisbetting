from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.data.loader import normalize_match_data
from src.features.build import add_features_with_history
from src.features.elo import compute_elo_features, compute_elo_with_history
from src.pricing.bankroll import recommend_stake
from src.pricing.markets import price_match_winner_market
from src.pricing.odds import decimal_to_american
from src.utils.config import load_config


def filter_daily_universe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    picks_cfg = cfg.get("daily_picks", {})
    include_atp = bool(picks_cfg.get("include_atp", True))
    include_wta = bool(picks_cfg.get("include_wta", True))
    include_challenger = bool(picks_cfg.get("include_challenger", False))

    out = df.copy()
    out["tour"] = out["tour"].astype(str).str.upper().str.strip()
    out["tournament"] = out["tournament"].astype(str)

    allowed_tours = set()
    if include_atp:
        allowed_tours.add("ATP")
    if include_wta:
        allowed_tours.add("WTA")

    out = out[out["tour"].isin(allowed_tours)].copy()

    if not include_challenger:
        challenger_mask = (
            out["tour"].str.contains("CHALLENGER", case=False, na=False)
            | out["tournament"].str.contains("CHALLENGER", case=False, na=False)
        )
        out = out[~challenger_mask].copy()

    return out


def main(config_path: str = "config/example_config.yaml"):
    cfg = load_config(config_path)

    if cfg.get("data_source", "real") == "real":
        hist_path = cfg["real_data"]["output_historical_merged_csv"]
        upcoming_path = cfg["real_data"]["output_upcoming_merged_csv"]
    else:
        hist_path = cfg["paths"]["historical_matches"]
        upcoming_path = cfg["paths"]["upcoming_matches"]

    hist = pd.read_csv(hist_path, parse_dates=["match_date"])
    upcoming = pd.read_csv(upcoming_path, parse_dates=["match_date"])
    hist = normalize_match_data(hist)
    upcoming = normalize_match_data(upcoming)

    upcoming = filter_daily_universe(upcoming, cfg)

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

    for c in artifacts.feature_columns:
        if c in upcoming_feat.columns:
            upcoming_feat[c] = pd.to_numeric(upcoming_feat[c], errors="coerce")

    probs = artifacts.pipeline.predict_proba(upcoming_feat[artifacts.feature_columns])[:, 1]
    upcoming_feat["model_prob_p1"] = probs

    priced = price_match_winner_market(upcoming_feat, model_prob_col="model_prob_p1")

    priced["odds_american"] = priced["odds_p1"].apply(decimal_to_american)
    priced["fair_odds_american"] = priced["fair_odds_decimal_p1"].apply(decimal_to_american)

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

    picks_cfg = cfg.get("daily_picks", {})
    min_edge = float(picks_cfg.get("minimum_edge", cfg["pricing"]["default_min_edge"]))
    picks = priced[priced["edge"] >= min_edge].sort_values("edge", ascending=False).reset_index(drop=True)

    out_dir = Path(cfg["paths"]["picks_output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "daily_picks.csv"
    picks.to_csv(out_csv, index=False)

    show_cols = [
        "match_date",
        "tour",
        "tournament",
        "player_1",
        "player_2",
        "odds_american",
        "no_vig_prob_p1",
        "model_prob_p1",
        "fair_odds_american",
        "ev",
        "edge",
        "recommended_stake",
    ]

    print("\n=== DAILY PICKS (PHASE 1 MONEYLINE) ===")
    print(f"Filters => ATP: {picks_cfg.get('include_atp', True)}, WTA: {picks_cfg.get('include_wta', True)}, Challenger: {picks_cfg.get('include_challenger', False)}")
    print(f"Minimum edge => {min_edge:.2%}")
    print(f"Matches after filters => {len(picks)}\n")
    if len(picks):
        print(picks[show_cols].to_string(index=False))
    else:
        print("No picks found for the configured filters/threshold.")
    print(f"\nSaved picks to {out_csv}")


if __name__ == "__main__":
    main()
