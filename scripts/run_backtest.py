from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.engine import walk_forward_backtest
from src.features.build import get_feature_columns
from src.utils.config import load_config


def main(config_path: str = "config/example_config.yaml"):
    cfg = load_config(config_path)
    df = pd.read_csv("data/processed/model_dataset.csv", parse_dates=["match_date"])
    feature_cols = get_feature_columns(df, target_col="p1_win")

    for edge in cfg["backtest"]["min_edge_thresholds"]:
        bets, metrics = walk_forward_backtest(
            df=df,
            feature_columns=feature_cols,
            start_date=cfg["backtest"]["start_date"],
            min_edge=edge,
            bankroll_start=cfg["bankroll"]["starting_bankroll"],
            staking_mode=cfg["bankroll"]["staking_mode"],
            flat_stake=cfg["bankroll"]["flat_stake"],
            max_stake_pct=cfg["bankroll"]["max_stake_pct"],
        )
        Path("outputs").mkdir(exist_ok=True)
        bets.to_csv(f"outputs/backtest_bets_edge_{int(edge*100)}.csv", index=False)
        print(f"Edge {edge:.2%} => {metrics}")


if __name__ == "__main__":
    main()
