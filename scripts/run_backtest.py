from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest.engine import walk_forward_backtest
from src.features.build import get_feature_columns
from src.utils.config import load_config


def main(config_path: str = "config/example_config.yaml"):
    cfg = load_config(config_path)
    dataset = Path("data/processed/model_dataset.csv")
    if not dataset.exists():
        raise FileNotFoundError("Missing data/processed/model_dataset.csv. Run: python scripts/prepare_data.py")

    df = pd.read_csv(dataset, parse_dates=["match_date"])
    feature_cols = get_feature_columns(df, target_col="p1_win")

    Path("outputs").mkdir(exist_ok=True)
    for edge in cfg["backtest"]["min_edge_thresholds"]:
        bets, metrics = walk_forward_backtest(
            df=df,
            feature_columns=feature_cols,
            start_date=cfg["backtest"]["start_date"],
            min_edge=edge,
            retrain_frequency_days=cfg["backtest"].get("retrain_frequency_days", 30),
            bankroll_start=cfg["bankroll"]["starting_bankroll"],
            staking_mode=cfg["bankroll"]["staking_mode"],
            flat_stake=cfg["bankroll"]["flat_stake"],
            max_stake_pct=cfg["bankroll"]["max_stake_pct"],
        )
        out_file = Path(f"outputs/backtest_bets_edge_{int(edge*100)}.csv")
        bets.to_csv(out_file, index=False)
        print(f"Edge {edge:.2%} => {metrics}")
        print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
