from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.features.build import get_feature_columns
from src.models.markets import RegressionMarketModel


def main():
    df = pd.read_csv("data/processed/model_dataset.csv", parse_dates=["match_date"])
    feature_cols = get_feature_columns(df, target_col="p1_win")
    train_mask = df["match_date"] < pd.Timestamp("2024-01-01")

    models = {}
    if "game_margin" in df.columns:
        gm = RegressionMarketModel(feature_cols, "game_margin").fit(df, train_mask)
        models["game_margin"] = gm
    if "total_games" in df.columns:
        tg = RegressionMarketModel(feature_cols, "total_games").fit(df, train_mask)
        models["total_games"] = tg

    joblib.dump(models, "outputs/models/side_market_models.joblib")
    print("Saved side-market baseline models:", list(models.keys()))


if __name__ == "__main__":
    main()
