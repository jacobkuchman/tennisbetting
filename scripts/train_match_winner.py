from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features.build import get_feature_columns
from src.models.match_winner import evaluate_binary_predictions, predict_proba, train_match_winner_model
from src.models.tree_model import train_tree_if_available
from src.utils.config import load_config


def main(config_path: str = "config/example_config.yaml"):
    cfg = load_config(config_path)
    dataset_path = Path("data/processed/model_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError("Run dataset build first: scripts/prepare_data.py")

    df = pd.read_csv(dataset_path, parse_dates=["match_date"])
    feature_cols = get_feature_columns(df, target_col="p1_win")

    split_date = pd.Timestamp(cfg["model"]["test_start_date"])
    train_mask = df["match_date"] < split_date
    test_mask = ~train_mask

    artifacts = train_match_winner_model(df, feature_cols, "p1_win", train_mask)
    y_prob = predict_proba(artifacts, df.loc[test_mask])
    metrics = evaluate_binary_predictions(df.loc[test_mask, "p1_win"], y_prob)

    Path(cfg["paths"]["model_output"]).mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, Path(cfg["paths"]["model_output"]) / "match_winner_logreg.joblib")

    tree = train_tree_if_available(df, feature_cols, "p1_win", train_mask)
    if tree is not None:
        tree_prob = tree["pipeline"].predict_proba(df.loc[test_mask, feature_cols])[:, 1]
        tree_metrics = evaluate_binary_predictions(df.loc[test_mask, "p1_win"], tree_prob)
        print("Tree metrics:", tree_metrics)

    print("Baseline metrics:", metrics)


if __name__ == "__main__":
    main()
