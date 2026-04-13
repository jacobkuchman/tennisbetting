from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        raise FileNotFoundError("Missing data/processed/model_dataset.csv. Run: python scripts/prepare_data.py")

    df = pd.read_csv(dataset_path, parse_dates=["match_date"])
    feature_cols = get_feature_columns(df, target_col="p1_win")

    split_date = pd.Timestamp(cfg["model"]["test_start_date"])
    train_mask = df["match_date"] < split_date
    test_mask = ~train_mask
    if train_mask.sum() < 50 or test_mask.sum() < 20:
        raise ValueError("Not enough train/test rows for baseline training. Adjust test_start_date or provide more data.")

    artifacts = train_match_winner_model(df, feature_cols, "p1_win", train_mask)
    y_prob = predict_proba(artifacts, df.loc[test_mask])
    metrics = evaluate_binary_predictions(df.loc[test_mask, "p1_win"], y_prob)

    model_dir = Path(cfg["paths"]["model_output"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "match_winner_logreg.joblib"
    joblib.dump(artifacts, model_file)

    eval_df = df.loc[test_mask, ["match_date", "player_1", "player_2", "p1_win", "odds_p1", "odds_p2"]].copy()
    eval_df["model_prob_p1"] = y_prob
    eval_df.to_csv(model_dir / "match_winner_eval_predictions.csv", index=False)

    tree = train_tree_if_available(df, feature_cols, "p1_win", train_mask)
    if tree is not None:
        tree_prob = tree["pipeline"].predict_proba(df.loc[test_mask, feature_cols])[:, 1]
        tree_metrics = evaluate_binary_predictions(df.loc[test_mask, "p1_win"], tree_prob)
        print("Tree metrics:", tree_metrics)

    print("Saved model:", model_file)
    print("Baseline metrics:", metrics)


if __name__ == "__main__":
    main()
