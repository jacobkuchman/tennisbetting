from __future__ import annotations

from src.data.pipeline import build_modeling_dataset
from src.utils.config import load_config


def main(config_path: str = "config/example_config.yaml"):
    cfg = load_config(config_path)
    out = "data/processed/model_dataset.csv"
    df = build_modeling_dataset(cfg["paths"]["historical_matches"], out)
    print(f"Prepared dataset rows={len(df)} saved to {out}")


if __name__ == "__main__":
    main()
