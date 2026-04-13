from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import load_csv, normalize_match_data, validate_time_order
from src.features.build import add_prematch_features
from src.features.elo import compute_elo_features


def build_modeling_dataset(input_csv: str | Path, output_csv: str | Path) -> pd.DataFrame:
    raw = load_csv(input_csv)
    normalized = normalize_match_data(raw)
    validate_time_order(normalized)

    with_elo = compute_elo_features(normalized)
    with_features = add_prematch_features(with_elo)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with_features.to_csv(output_csv, index=False)
    return with_features
