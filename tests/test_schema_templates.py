from pathlib import Path

import pandas as pd


TEMPLATE_FILES = [
    "historical_match_results.csv",
    "historical_moneyline_odds.csv",
    "upcoming_matches.csv",
    "upcoming_moneyline_odds.csv",
]


def test_template_files_exist_and_have_headers():
    base = Path("data/templates")
    for name in TEMPLATE_FILES:
        path = base / name
        assert path.exists(), f"Missing template: {path}"
        df = pd.read_csv(path)
        assert len(df.columns) > 0
