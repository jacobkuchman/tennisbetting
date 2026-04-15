from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

REQUIRED = {
    "historical_match_results.csv": [
        "match_date",
        "tour",
        "tournament",
        "surface",
        "round",
        "winner_name",
        "loser_name",
    ],
    "historical_moneyline_odds.csv": [
        "match_date",
        "tour",
        "tournament",
        "player_1",
        "player_2",
        "odds_p1",
        "odds_p2",
    ],
    "upcoming_matches.csv": [
        "match_date",
        "tour",
        "tournament",
        "surface",
        "round",
        "player_1",
        "player_2",
    ],
    "upcoming_moneyline_odds.csv": [
        "match_date",
        "tour",
        "tournament",
        "player_1",
        "player_2",
        "odds_p1",
        "odds_p2",
    ],
}


def validate_file(path: Path, required_cols: list[str]) -> list[str]:
    if not path.exists():
        return [f"File missing: {path}"]
    try:
        df = pd.read_csv(path, nrows=1)
    except Exception as exc:
        return [f"Could not read {path}: {exc}"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return [f"{path.name} missing columns: {missing}"]
    return []


def main(base_dir: str = "data/raw/real"):
    base = Path(base_dir)
    mapping = {
        "historical_match_results.csv": base / "historical_results.csv",
        "historical_moneyline_odds.csv": base / "historical_moneyline_odds.csv",
        "upcoming_matches.csv": base / "upcoming_matches.csv",
        "upcoming_moneyline_odds.csv": base / "upcoming_moneyline_odds.csv",
    }

    errors: list[str] = []
    for template_name, required_cols in REQUIRED.items():
        errors.extend(validate_file(mapping[template_name], required_cols))

    if errors:
        print("Schema validation FAILED:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("Schema validation passed for all Phase 1 moneyline files.")


if __name__ == "__main__":
    main()
