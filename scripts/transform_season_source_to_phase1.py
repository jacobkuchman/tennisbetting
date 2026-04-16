from __future__ import annotations

"""
Transform season-level ATP/WTA source files into Phase 1 moneyline schema files.

Expected input columns include:
- date_human
- date_timestamp
- tour_type_human
- tournament
- round
- surface
- home_name
- away_name
- winner_code
- home_odds_match_winner
- away_odds_match_winner
"""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.real_ingestion import normalize_player_name, normalize_surface, normalize_tournament_name


CRITICAL_FIELDS = ["match_date", "tour", "tournament", "round", "surface", "home_name", "away_name"]


def _norm_display_name(name: str) -> str:
    n = normalize_player_name(name)
    return " ".join(part.capitalize() for part in n.split())


def _norm_tournament(name: str) -> str:
    n = normalize_tournament_name(name)
    return " ".join(part.capitalize() for part in n.split())


def _norm_round(value: str) -> str:
    v = str(value).strip().upper()
    mapping = {
        "R128": "R128",
        "R64": "R64",
        "R32": "R32",
        "R16": "R16",
        "QF": "QF",
        "SF": "SF",
        "F": "F",
        "FINAL": "F",
        "SEMI FINAL": "SF",
        "SEMI-FINAL": "SF",
        "QUARTER FINAL": "QF",
        "QUARTER-FINAL": "QF",
    }
    return mapping.get(v, v)


def _norm_tour(value: str) -> str:
    v = str(value).strip().lower()
    if "wta" in v:
        return "WTA"
    return "ATP"


def _parse_date(df: pd.DataFrame) -> pd.Series:
    if "date_human" in df.columns:
        d1 = pd.to_datetime(df["date_human"], errors="coerce")
    else:
        d1 = pd.Series(pd.NaT, index=df.index)

    if "date_timestamp" in df.columns:
        d2 = pd.to_datetime(df["date_timestamp"], unit="s", errors="coerce")
    else:
        d2 = pd.Series(pd.NaT, index=df.index)

    out = d1.fillna(d2)
    return out.dt.strftime("%Y-%m-%d")


def _winner_from_code(row: pd.Series) -> tuple[str | None, str | None, bool]:
    code = str(row.get("winner_code", "")).strip().lower()
    home = row.get("home_name")
    away = row.get("away_name")

    home_codes = {"h", "home", "1", "p1", "player1"}
    away_codes = {"a", "away", "2", "p2", "player2"}

    if code in home_codes:
        return home, away, True
    if code in away_codes:
        return away, home, True
    return None, None, False


def transform_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    data = df.copy()
    data.columns = [c.strip() for c in data.columns]

    data["match_date"] = _parse_date(data)
    data["tour"] = data["tour_type_human"].map(_norm_tour)
    data["tournament"] = data["tournament"].map(_norm_tournament)
    data["round"] = data["round"].map(_norm_round)
    data["surface"] = data["surface"].map(normalize_surface)
    data["home_name"] = data["home_name"].map(_norm_display_name)
    data["away_name"] = data["away_name"].map(_norm_display_name)

    total_rows = len(data)
    dropped_rows = 0
    missing_odds_rows = 0

    completed_records = []
    historical_odds_records = []
    upcoming_matches_records = []
    upcoming_odds_records = []

    for _, row in data.iterrows():
        if any(pd.isna(row.get(col)) or str(row.get(col)).strip() == "" for col in CRITICAL_FIELDS):
            dropped_rows += 1
            continue

        winner_name, loser_name, is_completed = _winner_from_code(row)

        home_odds = pd.to_numeric(row.get("home_odds_match_winner"), errors="coerce")
        away_odds = pd.to_numeric(row.get("away_odds_match_winner"), errors="coerce")
        has_odds = pd.notna(home_odds) and pd.notna(away_odds)

        base_match = {
            "match_date": row["match_date"],
            "tour": row["tour"],
            "tournament": row["tournament"],
            "surface": row["surface"],
            "round": row["round"],
            "player_1": row["home_name"],
            "player_2": row["away_name"],
        }

        if is_completed and winner_name and loser_name:
            completed_records.append(
                {
                    "match_date": row["match_date"],
                    "tour": row["tour"],
                    "tournament": row["tournament"],
                    "surface": row["surface"],
                    "round": row["round"],
                    "winner_name": winner_name,
                    "loser_name": loser_name,
                }
            )
            if has_odds:
                historical_odds_records.append(
                    {
                        "match_date": row["match_date"],
                        "tour": row["tour"],
                        "tournament": row["tournament"],
                        "player_1": row["home_name"],
                        "player_2": row["away_name"],
                        "odds_p1": float(home_odds),
                        "odds_p2": float(away_odds),
                    }
                )
            else:
                missing_odds_rows += 1
        else:
            upcoming_matches_records.append(base_match)
            if has_odds:
                upcoming_odds_records.append(
                    {
                        "match_date": row["match_date"],
                        "tour": row["tour"],
                        "tournament": row["tournament"],
                        "player_1": row["home_name"],
                        "player_2": row["away_name"],
                        "odds_p1": float(home_odds),
                        "odds_p2": float(away_odds),
                    }
                )
            else:
                missing_odds_rows += 1

    historical_results = pd.DataFrame(completed_records)
    historical_odds = pd.DataFrame(historical_odds_records)
    upcoming_matches = pd.DataFrame(upcoming_matches_records)
    upcoming_odds = pd.DataFrame(upcoming_odds_records)

    summary = {
        "total_rows": total_rows,
        "completed_matches_used": len(historical_results),
        "upcoming_matches_used": len(upcoming_matches),
        "rows_dropped": dropped_rows,
        "rows_missing_odds": missing_odds_rows,
    }
    return historical_results, historical_odds, upcoming_matches, upcoming_odds, summary


def main(input_dir: str = "data/raw/source_seasons", output_dir: str = "data/raw/real") -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = ["*-atp-season.csv", "*-wta-season.csv"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(in_dir.glob(pattern)))

    if not files:
        raise FileNotFoundError(
            f"No season files found in {in_dir}. Expected names like 2024-atp-season.csv, 2025-wta-season.csv"
        )

    frames = [pd.read_csv(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)

    hist_res, hist_odds, up_matches, up_odds, summary = transform_dataframe(combined)

    hist_res.to_csv(out_dir / "historical_match_results.csv", index=False)
    hist_odds.to_csv(out_dir / "historical_moneyline_odds.csv", index=False)
    up_matches.to_csv(out_dir / "upcoming_matches.csv", index=False)
    up_odds.to_csv(out_dir / "upcoming_moneyline_odds.csv", index=False)

    print("Season transform complete:")
    for k, v in summary.items():
        print(f"- {k}: {v}")
    print(f"- outputs_dir: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform ATP/WTA season files to Phase 1 schema")
    parser.add_argument("--input-dir", default="data/raw/source_seasons")
    parser.add_argument("--output-dir", default="data/raw/real")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)
