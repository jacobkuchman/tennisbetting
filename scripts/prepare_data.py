from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.pipeline import build_modeling_dataset, build_real_moneyline_datasets
from src.data.real_ingestion import OddsSchema, RealSchema
from src.utils.config import load_config


def main(config_path: str = "config/example_config.yaml"):
    cfg = load_config(config_path)
    recent_windows = cfg.get("features", {}).get("recent_windows", [5, 10])

    source_mode = cfg.get("data_source", "real")
    if source_mode == "real":
        real = cfg["real_data"]
        hist_schema = RealSchema(**real["historical_results_schema"])
        odds_schema = OddsSchema(**real["odds_schema"])

        hist_merged, upcoming_merged, model_df, merge_summary = build_real_moneyline_datasets(
            historical_results_path=real["historical_results_path"],
            historical_odds_path=real["historical_odds_path"],
            upcoming_matches_path=real["upcoming_matches_path"],
            upcoming_odds_path=real["upcoming_odds_path"],
            output_historical_merged_csv=real["output_historical_merged_csv"],
            output_upcoming_merged_csv=real["output_upcoming_merged_csv"],
            recent_windows=recent_windows,
            historical_schema=hist_schema,
            odds_schema=odds_schema,
            upcoming_match_schema=real.get("upcoming_matches_schema"),
            unmatched_historical_debug_csv=real.get("unmatched_historical_debug_csv", "outputs/debug/unmatched_historical_results.csv"),
        )
        print(f"Real-data merged historical rows={len(hist_merged)} -> {real['output_historical_merged_csv']}")
        print(f"Real-data merged upcoming rows={len(upcoming_merged)} -> {real['output_upcoming_merged_csv']}")
        print(f"Model-ready rows={len(model_df)} -> data/processed/model_dataset.csv")
        print("Historical odds match summary:")
        print(f"- total historical rows: {merge_summary['historical_total_rows']}")
        print(f"- matched odds rows: {merge_summary['historical_matched_odds_rows']}")
        print(f"- unmatched odds rows: {merge_summary['historical_unmatched_odds_rows']}")
        print(f"- unmatched rows saved: {merge_summary['unmatched_debug_csv']}")
    else:
        out = "data/processed/model_dataset.csv"
        df = build_modeling_dataset(
            cfg["paths"]["historical_matches"],
            out,
            recent_windows=recent_windows,
        )
        print(f"Sample mode dataset rows={len(df)} saved to {out}")


if __name__ == "__main__":
    main()
