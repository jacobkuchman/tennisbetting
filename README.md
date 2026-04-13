# Tennis Betting Model - Phase 1 (Moneyline End-to-End)

This repo currently implements **Phase 1 only** for pre-match tennis betting:

1. Load/normalize historical match + moneyline odds data
2. Build leak-safe pre-match features (including Elo)
3. Train calibrated logistic regression for match winner
4. Remove vig from two-way moneyline odds
5. Compute fair odds and expected value (EV)
6. Run a simple walk-forward backtest
7. Generate daily moneyline picks from upcoming matches

> Side markets (spread/totals/first-set/correct-score) are intentionally deferred until this pipeline is stable.

---

## Project layout (relevant for Phase 1)

- `data/raw/historical_matches.csv` - sample historical input
- `data/raw/upcoming_matches.csv` - sample upcoming matches input
- `data/processed/model_dataset.csv` - generated training dataset
- `src/data/` - loading + normalization
- `src/features/` - Elo and feature engineering
- `src/models/match_winner.py` - baseline model + evaluation
- `src/pricing/odds.py` - vig removal, implied/fair odds, EV
- `src/backtest/engine.py` - walk-forward backtest
- `scripts/prepare_data.py`
- `scripts/train_match_winner.py`
- `scripts/run_backtest.py`
- `scripts/daily_picks.py`

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Exact run commands (Phase 1)

### 1) Build processed dataset
```bash
python scripts/prepare_data.py
```

### 2) Train baseline logistic regression
```bash
python scripts/train_match_winner.py
```

Outputs:
- `outputs/models/match_winner_logreg.joblib`
- `outputs/models/match_winner_eval_predictions.csv`

### 3) Run walk-forward backtest
```bash
python scripts/run_backtest.py
```

Output examples:
- `outputs/backtest_bets_edge_2.csv`
- `outputs/backtest_bets_edge_3.csv`
- `outputs/backtest_bets_edge_4.csv`

### 4) Generate daily picks from upcoming matches
```bash
python scripts/daily_picks.py
```

Output:
- `outputs/picks/daily_picks.csv`

---

## Input schema (minimum)

Historical and upcoming CSVs should include (as available):

- `match_date`, `tournament`, `tour`, `surface`, `round`
- `player_1`, `player_2`
- `rank_p1`, `rank_p2` (optional but recommended)
- `odds_p1`, `odds_p2` (required for pricing/EV)
- `p1_win` or `winner_name` (historical only)

Optional performance fields used if present:

- `p1_serve_points_won_pct`, `p2_serve_points_won_pct`
- `p1_return_points_won_pct`, `p2_return_points_won_pct`
- `p1_hold_pct`, `p2_hold_pct`
- `p1_break_pct`, `p2_break_pct`
- `total_games`

---

## Leakage controls

- Features are generated chronologically, row by row.
- Historical states update **after** feature creation for each row.
- Upcoming picks use `historical + upcoming` in time order, with upcoming outcomes blank, so future results cannot leak.
- Backtesting retrains using only matches before each prediction date.

---

## Current status

- ✅ Moneyline Phase 1 pipeline implemented end-to-end.
- ✅ Includes synthetic sample data for local dry runs.
- ⚠️ Sample data is placeholder/synthetic and must be replaced with real ATP/WTA + sportsbook feeds for production use.

