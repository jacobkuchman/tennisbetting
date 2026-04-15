# Tennis Betting Model - Phase 1 (Moneyline Only)

This repository currently supports **Phase 1 only**: pre-match tennis **moneyline** modeling and +EV detection.

It does **not** yet include production side-market pricing (spreads/totals/first-set/correct-score).

---

## What Phase 1 does end-to-end

1. Load and normalize historical match + odds data
2. Build leak-safe pre-match features (including Elo)
3. Train calibrated logistic regression for match winner probability
4. Remove vig from two-way moneyline odds
5. Convert probabilities to fair odds and compute EV/edge
6. Run walk-forward backtest (past-only training)
7. Generate daily picks for upcoming matches

---

## Clean local setup (exact commands)

Run from repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/train_match_winner.py
python scripts/run_backtest.py
python scripts/daily_picks.py
```

---

## Expected outputs by command

### `python scripts/prepare_data.py`
Creates:
- `data/processed/model_dataset.csv`

### `python scripts/train_match_winner.py`
Creates:
- `outputs/models/match_winner_logreg.joblib`
- `outputs/models/match_winner_eval_predictions.csv`

Prints:
- baseline test metrics (`log_loss`, `brier`, and `auc` when available)

### `python scripts/run_backtest.py`
Creates (per configured edge threshold):
- `outputs/backtest_bets_edge_2.csv`
- `outputs/backtest_bets_edge_3.csv`
- `outputs/backtest_bets_edge_4.csv`

Prints:
- bets, hit rate, average edge, stake totals, pnl, roi, max drawdown

### `python scripts/daily_picks.py`
Creates:
- `outputs/picks/daily_picks.csv`

Prints:
- readable picks table sorted by edge

---

## Real-data schema (required columns)

You can plug in any source as long as you provide the columns below.

### Required for `data/raw/historical_matches.csv`

| Column | Type | Required | Notes |
|---|---|---:|---|
| `match_date` | date/datetime | ✅ | parseable by pandas |
| `player_1` | string | ✅ | model is framed as P1 vs P2 |
| `player_2` | string | ✅ | |
| `tournament` | string | ✅ | |
| `tour` | string | ✅ | e.g. ATP, WTA |
| `surface` | string | ✅ | Hard/Clay/Grass/etc |
| `round` | string | ✅ | |
| `odds_p1` | float | ✅ | decimal moneyline for P1 |
| `odds_p2` | float | ✅ | decimal moneyline for P2 |
| `p1_win` | 0/1 | ✅* | required unless `winner_name` exists |
| `winner_name` | string | ✅* | optional if `p1_win` provided |

`*` At least one target source must exist: either `p1_win`, or `winner_name` from which `p1_win` can be derived.

### Required for `data/raw/upcoming_matches.csv`

| Column | Type | Required | Notes |
|---|---|---:|---|
| `match_date` | date/datetime | ✅ | upcoming/premarket date |
| `player_1` | string | ✅ | |
| `player_2` | string | ✅ | |
| `tournament` | string | ✅ | |
| `tour` | string | ✅ | ATP/WTA recommended |
| `surface` | string | ✅ | |
| `round` | string | ✅ | |
| `odds_p1` | float | ✅ | decimal moneyline |
| `odds_p2` | float | ✅ | decimal moneyline |

### Optional feature columns (used when present)

- `rank_p1`, `rank_p2`
- `tournament_tier`
- `p1_serve_points_won_pct`, `p2_serve_points_won_pct`
- `p1_return_points_won_pct`, `p2_return_points_won_pct`
- `p1_hold_pct`, `p2_hold_pct`
- `p1_break_pct`, `p2_break_pct`
- `total_games`
- `indoor`

---

## Leakage controls in this implementation

- Features are generated in chronological order.
- Player histories update **after** each row’s features are built.
- Upcoming rows are featurized with historical context while their outcomes are forced to null.
- Walk-forward backtest retrains only on matches before each prediction date.

---

## Troubleshooting

### `ModuleNotFoundError: pandas` / `numpy` / `sklearn`
Install dependencies inside an active virtual environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### `pip install -r requirements.txt` fails behind proxy / restricted network
Set your pip index/proxy explicitly (example):

```bash
python -m pip config set global.index-url https://pypi.org/simple
# or export HTTPS_PROXY / HTTP_PROXY if your environment requires it
```

### `Missing data/processed/model_dataset.csv`
Run:

```bash
python scripts/prepare_data.py
```

### `Model missing. Run: python scripts/train_match_winner.py`
Run:

```bash
python scripts/train_match_winner.py
```

### Not enough rows for train/test split
Adjust `model.test_start_date` in `config/example_config.yaml` or provide more historical rows.

### Odds format errors
Current pipeline expects **decimal odds** in `odds_p1` and `odds_p2`.

---

## Placeholder data warning

`data/raw/historical_matches.csv` and `data/raw/upcoming_matches.csv` are synthetic examples for dry-run structure checks only. Replace them with real ATP/WTA and sportsbook data before production use.

