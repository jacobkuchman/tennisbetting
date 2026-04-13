# Tennis Betting +EV Model (Baseline, Production-Oriented Scaffold)

This project provides a practical, modular Python baseline for identifying potential **+EV pre-match tennis bets** across:

1. Match winner
2. Game handicap/spread (baseline regression)
3. Total games (baseline regression)
4. First set winner (simulation-ready hook)
5. Correct score / set betting (simulation-ready hook)

## Project structure

- `data/raw/` – source historical and upcoming match + odds files
- `data/processed/` – normalized/model-ready datasets
- `src/data/` – ingestion, normalization, validation
- `src/features/` – Elo + pre-match feature generation
- `src/models/` – match winner and side-market models
- `src/pricing/` – odds conversion, vig removal, EV, staking, simulation
- `src/backtest/` – walk-forward backtest engine
- `src/utils/` – config loader
- `scripts/` – runnable CLI scripts
- `tests/` – unit tests
- `outputs/` – models, backtest outputs, daily picks
- `config/` – example config

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Prepare dataset
```bash
python scripts/prepare_data.py
```

### 2) Train baseline model
```bash
python scripts/train_match_winner.py
```

### 3) Backtest with edge thresholds
```bash
python scripts/run_backtest.py
```

### 4) Produce daily picks
```bash
python scripts/daily_picks.py
```

## Data expectations

Minimum fields (when available):

- `match_date`, `tournament`, `tour`, `surface`, `indoor`, `round`
- `player_1`, `player_2`
- `rank_p1`, `rank_p2`
- `p1_win` or winner/loser names
- odds columns: `odds_p1`, `odds_p2` (match winner market baseline)
- serve/return stats if available (`*_serve_points_won_pct`, `*_return_points_won_pct`, `*_hold_pct`, `*_break_pct`)

## Modeling design notes

- Time-safe feature generation only: each row uses prior match history.
- Time-based split only (no random leakage-prone split).
- Calibration included for match winner probabilities.
- Pricing compares model probabilities against **no-vig probabilities**.
- Bet filtering requires configurable min edge threshold.

## Side markets

Current baseline:

- `scripts/train_side_markets.py` trains regression baselines for:
  - `game_margin`
  - `total_games`

Advanced simulation hook:

- `src/pricing/simulation.py` offers simple match simulation from hold probabilities to derive:
  - match winner
  - first set winner
  - expected game margin
  - expected total games
  - correct-score distribution

## Bankroll management

Supported staking modes:

- `flat`
- `half_kelly` (default), with conservative cap (`max_stake_pct`)

## Manual integrations to add for production

1. Real historical match feeds (ATP/WTA + reliable player IDs)
2. Real pre-match sportsbook odds feed for all markets
3. Closing-line snapshot for CLV measurement
4. Better service-hold models and point-level simulation
5. Scheduled daily job + alerting (email/Slack/Telegram)

## Example files included

- `data/raw/historical_matches.csv` (synthetic example)
- `data/raw/upcoming_matches.csv` (synthetic example)
- `config/example_config.yaml`

## Reproducibility

- Central config in `config/example_config.yaml`
- deterministic model seeds where practical

