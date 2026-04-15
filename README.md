# Tennis Betting Model - Phase 1 (Moneyline Only)

This project is focused on **Phase 1 only**: pre-match tennis **moneyline** modeling, pricing, and +EV selection.

## Primary workflow (real data first)

1. Ingest real historical match results
2. Ingest real historical pre-match moneyline odds
3. Normalize players/tournaments/surfaces/dates
4. Match odds to results (robust join with normalized keys + fallback)
5. Validate duplicates/missing odds/impossible dates
6. Export cleaned merged datasets
7. Build leak-safe model dataset and run train/backtest/picks scripts

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run commands (Phase 1)

```bash
python scripts/prepare_data.py
python scripts/train_match_winner.py
python scripts/run_backtest.py
python scripts/daily_picks.py
```

---

## Real-data file formats supported

All files are CSV.

### 1) Historical match results (`real_data.historical_results_path`)
Required logical fields:
- match date
- tour (ATP/WTA)
- tournament
- surface
- round
- winner name
- loser name

Mapped via config:
`real_data.historical_results_schema`

### 2) Historical moneyline odds (`real_data.historical_odds_path`)
Required logical fields:
- odds date (match date)
- tour
- tournament
- player 1
- player 2
- decimal odds for player 1
- decimal odds for player 2

Mapped via config:
`real_data.odds_schema`

### 3) Upcoming matches (`real_data.upcoming_matches_path`)
Required logical fields:
- match date
- tour
- tournament
- surface
- round
- player 1
- player 2

Mapped via config:
`real_data.upcoming_matches_schema`

### 4) Upcoming moneyline odds (`real_data.upcoming_odds_path`)
Same logical fields as historical odds (date/tour/tournament/players/odds).

---

## Exact required columns (logical)

### Historical results required
- `match_date`
- `tour`
- `tournament`
- `surface`
- `round`
- `winner_name`
- `loser_name`

### Historical odds required
- `odds_date`
- `tour`
- `tournament`
- `player_1`
- `player_2`
- `odds_p1` (decimal)
- `odds_p2` (decimal)

### Upcoming matches required
- `match_date`
- `tour`
- `tournament`
- `surface`
- `round`
- `player_1`
- `player_2`

### Upcoming odds required
- `odds_date`
- `tour`
- `tournament`
- `player_1`
- `player_2`
- `odds_p1` (decimal)
- `odds_p2` (decimal)

---

## Matching logic between odds and results

1. Normalize fields:
   - player names: lowercase, accent/punctuation stripped
   - tournament names: alias normalization (e.g., masters naming)
   - surface standardization (Hard/Clay/Grass/Carpet)
   - parsed datetimes
2. Canonical player pairing:
   - player pairs are sorted alphabetically to reduce orientation mismatch
   - odds are swapped when needed to keep `odds_p1/odds_p2` aligned to canonical pairing
3. Primary join key:
   - `match_date + tour + tournament_norm + player_1_norm + player_2_norm`
4. Fallback join key (if primary fails):
   - `match_date + tour + player_1_norm + player_2_norm`
5. Validation:
   - reject duplicate normalized matches
   - reject missing odds after matching
   - reject impossible dates (pre-1960 or far-future)

---

## Output files produced by `prepare_data.py`

When `data_source: real`:
- `data/processed/historical_merged.csv` (cleaned results+odds)
- `data/processed/upcoming_merged.csv` (cleaned upcoming+odds)
- `data/processed/model_dataset.csv` (feature-engineered historical dataset for modeling)

When `data_source: sample`:
- `data/processed/model_dataset.csv`

---

## Config example (`config/example_config.yaml`)

- `data_source: real` is default.
- `real_data.*_schema` maps your external column names to logical fields.
- You can switch to `data_source: sample` for synthetic local dry-runs.

---

## Troubleshooting

### `pip install -r requirements.txt` fails in restricted network
Set pip index/proxy explicitly, then retry.

### `Missing odds for X ... after matching`
Your player/tournament names likely need better normalization or source-specific aliases.

### `Duplicate historical matches found after normalization`
Your results source has duplicate rows after canonicalization; dedupe upstream or add source-specific keys.

### `Found impossible future date`
Check timezone/date parsing and ensure historical files do not include upcoming events.

---

## Sample data support

Sample files are still included under `data/raw/`.
However, real-data ingestion is now the primary documented path.

