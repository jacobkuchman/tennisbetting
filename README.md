# Tennis Betting Model - Phase 1 (Moneyline Only)

This project is focused on **Phase 1 only**: pre-match tennis **moneyline** modeling, pricing, and +EV selection.

## Where to get data

For quickest start without your own private feeds:

1. **Tennis Abstract**
   - Use it for historical ATP/WTA match results (and optional stats enrichment).
   - Transform those downloads into `historical_results.csv` format.

2. **The Odds API**
   - Use it for upcoming and historical pre-match moneyline odds.
   - Transform exports into `historical_moneyline_odds.csv` and `upcoming_moneyline_odds.csv`.

> Side markets (spread/totals/first-set/correct score) are **not required yet**. Keep scope to moneyline only.

---

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

## Required input files and placement

Place your CSV files in `data/raw/real/` with these exact filenames:

- `historical_results.csv`
- `historical_moneyline_odds.csv`
- `upcoming_matches.csv`
- `upcoming_moneyline_odds.csv`

If you need starter headers, copy templates from `data/templates/`:

- `data/templates/historical_match_results.csv`
- `data/templates/historical_moneyline_odds.csv`
- `data/templates/upcoming_matches.csv`
- `data/templates/upcoming_moneyline_odds.csv`

---

## Run commands (Phase 1)

```bash
python scripts/validate_schema.py
python scripts/prepare_data.py
python scripts/train_match_winner.py
python scripts/run_backtest.py
python scripts/daily_picks.py
```

---

## Transform helper for raw downloads

Use `scripts/transform_raw_to_schema.py` as a starter mapping script.

It includes example functions for:
- Tennis Abstract historical results
- The Odds API moneyline odds
- upcoming matches formatting

You will likely need to edit column mappings to match your exact export format.

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

### `Schema validation FAILED`
Run `python scripts/validate_schema.py` and add any missing columns/files listed.

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

