from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.data.loader import load_csv

SURFACE_MAP = {
    "hard": "Hard",
    "h": "Hard",
    "clay": "Clay",
    "c": "Clay",
    "grass": "Grass",
    "g": "Grass",
    "carpet": "Carpet",
    "indoor hard": "Hard",
    "outdoor hard": "Hard",
}


@dataclass
class RealSchema:
    match_date: str
    tour: str
    tournament: str
    surface: str
    round: str
    winner_name: str
    loser_name: str


@dataclass
class OddsSchema:
    odds_date: str
    tour: str
    tournament: str
    player_1: str
    player_2: str
    odds_p1: str
    odds_p2: str


def normalize_text(value: str | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_player_name(value: str | None) -> str:
    txt = normalize_text(value)
    txt = txt.replace(" jr", "")
    txt = txt.replace(" sr", "")
    return txt


def normalize_tournament_name(value: str | None) -> str:
    txt = normalize_text(value)
    txt = txt.replace("atp ", "").replace("wta ", "")
    txt = txt.replace("masters 1000", "masters")
    return txt


def normalize_surface(value: str | None) -> str:
    txt = normalize_text(value)
    return SURFACE_MAP.get(txt, txt.title() if txt else "Unknown")


def _canonical_pair(p1: str, p2: str) -> tuple[str, str]:
    return (p1, p2) if p1 <= p2 else (p2, p1)


def _validate_date_bounds(series: pd.Series, allow_future_days: int = 7) -> None:
    min_date = pd.Timestamp("1960-01-01")
    max_date = pd.Timestamp(datetime.now(timezone.utc).date() + timedelta(days=allow_future_days))
    if (series < min_date).any():
        raise ValueError("Found impossible historical date before 1960-01-01")
    if (series > max_date).any():
        raise ValueError(f"Found impossible future date after {max_date.date()}")


def load_historical_results(path: str | Path, schema: RealSchema) -> pd.DataFrame:
    raw = load_csv(path)
    data = raw.rename(
        columns={
            schema.match_date: "match_date",
            schema.tour: "tour",
            schema.tournament: "tournament",
            schema.surface: "surface",
            schema.round: "round",
            schema.winner_name: "winner_name",
            schema.loser_name: "loser_name",
        }
    ).copy()

    required = ["match_date", "tour", "tournament", "surface", "round", "winner_name", "loser_name"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Historical results missing required columns: {missing}")

    data["match_date"] = pd.to_datetime(data["match_date"], errors="coerce")
    data = data.dropna(subset=["match_date", "winner_name", "loser_name"]).copy()
    _validate_date_bounds(data["match_date"], allow_future_days=1)

    data["tour"] = data["tour"].astype(str).str.upper()
    data["surface"] = data["surface"].map(normalize_surface)
    data["tournament_norm"] = data["tournament"].map(normalize_tournament_name)
    data["winner_norm"] = data["winner_name"].map(normalize_player_name)
    data["loser_norm"] = data["loser_name"].map(normalize_player_name)

    p1 = []
    p2 = []
    p1_win = []
    for _, r in data.iterrows():
        a, b = _canonical_pair(r["winner_norm"], r["loser_norm"])
        p1.append(a)
        p2.append(b)
        p1_win.append(1.0 if r["winner_norm"] == a else 0.0)

    data["player_1_norm"] = p1
    data["player_2_norm"] = p2
    data["p1_win"] = p1_win

    dup_cols = ["match_date", "tour", "tournament_norm", "round", "player_1_norm", "player_2_norm"]
    if data.duplicated(dup_cols).any():
        raise ValueError("Duplicate historical matches found after normalization.")

    return data


def load_moneyline_odds(path: str | Path, schema: OddsSchema, is_upcoming: bool = False) -> pd.DataFrame:
    raw = load_csv(path)
    data = raw.rename(
        columns={
            schema.odds_date: "match_date",
            schema.tour: "tour",
            schema.tournament: "tournament",
            schema.player_1: "player_1",
            schema.player_2: "player_2",
            schema.odds_p1: "odds_p1",
            schema.odds_p2: "odds_p2",
        }
    ).copy()

    required = ["match_date", "tour", "tournament", "player_1", "player_2", "odds_p1", "odds_p2"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Odds file missing required columns: {missing}")

    data["match_date"] = pd.to_datetime(data["match_date"], errors="coerce")
    data["odds_p1"] = pd.to_numeric(data["odds_p1"], errors="coerce")
    data["odds_p2"] = pd.to_numeric(data["odds_p2"], errors="coerce")

    data = data.dropna(subset=["match_date", "player_1", "player_2", "odds_p1", "odds_p2"]).copy()
    _validate_date_bounds(data["match_date"], allow_future_days=30 if is_upcoming else 1)

    data["tour"] = data["tour"].astype(str).str.upper()
    data["tournament_norm"] = data["tournament"].map(normalize_tournament_name)
    p1n = data["player_1"].map(normalize_player_name)
    p2n = data["player_2"].map(normalize_player_name)

    ordered_p1, ordered_p2, o1, o2 = [], [], [], []
    for a, b, oa, ob in zip(p1n, p2n, data["odds_p1"], data["odds_p2"]):
        c1, c2 = _canonical_pair(a, b)
        if a == c1:
            ordered_p1.append(c1)
            ordered_p2.append(c2)
            o1.append(oa)
            o2.append(ob)
        else:
            ordered_p1.append(c1)
            ordered_p2.append(c2)
            o1.append(ob)
            o2.append(oa)

    data["player_1_norm"] = ordered_p1
    data["player_2_norm"] = ordered_p2
    data["odds_p1"] = o1
    data["odds_p2"] = o2

    dup_cols = ["match_date", "tour", "tournament_norm", "player_1_norm", "player_2_norm"]
    data = data.sort_values("match_date").drop_duplicates(dup_cols, keep="last")

    return data


def merge_results_with_odds(results_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["match_date", "tour", "tournament_norm", "player_1_norm", "player_2_norm"]

    merged = results_df.merge(
        odds_df[key_cols + ["odds_p1", "odds_p2"]],
        on=key_cols,
        how="left",
        validate="1:1",
    )

    # fallback without tournament for rows that didn't match (name differences across books)
    missing = merged["odds_p1"].isna()
    if missing.any():
        fallback = results_df.loc[missing].merge(
            odds_df[["match_date", "tour", "player_1_norm", "player_2_norm", "odds_p1", "odds_p2"]],
            on=["match_date", "tour", "player_1_norm", "player_2_norm"],
            how="left",
        )
        merged.loc[missing, "odds_p1"] = fallback["odds_p1"].values
        merged.loc[missing, "odds_p2"] = fallback["odds_p2"].values

    merged["odds_matched"] = merged[["odds_p1", "odds_p2"]].notna().all(axis=1)

    merged = merged.rename(columns={"player_1_norm": "player_1", "player_2_norm": "player_2"})
    keep_cols = [
        "match_date",
        "tournament",
        "tour",
        "surface",
        "round",
        "player_1",
        "player_2",
        "winner_name",
        "loser_name",
        "p1_win",
        "odds_p1",
        "odds_p2",
        "odds_matched",
    ]
    for c in keep_cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    return merged[keep_cols].sort_values("match_date").reset_index(drop=True)


def merge_upcoming_with_odds(upcoming_matches_df: pd.DataFrame, upcoming_odds_df: pd.DataFrame) -> pd.DataFrame:
    m = upcoming_matches_df.copy()
    m["tour"] = m["tour"].astype(str).str.upper()
    m["tournament_norm"] = m["tournament"].map(normalize_tournament_name)
    m["player_1_norm"] = m["player_1"].map(normalize_player_name)
    m["player_2_norm"] = m["player_2"].map(normalize_player_name)

    # canonicalize player ordering in matches to align with odds
    p1, p2 = [], []
    for a, b in zip(m["player_1_norm"], m["player_2_norm"]):
        c1, c2 = _canonical_pair(a, b)
        p1.append(c1)
        p2.append(c2)
    m["player_1_norm"] = p1
    m["player_2_norm"] = p2

    key_cols = ["match_date", "tour", "tournament_norm", "player_1_norm", "player_2_norm"]
    out = m.merge(
        upcoming_odds_df[key_cols + ["odds_p1", "odds_p2"]],
        on=key_cols,
        how="left",
    )

    if out[["odds_p1", "odds_p2"]].isna().any().any():
        miss = int(out["odds_p1"].isna().sum())
        raise ValueError(f"Missing odds for {miss} upcoming matches after matching.")

    out["player_1"] = out["player_1_norm"]
    out["player_2"] = out["player_2_norm"]
    out = out.drop(columns=["player_1_norm", "player_2_norm", "tournament_norm"], errors="ignore")
    return out.sort_values("match_date").reset_index(drop=True)
