from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.match_winner import evaluate_binary_predictions, predict_proba, train_match_winner_model
from src.pricing.bankroll import recommend_stake
from src.pricing.odds import edge_vs_market, expected_value, remove_vig_two_way


def _compute_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max.replace(0, np.nan)
    return float(drawdowns.min()) if len(drawdowns) else 0.0


def walk_forward_backtest(
    df: pd.DataFrame,
    feature_columns: list[str],
    start_date: str,
    min_edge: float,
    bankroll_start: float = 10000.0,
    staking_mode: str = "half_kelly",
    flat_stake: float = 50.0,
    max_stake_pct: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    data = df.sort_values("match_date").copy()
    start_date_ts = pd.Timestamp(start_date)
    test_mask = data["match_date"] >= start_date_ts
    train_mask = data["match_date"] < start_date_ts

    if train_mask.sum() < 50 or test_mask.sum() == 0:
        raise ValueError("Not enough data for walk-forward split")

    artifacts = train_match_winner_model(data, feature_columns, "p1_win", train_mask)
    data.loc[test_mask, "model_prob_p1"] = predict_proba(artifacts, data.loc[test_mask])

    no_vig_probs = data.loc[test_mask].apply(
        lambda r: remove_vig_two_way(r["odds_p1"], r["odds_p2"])[0] if pd.notna(r.get("odds_p1")) and pd.notna(r.get("odds_p2")) else np.nan,
        axis=1,
    )
    data.loc[test_mask, "no_vig_prob_p1"] = no_vig_probs
    data.loc[test_mask, "edge"] = data.loc[test_mask].apply(
        lambda r: edge_vs_market(r["model_prob_p1"], r["no_vig_prob_p1"]) if pd.notna(r["no_vig_prob_p1"]) else np.nan,
        axis=1,
    )
    data.loc[test_mask, "ev"] = data.loc[test_mask].apply(
        lambda r: expected_value(r["model_prob_p1"], r["odds_p1"]) if pd.notna(r.get("odds_p1")) else np.nan,
        axis=1,
    )

    bets = data.loc[test_mask & (data["edge"] >= min_edge)].copy()

    bankroll = bankroll_start
    pnl = []
    stakes = []

    for _, row in bets.iterrows():
        stake = recommend_stake(
            bankroll=bankroll,
            prob=row["model_prob_p1"],
            decimal_odds=row["odds_p1"],
            mode=staking_mode,
            flat_stake=flat_stake,
            max_stake_pct=max_stake_pct,
        )
        stakes.append(stake)
        win = int(row["p1_win"])
        profit = stake * (row["odds_p1"] - 1) if win else -stake
        bankroll += profit
        pnl.append(profit)

    bets["stake"] = stakes
    bets["pnl"] = pnl
    bets["equity"] = bankroll_start + bets["pnl"].cumsum()

    metrics = evaluate_binary_predictions(data.loc[test_mask, "p1_win"], data.loc[test_mask, "model_prob_p1"].fillna(0.5))
    metrics.update(
        {
            "bets": int(len(bets)),
            "hit_rate": float((bets["pnl"] > 0).mean()) if len(bets) else 0.0,
            "avg_edge": float(bets["edge"].mean()) if len(bets) else 0.0,
            "total_staked": float(bets["stake"].sum()) if len(bets) else 0.0,
            "total_pnl": float(bets["pnl"].sum()) if len(bets) else 0.0,
            "roi": float(bets["pnl"].sum() / bets["stake"].sum()) if len(bets) and bets["stake"].sum() > 0 else 0.0,
            "max_drawdown": _compute_drawdown(bets["equity"]) if len(bets) else 0.0,
        }
    )
    return bets, metrics
