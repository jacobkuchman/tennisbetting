from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.match_winner import evaluate_binary_predictions, predict_proba, train_match_winner_model
from src.pricing.bankroll import recommend_stake
from src.pricing.odds import edge_vs_market, expected_value, implied_prob_to_decimal, remove_vig_two_way


def _compute_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max.replace(0, np.nan)
    return float(drawdowns.min()) if len(drawdowns) else 0.0


def walk_forward_backtest(
    df: pd.DataFrame,
    feature_columns: list[str],
    start_date: str,
    min_edge: float,
    retrain_frequency_days: int = 30,
    bankroll_start: float = 10000.0,
    staking_mode: str = "half_kelly",
    flat_stake: float = 50.0,
    max_stake_pct: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    data = df.sort_values("match_date").copy().reset_index(drop=True)
    start_date_ts = pd.Timestamp(start_date)

    test_idx = data.index[data["match_date"] >= start_date_ts].tolist()
    if not test_idx:
        raise ValueError("No test rows after start_date")

    preds = pd.Series(index=data.index, dtype=float)
    last_train_date = None
    artifacts = None

    for idx in test_idx:
        row_date = data.at[idx, "match_date"]
        need_retrain = (
            artifacts is None
            or last_train_date is None
            or (row_date - last_train_date).days >= retrain_frequency_days
        )

        if need_retrain:
            train_mask = data["match_date"] < row_date
            if train_mask.sum() < 50:
                continue
            artifacts = train_match_winner_model(data, feature_columns, "p1_win", train_mask)
            last_train_date = row_date

        preds.at[idx] = float(predict_proba(artifacts, data.loc[[idx]])[0])

    data["model_prob_p1"] = preds
    data = data.loc[test_idx].copy()
    data = data.dropna(subset=["model_prob_p1", "odds_p1", "odds_p2", "p1_win"]).copy()

    data[["no_vig_prob_p1", "no_vig_prob_p2"]] = data.apply(
        lambda r: pd.Series(remove_vig_two_way(r["odds_p1"], r["odds_p2"])),
        axis=1,
    )
    data["edge"] = data.apply(lambda r: edge_vs_market(r["model_prob_p1"], r["no_vig_prob_p1"]), axis=1)
    data["ev"] = data.apply(lambda r: expected_value(r["model_prob_p1"], r["odds_p1"]), axis=1)
    data["fair_odds_decimal_p1"] = data["model_prob_p1"].apply(implied_prob_to_decimal)

    bets = data.loc[data["edge"] >= min_edge].copy()
    bankroll = bankroll_start
    stakes = []
    pnl = []

    for _, row in bets.iterrows():
        stake = recommend_stake(
            bankroll=bankroll,
            prob=row["model_prob_p1"],
            decimal_odds=row["odds_p1"],
            mode=staking_mode,
            flat_stake=flat_stake,
            max_stake_pct=max_stake_pct,
        )
        profit = stake * (row["odds_p1"] - 1) if int(row["p1_win"]) == 1 else -stake
        bankroll += profit
        stakes.append(stake)
        pnl.append(profit)

    bets["stake"] = stakes
    bets["pnl"] = pnl
    bets["equity"] = bankroll_start + bets["pnl"].cumsum()

    metrics = evaluate_binary_predictions(data["p1_win"], data["model_prob_p1"])
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
