from __future__ import annotations


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    q = 1 - prob
    k = (b * prob - q) / b
    return max(0.0, k)


def recommend_stake(
    bankroll: float,
    prob: float,
    decimal_odds: float,
    mode: str = "half_kelly",
    flat_stake: float = 50.0,
    max_stake_pct: float = 0.02,
    min_bet_amount: float = 0.0,
) -> float:
    if bankroll <= 0 or decimal_odds <= 1.0:
        return 0.0

    cap = bankroll * max_stake_pct

    if mode == "flat":
        stake = min(flat_stake, cap)
    elif mode == "kelly":
        stake = bankroll * kelly_fraction(prob, decimal_odds)
    elif mode == "half_kelly":
        stake = bankroll * (kelly_fraction(prob, decimal_odds) * 0.5)
    elif mode == "quarter_kelly":
        stake = bankroll * (kelly_fraction(prob, decimal_odds) * 0.25)
    else:
        raise ValueError(f"Unsupported staking mode: {mode}")

    stake = min(max(0.0, stake), cap)
    if stake < min_bet_amount:
        return 0.0
    return stake
