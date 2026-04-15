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
) -> float:
    cap = bankroll * max_stake_pct
    if mode == "flat":
        return min(flat_stake, cap)
    if mode == "half_kelly":
        stake = bankroll * (kelly_fraction(prob, decimal_odds) * 0.5)
        return min(max(0.0, stake), cap)
    raise ValueError(f"Unsupported staking mode: {mode}")
