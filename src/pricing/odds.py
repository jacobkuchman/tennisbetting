from __future__ import annotations

import numpy as np


def decimal_to_implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be > 1")
    return 1.0 / decimal_odds


def implied_prob_to_decimal(prob: float) -> float:
    if prob <= 0 or prob >= 1:
        raise ValueError("Probability must be in (0, 1)")
    return 1.0 / prob


def decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds < 1:
        raise ValueError("Decimal odds must be >= 1")
    if decimal_odds >= 2:
        return int(round((decimal_odds - 1) * 100))
    return int(round(-100 / (decimal_odds - 1)))


def american_to_decimal(american_odds: int) -> float:
    if american_odds == 0:
        raise ValueError("American odds cannot be zero")
    if american_odds > 0:
        return 1 + american_odds / 100
    return 1 + 100 / abs(american_odds)


def remove_vig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    p_a = decimal_to_implied_prob(odds_a)
    p_b = decimal_to_implied_prob(odds_b)
    overround = p_a + p_b
    if overround <= 0:
        raise ValueError("Invalid overround")
    return p_a / overround, p_b / overround


def expected_value(model_prob: float, market_odds: float) -> float:
    """Return expected return per $1 stake."""
    return model_prob * (market_odds - 1) - (1 - model_prob)


def edge_vs_market(model_prob: float, no_vig_prob: float) -> float:
    return model_prob - no_vig_prob


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    total = probs.sum()
    if total <= 0:
        raise ValueError("Probability array sum must be > 0")
    return probs / total
