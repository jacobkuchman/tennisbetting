from src.pricing.bankroll import kelly_fraction, recommend_stake
from src.pricing.odds import (
    american_to_decimal,
    decimal_to_american,
    expected_value,
    implied_prob_to_decimal,
    remove_vig_two_way,
)


def test_odds_roundtrip():
    dec = 2.5
    am = decimal_to_american(dec)
    assert abs(american_to_decimal(am) - dec) < 0.05


def test_remove_vig_two_way_sums_to_one_and_preserves_order():
    p1, p2 = remove_vig_two_way(1.80, 2.10)
    assert round(p1 + p2, 8) == 1.0
    assert p1 > p2


def test_fair_odds_conversion_exact_cases():
    assert round(implied_prob_to_decimal(0.5), 2) == 2.00
    assert round(implied_prob_to_decimal(0.4), 2) == 2.50


def test_ev_is_zero_at_fair_price():
    prob = 0.55
    fair_odds = implied_prob_to_decimal(prob)
    ev = expected_value(prob, fair_odds)
    assert abs(ev) < 1e-9


def test_ev_positive_for_good_price():
    ev = expected_value(0.6, 2.0)
    assert ev > 0


def test_staking_caps():
    stake = recommend_stake(1000, 0.65, 2.2, mode="half_kelly", max_stake_pct=0.02)
    assert stake <= 20
    assert kelly_fraction(0.4, 1.8) >= 0
