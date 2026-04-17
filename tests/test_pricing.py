from src.pricing.bankroll import kelly_fraction, recommend_stake
from src.pricing.odds import (
    american_to_decimal,
    decimal_to_american,
    decimal_to_implied_prob,
    expected_value,
    implied_prob_to_decimal,
    remove_vig_two_way,
)


def test_odds_roundtrip():
    dec = 2.5
    am = decimal_to_american(dec)
    assert abs(american_to_decimal(am) - dec) < 0.05


def test_implied_probability_conversion():
    assert abs(decimal_to_implied_prob(2.0) - 0.5) < 1e-9
    assert abs(decimal_to_implied_prob(4.0) - 0.25) < 1e-9


def test_fair_odds_conversion_exact_cases():
    assert round(implied_prob_to_decimal(0.5), 2) == 2.00
    assert round(implied_prob_to_decimal(0.4), 2) == 2.50


def test_remove_vig_two_way_sums_to_one_and_preserves_order():
    p1, p2 = remove_vig_two_way(1.80, 2.10)
    assert round(p1 + p2, 8) == 1.0
    assert p1 > p2


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


def test_supported_staking_styles():
    stake_kelly = recommend_stake(1000, 0.6, 2.1, mode="kelly", max_stake_pct=0.05)
    stake_half = recommend_stake(1000, 0.6, 2.1, mode="half_kelly", max_stake_pct=0.05)
    stake_quarter = recommend_stake(1000, 0.6, 2.1, mode="quarter_kelly", max_stake_pct=0.05)
    stake_flat = recommend_stake(1000, 0.6, 2.1, mode="flat", flat_stake=25, max_stake_pct=0.05)

    assert stake_kelly >= stake_half >= stake_quarter >= 0
    assert stake_flat == 25


def test_min_bet_amount_sets_small_stakes_to_zero():
    stake = recommend_stake(1000, 0.51, 2.0, mode="quarter_kelly", max_stake_pct=0.05, min_bet_amount=10)
    assert stake == 0.0
