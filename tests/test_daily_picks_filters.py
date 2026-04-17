import pandas as pd

from scripts.daily_picks import filter_daily_universe


def test_daily_filters_exclude_challenger_and_non_main_tour():
    df = pd.DataFrame(
        {
            "tour": ["ATP", "WTA", "ATP CHALLENGER", "ITF"],
            "tournament": ["Rome", "Madrid", "Phoenix Challenger", "Futures"],
        }
    )
    cfg = {
        "daily_picks": {
            "include_atp": True,
            "include_wta": True,
            "include_challenger": False,
        }
    }
    out = filter_daily_universe(df, cfg)
    assert set(out["tour"].tolist()) == {"ATP", "WTA"}


def test_daily_filters_can_disable_wta():
    df = pd.DataFrame(
        {
            "tour": ["ATP", "WTA"],
            "tournament": ["Rome", "Madrid"],
        }
    )
    cfg = {
        "daily_picks": {
            "include_atp": True,
            "include_wta": False,
            "include_challenger": False,
        }
    }
    out = filter_daily_universe(df, cfg)
    assert out["tour"].tolist() == ["ATP"]
