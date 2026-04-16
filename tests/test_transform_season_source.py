import pandas as pd

from scripts.transform_season_source_to_phase1 import transform_dataframe


def test_transform_season_source_completed_and_upcoming_split():
    df = pd.DataFrame(
        [
            {
                "date_human": "2026-01-10",
                "tour_type_human": "ATP",
                "tournament": "ATP Masters 1000 Rome",
                "round": "R32",
                "surface": "Clay",
                "home_name": "Rafael Nadal",
                "away_name": "Novak Djokovic",
                "winner_code": "home",
                "home_odds_match_winner": 1.8,
                "away_odds_match_winner": 2.1,
            },
            {
                "date_human": "2026-12-30",
                "tour_type_human": "WTA",
                "tournament": "WTA Finals",
                "round": "SF",
                "surface": "Hard",
                "home_name": "Iga Swiatek",
                "away_name": "Coco Gauff",
                "winner_code": "",
                "home_odds_match_winner": 1.7,
                "away_odds_match_winner": 2.2,
            },
        ]
    )

    hist_res, hist_odds, up_matches, up_odds, summary = transform_dataframe(df)

    assert len(hist_res) == 1
    assert len(hist_odds) == 1
    assert len(up_matches) == 1
    assert len(up_odds) == 1
    assert summary["total_rows"] == 2
    assert hist_res.loc[0, "winner_name"] == "Rafael Nadal"
