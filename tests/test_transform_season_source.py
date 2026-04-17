import pandas as pd

from scripts.transform_season_source_to_phase1 import transform_dataframe


def test_transform_season_source_uses_status_and_numeric_winner_code():
    df = pd.DataFrame(
        [
            {
                "date_human": "2026-01-10",
                "tour_type_human": "ATP",
                "tournament": "ATP Masters 1000 Rome",
                "round": "R32",
                "surface": "Clay",
                "status": "FINISHED",
                "home_name": "Rafael Nadal",
                "away_name": "Novak Djokovic",
                "winner_code": 1,
                "home_odds_match_winner": 1.8,
                "away_odds_match_winner": 2.1,
            },
            {
                "date_human": "2026-01-11",
                "tour_type_human": "WTA",
                "tournament": "WTA Finals",
                "round": "SF",
                "surface": "Hard",
                "status": "completed",
                "home_name": "Iga Swiatek",
                "away_name": "Coco Gauff",
                "winner_code": "2",
                "home_odds_match_winner": 1.7,
                "away_odds_match_winner": 2.2,
            },
            {
                "date_human": "2026-12-30",
                "tour_type_human": "WTA",
                "tournament": "WTA Finals",
                "round": "SF",
                "surface": "Hard",
                "status": "scheduled",
                "home_name": "Iga Swiatek",
                "away_name": "Coco Gauff",
                "winner_code": "",
                "home_odds_match_winner": 1.7,
                "away_odds_match_winner": 2.2,
            },
        ]
    )

    hist_res, hist_odds, up_matches, up_odds, summary = transform_dataframe(df)

    assert len(hist_res) == 2
    assert len(hist_odds) == 2
    assert len(up_matches) == 1
    assert len(up_odds) == 1
    assert summary["total_rows"] == 3
    assert summary["completed_matches_used"] == 2
    assert summary["upcoming_matches_used"] == 1
    assert hist_res.loc[0, "winner_name"] == "Rafael Nadal"
    assert hist_res.loc[1, "winner_name"] == "Coco Gauff"


def test_empty_outputs_keep_headers():
    df = pd.DataFrame(
        [
            {
                "date_human": "2026-12-30",
                "tour_type_human": "WTA",
                "tournament": "WTA Finals",
                "round": "SF",
                "surface": "Hard",
                "status": "scheduled",
                "home_name": "Iga Swiatek",
                "away_name": "Coco Gauff",
                "winner_code": "",
                "home_odds_match_winner": 1.7,
                "away_odds_match_winner": 2.2,
            }
        ]
    )
    hist_res, hist_odds, up_matches, up_odds, _ = transform_dataframe(df)
    assert list(hist_res.columns) == ["match_date", "tour", "tournament", "surface", "round", "winner_name", "loser_name"]
    assert list(hist_odds.columns) == ["match_date", "tour", "tournament", "player_1", "player_2", "odds_p1", "odds_p2"]
    assert len(up_matches) == 1
    assert len(up_odds) == 1
