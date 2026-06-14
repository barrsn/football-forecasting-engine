import pandas as pd

from football_forecast.features.scorers import add_prior_scorer_features


def test_scorer_features_are_prior_only_and_batch_equal_kickoffs():
    matches = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3"],
            "kickoff_utc": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-02-01"], utc=True
            ),
            "available_at": pd.to_datetime(
                ["2020-01-01T03:00Z", "2020-01-01T03:00Z", "2020-02-01T03:00Z"],
                utc=True,
            ),
            "team1": ["A", "A", "A"],
            "team2": ["B", "C", "D"],
            "team1_goals": [2, 1, 0],
            "team2_goals": [0, 0, 0],
        }
    )
    goalscorers = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "home_team": ["A"] * 3,
            "away_team": ["B", "B", "C"],
            "team": ["A"] * 3,
            "scorer": ["One", "Two", "Three"],
            "own_goal": [False] * 3,
            "penalty": [False, True, False],
        }
    )
    out = add_prior_scorer_features(matches, goalscorers)

    assert out.loc[0, "team1_scorer_active_players"] == 0
    assert out.loc[1, "team1_scorer_active_players"] == 0
    assert out.loc[2, "team1_scorer_active_players"] == 3
    assert out.loc[2, "team1_scorer_complete_matches"] == 2
