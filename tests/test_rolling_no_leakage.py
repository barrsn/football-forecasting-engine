import pandas as pd

from football_forecast.features.rolling import add_rolling_team_features


def test_rolling_features_exclude_current_match():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "team1": ["A", "A", "A"],
            "team2": ["B", "C", "D"],
            "team1_goals": [3, 1, 2],
            "team2_goals": [0, 1, 1],
            "tournament": ["friendly"] * 3,
            "neutral": [True] * 3,
        }
    )
    out = add_rolling_team_features(df, windows=(2,))
    # First match for A has no prior goals.
    assert pd.isna(out.loc[0, "team1_goals_for_roll2"])
    # Second match for A uses only first match goals_for = 3, not current goals_for = 1.
    assert out.loc[1, "team1_goals_for_roll2"] == 3.0
    # Third match for A uses prior two matches: (3 + 1) / 2 = 2.
    assert out.loc[2, "team1_goals_for_roll2"] == 2.0
