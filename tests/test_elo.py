import pandas as pd

from football_forecast.features.elo import add_elo_features


def test_elo_features_are_pre_match():
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "team1": ["A", "A"],
            "team2": ["B", "B"],
            "team1_goals": [1, 0],
            "team2_goals": [0, 1],
            "tournament": ["friendly", "friendly"],
            "neutral": [True, True],
        }
    )
    out = add_elo_features(df)
    assert out.loc[0, "elo_team1_pre"] == 1500.0
    assert out.loc[0, "elo_team2_pre"] == 1500.0
    assert out.loc[1, "elo_team1_pre"] > 1500.0
    assert out.loc[1, "elo_team2_pre"] < 1500.0
