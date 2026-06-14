import pandas as pd

from football_forecast.features.elo import add_elo_features
from football_forecast.features.h2h import add_h2h_features
from football_forecast.features.rolling import add_rolling_team_features


def _same_time_matches():
    return pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "team1": ["A", "A", "A"],
            "team2": ["B", "C", "D"],
            "team1_goals": [5, 0, 1],
            "team2_goals": [0, 0, 0],
            "tournament": ["friendly"] * 3,
            "neutral": [True] * 3,
            "match_id": ["m1", "m2", "m3"],
            "available_at": pd.to_datetime(
                ["2020-01-01T03:00Z", "2020-01-01T03:00Z", "2020-01-02T03:00Z"],
                utc=True,
            ),
        }
    )


def test_equal_kickoff_matches_share_same_pre_match_elo():
    out = add_elo_features(_same_time_matches())
    assert out.loc[0, "elo_team1_pre"] == 1500.0
    assert out.loc[1, "elo_team1_pre"] == 1500.0
    assert out.loc[2, "elo_team1_pre"] != 1500.0


def test_equal_kickoff_match_does_not_enter_rolling_history():
    out = add_rolling_team_features(_same_time_matches(), windows=(2,))
    assert pd.isna(out.loc[0, "team1_goals_for_roll2"])
    assert pd.isna(out.loc[1, "team1_goals_for_roll2"])
    assert out.loc[2, "team1_goals_for_roll2"] == 2.5


def test_equal_kickoff_match_does_not_enter_h2h_history():
    matches = _same_time_matches()
    matches.loc[1, "team2"] = "B"
    matches.loc[2, "team2"] = "B"
    out = add_h2h_features(matches)
    assert out.loc[0, "h2h_total_matches"] == 0
    assert out.loc[1, "h2h_total_matches"] == 0
    assert out.loc[2, "h2h_total_matches"] == 2
