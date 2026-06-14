import pandas as pd

from football_forecast.features.fifa import add_fifa_ranking_features


def test_fifa_features_are_strictly_prior_and_include_differences():
    matches = pd.DataFrame(
        {
            "kickoff_utc": pd.to_datetime(["2024-01-02T12:00:00Z"], utc=True),
            "team1": ["A"],
            "team2": ["B"],
            "elo_diff_pre": [100.0],
            "neutral_int": [1],
            "tournament_importance": [2.0],
        }
    )
    ratings = pd.DataFrame(
        {
            "rating_date": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
                utc=True,
            ),
            "team": ["A", "B"],
            "confederation": ["UEFA", "CAF"],
            "fifa_rank": [1, 3],
            "fifa_points": [1600.0, 1400.0],
            "fifa_rated_matches": [20, 18],
            "fifa_rank_percentile": [1.0, 0.5],
            "fifa_points_z": [1.0, -1.0],
        }
    )
    out = add_fifa_ranking_features(matches, ratings)

    assert out.loc[0, "fifa_points_diff"] == 200.0
    assert out.loc[0, "fifa_rank_diff"] == -2
    assert out.loc[0, "team1_fifa_confederation_uefa"] == 1
    assert out.loc[0, "team2_fifa_confederation_caf"] == 1
    assert out.loc[0, "fifa_same_confederation_int"] == 0
    assert out.loc[0, "fifa_points_diff_abs"] == 200.0
    assert out.loc[0, "fifa_snapshot_age_days_max"] == 1.5
    assert out.loc[0, "fifa_any_missing_int"] == 0
