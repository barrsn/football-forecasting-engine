import pandas as pd
import pytest

from football_forecast.features.asof_join import (
    assert_prior_only_timestamps,
    join_team_rating_features_asof,
    join_team_ratings_asof,
)


def test_asof_join_is_strictly_before_kickoff():
    matches = pd.DataFrame(
        {
            "kickoff_utc": pd.to_datetime(
                ["2020-01-02T12:00Z", "2020-01-04T12:00Z"], utc=True
            ),
            "team1": ["A", "A"],
            "team2": ["B", "B"],
        }
    )
    ratings = pd.DataFrame(
        {
            "rating_date": pd.to_datetime(
                [
                    "2020-01-01T00:00Z",
                    "2020-01-02T12:00Z",
                    "2020-01-03T00:00Z",
                    "2020-01-01T00:00Z",
                ],
                utc=True,
            ),
            "team": ["A", "A", "A", "B"],
            "rating": [1500, 9999, 1520, 1490],
        }
    )
    out = join_team_ratings_asof(matches, ratings)
    assert out["team1_rating"].tolist() == [1500, 1520]
    assert 9999 not in out["team1_rating"].tolist()
    assert_prior_only_timestamps(
        out,
        ["team1_rating_available_at", "team2_rating_available_at"],
    )


def test_timestamp_assertion_rejects_exact_kickoff():
    frame = pd.DataFrame(
        {
            "kickoff_utc": pd.to_datetime(["2020-01-01T12:00Z"], utc=True),
            "feature_available_at": pd.to_datetime(["2020-01-01T12:00Z"], utc=True),
        }
    )
    with pytest.raises(ValueError, match="strictly before"):
        assert_prior_only_timestamps(frame, ["feature_available_at"])


def test_multi_rating_asof_join_keeps_one_strict_snapshot():
    matches = pd.DataFrame(
        {
            "kickoff_utc": pd.to_datetime(["2024-01-02T12:00:00Z"], utc=True),
            "team1": ["A"],
            "team2": ["B"],
        }
    )
    ratings = pd.DataFrame(
        {
            "rating_date": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-02T12:00:00Z",
                    "2024-01-01T00:00:00Z",
                ],
                utc=True,
            ),
            "team": ["A", "A", "B"],
            "points": [100.0, 999.0, 90.0],
            "rank": [1, 99, 2],
        }
    )
    out = join_team_rating_features_asof(
        matches,
        ratings,
        ["points", "rank"],
        prefix="fifa",
    )

    assert out.loc[0, "team1_fifa_points"] == 100.0
    assert out.loc[0, "team1_fifa_rank"] == 1
    assert out.loc[0, "fifa_points_diff"] == 10.0
    assert out.loc[0, "team1_fifa_available_at"] < out.loc[0, "kickoff_utc"]
