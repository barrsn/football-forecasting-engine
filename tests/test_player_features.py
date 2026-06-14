import pandas as pd
import pytest

from football_forecast.data.players import coerce_player_snapshots
from football_forecast.features.players import add_player_snapshot_features


def _snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "snapshot_id": ["a"] * 3 + ["b"] * 3,
            "player_id": ["p1", "p2", "p3", "q1", "q2", "q3"],
            "player_name": ["P1", "P2", "P3", "Q1", "Q2", "Q3"],
            "team": ["A"] * 3 + ["B"] * 3,
            "available_at": pd.to_datetime(["2024-01-01T10:00Z"] * 6, utc=True),
            "source": ["official"] * 6,
            "source_version": ["v1"] * 6,
            "position": ["GK", "DF", "FW", "GK", "MF", "FW"],
            "availability_status": [
                "available",
                "injured",
                "available",
                "available",
                "available",
                "available",
            ],
            "lineup_status": [
                "starter",
                "not_in_squad",
                "starter",
                "starter",
                "starter",
                "bench",
            ],
            "player_rating": [80, 90, 85, 75, 82, 78],
            "international_caps": [10, 20, 30, 8, 15, 25],
            "international_goals": [0, 1, 12, 0, 4, 10],
        }
    )


def test_player_snapshot_validation_rejects_duplicate_players():
    frame = pd.concat([_snapshots(), _snapshots().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate player_id"):
        coerce_player_snapshots(frame)


def test_player_features_measure_availability_and_quality_loss():
    matches = pd.DataFrame(
        {
            "kickoff_utc": pd.to_datetime(["2024-01-02T12:00Z"], utc=True),
            "team1": ["A"],
            "team2": ["B"],
        }
    )
    out = add_player_snapshot_features(matches, _snapshots())

    assert out.loc[0, "team1_player_unavailable_count"] == 1
    assert out.loc[0, "team1_player_official_starter_count"] == 2
    assert out.loc[0, "team1_player_absence_rating_loss"] > 0
    assert out.loc[0, "player_available_count_diff"] == -1
    assert out.loc[0, "team1_player_available_at"] < out.loc[0, "kickoff_utc"]


def test_player_snapshot_at_kickoff_is_not_joined():
    snapshots = _snapshots()
    snapshots["available_at"] = pd.Timestamp("2024-01-02T12:00Z")
    matches = pd.DataFrame(
        {
            "kickoff_utc": pd.to_datetime(["2024-01-02T12:00Z"], utc=True),
            "team1": ["A"],
            "team2": ["B"],
        }
    )
    out = add_player_snapshot_features(matches, snapshots)
    assert out.loc[0, "player_any_missing_int"] == 1
