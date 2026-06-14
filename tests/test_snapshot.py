import pandas as pd

from football_forecast.data.snapshot import create_snapshot, dataframe_sha256


def test_snapshot_uses_strict_availability_cutoff():
    matches = pd.DataFrame(
        {
            "kickoff_utc": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
            "available_at": pd.to_datetime(
                ["2020-01-01T03:00Z", "2020-01-03T00:00Z"], utc=True
            ),
            "source_version": ["abc", "abc"],
        }
    )
    snapshot, manifest = create_snapshot(matches, "2020-01-03T00:00Z")
    assert len(snapshot) == 1
    assert manifest.n_matches == 1
    assert manifest.source_versions == ("abc",)


def test_dataframe_hash_is_row_order_independent():
    frame = pd.DataFrame({"a": [2, 1], "b": ["x", "y"]})
    assert dataframe_sha256(frame) == dataframe_sha256(frame.iloc[::-1])
