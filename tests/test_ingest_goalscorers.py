import hashlib

import pandas as pd

from football_forecast.data.ingest_goalscorers import ingest_goalscorers


def test_ingest_goalscorers_normalizes_teams_and_booleans(tmp_path):
    source = tmp_path / "goalscorers.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "home_team": ["USA"],
            "away_team": ["Germany"],
            "team": ["USA"],
            "scorer": ["Player"],
            "minute": [10],
            "own_goal": ["False"],
            "penalty": ["0"],
        }
    ).to_csv(source, index=False)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    mapping = tmp_path / "teams.yaml"
    mapping.write_text(
        "United States:\n  - USA\nGermany:\n  - Germany\n",
        encoding="utf-8",
    )

    out, report = ingest_goalscorers(
        source,
        mapping_path=mapping,
        source_version="abc",
        expected_sha256=checksum,
        snapshot_at="2024-01-02T00:00Z",
    )

    assert out.loc[0, "home_team"] == "United States"
    assert out.loc[0, "team"] == "United States"
    assert not bool(out.loc[0, "own_goal"])
    assert not bool(out.loc[0, "penalty"])
    assert report.unresolved_names == ()
