from pathlib import Path

import pandas as pd

from football_forecast.data.prepare import prepare_completed_international_results
from football_forecast.data.provenance import file_sha256


def test_prepare_filters_future_incomplete_duplicate_and_ambiguous_rows(tmp_path: Path):
    source = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "date": [
                "1989-01-01",
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2026-06-11",
            ],
            "home_team": ["X", "A", "A", "D", "F", "H"],
            "away_team": ["Y", "B", "C", "E", "G", "I"],
            "home_score": [1, 1, 2, 1, None, None],
            "away_score": [0, 0, 0, 0, None, None],
            "tournament": ["Friendly"] * 6,
            "city": ["City"] * 6,
            "country": ["Country"] * 6,
            "neutral": [False] * 6,
        }
    ).to_csv(source, index=False)
    mapping = tmp_path / "teams.yaml"
    mapping.write_text("{}\n", encoding="utf-8")

    matches, audit = prepare_completed_international_results(
        source,
        source_version="test",
        expected_sha256=file_sha256(source),
        mapping_path=mapping,
        output_path=tmp_path / "processed.csv",
    )
    assert matches[["team1", "team2"]].values.tolist() == [["D", "E"]]
    assert audit.pre_start_rows == 1
    assert audit.incomplete_rows == 2
    assert audit.future_or_cutoff_rows == 1
    assert audit.ambiguous_same_day_rows == 2
