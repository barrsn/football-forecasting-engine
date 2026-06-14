from pathlib import Path

import pandas as pd
import pytest

from football_forecast.data.ingest_international import ingest_international_results
from football_forecast.data.provenance import file_sha256


def test_international_ingest_maps_schema_and_checks_checksum(tmp_path: Path):
    source = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "home_team": ["USA", "Holland"],
            "away_team": ["Holland", "USA"],
            "home_score": [1, 0],
            "away_score": [0, 1],
            "tournament": ["Friendly", "Friendly"],
            "city": ["X", "Y"],
            "country": ["Z", "Z"],
            "neutral": [True, False],
        }
    ).to_csv(source, index=False)
    mapping = tmp_path / "teams.yaml"
    mapping.write_text(
        "United States:\n  - USA\nNetherlands:\n  - Holland\n",
        encoding="utf-8",
    )
    matches, report = ingest_international_results(
        source,
        mapping_path=mapping,
        source_version="abc123",
        expected_sha256=file_sha256(source),
    )
    assert matches["team1"].tolist() == ["United States", "Netherlands"]
    assert matches["source_version"].unique().tolist() == ["abc123"]
    assert report.unresolved_names == ()


def test_international_ingest_rejects_checksum_mismatch(tmp_path: Path):
    source = tmp_path / "results.csv"
    source.write_text("date\n2020-01-01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        ingest_international_results(
            source,
            mapping_path=tmp_path / "unused.yaml",
            source_version="abc123",
            expected_sha256="0" * 64,
        )
