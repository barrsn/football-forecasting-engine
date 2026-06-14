from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_forecast.data.io import write_csv
from football_forecast.data.provenance import verify_file_sha256
from football_forecast.data.schema import coerce_matches
from football_forecast.data.teams import TeamNormalizationReport, normalize_match_teams


def ingest_international_results(
    input_path: str | Path,
    *,
    mapping_path: str | Path = "data/mapping/team_names.yaml",
    source_version: str,
    expected_sha256: str,
    snapshot_at: str | pd.Timestamp | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, TeamNormalizationReport]:
    """Convert the Mart Jürisoo/Kaggle results schema to the canonical schema."""
    input_path = Path(input_path)
    verify_file_sha256(input_path, expected_sha256)

    raw = pd.read_csv(input_path)
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing international results columns: {missing}")

    matches = raw.rename(
        columns={
            "home_team": "team1",
            "away_team": "team2",
            "home_score": "team1_goals",
            "away_score": "team2_goals",
        }
    )
    keep = [
        "date",
        "team1",
        "team2",
        "team1_goals",
        "team2_goals",
        "tournament",
        "neutral",
        *[column for column in ("city", "country") if column in matches],
    ]
    matches = matches[keep].copy()
    matches["source"] = "martj42_international_results"
    matches["source_version"] = source_version
    matches["snapshot_at"] = pd.to_datetime(snapshot_at, utc=True) if snapshot_at else pd.NaT
    matches, report = normalize_match_teams(matches, mapping_path)
    canonical = coerce_matches(matches)

    if output_path is not None:
        write_csv(canonical, output_path)
    return canonical, report
