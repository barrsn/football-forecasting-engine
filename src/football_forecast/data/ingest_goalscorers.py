from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from football_forecast.data.schema import parse_boolean_series
from football_forecast.data.teams import (
    TeamNormalizationReport,
    load_team_aliases,
    normalize_team_series,
)


def ingest_goalscorers(
    input_path: str | Path,
    *,
    mapping_path: str | Path,
    source_version: str,
    expected_sha256: str,
    snapshot_at: str | pd.Timestamp,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, TeamNormalizationReport]:
    """Validate and normalize the pinned Mart Jürisoo goalscorer table."""
    input_path = Path(input_path)
    actual_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for {input_path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    raw = pd.read_csv(input_path)
    required = {
        "date",
        "home_team",
        "away_team",
        "team",
        "scorer",
        "minute",
        "own_goal",
        "penalty",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing goalscorer columns: {missing}")

    out = raw.copy()
    aliases = load_team_aliases(mapping_path)
    reports: list[TeamNormalizationReport] = []
    for column in ("home_team", "away_team", "team"):
        out[column], report = normalize_team_series(out[column], aliases)
        reports.append(report)

    out["date"] = pd.to_datetime(out["date"], errors="raise", utc=True)
    out["minute"] = pd.to_numeric(out["minute"], errors="coerce")
    if (out["minute"].dropna() < 0).any():
        raise ValueError("minute must be non-negative")
    out["own_goal"] = parse_boolean_series(out["own_goal"], "own_goal")
    out["penalty"] = parse_boolean_series(out["penalty"], "penalty")
    out["scorer"] = out["scorer"].fillna("").astype(str).str.strip()
    out["source"] = "martj42_international_results"
    out["source_version"] = source_version
    out["snapshot_at"] = pd.to_datetime(snapshot_at, errors="raise", utc=True)

    unresolved = sorted(
        set().union(*(set(report.unresolved_names) for report in reports))
    )
    report = TeamNormalizationReport(
        total_values=sum(item.total_values for item in reports),
        alias_matches=sum(item.alias_matches for item in reports),
        unresolved_names=tuple(unresolved),
    )
    out = out.sort_values(
        ["date", "home_team", "away_team", "minute", "scorer"],
        na_position="last",
    ).reset_index(drop=True)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, index=False)
    return out, report


def download_pinned_goalscorers(
    *,
    commit: str,
    raw_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, object]:
    """Download goalscorers.csv from an immutable Git commit and record provenance."""
    url = (
        "https://raw.githubusercontent.com/martj42/"
        f"international_results/{commit}/goalscorers.csv"
    )
    content = urlopen(url, timeout=60).read()
    checksum = hashlib.sha256(content).hexdigest()
    raw_path = Path(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(content)

    manifest_file = Path(manifest_path)
    manifest = (
        json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest_file.exists()
        else {}
    )
    manifest["goalscorers"] = {
        "url": url,
        "commit": commit,
        "sha256": checksum,
        "bytes": len(content),
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest["goalscorers"]
