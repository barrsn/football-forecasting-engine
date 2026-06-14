from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd

from football_forecast.data.ingest_international import ingest_international_results
from football_forecast.data.provenance import file_sha256


@dataclass(frozen=True)
class PreparationAudit:
    source_path: str
    source_version: str
    source_sha256: str
    start_date: str
    cutoff: str
    input_rows: int
    future_or_cutoff_rows: int
    incomplete_rows: int
    pre_start_rows: int
    semantic_duplicate_rows: int
    ambiguous_same_day_rows: int
    output_rows: int
    output_min_date: str
    output_max_date: str
    unique_teams: int


def prepare_completed_international_results(
    source_path: str | Path,
    *,
    source_version: str,
    expected_sha256: str,
    start_date: str = "1990-01-01",
    cutoff: str = "2026-06-11",
    mapping_path: str | Path = "data/mapping/team_names.yaml",
    output_path: str | Path | None = None,
    audit_path: str | Path | None = None,
) -> tuple[pd.DataFrame, PreparationAudit]:
    """Prepare completed, chronologically unambiguous international matches."""
    source_path = Path(source_path)
    raw = pd.read_csv(source_path)
    input_rows = len(raw)
    dates = pd.to_datetime(raw["date"], errors="raise")
    start = pd.Timestamp(start_date)
    cutoff_time = pd.Timestamp(cutoff)

    incomplete = raw["home_score"].isna() | raw["away_score"].isna()
    before_start = dates < start
    at_or_after_cutoff = dates >= cutoff_time
    filtered = raw.loc[~incomplete & ~before_start & ~at_or_after_cutoff].copy()

    semantic_key = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    ]
    semantic_duplicates = filtered.duplicated(semantic_key, keep="first")
    semantic_duplicate_rows = int(semantic_duplicates.sum())
    filtered = filtered.loc[~semantic_duplicates].copy()

    appearances = pd.concat(
        [
            filtered[["date", "home_team"]].rename(columns={"home_team": "team"}),
            filtered[["date", "away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    )
    ambiguous_pairs = appearances.loc[
        appearances.duplicated(["date", "team"], keep=False), ["date", "team"]
    ].drop_duplicates()
    ambiguous_home = filtered.merge(
        ambiguous_pairs,
        left_on=["date", "home_team"],
        right_on=["date", "team"],
        how="left",
        indicator="_ambiguous_home",
    )["_ambiguous_home"].eq("both")
    ambiguous_away = filtered.merge(
        ambiguous_pairs,
        left_on=["date", "away_team"],
        right_on=["date", "team"],
        how="left",
        indicator="_ambiguous_away",
    )["_ambiguous_away"].eq("both")
    ambiguous_rows = ambiguous_home.to_numpy() | ambiguous_away.to_numpy()
    ambiguous_same_day_rows = int(ambiguous_rows.sum())
    filtered = filtered.loc[~ambiguous_rows].copy()

    prepared_source = source_path.with_name(f"{source_path.stem}.prepared.csv")
    filtered.to_csv(prepared_source, index=False)
    prepared_sha256 = file_sha256(prepared_source)
    canonical, _ = ingest_international_results(
        prepared_source,
        mapping_path=mapping_path,
        source_version=source_version,
        expected_sha256=prepared_sha256,
        snapshot_at=cutoff_time.tz_localize("UTC"),
        output_path=output_path,
    )

    audit = PreparationAudit(
        source_path=str(source_path),
        source_version=source_version,
        source_sha256=expected_sha256,
        start_date=start.isoformat(),
        cutoff=cutoff_time.isoformat(),
        input_rows=input_rows,
        future_or_cutoff_rows=int(at_or_after_cutoff.sum()),
        incomplete_rows=int(incomplete.sum()),
        pre_start_rows=int(before_start.sum()),
        semantic_duplicate_rows=semantic_duplicate_rows,
        ambiguous_same_day_rows=ambiguous_same_day_rows,
        output_rows=len(canonical),
        output_min_date=canonical["kickoff_utc"].min().isoformat(),
        output_max_date=canonical["kickoff_utc"].max().isoformat(),
        unique_teams=len(set(canonical["team1"]).union(canonical["team2"])),
    )
    if audit_path is not None:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(audit), indent=2), encoding="utf-8")
    return canonical, audit
