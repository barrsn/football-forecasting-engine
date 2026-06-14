from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from football_forecast.data.provenance import verify_file_sha256

PLACEHOLDER_PATTERN = re.compile(r"^(?:W|L|R)\d+$", re.IGNORECASE)


def _team_name(value: Any) -> tuple[str, bool]:
    if isinstance(value, str) and value.strip():
        name = value.strip()
        unresolved = bool(PLACEHOLDER_PATTERN.match(name)) or "winner" in name.casefold()
        return name, unresolved
    if isinstance(value, dict):
        for key in ("name", "title", "team", "code"):
            if value.get(key):
                return _team_name(str(value[key]))
    return "UNRESOLVED", True


def _parse_kickoff(match: dict[str, Any]) -> pd.Timestamp:
    if match.get("kickoff_utc"):
        return pd.to_datetime(match["kickoff_utc"], utc=True, errors="raise")
    date = match.get("date")
    if not date:
        return pd.NaT
    time_value = str(match.get("time") or "00:00").strip()
    timezone_match = re.search(r"UTC\s*([+-])\s*(\d{1,2})", time_value, re.IGNORECASE)
    if timezone_match:
        sign, hour = timezone_match.groups()
        offset = f"{sign}{int(hour):02d}:00"
        clock = re.sub(r"UTC\s*[+-]\s*\d{1,2}", "", time_value, flags=re.IGNORECASE).strip()
        value = f"{date} {clock} {offset}"
    else:
        value = f"{date} {time_value}"
    return pd.to_datetime(value, utc=True, errors="raise")


def _iter_matches(payload: dict[str, Any]):
    if payload.get("rounds"):
        for round_index, round_data in enumerate(payload["rounds"], start=1):
            stage = round_data.get("name") or round_data.get("round") or f"round_{round_index}"
            for match in round_data.get("matches", []):
                yield match, stage
        return
    for match in payload.get("matches", []):
        yield match, match.get("round") or match.get("stage") or "unknown"


def parse_openfootball_worldcup(path: str | Path) -> pd.DataFrame:
    """Parse local OpenFootball World Cup JSON without network access."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for match_index, (match, stage) in enumerate(_iter_matches(payload), start=1):
        team1, unresolved1 = _team_name(
            match.get("team1") or match.get("home") or match.get("team1name")
        )
        team2, unresolved2 = _team_name(
            match.get("team2") or match.get("away") or match.get("team2name")
        )
        group = match.get("group")
        if isinstance(group, str) and group.casefold().startswith("group "):
            group = group.split()[-1]
        match_id = str(
            match.get("match_id")
            or match.get("id")
            or match.get("num")
            or match_index
        )
        rows.append(
            {
                "match_id": match_id,
                "stage": str(match.get("stage") or stage),
                "group": group,
                "kickoff_utc": _parse_kickoff(match),
                "team1": team1,
                "team2": team2,
                "unresolved_team1": unresolved1,
                "unresolved_team2": unresolved2,
                "source": "openfootball_worldcup_json",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("OpenFootball JSON contains no matches")
    if frame["match_id"].duplicated().any():
        raise ValueError("OpenFootball JSON contains duplicate match IDs")
    return frame.sort_values(["kickoff_utc", "match_id"], na_position="last").reset_index(drop=True)


def ingest_openfootball_worldcup(
    path: str | Path,
    *,
    source_version: str,
    expected_sha256: str,
    snapshot_at: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    verify_file_sha256(path, expected_sha256)
    frame = parse_openfootball_worldcup(path)
    frame["source_version"] = source_version
    frame["snapshot_at"] = (
        pd.to_datetime(snapshot_at, utc=True) if snapshot_at is not None else pd.NaT
    )
    return frame


def compare_fixture_mirror(
    mirror: pd.DataFrame,
    official: pd.DataFrame,
) -> pd.DataFrame:
    """Return field-level mismatches between a mirror and an official fixture table."""
    fields = ("kickoff_utc", "team1", "team2", "group", "stage")
    merged = mirror.merge(
        official,
        on="match_id",
        how="outer",
        suffixes=("_mirror", "_official"),
        indicator=True,
    )
    mismatches: list[dict[str, object]] = []
    for row in merged.to_dict(orient="records"):
        match_id = row["match_id"]
        if row["_merge"] != "both":
            mismatches.append(
                {"match_id": match_id, "field": "match_id", "status": row["_merge"]}
            )
            continue
        for field in fields:
            mirror_value = row.get(f"{field}_mirror")
            official_value = row.get(f"{field}_official")
            both_missing = pd.isna(mirror_value) and pd.isna(official_value)
            if not both_missing and str(mirror_value) != str(official_value):
                mismatches.append(
                    {
                        "match_id": match_id,
                        "field": field,
                        "mirror": mirror_value,
                        "official": official_value,
                        "status": "mismatch",
                    }
                )
    return pd.DataFrame(mismatches)
