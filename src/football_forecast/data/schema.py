from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_MATCH_COLUMNS = {
    "date",
    "team1",
    "team2",
    "team1_goals",
    "team2_goals",
    "tournament",
    "neutral",
}

PROBABILITY_COLUMNS = ("p_team2_win", "p_draw", "p_team1_win")


class OutcomeClass(IntEnum):
    TEAM2_WIN = 0
    DRAW = 1
    TEAM1_WIN = 2


@dataclass(frozen=True)
class SchemaReport:
    is_valid: bool
    missing_columns: list[str]
    n_rows: int


def validate_matches(df: pd.DataFrame, required: Iterable[str] = REQUIRED_MATCH_COLUMNS) -> SchemaReport:
    required_set = set(required)
    missing = sorted(required_set.difference(df.columns))
    return SchemaReport(is_valid=not missing, missing_columns=missing, n_rows=len(df))


def parse_boolean_series(series: pd.Series, column_name: str) -> pd.Series:
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}

    def parse(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in true_values:
                return True
            if normalized in false_values:
                return False
        raise ValueError(f"Invalid boolean value in {column_name}: {value!r}")

    if series.isna().any():
        raise ValueError(f"Column {column_name} contains missing values")
    return series.map(parse).astype(bool)


def _coerce_nonnegative_integer(series: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any():
        raise ValueError(f"Column {column_name} contains missing values")
    if (numeric < 0).any() or not np.all(np.equal(numeric, np.floor(numeric))):
        raise ValueError(f"Column {column_name} must contain non-negative integers")
    return numeric.astype(int)


def _stable_match_id(row: pd.Series) -> str:
    key = "|".join(
        [
            str(row["source"]),
            row["kickoff_utc"].isoformat(),
            str(row["team1"]),
            str(row["team2"]),
        ]
    )
    return sha256(key.encode("utf-8")).hexdigest()[:20]


def _validate_duplicate_matches(df: pd.DataFrame) -> None:
    duplicate_ids = df["match_id"].duplicated(keep=False)
    if duplicate_ids.any():
        ids = sorted(df.loc[duplicate_ids, "match_id"].astype(str).unique())
        raise ValueError(f"Duplicate match_id values: {ids[:10]}")

    pair_key = pd.DataFrame(
        {
            "kickoff_utc": df["kickoff_utc"],
            "team_low": df[["team1", "team2"]].min(axis=1),
            "team_high": df[["team1", "team2"]].max(axis=1),
        }
    )
    duplicate_pairs = pair_key.duplicated(keep=False)
    if duplicate_pairs.any():
        raise ValueError("Duplicate matches found for the same teams and kickoff")

    appearances = pd.concat(
        [
            df[["kickoff_utc", "team1"]].rename(columns={"team1": "team"}),
            df[["kickoff_utc", "team2"]].rename(columns={"team2": "team"}),
        ],
        ignore_index=True,
    )
    duplicate_appearances = appearances.duplicated(["kickoff_utc", "team"], keep=False)
    if duplicate_appearances.any():
        bad = appearances.loc[duplicate_appearances].drop_duplicates()
        raise ValueError(f"A team appears more than once at the same kickoff:\n{bad}")


def coerce_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical, validated, chronologically sorted match table."""
    report = validate_matches(df)
    if not report.is_valid:
        raise ValueError(f"Missing required columns: {report.missing_columns}")

    out = df.copy()
    kickoff_source = out["kickoff_utc"] if "kickoff_utc" in out else out["date"]
    out["kickoff_utc"] = pd.to_datetime(kickoff_source, errors="raise", utc=True)
    out["date"] = out["kickoff_utc"]

    for column in ("team1", "team2", "tournament"):
        if out[column].isna().any():
            raise ValueError(f"Column {column} contains missing values")
        out[column] = out[column].astype(str).str.strip()
        if (out[column] == "").any():
            raise ValueError(f"Column {column} contains empty values")

    out["team1_goals"] = _coerce_nonnegative_integer(out["team1_goals"], "team1_goals")
    out["team2_goals"] = _coerce_nonnegative_integer(out["team2_goals"], "team2_goals")
    out["neutral"] = parse_boolean_series(out["neutral"], "neutral")

    if (out["team1"] == out["team2"]).any():
        bad = out.loc[out["team1"] == out["team2"], ["kickoff_utc", "team1", "team2"]]
        raise ValueError(f"Invalid matches with identical teams:\n{bad}")

    out["source"] = out.get("source", pd.Series("unknown", index=out.index)).fillna("unknown")
    out["source_version"] = out.get(
        "source_version", pd.Series("unversioned", index=out.index)
    ).fillna("unversioned")
    out["snapshot_at"] = pd.to_datetime(
        out.get("snapshot_at", pd.Series(pd.NaT, index=out.index)), errors="coerce", utc=True
    )
    default_available = out["kickoff_utc"] + pd.Timedelta(hours=3)
    out["available_at"] = pd.to_datetime(
        out.get("available_at", default_available), errors="raise", utc=True
    ).fillna(default_available)

    out["team1_regulation_goals"] = _coerce_nonnegative_integer(
        out.get("team1_regulation_goals", out["team1_goals"]), "team1_regulation_goals"
    )
    out["team2_regulation_goals"] = _coerce_nonnegative_integer(
        out.get("team2_regulation_goals", out["team2_goals"]), "team2_regulation_goals"
    )
    for column in (
        "team1_extra_time_goals",
        "team2_extra_time_goals",
        "team1_penalty_goals",
        "team2_penalty_goals",
    ):
        if column not in out:
            out[column] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        else:
            missing = out[column].isna()
            coerced = _coerce_nonnegative_integer(out.loc[~missing, column], column)
            out[column] = pd.Series(pd.NA, index=out.index, dtype="Int64")
            out.loc[~missing, column] = coerced.astype("Int64")

    if "match_id" not in out:
        out["match_id"] = out.apply(_stable_match_id, axis=1)
    else:
        if out["match_id"].isna().any():
            raise ValueError("Column match_id contains missing values")
        out["match_id"] = out["match_id"].astype(str).str.strip()

    _validate_duplicate_matches(out)
    return out.sort_values(["kickoff_utc", "match_id"]).reset_index(drop=True)


def add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Add outcome labels using the single repository-wide class definition."""
    out = df.copy()
    out["outcome"] = int(OutcomeClass.DRAW)
    out.loc[out["team1_goals"] > out["team2_goals"], "outcome"] = int(
        OutcomeClass.TEAM1_WIN
    )
    out.loc[out["team1_goals"] < out["team2_goals"], "outcome"] = int(
        OutcomeClass.TEAM2_WIN
    )
    return out
