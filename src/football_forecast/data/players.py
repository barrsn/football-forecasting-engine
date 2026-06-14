from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PLAYER_POSITIONS = {"GK", "DF", "MF", "FW", "UNKNOWN"}
AVAILABILITY_STATUSES = {
    "available",
    "doubtful",
    "injured",
    "suspended",
    "unavailable",
    "unknown",
}
LINEUP_STATUSES = {"starter", "bench", "not_in_squad", "unknown"}

REQUIRED_PLAYER_COLUMNS = {
    "snapshot_id",
    "player_id",
    "player_name",
    "team",
    "available_at",
    "source",
    "source_version",
}

NONNEGATIVE_PLAYER_COLUMNS = (
    "international_caps",
    "international_goals",
    "minutes_365d",
    "starts_365d",
    "goals_365d",
    "assists_365d",
    "yellow_cards_365d",
    "red_cards_365d",
    "player_rating_uncertainty",
)


@dataclass(frozen=True)
class PlayerSnapshotReport:
    rows: int
    snapshots: int
    teams: int
    players: int
    rating_coverage: float
    availability_coverage: float
    lineup_coverage: float


def _clean_required_text(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame[column].isna().any():
        raise ValueError(f"Column {column} contains missing values")
    values = frame[column].astype(str).str.strip()
    if (values == "").any():
        raise ValueError(f"Column {column} contains empty values")
    return values


def _optional_numeric(
    frame: pd.DataFrame,
    column: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="raise").astype(float)
    finite = values.notna()
    if not np.isfinite(values[finite]).all():
        raise ValueError(f"Column {column} contains non-finite values")
    if minimum is not None and (values[finite] < minimum).any():
        raise ValueError(f"Column {column} must be >= {minimum}")
    if maximum is not None and (values[finite] > maximum).any():
        raise ValueError(f"Column {column} must be <= {maximum}")
    return values


def coerce_player_snapshots(
    snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, PlayerSnapshotReport]:
    """Validate timestamped player, squad, availability, and lineup snapshots."""
    missing = sorted(REQUIRED_PLAYER_COLUMNS.difference(snapshots.columns))
    if missing:
        raise ValueError(f"Missing required player columns: {missing}")

    out = snapshots.copy()
    for column in (
        "snapshot_id",
        "player_id",
        "player_name",
        "team",
        "source",
        "source_version",
    ):
        out[column] = _clean_required_text(out, column)

    out["available_at"] = pd.to_datetime(out["available_at"], errors="raise", utc=True)
    if out["available_at"].isna().any():
        raise ValueError("Column available_at contains missing values")
    if "snapshot_at" in out:
        out["snapshot_at"] = pd.to_datetime(out["snapshot_at"], errors="raise", utc=True)
    else:
        out["snapshot_at"] = out["available_at"]
    if (out["snapshot_at"] > out["available_at"]).any():
        raise ValueError("snapshot_at cannot be after available_at")

    out["position"] = (
        out.get("position", pd.Series("UNKNOWN", index=out.index))
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    invalid_positions = sorted(set(out["position"]).difference(PLAYER_POSITIONS))
    if invalid_positions:
        raise ValueError(f"Invalid player positions: {invalid_positions}")

    out["availability_status"] = (
        out.get("availability_status", pd.Series("unknown", index=out.index))
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    invalid_availability = sorted(
        set(out["availability_status"]).difference(AVAILABILITY_STATUSES)
    )
    if invalid_availability:
        raise ValueError(f"Invalid availability statuses: {invalid_availability}")

    out["lineup_status"] = (
        out.get("lineup_status", pd.Series("unknown", index=out.index))
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.casefold()
    )
    invalid_lineups = sorted(set(out["lineup_status"]).difference(LINEUP_STATUSES))
    if invalid_lineups:
        raise ValueError(f"Invalid lineup statuses: {invalid_lineups}")

    out["expected_start_probability"] = _optional_numeric(
        out, "expected_start_probability", minimum=0.0, maximum=1.0
    )
    out["expected_minutes"] = _optional_numeric(
        out, "expected_minutes", minimum=0.0, maximum=130.0
    )
    out["player_rating"] = _optional_numeric(out, "player_rating")
    for column in NONNEGATIVE_PLAYER_COLUMNS:
        out[column] = _optional_numeric(out, column, minimum=0.0)

    if "birth_date" in out:
        out["birth_date"] = pd.to_datetime(out["birth_date"], errors="raise", utc=True)
        if (out["birth_date"] >= out["snapshot_at"]).fillna(False).any():
            raise ValueError("birth_date must be before snapshot_at")
        out["age_years"] = (
            (out["snapshot_at"] - out["birth_date"]).dt.total_seconds()
            / (365.25 * 86400)
        )
    else:
        out["birth_date"] = pd.NaT
        out["age_years"] = _optional_numeric(
            out, "age_years", minimum=14.0, maximum=60.0
        )

    duplicate = out.duplicated(["snapshot_id", "player_id"], keep=False)
    if duplicate.any():
        raise ValueError("Duplicate player_id values within a snapshot")

    snapshot_consistency = out.groupby("snapshot_id").agg(
        teams=("team", "nunique"),
        available_times=("available_at", "nunique"),
        sources=("source", "nunique"),
        versions=("source_version", "nunique"),
    )
    if (snapshot_consistency > 1).any(axis=None):
        raise ValueError(
            "Each snapshot_id must have one team, available_at, source, and source_version"
        )

    report = PlayerSnapshotReport(
        rows=len(out),
        snapshots=int(out["snapshot_id"].nunique()),
        teams=int(out["team"].nunique()),
        players=int(out["player_id"].nunique()),
        rating_coverage=float(out["player_rating"].notna().mean()),
        availability_coverage=float(
            (out["availability_status"] != "unknown").mean()
        ),
        lineup_coverage=float((out["lineup_status"] != "unknown").mean()),
    )
    return (
        out.sort_values(["available_at", "snapshot_id", "player_id"]).reset_index(
            drop=True
        ),
        report,
    )
