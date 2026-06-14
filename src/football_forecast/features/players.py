from __future__ import annotations

import numpy as np
import pandas as pd

from football_forecast.data.players import coerce_player_snapshots
from football_forecast.features.asof_join import (
    assert_prior_only_timestamps,
    join_team_rating_features_asof,
)

UNAVAILABLE_STATUSES = {"injured", "suspended", "unavailable"}


def _top_mean(values: pd.Series, count: int = 11) -> float:
    clean = values.dropna().astype(float)
    if clean.empty:
        return float("nan")
    return float(clean.nlargest(min(count, len(clean))).mean())


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def aggregate_player_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one row per timestamped team squad snapshot."""
    players, _ = coerce_player_snapshots(snapshots)
    players["_is_available"] = ~players["availability_status"].isin(
        UNAVAILABLE_STATUSES
    )
    players["_availability_known"] = players["availability_status"] != "unknown"
    players["_lineup_known"] = players["lineup_status"] != "unknown"
    players["_is_starter"] = players["lineup_status"] == "starter"

    inferred_start = players["expected_start_probability"].copy()
    inferred_start = inferred_start.mask(players["lineup_status"] == "starter", 1.0)
    inferred_start = inferred_start.mask(
        players["lineup_status"].isin({"bench", "not_in_squad"}), 0.0
    )
    inferred_start = inferred_start.mask(~players["_is_available"], 0.0)
    players["_start_probability"] = inferred_start

    rows: list[dict[str, object]] = []
    for snapshot_id, group in players.groupby("snapshot_id", sort=False):
        available = group.loc[group["_is_available"]]
        rating = group["player_rating"]
        available_rating = available["player_rating"]
        top_all = _top_mean(rating)
        top_available = _top_mean(available_rating)
        start_weight = group["_start_probability"]
        expected_minutes = group["expected_minutes"].copy()
        expected_minutes = expected_minutes.fillna(start_weight * 90.0)

        row: dict[str, object] = {
            "snapshot_id": snapshot_id,
            "team": group["team"].iloc[0],
            "rating_date": group["available_at"].iloc[0],
            "player_snapshot_at": group["snapshot_at"].max(),
            "player_source": group["source"].iloc[0],
            "player_source_version": group["source_version"].iloc[0],
            "player_squad_size": float(len(group)),
            "player_available_count": float(group["_is_available"].sum()),
            "player_unavailable_count": float((~group["_is_available"]).sum()),
            "player_official_starter_count": float(group["_is_starter"].sum()),
            "player_expected_starter_count": float(start_weight.sum(min_count=1)),
            "player_rating_coverage": float(rating.notna().mean()),
            "player_availability_coverage": float(group["_availability_known"].mean()),
            "player_lineup_coverage": float(group["_lineup_known"].mean()),
            "player_rating_mean": float(rating.mean()),
            "player_rating_top11": top_all,
            "player_available_rating_top11": top_available,
            "player_absence_rating_loss": (
                top_all - top_available
                if np.isfinite(top_all) and np.isfinite(top_available)
                else float("nan")
            ),
            "player_expected_xi_rating": _weighted_mean(rating, start_weight),
            "player_expected_minutes_rating": _weighted_mean(
                rating, expected_minutes
            ),
            "player_age_mean": float(group["age_years"].mean()),
            "player_age_std": float(group["age_years"].std(ddof=0)),
            "player_caps_total": float(group["international_caps"].sum(min_count=1)),
            "player_goals_total": float(
                group["international_goals"].sum(min_count=1)
            ),
            "player_minutes_365d_total": float(
                group["minutes_365d"].sum(min_count=1)
            ),
            "player_starts_365d_total": float(
                group["starts_365d"].sum(min_count=1)
            ),
            "player_goals_365d_total": float(group["goals_365d"].sum(min_count=1)),
            "player_assists_365d_total": float(
                group["assists_365d"].sum(min_count=1)
            ),
            "player_club_diversity": float(
                group.get("club", pd.Series(dtype=object)).dropna().nunique()
            ),
        }
        for position in ("GK", "DF", "MF", "FW"):
            position_group = available.loc[available["position"] == position]
            row[f"player_available_{position.lower()}_count"] = float(
                len(position_group)
            )
            row[f"player_{position.lower()}_rating_top"] = _top_mean(
                position_group["player_rating"], count=1
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rating_date", "team"]).reset_index(
        drop=True
    )


def add_player_snapshot_features(
    matches: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Join the latest strictly pre-kickoff player snapshot for each team."""
    aggregated = aggregate_player_snapshots(snapshots)
    rating_columns = [
        column
        for column in aggregated.columns
        if column.startswith("player_")
        and column
        not in {
            "player_snapshot_at",
            "player_source",
            "player_source_version",
        }
    ]
    out = join_team_rating_features_asof(
        matches,
        aggregated,
        rating_columns,
        prefix="player",
    )
    out["player_any_missing_int"] = (
        out["team1_player_squad_size"].isna()
        | out["team2_player_squad_size"].isna()
    ).astype(int)
    assert_prior_only_timestamps(
        out,
        ["team1_player_available_at", "team2_player_available_at"],
    )
    return out


def player_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded_suffixes = ("_available_at",)
    return [
        column
        for column in frame.columns
        if (
            column.startswith(("team1_player_", "team2_player_", "player_"))
            and not column.endswith(excluded_suffixes)
            and pd.api.types.is_numeric_dtype(frame[column])
        )
    ]
