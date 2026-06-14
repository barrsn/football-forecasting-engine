from __future__ import annotations

import numpy as np
import pandas as pd

from football_forecast.features.asof_join import (
    assert_prior_only_timestamps,
    join_team_rating_features_asof,
)

FIFA_NUMERIC_COLUMNS = [
    "fifa_rank",
    "fifa_points",
    "fifa_rated_matches",
    "fifa_rank_percentile",
    "fifa_points_z",
]
FIFA_CONFEDERATIONS = ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA")


def add_fifa_ranking_features(
    matches: pd.DataFrame,
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    columns = [*FIFA_NUMERIC_COLUMNS, "confederation"]
    out = join_team_rating_features_asof(
        matches,
        rankings,
        columns,
        prefix="fifa",
    )
    team1_confederation = out["team1_fifa_confederation"].fillna("").str.upper()
    team2_confederation = out["team2_fifa_confederation"].fillna("").str.upper()
    out["fifa_same_confederation_int"] = (
        (team1_confederation == team2_confederation)
        & team1_confederation.ne("")
    ).astype(int)
    for confederation in FIFA_CONFEDERATIONS:
        suffix = confederation.casefold()
        out[f"team1_fifa_confederation_{suffix}"] = (
            team1_confederation == confederation
        ).astype(int)
        out[f"team2_fifa_confederation_{suffix}"] = (
            team2_confederation == confederation
        ).astype(int)
    for column in ("rank", "points", "rated_matches", "rank_percentile", "points_z"):
        left = out[f"team1_fifa_{column}"]
        right = out[f"team2_fifa_{column}"]
        out[f"fifa_{column}_diff_abs"] = (left - right).abs()
        out[f"fifa_{column}_mean"] = (left + right) / 2.0
    out["fifa_rated_matches_min_log1p"] = np.log1p(
        out[["team1_fifa_rated_matches", "team2_fifa_rated_matches"]]
        .min(axis=1)
        .clip(lower=0)
    )
    kickoff = pd.to_datetime(out["kickoff_utc"], utc=True)
    team1_available = pd.to_datetime(out["team1_fifa_available_at"], utc=True)
    team2_available = pd.to_datetime(out["team2_fifa_available_at"], utc=True)
    out["team1_fifa_snapshot_age_days"] = (
        kickoff - team1_available
    ).dt.total_seconds() / 86400.0
    out["team2_fifa_snapshot_age_days"] = (
        kickoff - team2_available
    ).dt.total_seconds() / 86400.0
    out["fifa_snapshot_age_days_max"] = out[
        ["team1_fifa_snapshot_age_days", "team2_fifa_snapshot_age_days"]
    ].max(axis=1)
    out["team1_fifa_missing_int"] = out["team1_fifa_points"].isna().astype(int)
    out["team2_fifa_missing_int"] = out["team2_fifa_points"].isna().astype(int)
    out["fifa_any_missing_int"] = (
        out["team1_fifa_missing_int"] | out["team2_fifa_missing_int"]
    ).astype(int)
    if "elo_diff_pre" in out:
        elo_scaled = out["elo_diff_pre"] / 400.0
        out["fifa_elo_consensus_diff"] = out["fifa_points_z_diff"] + elo_scaled
        out["fifa_elo_disagreement"] = out["fifa_points_z_diff"] - elo_scaled
        out["fifa_points_neutral_interaction"] = (
            out["fifa_points_z_diff"] * out.get("neutral_int", 0)
        )
        out["fifa_points_importance_interaction"] = (
            out["fifa_points_z_diff"] * out.get("tournament_importance", 1.0)
        )
    assert_prior_only_timestamps(
        out,
        ["team1_fifa_available_at", "team2_fifa_available_at"],
    )
    return out


def fifa_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for side in ("team1", "team2"):
        columns.extend(f"{side}_{column}" for column in FIFA_NUMERIC_COLUMNS)
    columns.extend(f"fifa_{column[5:]}_diff" for column in FIFA_NUMERIC_COLUMNS)
    columns.append("fifa_same_confederation_int")
    for side in ("team1", "team2"):
        columns.extend(
            f"{side}_fifa_confederation_{confederation.casefold()}"
            for confederation in FIFA_CONFEDERATIONS
        )
    return [column for column in columns if column in frame]
