from __future__ import annotations

import pandas as pd

from football_forecast.data.schema import add_outcome, coerce_matches
from football_forecast.features.asof_join import assert_prior_only_timestamps
from football_forecast.features.elo import EloConfig, add_elo_features, tournament_importance
from football_forecast.features.fifa import add_fifa_ranking_features, fifa_feature_columns
from football_forecast.features.h2h import add_h2h_features
from football_forecast.features.players import (
    add_player_snapshot_features,
    player_feature_columns,
)
from football_forecast.features.rolling import add_rolling_team_features
from football_forecast.features.scorers import (
    add_prior_scorer_features,
    scorer_feature_columns,
)

CONFEDERATIONS = ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA")


def build_feature_table(
    matches: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
    *,
    elo_config: EloConfig | None = None,
    rolling_half_life_days: float = 365.0,
    fifa_rankings: pd.DataFrame | None = None,
    goalscorers: pd.DataFrame | None = None,
    player_snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = coerce_matches(matches)
    base = add_elo_features(base, elo_config or EloConfig(), windows=windows)
    base = add_rolling_team_features(
        base,
        windows=windows,
        half_life_days=rolling_half_life_days,
    )
    base = add_h2h_features(base)
    if goalscorers is not None:
        base = add_prior_scorer_features(base, goalscorers)
    base = add_outcome(base)
    base["neutral_int"] = base["neutral"].astype(int)
    
    if fifa_rankings is not None:
        base = add_fifa_ranking_features(base, fifa_rankings)
    if player_snapshots is not None:
        base = add_player_snapshot_features(base, player_snapshots)

    host_team = base.get("host_team", pd.Series("", index=base.index))
    base["host_team1_int"] = (host_team == base["team1"]).astype(int)
    base["host_team2_int"] = (host_team == base["team2"]).astype(int)
    
    base["tournament_importance"] = base["tournament"].apply(
        lambda t: tournament_importance(t)
    )
    base["elo_diff_x_tournament_importance"] = base["elo_diff_pre"] * base["tournament_importance"]
    base["elo_diff_x_days_rest_diff"] = base["elo_diff_pre"] * base.get("days_rest_diff", 0.0)
    if "fifa_points_diff" in base.columns:
        base["elo_diff_x_fifa_points_diff"] = base["elo_diff_pre"] * base["fifa_points_diff"]

    if {"team1_confederation", "team2_confederation"}.issubset(base.columns):
        team1_confederation = base["team1_confederation"].astype(str).str.upper()
        team2_confederation = base["team2_confederation"].astype(str).str.upper()
        base["same_confederation_int"] = (
            team1_confederation == team2_confederation
        ).astype(int)
        for confederation in CONFEDERATIONS:
            suffix = confederation.casefold()
            base[f"team1_confederation_{suffix}"] = (
                team1_confederation == confederation
            ).astype(int)
            base[f"team2_confederation_{suffix}"] = (
                team2_confederation == confederation
            ).astype(int)
    assert_prior_only_timestamps(
        base,
        [
            "elo_team1_available_at",
            "elo_team2_available_at",
            "team1_history_available_at",
            "team2_history_available_at",
        ],
    )
    return base


def default_feature_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "elo_team1_pre",
        "elo_team2_pre",
        "elo_diff_pre",
        "elo_exp_team1",
        "neutral_int",
        "host_team1_int",
        "host_team2_int",
        "tournament_importance",
        "team1_days_rest",
        "team2_days_rest",
        "days_rest_diff",
        "h2h_total_matches",
        "h2h_team1_win_pct",
        "h2h_team2_win_pct",
        "h2h_team1_goals_mean",
        "h2h_team2_goals_mean",
        "h2h_last_outcome",
    ]
    rolling = [
        c
        for c in df.columns
        if ("_roll" in c or "_ewm" in c) and c not in preferred and not c.startswith("elo_momentum_")
    ]
    elo_momentum = [c for c in df.columns if c.startswith("elo_momentum_")]
    confederation = [
        c for c in df.columns if c == "same_confederation_int" or "_confederation_" in c
    ]
    cross = [
        "elo_diff_x_tournament_importance",
        "elo_diff_x_days_rest_diff",
    ]
    if "elo_diff_x_fifa_points_diff" in df.columns:
        cross.append("elo_diff_x_fifa_points_diff")
        
    fifa = fifa_feature_columns(df)
    players = player_feature_columns(df)
    scorers = scorer_feature_columns(df)
    cols = [
        c
        for c in [
            *preferred,
            *rolling,
            *elo_momentum,
            *confederation,
            *cross,
            *fifa,
            *players,
            *scorers,
        ]
        if c in df.columns
    ]
    return sorted(set(cols), key=cols.index)
