from __future__ import annotations

import numpy as np
import pandas as pd

from football_forecast.features.strength import (
    OnlinePoissonConfig,
    add_online_poisson_strength,
    add_prior_match_counts,
)


def tournament_category(value: object) -> str:
    name = str(value).casefold()
    if "friendly" in name:
        return "friendly"
    if "nations league" in name:
        return "nations_league"
    if "qualification" in name or "qualifier" in name:
        if "world cup" in name:
            return "world_cup_qualifier"
        return "continental_qualifier"
    if "world cup" in name and "conifa" not in name:
        return "world_cup"
    major_tokens = (
        "uefa euro",
        "african cup of nations",
        "asian cup",
        "copa am",
        "gold cup",
        "oceania nations cup",
        "confederations cup",
    )
    if any(token in name for token in major_tokens):
        return "continental_final"
    return "other"


def add_advanced_context_features(
    frame: pd.DataFrame,
    *,
    online_config: OnlinePoissonConfig | None = None,
) -> pd.DataFrame:
    """Add deterministic, pre-match-only context and nonlinear features."""
    out = add_prior_match_counts(frame)
    out = add_online_poisson_strength(out, online_config)
    kickoff = pd.to_datetime(out["kickoff_utc"], utc=True)
    year_fraction = kickoff.dt.year + (kickoff.dt.dayofyear - 1) / 365.25
    out["calendar_year_scaled"] = (year_fraction - 2000.0) / 10.0
    out["month_sin"] = np.sin(2.0 * np.pi * kickoff.dt.month / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * kickoff.dt.month / 12.0)

    categories = out["tournament"].map(tournament_category)
    for category in (
        "friendly",
        "nations_league",
        "world_cup_qualifier",
        "continental_qualifier",
        "world_cup",
        "continental_final",
    ):
        out[f"tournament_category_{category}"] = (categories == category).astype(int)

    elo_scaled = out["elo_diff_pre"] / 400.0
    out["elo_diff_scaled"] = elo_scaled
    out["elo_diff_abs_scaled"] = elo_scaled.abs()
    out["elo_diff_squared"] = elo_scaled**2
    out["elo_diff_cubic"] = elo_scaled**3
    out["elo_diff_neutral_interaction"] = elo_scaled * out["neutral_int"]
    out["elo_diff_importance_interaction"] = (
        elo_scaled * out["tournament_importance"]
    )
    out["online_attack_diff_pre"] = (
        out["online_attack_team1_pre"] - out["online_attack_team2_pre"]
    )
    out["online_defence_diff_pre"] = (
        out["online_defence_team1_pre"] - out["online_defence_team2_pre"]
    )
    out["online_lambda_diff_abs_pre"] = out["online_lambda_diff_pre"].abs()
    out["online_log_lambda_ratio_abs_pre"] = out[
        "online_log_lambda_ratio_pre"
    ].abs()

    for suffix in ("roll5", "roll10", "roll20", "ewm"):
        required = {
            f"team1_goals_for_{suffix}",
            f"team1_goals_against_{suffix}",
            f"team2_goals_for_{suffix}",
            f"team2_goals_against_{suffix}",
        }
        if not required.issubset(out.columns):
            continue
        team1_matchup = (
            out[f"team1_goals_for_{suffix}"]
            + out[f"team2_goals_against_{suffix}"]
        ) / 2.0
        team2_matchup = (
            out[f"team2_goals_for_{suffix}"]
            + out[f"team1_goals_against_{suffix}"]
        ) / 2.0
        out[f"team1_matchup_goals_{suffix}"] = team1_matchup
        out[f"team2_matchup_goals_{suffix}"] = team2_matchup
        out[f"matchup_goals_{suffix}_diff"] = team1_matchup - team2_matchup
        out[f"matchup_goals_{suffix}_diff_abs"] = (
            team1_matchup - team2_matchup
        ).abs()
        out[f"matchup_goals_{suffix}_total"] = team1_matchup + team2_matchup
        if {
            f"team1_win_{suffix}",
            f"team2_win_{suffix}",
            f"team1_draw_{suffix}",
            f"team2_draw_{suffix}",
        }.issubset(out.columns):
            out[f"win_form_{suffix}_diff"] = (
                out[f"team1_win_{suffix}"] - out[f"team2_win_{suffix}"]
            )
            out[f"draw_form_{suffix}_diff"] = (
                out[f"team1_draw_{suffix}"] - out[f"team2_draw_{suffix}"]
            )
            out[f"draw_form_{suffix}_mean"] = (
                out[f"team1_draw_{suffix}"] + out[f"team2_draw_{suffix}"]
            ) / 2.0

    for prefix in ("goal_diff", "goals_for", "opponent_adjusted_goal_diff"):
        short = f"{prefix}_roll5_diff"
        long = f"{prefix}_roll20_diff"
        if short in out and long in out:
            out[f"{prefix}_momentum_diff"] = out[short] - out[long]
            out[f"{prefix}_momentum_abs"] = (out[short] - out[long]).abs()

    for column in (
        "goal_diff_roll5_diff",
        "goal_diff_roll10_diff",
        "goal_diff_roll20_diff",
        "goal_diff_ewm_diff",
        "opponent_adjusted_goal_diff_roll5_diff",
        "opponent_adjusted_goal_diff_roll10_diff",
        "opponent_adjusted_goal_diff_roll20_diff",
        "opponent_adjusted_goal_diff_ewm_diff",
    ):
        if column in out:
            out[f"{column}_abs"] = out[column].abs()

    out["team1_days_rest_log1p"] = np.log1p(
        out["team1_days_rest"].clip(lower=0, upper=365)
    )
    out["team2_days_rest_log1p"] = np.log1p(
        out["team2_days_rest"].clip(lower=0, upper=365)
    )
    out["days_rest_diff_clipped"] = out["days_rest_diff"].clip(-60, 60)
    return out


def advanced_feature_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = (
        "online_",
        "tournament_category_",
    )
    exact = {
        "team1_prior_match_count",
        "team2_prior_match_count",
        "prior_match_count_min",
        "prior_match_count_diff",
        "team1_prior_match_log1p",
        "team2_prior_match_log1p",
        "prior_match_count_min_log1p",
        "calendar_year_scaled",
        "month_sin",
        "month_cos",
        "elo_diff_scaled",
        "elo_diff_abs_scaled",
        "elo_diff_squared",
        "elo_diff_cubic",
        "elo_diff_neutral_interaction",
        "elo_diff_importance_interaction",
        "team1_days_rest_log1p",
        "team2_days_rest_log1p",
        "days_rest_diff_clipped",
    }
    columns = []
    for column in frame.columns:
        if (
            column in exact
            or column.startswith(prefixes)
            or "matchup_goals_" in column
            or "win_form_" in column
            or "draw_form_" in column
            or column.endswith("_momentum_diff")
        ):
            columns.append(column)
    return columns


def compact_feature_columns(frame: pd.DataFrame) -> list[str]:
    exact = {
        "elo_diff_pre",
        "elo_exp_team1",
        "neutral_int",
        "tournament_importance",
        "days_rest_diff",
        "days_rest_diff_clipped",
        "prior_match_count_min_log1p",
        "prior_match_count_diff",
        "calendar_year_scaled",
        "month_sin",
        "month_cos",
        "online_attack_diff_pre",
        "online_defence_diff_pre",
        "online_lambda_team1_pre",
        "online_lambda_team2_pre",
        "online_lambda_diff_pre",
        "online_lambda_total_pre",
        "online_log_lambda_ratio_pre",
        "online_lambda_diff_abs_pre",
        "online_log_lambda_ratio_abs_pre",
        "elo_diff_scaled",
        "elo_diff_abs_scaled",
        "elo_diff_squared",
        "elo_diff_cubic",
        "elo_diff_neutral_interaction",
        "elo_diff_importance_interaction",
    }
    return [
        column
        for column in frame.columns
        if column in exact
        or column.startswith("tournament_category_")
        or column.endswith("_diff")
        and (
            "_roll" in column
            or "_ewm" in column
            or "_momentum_" in column
            or column.startswith(("win_form_", "draw_form_", "matchup_goals_"))
        )
        or column.startswith("draw_form_")
        or column.startswith("matchup_goals_")
        or (column.endswith("_diff_abs") and not column.startswith("fifa_"))
        or column.endswith("_momentum_abs")
    ]
