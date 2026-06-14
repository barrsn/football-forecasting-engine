from __future__ import annotations

from bisect import insort
from dataclasses import dataclass
from math import exp, log

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OnlinePoissonConfig:
    initial_goals: float = 1.25
    prior_matches: float = 100.0
    learning_rate: float = 0.035
    home_log_advantage: float = 0.16
    half_life_years: float = 4.0
    max_abs_residual: float = 3.0


def _decay_state(
    value: float,
    last_time: pd.Timestamp | None,
    current_time: pd.Timestamp,
    half_life_years: float,
) -> float:
    if last_time is None or half_life_years <= 0:
        return value
    elapsed_years = max(
        0.0,
        (current_time - last_time).total_seconds() / (365.25 * 86400.0),
    )
    return value * (0.5 ** (elapsed_years / half_life_years))


def add_online_poisson_strength(
    matches: pd.DataFrame,
    config: OnlinePoissonConfig | None = None,
) -> pd.DataFrame:
    """Add prior-only attack, defence and expected-goal strength features.

    Results become eligible only when ``available_at < kickoff_utc``. Matches
    sharing a kickoff are scored from the same state and applied as one batch.
    """
    cfg = config or OnlinePoissonConfig()
    required = {
        "team1",
        "team2",
        "team1_goals",
        "team2_goals",
        "neutral",
    }
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError(f"Missing columns for online strength: {sorted(missing)}")

    time_column = "kickoff_utc" if "kickoff_utc" in matches else "date"
    df = matches.copy()
    df[time_column] = pd.to_datetime(df[time_column], utc=True)
    if "available_at" in df:
        df["available_at"] = pd.to_datetime(df["available_at"], utc=True)
    else:
        df["available_at"] = df[time_column]
    sort_columns = [time_column]
    if "match_id" in df:
        sort_columns.append("match_id")
    df = df.sort_values(sort_columns).reset_index(drop=True)
    df["strength_row_id"] = np.arange(len(df))

    attack: dict[str, float] = {}
    defence: dict[str, float] = {}
    state_time: dict[str, pd.Timestamp] = {}
    pending: list[dict[str, object]] = []
    observed_goals = 2.0 * cfg.initial_goals * cfg.prior_matches
    observed_team_games = 2.0 * cfg.prior_matches
    feature_rows: list[dict[str, float | int]] = []

    def apply_available(cutoff: pd.Timestamp) -> None:
        nonlocal observed_goals, observed_team_games
        eligible = [event for event in pending if event["available_at"] < cutoff]
        if not eligible:
            return
        pending[:] = [event for event in pending if event["available_at"] >= cutoff]
        team_deltas: dict[str, list[float]] = {}
        for event in eligible:
            team1 = str(event["team1"])
            team2 = str(event["team2"])
            residual1 = float(event["residual1"])
            residual2 = float(event["residual2"])
            team_deltas.setdefault(team1, [0.0, 0.0])
            team_deltas.setdefault(team2, [0.0, 0.0])
            team_deltas[team1][0] += cfg.learning_rate * residual1
            team_deltas[team1][1] -= cfg.learning_rate * residual2
            team_deltas[team2][0] += cfg.learning_rate * residual2
            team_deltas[team2][1] -= cfg.learning_rate * residual1
            observed_goals += float(event["goals1"]) + float(event["goals2"])
            observed_team_games += 2.0
        for team, (attack_delta, defence_delta) in team_deltas.items():
            attack[team] = attack.get(team, 0.0) + attack_delta
            defence[team] = defence.get(team, 0.0) + defence_delta

    for kickoff, batch in df.groupby(time_column, sort=True):
        apply_available(kickoff)
        teams = set(batch["team1"]).union(batch["team2"])
        for team in teams:
            attack[team] = _decay_state(
                attack.get(team, 0.0),
                state_time.get(team),
                kickoff,
                cfg.half_life_years,
            )
            defence[team] = _decay_state(
                defence.get(team, 0.0),
                state_time.get(team),
                kickoff,
                cfg.half_life_years,
            )
            state_time[team] = kickoff

        global_goals = observed_goals / observed_team_games
        for row in batch.itertuples(index=False):
            attack1 = attack.get(row.team1, 0.0)
            defence1 = defence.get(row.team1, 0.0)
            attack2 = attack.get(row.team2, 0.0)
            defence2 = defence.get(row.team2, 0.0)
            home_adjustment = 0.0 if bool(row.neutral) else cfg.home_log_advantage
            lambda1 = float(
                np.clip(
                    global_goals * exp(attack1 - defence2 + home_adjustment),
                    0.15,
                    5.0,
                )
            )
            lambda2 = float(
                np.clip(global_goals * exp(attack2 - defence1), 0.15, 5.0)
            )
            residual1 = float(
                np.clip(
                    float(row.team1_goals) - lambda1,
                    -cfg.max_abs_residual,
                    cfg.max_abs_residual,
                )
            )
            residual2 = float(
                np.clip(
                    float(row.team2_goals) - lambda2,
                    -cfg.max_abs_residual,
                    cfg.max_abs_residual,
                )
            )
            feature_rows.append(
                {
                    "strength_row_id": int(row.strength_row_id),
                    "online_attack_team1_pre": attack1,
                    "online_defence_team1_pre": defence1,
                    "online_attack_team2_pre": attack2,
                    "online_defence_team2_pre": defence2,
                    "online_global_goals_pre": global_goals,
                    "online_lambda_team1_pre": lambda1,
                    "online_lambda_team2_pre": lambda2,
                    "online_lambda_diff_pre": lambda1 - lambda2,
                    "online_lambda_total_pre": lambda1 + lambda2,
                    "online_log_lambda_ratio_pre": log(lambda1 / lambda2),
                }
            )
            pending.append(
                {
                    "available_at": row.available_at,
                    "team1": row.team1,
                    "team2": row.team2,
                    "residual1": residual1,
                    "residual2": residual2,
                    "goals1": float(row.team1_goals),
                    "goals2": float(row.team2_goals),
                }
            )

    features = pd.DataFrame(feature_rows)
    return (
        df.merge(features, on="strength_row_id", how="left")
        .drop(columns="strength_row_id")
    )


def add_prior_match_counts(matches: pd.DataFrame) -> pd.DataFrame:
    """Count strictly available prior matches for both teams."""
    time_column = "kickoff_utc" if "kickoff_utc" in matches else "date"
    df = matches.copy()
    kickoff = pd.to_datetime(df[time_column], utc=True)
    available = (
        pd.to_datetime(df["available_at"], utc=True)
        if "available_at" in df
        else kickoff
    )
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "row_id": np.arange(len(df)),
                    "team": df["team1"].to_numpy(),
                    "side": "team1",
                    "kickoff": kickoff.to_numpy(),
                    "available": available.to_numpy(),
                }
            ),
            pd.DataFrame(
                {
                    "row_id": np.arange(len(df)),
                    "team": df["team2"].to_numpy(),
                    "side": "team2",
                    "kickoff": kickoff.to_numpy(),
                    "available": available.to_numpy(),
                }
            ),
        ],
        ignore_index=True,
    )
    long["kickoff"] = pd.to_datetime(long["kickoff"], utc=True)
    long["available"] = pd.to_datetime(long["available"], utc=True)
    rows: list[dict[str, object]] = []
    for _, team_rows in long.groupby("team", sort=False):
        available_history: list[int] = []
        for current_time, batch in team_rows.sort_values(
                ["kickoff", "row_id"]
        ).groupby("kickoff", sort=True):
            cutoff = int(current_time.value)
            prior_count = int(np.searchsorted(available_history, cutoff, side="left"))
            for row in batch.itertuples(index=False):
                rows.append(
                    {
                        "row_id": row.row_id,
                        "side": row.side,
                        "prior_match_count": prior_count,
                    }
                )
            for value in batch["available"].astype("int64"):
                insort(available_history, int(value))

    counts = pd.DataFrame(rows)
    team1 = (
        counts.loc[counts["side"] == "team1", ["row_id", "prior_match_count"]]
        .set_index("row_id")["prior_match_count"]
        .reindex(range(len(df)))
        .to_numpy()
    )
    team2 = (
        counts.loc[counts["side"] == "team2", ["row_id", "prior_match_count"]]
        .set_index("row_id")["prior_match_count"]
        .reindex(range(len(df)))
        .to_numpy()
    )
    df["team1_prior_match_count"] = team1
    df["team2_prior_match_count"] = team2
    df["prior_match_count_min"] = np.minimum(team1, team2)
    df["prior_match_count_diff"] = team1 - team2
    df["team1_prior_match_log1p"] = np.log1p(team1)
    df["team2_prior_match_log1p"] = np.log1p(team2)
    df["prior_match_count_min_log1p"] = np.log1p(df["prior_match_count_min"])
    return df
