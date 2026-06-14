from __future__ import annotations

from dataclasses import dataclass, field
from math import log

import pandas as pd


DEFAULT_IMPORTANCE_WEIGHTS = {
    "friendly": 0.5,
    "qualification": 1.0,
    "qualifier": 1.0,
    "nations league": 1.0,
    "continental": 1.5,
    "copa": 1.5,
    "euro": 1.5,
    "world cup": 2.0,
}


@dataclass(frozen=True)
class EloConfig:
    initial_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 60.0
    host_advantage: float = 35.0
    goal_diff_weight: bool = True
    regression_per_year: float = 0.10
    importance_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_IMPORTANCE_WEIGHTS)
    )


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / 400.0))


def actual_score(goals_a: int, goals_b: int) -> float:
    if goals_a > goals_b:
        return 1.0
    if goals_a < goals_b:
        return 0.0
    return 0.5


def goal_diff_multiplier(goal_diff: int) -> float:
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    return 1.0 + log(gd)


def tournament_importance(tournament: str, weights: dict[str, float] | None = None) -> float:
    mapping = weights or DEFAULT_IMPORTANCE_WEIGHTS
    normalized = str(tournament).casefold()
    matches = [weight for token, weight in mapping.items() if token in normalized]
    return max(matches, default=1.0)


def _regress_rating(
    rating: float,
    last_played: pd.Timestamp | None,
    current_time: pd.Timestamp,
    config: EloConfig,
) -> float:
    if last_played is None or config.regression_per_year <= 0:
        return rating
    elapsed_years = max(0.0, (current_time - last_played).total_seconds() / (365.25 * 86400))
    retention = (1.0 - config.regression_per_year) ** elapsed_years
    return config.initial_rating + (rating - config.initial_rating) * retention


def add_elo_features(
    matches: pd.DataFrame, 
    config: EloConfig | None = None,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """Compute pre-match Elo features, batching equal kickoffs to prevent leakage."""
    cfg = config or EloConfig()
    time_column = "kickoff_utc" if "kickoff_utc" in matches else "date"
    df = matches.copy()
    df[time_column] = pd.to_datetime(df[time_column], utc=True)
    df = df.sort_values([time_column, "match_id"] if "match_id" in df else [time_column]).reset_index(
        drop=True
    )
    ratings: dict[str, float] = {}
    pre_match_history: dict[str, list[float]] = {}
    last_played: dict[str, pd.Timestamp] = {}
    last_available: dict[str, pd.Timestamp] = {}
    feature_rows: list[dict[str, object]] = []

    for kickoff, batch in df.groupby(time_column, sort=True):
        teams = set(batch["team1"]).union(batch["team2"])
        batch_ratings = {
            team: _regress_rating(
                ratings.get(team, cfg.initial_rating),
                last_played.get(team),
                kickoff,
                cfg,
            )
            for team in teams
        }
        deltas = {team: 0.0 for team in teams}

        for row in batch.itertuples(index=False):
            r1 = batch_ratings[row.team1]
            r2 = batch_ratings[row.team2]
            neutral = bool(row.neutral)
            host_team = getattr(row, "host_team", None)
            adjustment1 = 0.0 if neutral else cfg.home_advantage
            if host_team == row.team1:
                adjustment1 += cfg.host_advantage
            adjustment2 = cfg.host_advantage if host_team == row.team2 else 0.0

            exp1 = expected_score(r1 + adjustment1, r2 + adjustment2)
            act1 = actual_score(row.team1_goals, row.team2_goals)
            multiplier = (
                goal_diff_multiplier(row.team1_goals - row.team2_goals)
                if cfg.goal_diff_weight
                else 1.0
            )
            importance = tournament_importance(row.tournament, cfg.importance_weights)
            delta = cfg.k_factor * importance * multiplier * (act1 - exp1)
            deltas[row.team1] += delta
            deltas[row.team2] -= delta

            feature_row = {
                "_row_index": row.Index if hasattr(row, "Index") else len(feature_rows),
                "elo_team1_pre": r1,
                "elo_team2_pre": r2,
                "elo_diff_pre": r1 - r2,
                "elo_exp_team1": exp1,
                "elo_importance": importance,
                "elo_team1_available_at": last_available.get(row.team1, pd.NaT),
                "elo_team2_available_at": last_available.get(row.team2, pd.NaT),
            }
            for window in windows:
                hist1 = pre_match_history.get(row.team1, [])
                momentum1 = r1 - hist1[-window] if len(hist1) >= window else r1 - cfg.initial_rating
                
                hist2 = pre_match_history.get(row.team2, [])
                momentum2 = r2 - hist2[-window] if len(hist2) >= window else r2 - cfg.initial_rating
                
                feature_row[f"elo_momentum_team1_roll{window}"] = momentum1
                feature_row[f"elo_momentum_team2_roll{window}"] = momentum2
                feature_row[f"elo_momentum_roll{window}_diff"] = momentum1 - momentum2
            
            feature_rows.append(feature_row)

        for team in teams:
            ratings[team] = batch_ratings[team] + deltas[team]
            last_played[team] = kickoff

        for row in batch.itertuples(index=False):
            available_at = getattr(row, "available_at", kickoff)
            available_at = pd.to_datetime(available_at, utc=True)
            last_available[row.team1] = max(
                available_at, last_available.get(row.team1, available_at)
            )
            last_available[row.team2] = max(
                available_at, last_available.get(row.team2, available_at)
            )
            if row.team1 not in pre_match_history:
                pre_match_history[row.team1] = []
            pre_match_history[row.team1].append(batch_ratings[row.team1])
            if row.team2 not in pre_match_history:
                pre_match_history[row.team2] = []
            pre_match_history[row.team2].append(batch_ratings[row.team2])

    features = pd.DataFrame(feature_rows).drop(columns="_row_index")
    return pd.concat([df, features], axis=1)
