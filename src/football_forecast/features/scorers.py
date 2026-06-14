from __future__ import annotations

from collections import defaultdict, deque
from math import log

import numpy as np
import pandas as pd

from football_forecast.data.schema import parse_boolean_series


def _decay_weight(age_days: float, half_life_days: float) -> float:
    return float(np.exp(-log(2.0) * max(age_days, 0.0) / half_life_days))


def add_prior_scorer_features(
    matches: pd.DataFrame,
    goalscorers: pd.DataFrame,
    *,
    half_life_days: float = 730.5,
    active_window_days: float = 730.5,
) -> pd.DataFrame:
    """Add prior-only player scoring-pool features.

    The source has scorer events, not appearances. These columns measure scoring
    threat and concentration; they are deliberately not labelled player ratings.
    """
    required = {
        "date",
        "home_team",
        "away_team",
        "team",
        "scorer",
        "own_goal",
        "penalty",
    }
    missing = sorted(required.difference(goalscorers.columns))
    if missing:
        raise ValueError(f"Missing goalscorer columns: {missing}")

    base = matches.copy()
    base["_original_order"] = np.arange(len(base))
    base["kickoff_utc"] = pd.to_datetime(base["kickoff_utc"], utc=True)
    sort_columns = ["kickoff_utc"]
    if "match_id" in base:
        sort_columns.append("match_id")
    base = base.sort_values(sort_columns).reset_index(drop=True)
    work_columns = [
        "kickoff_utc",
        "team1",
        "team2",
        "team1_goals",
        "team2_goals",
    ]
    for optional in ("match_id", "available_at"):
        if optional in base:
            work_columns.append(optional)
    work = base[work_columns].copy()

    event_data = goalscorers.copy()
    event_data["date"] = pd.to_datetime(event_data["date"], errors="raise", utc=True)
    event_data["scorer"] = event_data["scorer"].fillna("").astype(str).str.strip()
    event_data["own_goal"] = parse_boolean_series(
        event_data["own_goal"], "own_goal"
    )
    event_data["penalty"] = parse_boolean_series(event_data["penalty"], "penalty")

    event_groups: dict[
        tuple[pd.Timestamp, str, str], list[tuple[str, str, bool, bool]]
    ] = defaultdict(list)
    for event in event_data.itertuples(index=False):
        event_groups[
            (
                pd.Timestamp(event.date).normalize(),
                str(event.home_team),
                str(event.away_team),
            )
        ].append(
            (
                str(event.team),
                str(event.scorer),
                bool(event.own_goal),
                bool(event.penalty),
            )
        )

    goal_history: dict[str, deque[dict[str, object]]] = defaultdict(deque)
    match_history: dict[str, deque[dict[str, object]]] = defaultdict(deque)
    feature_rows: list[dict[str, float]] = []

    for kickoff, batch in work.groupby("kickoff_utc", sort=True):
        for row in batch.itertuples(index=False):
            side_features: dict[str, dict[str, float]] = {}
            for side, team in (("team1", row.team1), ("team2", row.team2)):
                team_goals = goal_history[str(team)]
                team_matches = match_history[str(team)]
                while team_goals:
                    age_days = (
                        kickoff - pd.Timestamp(team_goals[0]["kickoff_utc"])
                    ).total_seconds() / 86400
                    if age_days <= active_window_days:
                        break
                    team_goals.popleft()
                while team_matches:
                    age_days = (
                        kickoff - pd.Timestamp(team_matches[0]["kickoff_utc"])
                    ).total_seconds() / 86400
                    if age_days <= active_window_days:
                        break
                    team_matches.popleft()

                player_scores: dict[str, float] = defaultdict(float)
                penalty_weight = 0.0
                total_weight = 0.0
                for item in team_goals:
                    if pd.Timestamp(item["available_at"]) >= kickoff:
                        continue
                    age_days = (
                        kickoff - pd.Timestamp(item["kickoff_utc"])
                    ).total_seconds() / 86400
                    goal_weight = _decay_weight(age_days, half_life_days)
                    player_scores[str(item["scorer"])] += goal_weight
                    total_weight += goal_weight
                    penalty_weight += goal_weight * int(item["penalty"])

                eligible_matches = [
                    item
                    for item in team_matches
                    if pd.Timestamp(item["available_at"]) < kickoff
                ]
                complete_matches = sum(
                    int(item["complete"]) for item in eligible_matches
                )
                values = np.sort(np.asarray(list(player_scores.values()), dtype=float))[::-1]
                shares = values / values.sum() if values.size and values.sum() > 0 else values
                side_features[side] = {
                    "scorer_active_players": float(len(values)),
                    "scorer_goals_decay": float(values.sum()) if values.size else 0.0,
                    "scorer_top1_goals_decay": float(values[0]) if values.size else 0.0,
                    "scorer_top3_share": float(shares[:3].sum()) if values.size else 0.0,
                    "scorer_hhi": float(np.square(shares).sum()) if values.size else 0.0,
                    "scorer_penalty_share": (
                        float(penalty_weight / total_weight) if total_weight else 0.0
                    ),
                    "scorer_complete_matches": float(complete_matches),
                    "scorer_history_matches": float(len(eligible_matches)),
                    "scorer_coverage": (
                        float(complete_matches / len(eligible_matches))
                        if eligible_matches
                        else 0.0
                    ),
                }
            features: dict[str, float] = {}
            for side in ("team1", "team2"):
                for name, value in side_features[side].items():
                    features[f"{side}_{name}"] = value
            for name in side_features["team1"]:
                features[f"{name}_diff"] = (
                    side_features["team1"][name] - side_features["team2"][name]
                )
            feature_rows.append(features)

        for row in batch.itertuples(index=False):
            key = (kickoff.normalize(), str(row.team1), str(row.team2))
            events = event_groups.get(key, [])
            expected_goals = int(row.team1_goals + row.team2_goals)
            complete = len(events) == expected_goals
            available_at = pd.to_datetime(
                getattr(row, "available_at", kickoff), utc=True
            )
            for team in (str(row.team1), str(row.team2)):
                scorers: list[tuple[str, bool]] = []
                if complete and events:
                    scorers = [
                        (scorer, is_penalty)
                        for event_team, scorer, own_goal, is_penalty in events
                        if event_team == team and not own_goal and scorer
                    ]
                match_history[team].append(
                    {
                        "kickoff_utc": kickoff,
                        "available_at": available_at,
                        "complete": complete,
                    }
                )
                for scorer, is_penalty in scorers:
                    goal_history[team].append(
                        {
                            "kickoff_utc": kickoff,
                            "available_at": available_at,
                            "scorer": scorer,
                            "penalty": is_penalty,
                        }
                    )

    features = pd.DataFrame(feature_rows, index=base.index)
    out = pd.concat([base, features], axis=1)
    return (
        out.sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )


def scorer_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if "scorer_" in column and pd.api.types.is_numeric_dtype(frame[column])
    ]
