from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _pair_key(team1: str, team2: str) -> tuple[str, str]:
    return tuple(sorted((str(team1), str(team2))))


def add_h2h_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add strictly prior head-to-head features.

    Matches at the same kickoff are evaluated from the same history snapshot.
    Results also remain unavailable until their explicit ``available_at`` time.
    """
    base = matches.copy()
    base["_original_order"] = np.arange(len(base))
    existing_features = [column for column in base if column.startswith("h2h_")]
    if existing_features:
        base = base.drop(columns=existing_features)
    time_column = "kickoff_utc" if "kickoff_utc" in base else "date"
    base["kickoff_utc"] = pd.to_datetime(base[time_column], utc=True)
    available = base.get("available_at", base["kickoff_utc"])
    base["available_at"] = pd.to_datetime(available, utc=True)
    sort_columns = ["kickoff_utc"]
    if "match_id" in base:
        sort_columns.append("match_id")
    base = base.sort_values(sort_columns).reset_index(drop=True)
    work_columns = [
        "kickoff_utc",
        "available_at",
        "team1",
        "team2",
        "team1_goals",
        "team2_goals",
    ]
    if "match_id" in base:
        work_columns.append("match_id")
    work = base[work_columns]

    history: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    feature_rows: list[dict[str, float]] = []

    for kickoff, batch in work.groupby("kickoff_utc", sort=True):
        for row in batch.itertuples(index=False):
            first, second = _pair_key(row.team1, row.team2)
            prior = [
                item
                for item in history[(first, second)]
                if pd.Timestamp(item["available_at"]) < kickoff
            ]
            total = len(prior)
            if total:
                first_goals = np.asarray(
                    [float(item["first_goals"]) for item in prior], dtype=float
                )
                second_goals = np.asarray(
                    [float(item["second_goals"]) for item in prior], dtype=float
                )
                first_win_pct = float(np.mean(first_goals > second_goals))
                second_win_pct = float(np.mean(second_goals > first_goals))
                first_goals_mean = float(first_goals.mean())
                second_goals_mean = float(second_goals.mean())
                last_first_outcome = float(prior[-1]["first_outcome"])
            else:
                first_win_pct = 0.0
                second_win_pct = 0.0
                first_goals_mean = 0.0
                second_goals_mean = 0.0
                last_first_outcome = 0.0

            team1_is_first = str(row.team1) == first
            feature_rows.append(
                {
                    "h2h_total_matches": float(total),
                    "h2h_team1_win_pct": (
                        first_win_pct if team1_is_first else second_win_pct
                    ),
                    "h2h_team2_win_pct": (
                        second_win_pct if team1_is_first else first_win_pct
                    ),
                    "h2h_team1_goals_mean": (
                        first_goals_mean if team1_is_first else second_goals_mean
                    ),
                    "h2h_team2_goals_mean": (
                        second_goals_mean if team1_is_first else first_goals_mean
                    ),
                    "h2h_last_outcome": (
                        last_first_outcome if team1_is_first else -last_first_outcome
                    ),
                }
            )

        for row in batch.itertuples(index=False):
            first, second = _pair_key(row.team1, row.team2)
            team1_is_first = str(row.team1) == first
            first_goals = row.team1_goals if team1_is_first else row.team2_goals
            second_goals = row.team2_goals if team1_is_first else row.team1_goals
            history[(first, second)].append(
                {
                    "available_at": row.available_at,
                    "first_goals": first_goals,
                    "second_goals": second_goals,
                    "first_outcome": float(
                        1 if first_goals > second_goals else -1 if first_goals < second_goals else 0
                    ),
                }
            )

    features = pd.DataFrame(feature_rows, index=base.index)
    out = pd.concat([base, features], axis=1)
    return (
        out.sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )
