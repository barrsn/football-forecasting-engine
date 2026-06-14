from __future__ import annotations

from math import log

import numpy as np
import pandas as pd

ROLLING_STATS = (
    "goals_for",
    "goals_against",
    "goal_diff",
    "win",
    "draw",
    "opponent_adjusted_goal_diff",
    "clean_sheet",
    "failed_to_score",
)


def _long_team_frame(matches: pd.DataFrame) -> pd.DataFrame:
    time_column = "kickoff_utc" if "kickoff_utc" in matches else "date"
    available = (
        pd.to_datetime(matches["available_at"], utc=True)
        if "available_at" in matches
        else pd.to_datetime(matches[time_column], utc=True)
    )
    team1_opponent_elo = matches.get("elo_team2_pre", pd.Series(1500.0, index=matches.index))
    team2_opponent_elo = matches.get("elo_team1_pre", pd.Series(1500.0, index=matches.index))
    team1 = pd.DataFrame(
        {
            "_match_row_id": matches["_match_row_id"],
            "kickoff_utc": pd.to_datetime(matches[time_column], utc=True),
            "available_at": available,
            "team": matches["team1"],
            "side": "team1",
            "goals_for": matches["team1_goals"],
            "goals_against": matches["team2_goals"],
            "opponent_elo_pre": team1_opponent_elo,
        }
    )
    team2 = pd.DataFrame(
        {
            "_match_row_id": matches["_match_row_id"],
            "kickoff_utc": pd.to_datetime(matches[time_column], utc=True),
            "available_at": available,
            "team": matches["team2"],
            "side": "team2",
            "goals_for": matches["team2_goals"],
            "goals_against": matches["team1_goals"],
            "opponent_elo_pre": team2_opponent_elo,
        }
    )
    long = pd.concat([team1, team2], ignore_index=True)
    long["goal_diff"] = long["goals_for"] - long["goals_against"]
    long["win"] = (long["goal_diff"] > 0).astype(float)
    long["draw"] = (long["goal_diff"] == 0).astype(float)
    long["clean_sheet"] = (long["goals_against"] == 0).astype(float)
    long["failed_to_score"] = (long["goals_for"] == 0).astype(float)
    long["opponent_adjusted_goal_diff"] = (
        long["goal_diff"] + (long["opponent_elo_pre"] - 1500.0) / 400.0
    )
    return long.sort_values(["team", "kickoff_utc", "_match_row_id"]).reset_index(drop=True)


def _weighted_mean(
    history: list[dict[str, object]],
    column: str,
    current_time: pd.Timestamp,
    half_life_days: float,
) -> float:
    if not history:
        return float("nan")
    ages = np.asarray(
        [(current_time - item["kickoff_utc"]).total_seconds() / 86400 for item in history],
        dtype=float,
    )
    weights = np.exp(-log(2.0) * ages / half_life_days)
    values = np.asarray([item[column] for item in history], dtype=float)
    return float(np.average(values, weights=weights))


def add_rolling_team_features(
    matches: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
    *,
    half_life_days: float = 365.0,
) -> pd.DataFrame:
    """Add prior-only rolling features, with equal kickoffs treated as one batch."""
    time_column = "kickoff_utc" if "kickoff_utc" in matches else "date"
    df = matches.copy()
    df[time_column] = pd.to_datetime(df[time_column], utc=True)
    df = df.sort_values([time_column, "match_id"] if "match_id" in df else [time_column]).reset_index(
        drop=True
    )
    df["_match_row_id"] = range(len(df))
    long = _long_team_frame(df)
    feature_rows: list[dict[str, object]] = []

    for team, team_rows in long.groupby("team", sort=False):
        history: list[dict[str, object]] = []
        for kickoff, batch in team_rows.groupby("kickoff_utc", sort=True):
            eligible = [item for item in history if item["available_at"] < kickoff]
            history_available_at = (
                max((item["available_at"] for item in eligible), default=pd.NaT)
            )
            days_rest = (
                float((kickoff - eligible[-1]["kickoff_utc"]).total_seconds() / 86400)
                if eligible
                else float("nan")
            )
            shared: dict[str, object] = {
                "history_available_at": history_available_at,
                "days_rest": days_rest,
            }
            win_streak = 0
            loss_streak = 0
            unbeaten_streak = 0
            for item in reversed(eligible):
                if item["win"] == 1.0:
                    win_streak += 1
                else:
                    break
            for item in reversed(eligible):
                if item["goal_diff"] < 0:
                    loss_streak += 1
                else:
                    break
            for item in reversed(eligible):
                if item["goal_diff"] >= 0:
                    unbeaten_streak += 1
                else:
                    break

            shared["win_streak"] = float(win_streak)
            shared["loss_streak"] = float(loss_streak)
            shared["unbeaten_streak"] = float(unbeaten_streak)
            
            for window in windows:
                recent = eligible[-window:]
                for column in ROLLING_STATS:
                    shared[f"{column}_roll{window}"] = (
                        float(np.mean([item[column] for item in recent]))
                        if recent
                        else float("nan")
                    )
            for column in ROLLING_STATS:
                shared[f"{column}_ewm"] = _weighted_mean(
                    eligible, column, kickoff, half_life_days
                )

            for row in batch.to_dict(orient="records"):
                feature_rows.append(
                    {
                        "_match_row_id": row["_match_row_id"],
                        "side": row["side"],
                        **shared,
                    }
                )

            for row in batch.to_dict(orient="records"):
                history.append(
                    {
                        "kickoff_utc": row["kickoff_utc"],
                        "available_at": row["available_at"],
                        **{column: row[column] for column in ROLLING_STATS},
                    }
                )
            history.sort(key=lambda item: (item["kickoff_utc"], item["available_at"]))

    features = pd.DataFrame(feature_rows)
    value_columns = [
        column for column in features.columns if column not in {"_match_row_id", "side"}
    ]
    team1_features = features.loc[
        features["side"] == "team1", ["_match_row_id", *value_columns]
    ].rename(columns={column: f"team1_{column}" for column in value_columns})
    team2_features = features.loc[
        features["side"] == "team2", ["_match_row_id", *value_columns]
    ].rename(columns={column: f"team2_{column}" for column in value_columns})

    out = df.merge(team1_features, on="_match_row_id", how="left")
    out = out.merge(team2_features, on="_match_row_id", how="left")
    for window in windows:
        for column in (
            "goal_diff",
            "goals_for",
            "goals_against",
            "opponent_adjusted_goal_diff",
            "clean_sheet",
            "failed_to_score",
        ):
            out[f"{column}_roll{window}_diff"] = (
                out[f"team1_{column}_roll{window}"] - out[f"team2_{column}_roll{window}"]
            )
    for column in (
        "goal_diff",
        "goals_for",
        "goals_against",
        "opponent_adjusted_goal_diff",
        "clean_sheet",
        "failed_to_score",
    ):
        out[f"{column}_ewm_diff"] = (
            out[f"team1_{column}_ewm"] - out[f"team2_{column}_ewm"]
        )
    for column in ("win_streak", "loss_streak", "unbeaten_streak"):
        out[f"{column}_diff"] = out[f"team1_{column}"] - out[f"team2_{column}"]
    out["days_rest_diff"] = out["team1_days_rest"] - out["team2_days_rest"]
    return out.drop(columns=["_match_row_id"])
