from __future__ import annotations

import pandas as pd


def join_team_ratings_asof(
    matches: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    rating_column: str = "rating",
) -> pd.DataFrame:
    """Join the most recent strictly pre-kickoff team rating for both teams."""
    required_matches = {"kickoff_utc", "team1", "team2"}
    required_ratings = {"rating_date", "team", rating_column}
    if missing := sorted(required_matches.difference(matches.columns)):
        raise ValueError(f"Missing match columns: {missing}")
    if missing := sorted(required_ratings.difference(ratings.columns)):
        raise ValueError(f"Missing rating columns: {missing}")

    base = matches.copy()
    base["_original_order"] = range(len(base))
    rating_data = ratings.copy()
    base["kickoff_utc"] = pd.to_datetime(base["kickoff_utc"], utc=True)
    rating_data["rating_date"] = pd.to_datetime(rating_data["rating_date"], utc=True)
    rating_data = rating_data.sort_values(["rating_date", "team"])

    for side in ("team1", "team2"):
        left = base[["_original_order", "kickoff_utc", side]].rename(columns={side: "team"})
        left = left.sort_values(["kickoff_utc", "team"])
        joined = pd.merge_asof(
            left,
            rating_data[["team", "rating_date", rating_column]],
            left_on="kickoff_utc",
            right_on="rating_date",
            by="team",
            direction="backward",
            allow_exact_matches=False,
        )
        joined = joined.set_index("_original_order")
        base[f"{side}_{rating_column}"] = joined[rating_column]
        base[f"{side}_{rating_column}_available_at"] = joined["rating_date"]

    return base.sort_values("_original_order").drop(columns="_original_order").reset_index(drop=True)


def join_team_rating_features_asof(
    matches: pd.DataFrame,
    ratings: pd.DataFrame,
    rating_columns: list[str],
    *,
    prefix: str,
) -> pd.DataFrame:
    """Join several rating fields from one strictly prior team snapshot."""
    required_matches = {"kickoff_utc", "team1", "team2"}
    required_ratings = {"rating_date", "team", *rating_columns}
    if missing := sorted(required_matches.difference(matches.columns)):
        raise ValueError(f"Missing match columns: {missing}")
    if missing := sorted(required_ratings.difference(ratings.columns)):
        raise ValueError(f"Missing rating columns: {missing}")

    base = matches.copy()
    base["_original_order"] = range(len(base))
    base["kickoff_utc"] = pd.to_datetime(base["kickoff_utc"], utc=True)
    rating_data = ratings.copy()
    rating_data["rating_date"] = pd.to_datetime(rating_data["rating_date"], utc=True)
    rating_data = rating_data.sort_values(["rating_date", "team"])

    for side in ("team1", "team2"):
        left = base[["_original_order", "kickoff_utc", side]].rename(
            columns={side: "team"}
        )
        joined = pd.merge_asof(
            left.sort_values(["kickoff_utc", "team"]),
            rating_data[["team", "rating_date", *rating_columns]],
            left_on="kickoff_utc",
            right_on="rating_date",
            by="team",
            direction="backward",
            allow_exact_matches=False,
        ).set_index("_original_order")
        for column in rating_columns:
            suffix = (
                column[len(prefix) + 1 :]
                if prefix and column.startswith(f"{prefix}_")
                else column
            )
            base[f"{side}_{prefix}_{suffix}"] = joined[column]
        base[f"{side}_{prefix}_available_at"] = joined["rating_date"]

    for column in rating_columns:
        suffix = (
            column[len(prefix) + 1 :]
            if prefix and column.startswith(f"{prefix}_")
            else column
        )
        left_column = f"team1_{prefix}_{suffix}"
        right_column = f"team2_{prefix}_{suffix}"
        if pd.api.types.is_numeric_dtype(base[left_column]):
            base[f"{prefix}_{suffix}_diff"] = (
                base[left_column] - base[right_column]
            )
    return (
        base.sort_values("_original_order")
        .drop(columns="_original_order")
        .reset_index(drop=True)
    )


def assert_prior_only_timestamps(
    frame: pd.DataFrame,
    timestamp_columns: list[str],
    *,
    kickoff_column: str = "kickoff_utc",
) -> None:
    kickoff = pd.to_datetime(frame[kickoff_column], utc=True)
    for column in timestamp_columns:
        timestamps = pd.to_datetime(frame[column], utc=True)
        invalid = timestamps.notna() & (timestamps >= kickoff)
        if invalid.any():
            raise ValueError(f"Feature timestamp {column} is not strictly before kickoff")
