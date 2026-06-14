from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import json
import re
from typing import Any

import pandas as pd

from football_forecast.data.teams import normalize_team_series


@dataclass(frozen=True)
class FifaRankingSchedule:
    schedule_id: str
    published_at: pd.Timestamp
    ranking_date: str


def extract_ranking_schedules(html: str) -> tuple[list[FifaRankingSchedule], str]:
    match = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError("FIFA page does not contain __NEXT_DATA__")
    data = json.loads(unescape(match.group(1)))
    ranking = data["props"]["pageProps"]["pageData"]["ranking"]
    exact_timestamps = {
        item["id"]: item.get("iso")
        for year in ranking.get("dates", [])
        for item in year.get("dates", [])
    }
    schedules: list[FifaRankingSchedule] = []
    seen: set[str] = set()
    for item in ranking["allAvailableDates"]:
        schedule_id = str(item["id"])
        if schedule_id in seen:
            continue
        seen.add(schedule_id)
        ranking_date = str(item.get("matchWindowEndDate") or item["date"])
        timestamp = exact_timestamps.get(schedule_id)
        if timestamp:
            published_at = pd.to_datetime(timestamp, utc=True)
        else:
            # Unknown publication time: make the rating eligible only the next day.
            published_at = pd.Timestamp(ranking_date, tz="UTC") + pd.Timedelta(days=1)
        schedules.append(
            FifaRankingSchedule(
                schedule_id=schedule_id,
                published_at=published_at,
                ranking_date=ranking_date,
            )
        )
    schedules.sort(key=lambda value: value.published_at)
    return schedules, str(data.get("buildId", "unknown"))


def _team_name(record: dict[str, Any]) -> str:
    names = record.get("TeamName") or []
    english = next(
        (
            item.get("Description")
            for item in names
            if str(item.get("Locale", "")).casefold().startswith("en")
        ),
        None,
    )
    if not english:
        raise ValueError(f"FIFA ranking row has no English team name: {record!r}")
    return str(english)


def parse_ranking_snapshot(
    payload: dict[str, Any],
    schedule: FifaRankingSchedule,
    aliases: dict[str, str],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    records = payload.get("Results")
    if not isinstance(records, list) or not records:
        raise ValueError(f"FIFA snapshot {schedule.schedule_id} contains no Results")
    rows = []
    for record in records:
        rows.append(
            {
                "schedule_id": schedule.schedule_id,
                "rating_date": schedule.published_at,
                "ranking_date": schedule.ranking_date,
                "team": _team_name(record),
                "country_code": record.get("IdCountry"),
                "confederation": record.get("ConfederationName"),
                "fifa_rank": record.get("Rank"),
                "fifa_points": record.get("TotalPoints"),
                "fifa_previous_rank": record.get("PrevRank"),
                "fifa_previous_points": record.get("PrevPoints"),
                "fifa_rated_matches": record.get("RatedMatches"),
            }
        )
    frame = pd.DataFrame(rows)
    frame["team"], report = normalize_team_series(frame["team"], aliases)
    numeric = [
        "fifa_rank",
        "fifa_points",
        "fifa_previous_rank",
        "fifa_previous_points",
        "fifa_rated_matches",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    team_count = frame["fifa_rank"].notna().sum()
    frame["fifa_rank_percentile"] = 1.0 - (
        (frame["fifa_rank"] - 1.0) / max(team_count - 1, 1)
    )
    points_mean = frame["fifa_points"].mean()
    points_std = frame["fifa_points"].std(ddof=0)
    frame["fifa_points_z"] = (frame["fifa_points"] - points_mean) / max(
        float(points_std), 1e-9
    )
    return frame, report.unresolved_names
