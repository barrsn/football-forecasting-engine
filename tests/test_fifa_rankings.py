import json

import pandas as pd

from football_forecast.data.fifa_rankings import (
    FifaRankingSchedule,
    extract_ranking_schedules,
    parse_ranking_snapshot,
)


def test_extract_schedules_uses_exact_timestamp_and_conservative_fallback():
    next_data = {
        "buildId": "build-1",
        "props": {
            "pageProps": {
                "pageData": {
                    "ranking": {
                        "dates": [
                            {
                                "dates": [
                                    {
                                        "id": "exact",
                                        "iso": "2024-04-04T12:30:00Z",
                                    }
                                ]
                            }
                        ],
                        "allAvailableDates": [
                            {
                                "id": "fallback",
                                "date": "2024-01-01",
                                "matchWindowEndDate": "2024-01-01",
                            },
                            {
                                "id": "exact",
                                "date": "2024-04-04",
                                "matchWindowEndDate": "2024-04-04",
                            },
                        ],
                    }
                }
            }
        },
    }
    html = f'<script id="__NEXT_DATA__">{json.dumps(next_data)}</script>'
    schedules, build_id = extract_ranking_schedules(html)

    assert build_id == "build-1"
    assert schedules[0].published_at == pd.Timestamp("2024-01-02T00:00:00Z")
    assert schedules[1].published_at == pd.Timestamp("2024-04-04T12:30:00Z")


def test_parse_snapshot_normalizes_team_and_computes_comparable_features():
    payload = {
        "Results": [
            {
                "TeamName": [{"Locale": "en-GB", "Description": "Korea Republic"}],
                "IdCountry": "KOR",
                "ConfederationName": "AFC",
                "Rank": 1,
                "PrevRank": 2,
                "TotalPoints": 1600.0,
                "PrevPoints": 1590.0,
                "RatedMatches": 20,
            },
            {
                "TeamName": [{"Locale": "en-GB", "Description": "Japan"}],
                "IdCountry": "JPN",
                "ConfederationName": "AFC",
                "Rank": 2,
                "PrevRank": 1,
                "TotalPoints": 1500.0,
                "PrevPoints": 1510.0,
                "RatedMatches": 20,
            },
        ]
    }
    schedule = FifaRankingSchedule(
        "id1",
        pd.Timestamp("2024-01-01T12:00:00Z"),
        "2024-01-01",
    )
    frame, unresolved = parse_ranking_snapshot(
        payload,
        schedule,
        {"korea republic": "South Korea"},
    )

    assert frame["team"].tolist() == ["South Korea", "Japan"]
    assert frame["fifa_rank_percentile"].tolist() == [1.0, 0.0]
    assert abs(frame["fifa_points_z"].mean()) < 1e-12
    assert unresolved == ("Japan",)
