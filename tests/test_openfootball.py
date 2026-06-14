import json
from pathlib import Path

from football_forecast.data.ingest_openfootball import parse_openfootball_worldcup


def test_openfootball_matches_schema_and_placeholders(tmp_path: Path):
    path = tmp_path / "worldcup.json"
    path.write_text(
        json.dumps(
            {
                "name": "World Cup 2026",
                "matches": [
                    {
                        "num": 1,
                        "round": "Matchday 1",
                        "date": "2026-06-11",
                        "time": "13:00 UTC-6",
                        "team1": "Mexico",
                        "team2": "South Africa",
                        "group": "Group A",
                    },
                    {
                        "num": 104,
                        "round": "Final",
                        "date": "2026-07-19",
                        "time": "15:00 UTC-4",
                        "team1": "W101",
                        "team2": "W102",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    frame = parse_openfootball_worldcup(path)
    assert frame.loc[0, "group"] == "A"
    assert frame.loc[0, "kickoff_utc"].isoformat() == "2026-06-11T19:00:00+00:00"
    assert frame.loc[1, "unresolved_team1"]
    assert frame.loc[1, "unresolved_team2"]
