from pathlib import Path

import pandas as pd

from football_forecast.data.teams import load_team_aliases, normalize_match_teams


def test_team_aliases_and_unresolved_report(tmp_path: Path):
    mapping = tmp_path / "teams.yaml"
    mapping.write_text(
        "United States:\n  - USA\nNetherlands:\n  - Holland\n",
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "team1": ["USA", "Unknown Team"],
            "team2": ["Holland", "United States"],
        }
    )
    normalized, report = normalize_match_teams(frame, mapping)
    assert normalized["team1"].tolist() == ["United States", "Unknown Team"]
    assert normalized["team2"].tolist() == ["Netherlands", "United States"]
    assert report.unresolved_names == ("Unknown Team",)
    assert load_team_aliases(mapping)["usa"] == "United States"
