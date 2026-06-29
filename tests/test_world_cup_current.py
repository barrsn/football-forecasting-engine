import pandas as pd

from football_forecast.world_cup.current import infer_fixture_groups


def test_infer_fixture_groups_from_connected_fixtures():
    fixtures = pd.DataFrame(
        {
            "team1": ["A1", "A3", "A1", "B1", "B3", "B1"],
            "team2": ["A2", "A4", "A3", "B2", "B4", "B3"],
        }
    )

    assert infer_fixture_groups(fixtures) == ["A", "A", "A", "B", "B", "B"]
