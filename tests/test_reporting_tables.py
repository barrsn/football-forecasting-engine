import numpy as np
import pandas as pd

from football_forecast.reporting.tables import prediction_table


def test_prediction_table_can_include_high_confidence_policy():
    matches = pd.DataFrame(
        {
            "match_id": ["a", "b"],
            "team1": ["A", "C"],
            "team2": ["B", "D"],
        }
    )
    out = prediction_table(
        matches,
        np.array([[0.1, 0.2, 0.7], [0.34, 0.33, 0.33]]),
        confidence_threshold=0.5,
    )

    assert out.loc[0, "predicted_label"] == "team1_win"
    assert out.loc[1, "predicted_label"] == "abstain"
