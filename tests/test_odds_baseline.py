import numpy as np
import pandas as pd
import pytest

from football_forecast.evaluation.baselines import BookmakerOddsBaseline


def test_bookmaker_odds_are_de_vigged():
    frame = pd.DataFrame(
        {
            "odds_team2_win": [4.0],
            "odds_draw": [3.0],
            "odds_team1_win": [2.0],
        }
    )
    proba = BookmakerOddsBaseline().predict_proba(frame)
    assert np.isclose(proba.sum(), 1.0)
    assert proba[0, 2] > proba[0, 0]


def test_invalid_decimal_odds_are_rejected():
    frame = pd.DataFrame(
        {
            "odds_team2_win": [1.0],
            "odds_draw": [3.0],
            "odds_team1_win": [2.0],
        }
    )
    with pytest.raises(ValueError, match="greater than 1"):
        BookmakerOddsBaseline().predict_proba(frame)
