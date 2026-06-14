import numpy as np

from football_forecast.evaluation.metrics import brier_score_multiclass, ranked_probability_score


def test_brier_score_perfect_is_zero():
    y = np.array([0, 1, 2])
    p = np.eye(3)
    assert brier_score_multiclass(y, p) == 0.0


def test_rps_perfect_is_zero():
    y = np.array([0, 1, 2])
    p = np.eye(3)
    assert ranked_probability_score(y, p) == 0.0
