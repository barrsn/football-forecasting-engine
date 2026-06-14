import numpy as np

from football_forecast.models.poisson import TwoPoissonScoreModel


def test_score_matrix_sums_to_one():
    m = TwoPoissonScoreModel.score_matrix(1.2, 0.8, max_goals=7)
    assert np.isclose(m.sum(), 1.0)
    assert m.shape == (8, 8)
