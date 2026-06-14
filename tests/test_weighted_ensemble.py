import numpy as np
import pandas as pd

from football_forecast.models.ensemble import WeightedOutcomeEnsemble


class _FixedModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, X):
        return np.repeat(self.probabilities[None, :], len(X), axis=0)


def test_weighted_ensemble_is_serializable_probability_model():
    ensemble = WeightedOutcomeEnsemble(
        [_FixedModel([0.2, 0.3, 0.5]), _FixedModel([0.4, 0.2, 0.4])],
        np.array([0.75, 0.25]),
    )
    probabilities = ensemble.predict_proba(pd.DataFrame({"x": [1, 2]}))

    assert np.allclose(probabilities[0], [0.25, 0.275, 0.475])
    assert np.allclose(probabilities.sum(axis=1), 1.0)
