import numpy as np
import pandas as pd

from football_forecast.models.outcome import StructuredOutcomeModel


def test_structured_outcome_returns_ordered_probabilities():
    X = pd.DataFrame(
        {
            "strength_diff": [-2.0, -1.0, 0.0, 1.0, 2.0] * 12,
            "strength_abs": [2.0, 1.0, 0.0, 1.0, 2.0] * 12,
        }
    )
    y = pd.Series([0, 0, 1, 2, 2] * 12)
    model = StructuredOutcomeModel(draw_c=0.1, decisive_c=0.1).fit(X, y)
    probabilities = model.predict_proba(X.iloc[:5])

    assert probabilities.shape == (5, 3)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0, 0] > probabilities[0, 2]
    assert probabilities[-1, 2] > probabilities[-1, 0]
