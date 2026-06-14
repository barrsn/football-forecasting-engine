from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from football_forecast.evaluation.metrics import log_loss_multiclass, normalize_probabilities
from football_forecast.evaluation.probabilities import probability_frame


def blend_probabilities(probability_sets: list[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    if not probability_sets:
        raise ValueError("probability_sets must not be empty")
    arr = np.stack(probability_sets, axis=0)
    if weights is None:
        weights = np.ones(arr.shape[0]) / arr.shape[0]
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    blended = np.tensordot(weights, arr, axes=(0, 0))
    return normalize_probabilities(blended)


def optimize_ensemble_weights(y_true: np.ndarray, probability_sets: list[np.ndarray]) -> np.ndarray:
    n = len(probability_sets)
    if n == 0:
        raise ValueError("Need at least one probability set")

    def objective(w: np.ndarray) -> float:
        return log_loss_multiclass(y_true, blend_probabilities(probability_sets, w))

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n
    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")
    if not result.success:
        return x0
    return result.x / result.x.sum()


def blend_probability_frames(
    probability_sets: list[np.ndarray | pd.DataFrame],
    weights: np.ndarray | None = None,
    *,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    arrays = [np.asarray(probability_frame(item), dtype=float) for item in probability_sets]
    return probability_frame(blend_probabilities(arrays, weights), index=index)


class WeightedOutcomeEnsemble:
    """Serializable weighted ensemble of fitted probability models."""

    def __init__(self, models: list[object], weights: np.ndarray) -> None:
        if len(models) != len(weights):
            raise ValueError("models and weights must have the same length")
        if not models:
            raise ValueError("models must not be empty")
        self.models = models
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = [model.predict_proba(X) for model in self.models]
        return blend_probabilities(probabilities, self.weights)

    def predict_proba_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return probability_frame(self.predict_proba(X), index=X.index)
