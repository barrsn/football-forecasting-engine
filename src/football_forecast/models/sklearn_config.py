from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

LOGISTIC_MAX_ITER = 5000
LOGISTIC_SOLVER = "sag"


def predict_logistic_pipeline_proba(
    pipeline: Pipeline,
    x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict logistic probabilities without sklearn's native decision path."""
    model = pipeline[-1]
    values = pipeline[:-1].transform(x)
    values = np.asarray(values, dtype=float)
    coef = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float)
    classes = np.asarray(model.classes_)

    if coef.shape[0] == 1:
        scores = np.sum(values * coef[0], axis=1) + intercept[0]
        positive = 1.0 / (1.0 + np.exp(-scores))
        probabilities = np.column_stack([1.0 - positive, positive])
    else:
        scores = np.sum(values[:, None, :] * coef[None, :, :], axis=2)
        scores = scores + intercept[None, :]
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return probabilities, classes
