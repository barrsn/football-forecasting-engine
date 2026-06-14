from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss as sklearn_log_loss

from football_forecast.evaluation.probabilities import validate_probabilities


def log_loss_multiclass(y_true: np.ndarray, proba: np.ndarray, labels: list[int] | None = None) -> float:
    if labels is None:
        labels = [0, 1, 2]
    return float(sklearn_log_loss(y_true, validate_probabilities(proba), labels=labels))


def brier_score_multiclass(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    y = np.asarray(y_true, dtype=int)
    p = validate_probabilities(proba)
    if p.shape[1] != n_classes:
        raise ValueError(f"Expected {n_classes} classes, got {p.shape[1]}")
    one_hot = np.eye(n_classes)[y]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Ranked Probability Score for ordered classes [team2 win, draw, team1 win]."""
    y = np.asarray(y_true, dtype=int)
    p = validate_probabilities(proba)
    if p.shape[1] != 3:
        raise ValueError("RPS implementation expects exactly 3 ordered classes")
    one_hot = np.eye(3)[y]
    cum_p = np.cumsum(p, axis=1)
    cum_y = np.cumsum(one_hot, axis=1)
    # Divide by K-1 following common RPS normalization.
    return float(np.mean(np.sum((cum_p - cum_y) ** 2, axis=1) / 2.0))


def normalize_probabilities(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2:
        raise ValueError("Probabilities must be a two-dimensional array")
    if not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite")
    p = np.clip(p, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def accuracy_score_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = validate_probabilities(proba)
    return float(np.mean(np.argmax(p, axis=1) == y))


def expected_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=int)
    p = validate_probabilities(proba)
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    bin_ids = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += float(mask.mean()) * abs(float(confidence[mask].mean() - correct[mask].mean()))
    return error


def probability_sharpness(proba: np.ndarray) -> float:
    p = validate_probabilities(proba)
    return float(np.mean(np.max(p, axis=1)))
