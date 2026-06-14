from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from football_forecast.evaluation.metrics import log_loss_multiclass, normalize_probabilities
from football_forecast.evaluation.probabilities import validate_probabilities


def reliability_table(
    y_true: np.ndarray,
    proba_positive: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Binary reliability table that includes exact probabilities 0 and 1."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba_positive, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and proba_positive must have the same shape")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        rows.append(
            {
                "bin": bin_id + 1,
                "p_min": bins[bin_id],
                "p_max": bins[bin_id + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "empirical_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def multiclass_reliability_tables(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> dict[int, pd.DataFrame]:
    y = np.asarray(y_true, dtype=int)
    p = validate_probabilities(proba)
    return {
        klass: reliability_table((y == klass).astype(int), p[:, klass], n_bins=n_bins)
        for klass in range(p.shape[1])
    }


@dataclass
class TemperatureScaler:
    temperature: float = 1.0
    alpha: float = 0.05

    def fit(self, proba: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        p = validate_probabilities(proba)
        y = np.asarray(y_true, dtype=int)
        log_p = np.log(np.clip(p, 1e-12, 1.0))

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            return log_loss_multiclass(y, _softmax(log_p / temperature)) + self.alpha * (log_temperature ** 2)

        result = minimize_scalar(objective, bounds=(-1.5, 1.5), method="bounded")
        self.temperature = float(np.exp(result.x)) if result.success else 1.0
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        p = validate_probabilities(proba)
        logits = np.log(np.clip(p, 1e-12, 1.0)) / self.temperature
        return _softmax(logits)


class SigmoidMulticlassCalibrator:
    def __init__(self) -> None:
        self.models: list[LogisticRegression | None] = []
        self.constants: list[float | None] = []

    def fit(self, proba: np.ndarray, y_true: np.ndarray) -> "SigmoidMulticlassCalibrator":
        p = validate_probabilities(proba)
        y = np.asarray(y_true, dtype=int)
        logits = np.log(np.clip(p, 1e-12, 1.0) / np.clip(1.0 - p, 1e-12, 1.0))
        self.models = []
        self.constants = []
        for klass in range(p.shape[1]):
            target = (y == klass).astype(int)
            if np.unique(target).size < 2:
                self.models.append(None)
                self.constants.append(float(target.mean()))
                continue
            model = LogisticRegression(C=1e6, max_iter=2000, solver="liblinear")
            model.fit(logits[:, [klass]], target)
            self.models.append(model)
            self.constants.append(None)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        p = validate_probabilities(proba)
        logits = np.log(np.clip(p, 1e-12, 1.0) / np.clip(1.0 - p, 1e-12, 1.0))
        calibrated = np.zeros_like(p)
        for klass, (model, constant) in enumerate(zip(self.models, self.constants)):
            calibrated[:, klass] = (
                constant
                if model is None
                else model.predict_proba(logits[:, [klass]])[:, 1]
            )
        return normalize_probabilities(calibrated)


class IsotonicMulticlassCalibrator:
    def __init__(self, min_samples: int = 1000) -> None:
        self.min_samples = min_samples
        self.models: list[IsotonicRegression] = []

    def fit(self, proba: np.ndarray, y_true: np.ndarray) -> "IsotonicMulticlassCalibrator":
        p = validate_probabilities(proba)
        if len(p) < self.min_samples:
            raise ValueError(
                f"Isotonic calibration requires at least {self.min_samples} samples"
            )
        y = np.asarray(y_true, dtype=int)
        self.models = []
        for klass in range(p.shape[1]):
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(p[:, klass], (y == klass).astype(int))
            self.models.append(model)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        p = validate_probabilities(proba)
        calibrated = np.column_stack(
            [model.predict(p[:, klass]) for klass, model in enumerate(self.models)]
        )
        return normalize_probabilities(calibrated)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)
