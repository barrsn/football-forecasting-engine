from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_forecast.data.schema import OutcomeClass
from football_forecast.evaluation.probabilities import validate_probabilities

OUTCOME_LABELS = {
    int(OutcomeClass.TEAM2_WIN): "team2_win",
    int(OutcomeClass.DRAW): "draw",
    int(OutcomeClass.TEAM1_WIN): "team1_win",
}


@dataclass(frozen=True)
class SelectiveAccuracyReport:
    threshold: float
    coverage: float
    selective_accuracy: float
    full_accuracy: float
    selected_matches: int
    total_matches: int


def selective_predictions(
    proba: np.ndarray,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Return hard picks only when maximum probability reaches ``threshold``."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    probabilities = validate_probabilities(proba)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    selected = confidence >= threshold
    hard_prediction = pd.array(
        np.where(selected, prediction, pd.NA),
        dtype="Int64",
    )
    labels = pd.Series(hard_prediction).map(OUTCOME_LABELS).fillna("abstain")
    return pd.DataFrame(
        {
            "predicted_outcome": hard_prediction,
            "predicted_label": labels,
            "confidence": confidence,
            "is_high_confidence": selected,
        }
    )


def evaluate_selective_accuracy(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    threshold: float,
) -> SelectiveAccuracyReport:
    y = np.asarray(y_true, dtype=int)
    probabilities = validate_probabilities(proba)
    if len(y) != len(probabilities):
        raise ValueError("y_true and proba must have the same number of rows")
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    selected = confidence >= threshold
    return SelectiveAccuracyReport(
        threshold=float(threshold),
        coverage=float(selected.mean()),
        selective_accuracy=(
            float(np.mean(prediction[selected] == y[selected]))
            if selected.any()
            else float("nan")
        ),
        full_accuracy=float(np.mean(prediction == y)),
        selected_matches=int(selected.sum()),
        total_matches=len(y),
    )


def choose_stable_confidence_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    groups: np.ndarray | pd.Series,
    *,
    target_accuracy: float = 0.65,
    thresholds: tuple[float, ...] = tuple(
        round(float(value), 2) for value in np.arange(0.40, 0.81, 0.01)
    ),
    min_group_predictions: int = 100,
) -> float:
    """Choose the highest-coverage threshold meeting the target in every group."""
    y = np.asarray(y_true, dtype=int)
    probabilities = validate_probabilities(proba)
    group_values = np.asarray(groups)
    if len(y) != len(probabilities) or len(y) != len(group_values):
        raise ValueError("y_true, proba, and groups must have equal length")

    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    for threshold in sorted(thresholds):
        valid = True
        for group in pd.unique(group_values):
            selected = (group_values == group) & (confidence >= threshold)
            if selected.sum() < min_group_predictions:
                valid = False
                break
            accuracy = float(np.mean(prediction[selected] == y[selected]))
            if accuracy < target_accuracy:
                valid = False
                break
        if valid:
            return float(threshold)
    raise ValueError("No confidence threshold satisfies the requested constraints")
