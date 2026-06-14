import numpy as np
import pytest

from football_forecast.evaluation.calibration import (
    IsotonicMulticlassCalibrator,
    TemperatureScaler,
    reliability_table,
)
from football_forecast.evaluation.probabilities import probability_frame, validate_probabilities


def test_probability_contract_has_named_columns():
    frame = probability_frame(np.array([[0.2, 0.3, 0.5]]))
    assert frame.columns.tolist() == ["p_team2_win", "p_draw", "p_team1_win"]
    validate_probabilities(frame)


@pytest.mark.parametrize(
    "values",
    [
        [[0.2, 0.2, 0.2]],
        [[-0.1, 0.5, 0.6]],
        [[np.nan, 0.5, 0.5]],
    ],
)
def test_invalid_probabilities_are_rejected(values):
    with pytest.raises(ValueError):
        validate_probabilities(np.asarray(values, dtype=float))


def test_reliability_table_includes_zero_and_one():
    table = reliability_table(np.array([0, 1, 1]), np.array([0.0, 0.5, 1.0]), n_bins=2)
    assert table["n"].sum() == 3
    assert table.iloc[0]["mean_predicted"] == 0.0
    assert table.iloc[-1]["mean_predicted"] == 0.75


def test_temperature_scaler_preserves_probability_contract():
    proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.2, 0.2],
        ]
    )
    calibrated = TemperatureScaler().fit(proba, np.array([0, 1, 2, 1])).transform(proba)
    validate_probabilities(calibrated)


def test_isotonic_requires_large_calibration_set():
    with pytest.raises(ValueError, match="at least 1000"):
        IsotonicMulticlassCalibrator().fit(
            np.tile([[0.3, 0.3, 0.4]], (20, 1)),
            np.zeros(20, dtype=int),
        )
