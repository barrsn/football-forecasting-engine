from __future__ import annotations

import numpy as np
import pandas as pd

from football_forecast.data.schema import PROBABILITY_COLUMNS


def validate_probabilities(proba: np.ndarray | pd.DataFrame, *, atol: float = 1e-8) -> np.ndarray:
    values = np.asarray(proba, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(PROBABILITY_COLUMNS):
        raise ValueError(f"Expected probability shape (n, 3), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Probabilities must be finite")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("Probabilities must be within [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=atol):
        raise ValueError("Each probability row must sum to 1")
    return values


def probability_frame(
    proba: np.ndarray | pd.DataFrame,
    *,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    if isinstance(proba, pd.DataFrame):
        missing = [column for column in PROBABILITY_COLUMNS if column not in proba]
        if missing:
            raise ValueError(f"Missing probability columns: {missing}")
        frame = proba.loc[:, PROBABILITY_COLUMNS].copy()
        validate_probabilities(frame)
        return frame
    values = validate_probabilities(proba)
    return pd.DataFrame(values, columns=PROBABILITY_COLUMNS, index=index)
