from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from football_forecast.evaluation.backtesting import evaluate_probabilities


def date_mask(
    dates: pd.Series,
    start: str | None = None,
    end: str | None = None,
) -> np.ndarray:
    timestamps = pd.to_datetime(dates, utc=True)
    mask = np.ones(len(timestamps), dtype=bool)
    if start is not None:
        mask &= (timestamps >= pd.Timestamp(start, tz="UTC")).to_numpy()
    if end is not None:
        mask &= (timestamps < pd.Timestamp(end, tz="UTC")).to_numpy()
    return mask


def recency_sample_weights(
    dates: pd.Series,
    reference_date: str | pd.Timestamp,
    *,
    half_life_years: float | None,
) -> np.ndarray:
    if half_life_years is None:
        return np.ones(len(dates), dtype=float)
    timestamps = pd.to_datetime(dates, utc=True)
    reference = pd.to_datetime(reference_date, utc=True)
    ages_years = (reference - timestamps).dt.total_seconds().to_numpy() / (
        365.25 * 86400
    )
    ages_years = np.clip(ages_years, 0.0, None)
    weights = np.exp(-np.log(2.0) * ages_years / half_life_years)
    return weights / weights.mean()


def metrics_row(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    **metadata: object,
) -> dict[str, object]:
    return {
        "model": model_name,
        "split": split_name,
        **asdict(evaluate_probabilities(y_true, proba)),
        **metadata,
    }
