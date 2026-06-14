import numpy as np
import pandas as pd

from football_forecast.evaluation.experiments import date_mask, recency_sample_weights


def test_date_mask_is_start_inclusive_end_exclusive():
    dates = pd.Series(pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"], utc=True))
    assert date_mask(dates, "2024-01-01", "2025-01-01").tolist() == [False, True, False]


def test_recency_weights_favor_recent_matches_and_are_normalized():
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2023-01-01"], utc=True))
    weights = recency_sample_weights(dates, "2024-01-01", half_life_years=4.0)
    assert weights[1] > weights[0]
    assert np.isclose(weights.mean(), 1.0)
