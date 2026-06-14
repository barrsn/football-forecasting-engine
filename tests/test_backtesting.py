import pandas as pd

from football_forecast.evaluation.backtesting import (
    annual_rolling_origin_folds,
    fixed_holdout_indices,
)


def test_annual_folds_are_strictly_chronological():
    dates = pd.Series(pd.to_datetime(["2018-01-01", "2019-01-01", "2020-01-01"], utc=True))
    folds = annual_rolling_origin_folds(dates, first_test_year=2019)
    assert len(folds) == 2
    assert folds[0].train_index.tolist() == [0]
    assert folds[0].test_index.tolist() == [1]


def test_fixed_holdout_starts_in_2025():
    dates = pd.Series(
        pd.to_datetime(["2024-12-31", "2025-01-01", "2026-06-10", "2026-06-11"], utc=True)
    )
    train, holdout = fixed_holdout_indices(dates)
    assert train.tolist() == [0]
    assert holdout.tolist() == [1, 2]
