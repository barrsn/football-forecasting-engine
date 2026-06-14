from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from football_forecast.evaluation.metrics import (
    accuracy_score_multiclass,
    brier_score_multiclass,
    expected_calibration_error,
    log_loss_multiclass,
    probability_sharpness,
    ranked_probability_score,
)


@dataclass(frozen=True)
class ChronologicalFold:
    name: str
    train_index: np.ndarray
    test_index: np.ndarray


@dataclass(frozen=True)
class ModelMetrics:
    log_loss: float
    brier: float
    rps: float
    accuracy: float
    calibration_error: float
    sharpness: float
    n_matches: int


TOURNAMENT_REPLAYS = {
    "world_cup_2014": ("2014-06-12", "2014-07-14"),
    "world_cup_2018": ("2018-06-14", "2018-07-16"),
    "world_cup_2022": ("2022-11-20", "2022-12-19"),
    "tournaments_2024": ("2024-06-01", "2024-08-01"),
}


def annual_rolling_origin_folds(
    dates: pd.Series,
    *,
    first_test_year: int,
    last_test_year: int | None = None,
) -> list[ChronologicalFold]:
    timestamps = pd.to_datetime(dates, utc=True)
    final_year = last_test_year or int(timestamps.dt.year.max())
    folds: list[ChronologicalFold] = []
    for year in range(first_test_year, final_year + 1):
        train_index = np.flatnonzero((timestamps < pd.Timestamp(f"{year}-01-01", tz="UTC")).to_numpy())
        test_index = np.flatnonzero(
            (
                (timestamps >= pd.Timestamp(f"{year}-01-01", tz="UTC"))
                & (timestamps < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
            ).to_numpy()
        )
        if len(train_index) and len(test_index):
            folds.append(ChronologicalFold(str(year), train_index, test_index))
    return folds


def replay_folds(dates: pd.Series) -> list[ChronologicalFold]:
    timestamps = pd.to_datetime(dates, utc=True)
    folds: list[ChronologicalFold] = []
    for name, (start, end) in TOURNAMENT_REPLAYS.items():
        start_time = pd.Timestamp(start, tz="UTC")
        end_time = pd.Timestamp(end, tz="UTC")
        train_index = np.flatnonzero((timestamps < start_time).to_numpy())
        test_index = np.flatnonzero(
            ((timestamps >= start_time) & (timestamps < end_time)).to_numpy()
        )
        if len(train_index) and len(test_index):
            folds.append(ChronologicalFold(name, train_index, test_index))
    return folds


def fixed_holdout_indices(
    dates: pd.Series,
    *,
    start: str = "2025-01-01",
    end: str = "2026-06-11",
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = pd.to_datetime(dates, utc=True)
    start_time = pd.Timestamp(start, tz="UTC")
    end_time = pd.Timestamp(end, tz="UTC")
    train = np.flatnonzero((timestamps < start_time).to_numpy())
    holdout = np.flatnonzero(
        ((timestamps >= start_time) & (timestamps < end_time)).to_numpy()
    )
    return train, holdout


def out_of_fold_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    folds: list[ChronologicalFold],
    model_factory: Callable[[], object],
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for fold in folds:
        model = model_factory()
        train = frame.iloc[fold.train_index]
        test = frame.iloc[fold.test_index]
        model.fit(train[feature_columns], train[target_column])
        predictions.append(np.asarray(model.predict_proba(test[feature_columns]), dtype=float))
        targets.append(test[target_column].to_numpy(dtype=int))
    if not predictions:
        raise ValueError("No non-empty chronological folds were supplied")
    return np.vstack(predictions), np.concatenate(targets)


def evaluate_probabilities(y_true: np.ndarray, proba: np.ndarray) -> ModelMetrics:
    return ModelMetrics(
        log_loss=log_loss_multiclass(y_true, proba),
        brier=brier_score_multiclass(y_true, proba),
        rps=ranked_probability_score(y_true, proba),
        accuracy=accuracy_score_multiclass(y_true, proba),
        calibration_error=expected_calibration_error(y_true, proba),
        sharpness=probability_sharpness(proba),
        n_matches=len(y_true),
    )


def sliced_evaluation(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    proba: np.ndarray,
) -> dict[str, ModelMetrics]:
    masks: dict[str, np.ndarray] = {"overall": np.ones(len(frame), dtype=bool)}
    if "neutral" in frame:
        masks["neutral"] = frame["neutral"].to_numpy(dtype=bool)
        masks["non_neutral"] = ~masks["neutral"]
    if "elo_diff_pre" in frame:
        gap = frame["elo_diff_pre"].abs().to_numpy()
        masks["low_elo_gap"] = gap < 100
        masks["high_elo_gap"] = gap >= 200
    masks["draws"] = np.asarray(y_true) == 1
    return {
        name: evaluate_probabilities(np.asarray(y_true)[mask], np.asarray(proba)[mask])
        for name, mask in masks.items()
        if mask.any()
    }
