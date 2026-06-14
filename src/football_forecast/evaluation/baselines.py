from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from football_forecast.evaluation.metrics import normalize_probabilities


class ClassPriorBaseline:
    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = smoothing
        self.class_probabilities = np.ones(3) / 3.0

    def fit(self, y: pd.Series | np.ndarray) -> "ClassPriorBaseline":
        counts = np.bincount(np.asarray(y, dtype=int), minlength=3).astype(float)
        counts += self.smoothing
        self.class_probabilities = counts / counts.sum()
        return self

    def predict_proba(self, n_rows: int) -> np.ndarray:
        return np.tile(self.class_probabilities, (n_rows, 1))


class EloLogisticBaseline:
    feature_columns = ("elo_diff_pre", "neutral_int", "host_team1_int", "host_team2_int")

    def __init__(self, random_state: int = 42) -> None:
        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        random_state=random_state,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def fit(self, frame: pd.DataFrame, y: pd.Series) -> "EloLogisticBaseline":
        columns = [column for column in self.feature_columns if column in frame]
        self.columns_ = columns
        self.model.fit(frame[columns], y)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(frame[self.columns_])
        classes = self.model[-1].classes_
        out = np.zeros((len(frame), 3), dtype=float)
        for index, klass in enumerate(classes):
            out[:, int(klass)] = raw[:, index]
        return normalize_probabilities(out)


class BookmakerOddsBaseline:
    """Convert decimal odds to de-vigged probabilities for benchmarking only."""

    columns = ("odds_team2_win", "odds_draw", "odds_team1_win")

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.columns if column not in frame]
        if missing:
            raise ValueError(f"Missing odds columns: {missing}")
        odds = frame.loc[:, self.columns].to_numpy(dtype=float)
        if not np.isfinite(odds).all() or (odds <= 1.0).any():
            raise ValueError("Decimal odds must be finite and greater than 1")
        return normalize_probabilities(1.0 / odds)
