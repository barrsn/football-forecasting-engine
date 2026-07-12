from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from football_forecast.evaluation.metrics import normalize_probabilities
from football_forecast.evaluation.probabilities import probability_frame


class StablePoissonRegressor(BaseEstimator, RegressorMixin):
    """Small Poisson GLM trained without scipy/sklearn GLM optimizers."""

    def __init__(
        self,
        alpha: float = 0.1,
        max_iter: int = 120,
        learning_rate: float = 0.03,
        tol: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "StablePoissonRegressor":
        values = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        weights = (
            np.ones(len(target), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        weights = weights / weights.mean()
        n_features = values.shape[1]
        coef = np.zeros(n_features, dtype=float)
        intercept = float(np.log(np.clip(np.average(target, weights=weights), 0.05, None)))
        previous_loss = float("inf")
        for _ in range(self.max_iter):
            linear = np.sum(values * coef, axis=1) + intercept
            linear = np.clip(linear, -6.0, 3.0)
            expected = np.exp(linear)
            residual = (expected - target) * weights
            grad_intercept = float(residual.mean())
            grad_coef = np.mean(values * residual[:, None], axis=0) + self.alpha * coef
            step = self.learning_rate
            coef_next = coef - step * grad_coef
            intercept_next = intercept - step * grad_intercept
            loss = float(
                np.mean(weights * (expected - target * linear))
                + 0.5 * self.alpha * np.sum(coef * coef)
            )
            coef = coef_next
            intercept = intercept_next
            if abs(previous_loss - loss) < self.tol:
                break
            previous_loss = loss
        self.coef_ = coef
        self.intercept_ = intercept
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        linear = np.sum(values * self.coef_, axis=1) + self.intercept_
        return np.exp(np.clip(linear, -6.0, 3.0))


class TwoPoissonScoreModel:
    """Two independent Poisson regressions for football scorelines.

    This is a strong transparent baseline. It ignores correlation between team scores;
    bivariate/Dixon-Coles extensions can be added later.
    """

    def __init__(self, alpha: float = 0.1, max_iter: int = 120):
        self.alpha = alpha
        self.max_iter = max_iter
        self.model_team1 = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    StablePoissonRegressor(
                        alpha=alpha,
                        max_iter=max_iter,
                    ),
                ),
            ]
        )
        self.model_team2 = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    StablePoissonRegressor(
                        alpha=alpha,
                        max_iter=max_iter,
                    ),
                ),
            ]
        )

    def fit(
        self,
        X: pd.DataFrame,
        y_team1_goals: pd.Series,
        y_team2_goals: pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> "TwoPoissonScoreModel":
        fit_params = (
            {}
            if sample_weight is None
            else {"model__sample_weight": np.asarray(sample_weight, dtype=float)}
        )
        self.model_team1.fit(X, y_team1_goals, **fit_params)
        self.model_team2.fit(X, y_team2_goals, **fit_params)
        return self

    def predict_lambdas(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lam1 = np.clip(self.model_team1.predict(X), 0.02, 5.0)
        lam2 = np.clip(self.model_team2.predict(X), 0.02, 5.0)
        return lam1, lam2

    @staticmethod
    def score_matrix(lambda1: float, lambda2: float, max_goals: int = 7) -> np.ndarray:
        goals = np.arange(max_goals + 1)
        p1 = poisson.pmf(goals, lambda1)
        p2 = poisson.pmf(goals, lambda2)
        matrix = np.outer(p1, p2)
        return matrix / matrix.sum()

    def predict_score_matrices(self, X: pd.DataFrame, max_goals: int = 7) -> np.ndarray:
        lam1, lam2 = self.predict_lambdas(X)
        return np.asarray([self.score_matrix(a, b, max_goals=max_goals) for a, b in zip(lam1, lam2)])

    def predict_outcome_proba(self, X: pd.DataFrame, max_goals: int = 7) -> np.ndarray:
        matrices = self.predict_score_matrices(X, max_goals=max_goals)
        rows = []
        for m in matrices:
            team1_win = np.tril(m, k=-1).sum()  # rows=team1 goals, cols=team2 goals; row > col
            team2_win = np.triu(m, k=1).sum()
            draw = np.trace(m)
            rows.append([team2_win, draw, team1_win])
        return normalize_probabilities(np.asarray(rows))

    def predict_proba_frame(self, X: pd.DataFrame, max_goals: int = 7) -> pd.DataFrame:
        return probability_frame(self.predict_outcome_proba(X, max_goals=max_goals), index=X.index)


class DixonColesScoreModel(TwoPoissonScoreModel):
    """Independent Poisson model with a fitted low-score Dixon-Coles correction."""

    def __init__(self, alpha: float = 0.1, max_iter: int = 200) -> None:
        super().__init__(alpha=alpha, max_iter=max_iter)
        self.rho = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y_team1_goals: pd.Series,
        y_team2_goals: pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> "DixonColesScoreModel":
        super().fit(
            X,
            y_team1_goals,
            y_team2_goals,
            sample_weight=sample_weight,
        )
        lambda1, lambda2 = self.predict_lambdas(X)
        goals1 = np.asarray(y_team1_goals, dtype=int)
        goals2 = np.asarray(y_team2_goals, dtype=int)

        def objective(rho: float) -> float:
            correction = np.ones(len(goals1), dtype=float)
            mask00 = (goals1 == 0) & (goals2 == 0)
            mask01 = (goals1 == 0) & (goals2 == 1)
            mask10 = (goals1 == 1) & (goals2 == 0)
            mask11 = (goals1 == 1) & (goals2 == 1)
            correction[mask00] = 1.0 - lambda1[mask00] * lambda2[mask00] * rho
            correction[mask01] = 1.0 + lambda1[mask01] * rho
            correction[mask10] = 1.0 + lambda2[mask10] * rho
            correction[mask11] = 1.0 - rho
            if (correction <= 0).any():
                return float("inf")
            return float(-np.log(correction).sum())

        result = minimize_scalar(objective, bounds=(-0.2, 0.2), method="bounded")
        self.rho = float(result.x) if result.success else 0.0
        return self

    def score_matrix(self, lambda1: float, lambda2: float, max_goals: int = 7) -> np.ndarray:
        matrix = super().score_matrix(lambda1, lambda2, max_goals=max_goals)
        if max_goals >= 1:
            matrix[0, 0] *= 1.0 - lambda1 * lambda2 * self.rho
            matrix[0, 1] *= 1.0 + lambda1 * self.rho
            matrix[1, 0] *= 1.0 + lambda2 * self.rho
            matrix[1, 1] *= 1.0 - self.rho
        matrix = np.clip(matrix, 0.0, None)
        return matrix / matrix.sum()
