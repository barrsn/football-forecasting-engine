from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from football_forecast.evaluation.metrics import normalize_probabilities
from football_forecast.evaluation.probabilities import probability_frame
from football_forecast.models.sklearn_config import (
    LOGISTIC_MAX_ITER,
    LOGISTIC_SOLVER,
    predict_logistic_pipeline_proba,
)


class OutcomeModel:
    """Win/draw/loss probability model.

    Uses sklearn by default so the repository is runnable without heavy optional dependencies.
    Class order is [0=team2 win, 1=draw, 2=team1 win].
    """

    def __init__(
        self,
        model_type: str = "hist_gbm",
        random_state: int = 42,
        model_params: dict[str, object] | None = None,
    ):
        self.model_type = model_type
        self.random_state = random_state
        params = dict(model_params or {})
        if model_type == "logistic":
            defaults = {
                "C": 1.0,
                "max_iter": LOGISTIC_MAX_ITER,
                "class_weight": None,
                "random_state": random_state,
                "solver": LOGISTIC_SOLVER,
            }
            defaults.update(params)
            self.model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(**defaults),
                    ),
                ]
            )
        elif model_type == "hist_gbm":
            defaults = {
                "max_iter": 300,
                "learning_rate": 0.03,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 30,
                "l2_regularization": 1.0,
                "random_state": random_state,
            }
            defaults.update(params)
            self.model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(**defaults),
                    ),
                ]
            )
        elif model_type == "catboost":
            try:
                from catboost import CatBoostClassifier
            except ImportError as exc:
                raise ImportError("Install the optional 'boosting' dependencies for CatBoost") from exc
            defaults = {
                "loss_function": "MultiClass",
                "iterations": 800,
                "depth": 5,
                "learning_rate": 0.04,
                "random_seed": random_state,
                "verbose": False,
            }
            defaults.update(params)
            self.model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        CatBoostClassifier(**defaults),
                    ),
                ]
            )
        elif model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise ImportError("Install the optional 'boosting' dependencies for LightGBM") from exc
            defaults = {
                "objective": "multiclass",
                "n_estimators": 800,
                "max_depth": 5,
                "num_leaves": 15,
                "learning_rate": 0.04,
                "min_child_samples": 30,
                "reg_lambda": 1.0,
                "random_state": random_state,
                "verbosity": -1,
                "n_jobs": 1,
            }
            defaults.update(params)
            self.model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMClassifier(**defaults),
                    ),
                ]
            )
        else:
            raise ValueError(
                "model_type must be 'hist_gbm', 'logistic', 'catboost', or 'lightgbm'"
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> "OutcomeModel":
        fit_params = (
            {}
            if sample_weight is None
            else {"model__sample_weight": np.asarray(sample_weight, dtype=float)}
        )
        self.model.fit(X, y, **fit_params)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_type == "logistic":
            raw, classes = predict_logistic_pipeline_proba(self.model, X)
        else:
            raw = self.model.predict_proba(X)
            classes = list(self.model[-1].classes_)
        out = np.zeros((len(X), 3), dtype=float)
        for idx, klass in enumerate(classes):
            out[:, int(klass)] = raw[:, idx]
        return normalize_probabilities(out)

    def predict_proba_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return probability_frame(self.predict_proba(X), index=X.index)


class StructuredOutcomeModel:
    """Two-stage outcome model: draw probability, then winner if decisive."""

    def __init__(
        self,
        *,
        draw_c: float = 0.1,
        decisive_c: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.draw_model = self._pipeline(draw_c, random_state)
        self.decisive_model = self._pipeline(decisive_c, random_state)

    @staticmethod
    def _pipeline(c_value: float, random_state: int) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=c_value,
                        max_iter=LOGISTIC_MAX_ITER,
                        random_state=random_state,
                        solver=LOGISTIC_SOLVER,
                    ),
                ),
            ]
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> "StructuredOutcomeModel":
        target = np.asarray(y, dtype=int)
        weights = (
            np.ones(len(target), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        draw_target = (target == 1).astype(int)
        decisive_mask = target != 1
        self.draw_model.fit(
            X,
            draw_target,
            model__sample_weight=weights,
        )
        self.decisive_model.fit(
            X.loc[decisive_mask],
            (target[decisive_mask] == 2).astype(int),
            model__sample_weight=weights[decisive_mask],
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        draw_raw, draw_classes = predict_logistic_pipeline_proba(self.draw_model, X)
        draw = draw_raw[:, list(draw_classes).index(1)]
        decisive_raw, decisive_classes = predict_logistic_pipeline_proba(
            self.decisive_model,
            X,
        )
        team1_given_decisive = decisive_raw[:, list(decisive_classes).index(1)]
        decisive = 1.0 - draw
        probabilities = np.column_stack(
            [
                decisive * (1.0 - team1_given_decisive),
                draw,
                decisive * team1_given_decisive,
            ]
        )
        return normalize_probabilities(probabilities)

    def predict_proba_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return probability_frame(self.predict_proba(X), index=X.index)
