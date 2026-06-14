from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_forecast.evaluation.baselines import ClassPriorBaseline, EloLogisticBaseline
from football_forecast.evaluation.experiments import (
    date_mask,
    metrics_row,
    recency_sample_weights,
)
from football_forecast.models.outcome import OutcomeModel, StructuredOutcomeModel


@dataclass(frozen=True)
class RollingSearchConfig:
    validation_years: tuple[int, ...] = (2018, 2021, 2022, 2023, 2024)
    c_values: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0)
    half_life_years: tuple[float | None, ...] = (4.0, 8.0, 12.0)
    random_state: int = 42


def run_rolling_logistic_search(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    config: RollingSearchConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate feature and regularization choices on expanding annual folds."""
    cfg = config or RollingSearchConfig()
    dates = pd.to_datetime(frame["kickoff_utc"], utc=True)
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for year in cfg.validation_years:
        train_mask = date_mask(dates, end=f"{year}-01-01")
        validation_mask = date_mask(
            dates,
            start=f"{year}-01-01",
            end=f"{year + 1}-01-01",
        )
        train = frame.loc[train_mask]
        validation = frame.loc[validation_mask]
        if train.empty or validation.empty:
            raise ValueError(f"Rolling fold {year} is empty")
        y_validation = validation["outcome"].to_numpy()

        prior = ClassPriorBaseline().fit(train["outcome"])
        baseline_models = {
            "class_prior": prior.predict_proba(len(validation)),
            "elo_logistic": EloLogisticBaseline()
            .fit(train, train["outcome"])
            .predict_proba(validation),
        }
        for name, probabilities in baseline_models.items():
            fold_rows.append(
                metrics_row(
                    name,
                    str(year),
                    y_validation,
                    probabilities,
                    feature_set="baseline",
                    C=np.nan,
                    half_life_years=np.nan,
                )
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "candidate": name,
                        "year": year,
                        "row_index": validation.index,
                        "outcome": y_validation,
                        "p_team2_win": probabilities[:, 0],
                        "p_draw": probabilities[:, 1],
                        "p_team1_win": probabilities[:, 2],
                    }
                )
            )

        for feature_set_name, columns in feature_sets.items():
            for c_value in cfg.c_values:
                for half_life in cfg.half_life_years:
                    candidate_name = (
                        f"{feature_set_name}__c{c_value}__hl{half_life}"
                    )
                    weights = recency_sample_weights(
                        train["kickoff_utc"],
                        f"{year}-01-01",
                        half_life_years=half_life,
                    )
                    model = OutcomeModel(
                        "logistic",
                        random_state=cfg.random_state,
                        model_params={"C": c_value},
                    ).fit(
                        train[columns],
                        train["outcome"],
                        sample_weight=weights,
                    )
                    probabilities = model.predict_proba(validation[columns])
                    fold_rows.append(
                        metrics_row(
                            candidate_name,
                            str(year),
                            y_validation,
                            probabilities,
                            feature_set=feature_set_name,
                            C=c_value,
                            half_life_years=half_life,
                        )
                    )
                    prediction_rows.append(
                        pd.DataFrame(
                            {
                                "candidate": candidate_name,
                                "year": year,
                                "row_index": validation.index,
                                "outcome": y_validation,
                                "p_team2_win": probabilities[:, 0],
                                "p_draw": probabilities[:, 1],
                                "p_team1_win": probabilities[:, 2],
                            }
                        )
                    )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    aggregate_rows = []
    for candidate, rows in predictions.groupby("candidate", sort=False):
        probabilities = rows[
            ["p_team2_win", "p_draw", "p_team1_win"]
        ].to_numpy()
        metadata = folds.loc[folds["model"] == candidate].iloc[0]
        aggregate_rows.append(
            metrics_row(
                candidate,
                "pooled_rolling_origin",
                rows["outcome"].to_numpy(),
                probabilities,
                feature_set=metadata["feature_set"],
                C=metadata["C"],
                half_life_years=metadata["half_life_years"],
                fold_log_loss_std=float(
                    folds.loc[folds["model"] == candidate, "log_loss"].std(ddof=0)
                ),
            )
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["log_loss", "brier", "rps"],
        ignore_index=True,
    )
    return folds, aggregate


def run_rolling_structured_search(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    validation_years: tuple[int, ...] = (2018, 2021, 2022, 2023, 2024),
    draw_c_values: tuple[float, ...] = (0.03, 0.1, 0.3),
    decisive_c_values: tuple[float, ...] = (0.3, 1.0),
    half_life_values: tuple[float | None, ...] = (8.0, 12.0, 20.0, None),
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune the two-stage draw/decisive model on expanding annual folds."""
    dates = pd.to_datetime(frame["kickoff_utc"], utc=True)
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for year in validation_years:
        train = frame.loc[date_mask(dates, end=f"{year}-01-01")]
        validation = frame.loc[
            date_mask(
                dates,
                start=f"{year}-01-01",
                end=f"{year + 1}-01-01",
            )
        ]
        for draw_c in draw_c_values:
            for decisive_c in decisive_c_values:
                for half_life in half_life_values:
                    half_life_name = "none" if half_life is None else str(half_life)
                    name = (
                        f"structured__draw{draw_c}__decisive{decisive_c}"
                        f"__hl{half_life_name}"
                    )
                    weights = recency_sample_weights(
                        train["kickoff_utc"],
                        f"{year}-01-01",
                        half_life_years=half_life,
                    )
                    model = StructuredOutcomeModel(
                        draw_c=draw_c,
                        decisive_c=decisive_c,
                        random_state=random_state,
                    ).fit(
                        train[feature_columns],
                        train["outcome"],
                        sample_weight=weights,
                    )
                    probabilities = model.predict_proba(validation[feature_columns])
                    fold_rows.append(
                        metrics_row(
                            name,
                            str(year),
                            validation["outcome"].to_numpy(),
                            probabilities,
                            draw_C=draw_c,
                            decisive_C=decisive_c,
                            half_life_years=half_life,
                        )
                    )
                    prediction_rows.append(
                        pd.DataFrame(
                            {
                                "candidate": name,
                                "year": year,
                                "row_index": validation.index,
                                "outcome": validation["outcome"].to_numpy(),
                                "p_team2_win": probabilities[:, 0],
                                "p_draw": probabilities[:, 1],
                                "p_team1_win": probabilities[:, 2],
                            }
                        )
                    )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    aggregate_rows = []
    for candidate, rows in predictions.groupby("candidate", sort=False):
        metadata = folds.loc[folds["model"] == candidate].iloc[0]
        aggregate_rows.append(
            metrics_row(
                candidate,
                "pooled_rolling_origin",
                rows["outcome"].to_numpy(),
                rows[["p_team2_win", "p_draw", "p_team1_win"]].to_numpy(),
                draw_C=metadata["draw_C"],
                decisive_C=metadata["decisive_C"],
                half_life_years=metadata["half_life_years"],
                fold_log_loss_std=float(
                    folds.loc[folds["model"] == candidate, "log_loss"].std(ddof=0)
                ),
            )
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["log_loss", "brier", "rps"],
        ignore_index=True,
    )
    return folds, aggregate, predictions
