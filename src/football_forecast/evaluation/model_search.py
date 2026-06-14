from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from football_forecast.evaluation.backtesting import evaluate_probabilities
from football_forecast.evaluation.baselines import ClassPriorBaseline, EloLogisticBaseline
from football_forecast.evaluation.calibration import (
    SigmoidMulticlassCalibrator,
    TemperatureScaler,
)
from football_forecast.evaluation.experiments import (
    date_mask,
    metrics_row,
    recency_sample_weights,
)
from football_forecast.evaluation.selection import select_candidate_model
from football_forecast.models.ensemble import (
    blend_probabilities,
    optimize_ensemble_weights,
)
from football_forecast.models.outcome import OutcomeModel
from football_forecast.models.poisson import DixonColesScoreModel, TwoPoissonScoreModel
from football_forecast.reporting.tables import prediction_table


@dataclass(frozen=True)
class SearchConfig:
    tuning_start: str = "2023-01-01"
    final_training_end: str = "2024-01-01"
    calibration_end: str = "2024-07-01"
    ensemble_end: str = "2025-01-01"
    test_end: str = "2026-06-11"
    random_state: int = 42


@dataclass
class SearchResult:
    tuning_results: pd.DataFrame
    validation_results: pd.DataFrame
    test_results: pd.DataFrame
    test_predictions: pd.DataFrame
    selection: dict[str, Any]


def default_outcome_candidates(include_lightgbm: bool = True) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for half_life in (None, 8.0):
        for c_value in (0.03, 0.1, 0.3, 1.0):
            candidates.append(
                {
                    "name": f"logistic_c{c_value}_hl{half_life}",
                    "model_type": "logistic",
                    "model_params": {"C": c_value},
                    "half_life_years": half_life,
                }
            )
        hist_params = [
            {
                "max_iter": 300,
                "learning_rate": 0.03,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 50,
                "l2_regularization": 5.0,
            },
            {
                "max_iter": 400,
                "learning_rate": 0.03,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 50,
                "l2_regularization": 5.0,
            },
            {
                "max_iter": 300,
                "learning_rate": 0.05,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 30,
                "l2_regularization": 2.0,
            },
            {
                "max_iter": 400,
                "learning_rate": 0.03,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 50,
                "l2_regularization": 10.0,
            },
        ]
        for index, params in enumerate(hist_params, start=1):
            candidates.append(
                {
                    "name": f"hist_gbm_{index}_hl{half_life}",
                    "model_type": "hist_gbm",
                    "model_params": params,
                    "half_life_years": half_life,
                }
            )
        if include_lightgbm:
            lightgbm_params = [
                {
                    "n_estimators": 300,
                    "learning_rate": 0.03,
                    "num_leaves": 7,
                    "max_depth": 4,
                    "min_child_samples": 50,
                    "reg_lambda": 5.0,
                },
                {
                    "n_estimators": 500,
                    "learning_rate": 0.03,
                    "num_leaves": 15,
                    "max_depth": 5,
                    "min_child_samples": 50,
                    "reg_lambda": 5.0,
                },
                {
                    "n_estimators": 400,
                    "learning_rate": 0.05,
                    "num_leaves": 15,
                    "max_depth": 5,
                    "min_child_samples": 30,
                    "reg_lambda": 2.0,
                },
                {
                    "n_estimators": 600,
                    "learning_rate": 0.02,
                    "num_leaves": 31,
                    "max_depth": 6,
                    "min_child_samples": 50,
                    "reg_lambda": 10.0,
                },
            ]
            for index, params in enumerate(lightgbm_params, start=1):
                candidates.append(
                    {
                        "name": f"lightgbm_{index}_hl{half_life}",
                        "model_type": "lightgbm",
                        "model_params": params,
                        "half_life_years": half_life,
                    }
                )
    return candidates


def default_score_candidates() -> list[dict[str, Any]]:
    candidates = []
    for model_type in ("poisson", "dixon_coles"):
        for alpha in (0.03, 0.1, 0.3):
            for half_life in (None, 8.0):
                candidates.append(
                    {
                        "name": f"{model_type}_a{alpha}_hl{half_life}",
                        "model_type": model_type,
                        "alpha": alpha,
                        "half_life_years": half_life,
                    }
                )
    return candidates


def _fit_outcome(
    candidate: dict[str, Any],
    train: pd.DataFrame,
    feature_columns: list[str],
    reference_date: str,
    random_state: int,
) -> OutcomeModel:
    weights = recency_sample_weights(
        train["kickoff_utc"],
        reference_date,
        half_life_years=candidate["half_life_years"],
    )
    return OutcomeModel(
        candidate["model_type"],
        random_state=random_state,
        model_params=candidate["model_params"],
    ).fit(train[feature_columns], train["outcome"], sample_weight=weights)


def _fit_score(
    candidate: dict[str, Any],
    train: pd.DataFrame,
    feature_columns: list[str],
    reference_date: str,
) -> TwoPoissonScoreModel:
    model_class = (
        DixonColesScoreModel
        if candidate["model_type"] == "dixon_coles"
        else TwoPoissonScoreModel
    )
    weights = recency_sample_weights(
        train["kickoff_utc"],
        reference_date,
        half_life_years=candidate["half_life_years"],
    )
    return model_class(alpha=candidate["alpha"]).fit(
        train[feature_columns],
        train["team1_goals"],
        train["team2_goals"],
        sample_weight=weights,
    )


def _search_candidates(
    frame: pd.DataFrame,
    feature_columns: list[str],
    train_mask: np.ndarray,
    tune_mask: np.ndarray,
    candidates: list[dict[str, Any]],
    config: SearchConfig,
    *,
    kind: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = frame.loc[train_mask]
    tune = frame.loc[tune_mask]
    y_tune = tune["outcome"].to_numpy()
    rows = []
    for candidate in candidates:
        started = perf_counter()
        if kind == "outcome":
            model = _fit_outcome(
                candidate,
                train,
                feature_columns,
                config.tuning_start,
                config.random_state,
            )
            proba = model.predict_proba(tune[feature_columns])
        else:
            model = _fit_score(
                candidate,
                train,
                feature_columns,
                config.tuning_start,
            )
            proba = model.predict_outcome_proba(tune[feature_columns])
        rows.append(
            metrics_row(
                candidate["name"],
                "tuning_2023",
                y_tune,
                proba,
                kind=kind,
                seconds=perf_counter() - started,
                parameters=json.dumps(candidate, sort_keys=True),
            )
        )
    results = pd.DataFrame(rows).sort_values(
        ["log_loss", "brier", "rps"], ignore_index=True
    )
    best_name = results.iloc[0]["model"]
    best = next(candidate for candidate in candidates if candidate["name"] == best_name)
    return results, best


def run_chronological_model_search(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    config: SearchConfig | None = None,
    include_lightgbm: bool = True,
) -> SearchResult:
    cfg = config or SearchConfig()
    dates = frame["kickoff_utc"]
    tuning_train = date_mask(dates, end=cfg.tuning_start)
    tuning = date_mask(dates, cfg.tuning_start, cfg.final_training_end)
    final_train = date_mask(dates, end=cfg.final_training_end)
    calibration = date_mask(dates, cfg.final_training_end, cfg.calibration_end)
    ensemble_validation = date_mask(dates, cfg.calibration_end, cfg.ensemble_end)
    test = date_mask(dates, cfg.ensemble_end, cfg.test_end)
    for name, mask in {
        "tuning_train": tuning_train,
        "tuning": tuning,
        "final_train": final_train,
        "calibration": calibration,
        "ensemble_validation": ensemble_validation,
        "test": test,
    }.items():
        if not mask.any():
            raise ValueError(f"Chronological split {name} is empty")

    outcome_results, best_outcome = _search_candidates(
        frame,
        feature_columns,
        tuning_train,
        tuning,
        default_outcome_candidates(include_lightgbm=include_lightgbm),
        cfg,
        kind="outcome",
    )
    score_results, best_score = _search_candidates(
        frame,
        feature_columns,
        tuning_train,
        tuning,
        default_score_candidates(),
        cfg,
        kind="score",
    )
    tuning_results = pd.concat([outcome_results, score_results], ignore_index=True)

    train_frame = frame.loc[final_train]
    calibration_frame = frame.loc[calibration]
    ensemble_frame = frame.loc[ensemble_validation]
    test_frame = frame.loc[test]
    best_outcome_model = _fit_outcome(
        best_outcome,
        train_frame,
        feature_columns,
        cfg.final_training_end,
        cfg.random_state,
    )
    best_score_model = _fit_score(
        best_score,
        train_frame,
        feature_columns,
        cfg.final_training_end,
    )
    elo_model = EloLogisticBaseline().fit(train_frame, train_frame["outcome"])
    prior_model = ClassPriorBaseline().fit(train_frame["outcome"])

    outcome_calibration = best_outcome_model.predict_proba(
        calibration_frame[feature_columns]
    )
    outcome_validation_raw = best_outcome_model.predict_proba(
        ensemble_frame[feature_columns]
    )
    outcome_test_raw = best_outcome_model.predict_proba(test_frame[feature_columns])
    calibrators = {
        "raw": None,
        "temperature": TemperatureScaler().fit(
            outcome_calibration,
            calibration_frame["outcome"].to_numpy(),
        ),
        "sigmoid": SigmoidMulticlassCalibrator().fit(
            outcome_calibration,
            calibration_frame["outcome"].to_numpy(),
        ),
    }
    validation_calibration_rows = []
    calibrated_validation: dict[str, np.ndarray] = {}
    calibrated_test: dict[str, np.ndarray] = {}
    for name, calibrator in calibrators.items():
        validation_proba = (
            outcome_validation_raw
            if calibrator is None
            else calibrator.transform(outcome_validation_raw)
        )
        test_proba = (
            outcome_test_raw
            if calibrator is None
            else calibrator.transform(outcome_test_raw)
        )
        calibrated_validation[name] = validation_proba
        calibrated_test[name] = test_proba
        validation_calibration_rows.append(
            metrics_row(
                f"{best_outcome['name']}__{name}",
                "ensemble_validation_2024_h2",
                ensemble_frame["outcome"].to_numpy(),
                validation_proba,
                kind="calibration",
            )
        )
    calibration_results = pd.DataFrame(validation_calibration_rows).sort_values(
        ["log_loss", "brier", "rps"], ignore_index=True
    )
    selected_calibration = str(calibration_results.iloc[0]["model"]).split("__")[-1]

    validation_probabilities = {
        "class_prior": prior_model.predict_proba(len(ensemble_frame)),
        "elo_logistic": elo_model.predict_proba(ensemble_frame),
        best_score["name"]: best_score_model.predict_outcome_proba(
            ensemble_frame[feature_columns]
        ),
        f"{best_outcome['name']}__raw": outcome_validation_raw,
        f"{best_outcome['name']}__{selected_calibration}": calibrated_validation[
            selected_calibration
        ],
    }
    test_probabilities = {
        "class_prior": prior_model.predict_proba(len(test_frame)),
        "elo_logistic": elo_model.predict_proba(test_frame),
        best_score["name"]: best_score_model.predict_outcome_proba(
            test_frame[feature_columns]
        ),
        f"{best_outcome['name']}__raw": outcome_test_raw,
        f"{best_outcome['name']}__{selected_calibration}": calibrated_test[
            selected_calibration
        ],
    }
    ensemble_members = [
        "elo_logistic",
        best_score["name"],
        f"{best_outcome['name']}__{selected_calibration}",
    ]
    weights = optimize_ensemble_weights(
        ensemble_frame["outcome"].to_numpy(),
        [validation_probabilities[name] for name in ensemble_members],
    )
    validation_probabilities["optimized_ensemble"] = blend_probabilities(
        [validation_probabilities[name] for name in ensemble_members],
        weights,
    )
    test_probabilities["optimized_ensemble"] = blend_probabilities(
        [test_probabilities[name] for name in ensemble_members],
        weights,
    )
    validation_probabilities["equal_ensemble"] = blend_probabilities(
        [validation_probabilities[name] for name in ensemble_members]
    )
    test_probabilities["equal_ensemble"] = blend_probabilities(
        [test_probabilities[name] for name in ensemble_members]
    )

    validation_results = pd.DataFrame(
        [
            metrics_row(
                name,
                "ensemble_validation_2024_h2",
                ensemble_frame["outcome"].to_numpy(),
                proba,
            )
            for name, proba in validation_probabilities.items()
        ]
    ).sort_values(["log_loss", "brier", "rps"], ignore_index=True)
    test_results = pd.DataFrame(
        [
            metrics_row(
                name,
                "holdout_2025_2026",
                test_frame["outcome"].to_numpy(),
                proba,
            )
            for name, proba in test_probabilities.items()
        ]
    ).sort_values(["log_loss", "brier", "rps"], ignore_index=True)

    selected_validation_model = str(validation_results.iloc[0]["model"])
    baseline_names = [
        "class_prior",
        "elo_logistic",
        best_score["name"],
        f"{best_outcome['name']}__raw",
    ]
    strongest_baseline = (
        validation_results.loc[validation_results["model"].isin(baseline_names)]
        .sort_values(["log_loss", "brier", "rps"])
        .iloc[0]["model"]
    )
    validation_candidate_metrics = evaluate_probabilities(
        ensemble_frame["outcome"].to_numpy(),
        validation_probabilities[selected_validation_model],
    )
    validation_baseline_metrics = evaluate_probabilities(
        ensemble_frame["outcome"].to_numpy(),
        validation_probabilities[str(strongest_baseline)],
    )
    test_candidate_metrics = evaluate_probabilities(
        test_frame["outcome"].to_numpy(),
        test_probabilities[selected_validation_model],
    )
    test_baseline_metrics = evaluate_probabilities(
        test_frame["outcome"].to_numpy(),
        test_probabilities[str(strongest_baseline)],
    )
    validation_decision = select_candidate_model(
        selected_validation_model,
        validation_candidate_metrics,
        str(strongest_baseline),
        validation_baseline_metrics,
    )
    test_decision = select_candidate_model(
        selected_validation_model,
        test_candidate_metrics,
        str(strongest_baseline),
        test_baseline_metrics,
    )
    promoted = validation_decision.approved and test_decision.approved
    deployment_model = selected_validation_model if promoted else str(strongest_baseline)
    selected_test_proba = test_probabilities[deployment_model]
    predictions = prediction_table(test_frame, selected_test_proba)
    predictions["outcome"] = test_frame["outcome"].to_numpy()
    predictions["team1_goals"] = test_frame["team1_goals"].to_numpy()
    predictions["team2_goals"] = test_frame["team2_goals"].to_numpy()

    selection = {
        "config": asdict(cfg),
        "feature_count": len(feature_columns),
        "split_sizes": {
            "tuning_train": int(tuning_train.sum()),
            "tuning_2023": int(tuning.sum()),
            "final_train": int(final_train.sum()),
            "calibration_2024_h1": int(calibration.sum()),
            "ensemble_validation_2024_h2": int(ensemble_validation.sum()),
            "holdout_2025_2026": int(test.sum()),
        },
        "best_outcome_candidate": best_outcome,
        "best_score_candidate": best_score,
        "selected_calibration": selected_calibration,
        "ensemble_members": ensemble_members,
        "ensemble_weights": {
            name: float(weight) for name, weight in zip(ensemble_members, weights)
        },
        "selected_validation_model": selected_validation_model,
        "selected_validation_metrics": validation_results.iloc[0].to_dict(),
        "strongest_validation_baseline": str(strongest_baseline),
        "promotion_validation_pass": validation_decision.approved,
        "promotion_holdout_pass": test_decision.approved,
        "promoted": promoted,
        "deployment_model": deployment_model,
        "deployment_reason": (
            validation_decision.reason
            if not validation_decision.approved
            else test_decision.reason
        ),
        "deployment_holdout_metrics": test_results.loc[
            test_results["model"] == deployment_model
        ].iloc[0].to_dict(),
        "best_holdout_model_for_reference": test_results.iloc[0]["model"],
    }
    return SearchResult(
        tuning_results=tuning_results,
        validation_results=pd.concat(
            [calibration_results, validation_results], ignore_index=True
        ),
        test_results=test_results,
        test_predictions=predictions,
        selection=selection,
    )


def write_search_result(result: SearchResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.tuning_results.to_csv(output / "tuning_results.csv", index=False)
    result.validation_results.to_csv(output / "validation_results.csv", index=False)
    result.test_results.to_csv(output / "test_results.csv", index=False)
    result.test_predictions.to_csv(output / "test_predictions.csv", index=False)
    (output / "selection.json").write_text(
        json.dumps(result.selection, indent=2, default=str),
        encoding="utf-8",
    )
