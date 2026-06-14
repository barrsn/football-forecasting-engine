# ruff: noqa: E402
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

from football_forecast.data.io import read_matches_csv
from football_forecast.evaluation.backtesting import evaluate_probabilities
from football_forecast.evaluation.baselines import ClassPriorBaseline, EloLogisticBaseline
from football_forecast.evaluation.calibration import (
    SigmoidMulticlassCalibrator,
    TemperatureScaler,
)
from football_forecast.evaluation.selection import select_candidate_model
from football_forecast.features.build import build_feature_table, default_feature_columns
from football_forecast.models.ensemble import blend_probabilities, optimize_ensemble_weights
from football_forecast.models.outcome import OutcomeModel
from football_forecast.models.poisson import DixonColesScoreModel, TwoPoissonScoreModel


def chronological_three_way_split(
    n_rows: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_rows < 6:
        raise ValueError("At least six rows are required for train/validation/test")
    train_end = max(1, int(n_rows * train_fraction))
    validation_end = max(train_end + 1, int(n_rows * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, n_rows - 1)
    return (
        np.arange(0, train_end),
        np.arange(train_end, validation_end),
        np.arange(validation_end, n_rows),
    )


def _format_metrics(name: str, y_true: np.ndarray, proba: np.ndarray) -> dict[str, object]:
    metrics = evaluate_probabilities(y_true, proba)
    return {
        "model": name,
        "log_loss": metrics.log_loss,
        "brier": metrics.brier,
        "rps": metrics.rps,
        "accuracy": metrics.accuracy,
        "ece": metrics.calibration_error,
        "sharpness": metrics.sharpness,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", default="data/sample/international_results_sample.csv")
    args = parser.parse_args()

    matches = read_matches_csv(Path(args.matches))
    features = build_feature_table(matches, windows=(3, 5))
    feature_columns = default_feature_columns(features)
    train_index, validation_index, test_index = chronological_three_way_split(len(features))
    train = features.iloc[train_index]
    validation = features.iloc[validation_index]
    test = features.iloc[test_index]
    y_validation = validation["outcome"].to_numpy()
    y_test = test["outcome"].to_numpy()

    validation_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}

    prior = ClassPriorBaseline().fit(train["outcome"])
    validation_probabilities["class_prior"] = prior.predict_proba(len(validation))
    test_probabilities["class_prior"] = prior.predict_proba(len(test))

    elo = EloLogisticBaseline().fit(train, train["outcome"])
    validation_probabilities["elo_logistic"] = elo.predict_proba(validation)
    test_probabilities["elo_logistic"] = elo.predict_proba(test)

    for model_type in ("logistic", "hist_gbm"):
        model = OutcomeModel(model_type=model_type, random_state=42).fit(
            train[feature_columns], train["outcome"]
        )
        validation_probabilities[model_type] = model.predict_proba(
            validation[feature_columns]
        )
        test_probabilities[model_type] = model.predict_proba(test[feature_columns])

    for name, model in (
        ("poisson", TwoPoissonScoreModel()),
        ("dixon_coles", DixonColesScoreModel()),
    ):
        model.fit(
            train[feature_columns],
            train["team1_goals"],
            train["team2_goals"],
        )
        validation_probabilities[name] = model.predict_outcome_proba(
            validation[feature_columns]
        )
        test_probabilities[name] = model.predict_outcome_proba(test[feature_columns])

    calibration_base = "hist_gbm"
    temperature = TemperatureScaler().fit(
        validation_probabilities[calibration_base], y_validation
    )
    validation_probabilities["hist_gbm_temperature"] = temperature.transform(
        validation_probabilities[calibration_base]
    )
    test_probabilities["hist_gbm_temperature"] = temperature.transform(
        test_probabilities[calibration_base]
    )
    sigmoid = SigmoidMulticlassCalibrator().fit(
        validation_probabilities[calibration_base], y_validation
    )
    validation_probabilities["hist_gbm_sigmoid"] = sigmoid.transform(
        validation_probabilities[calibration_base]
    )
    test_probabilities["hist_gbm_sigmoid"] = sigmoid.transform(
        test_probabilities[calibration_base]
    )

    ensemble_names = ["elo_logistic", "poisson", "hist_gbm_temperature"]
    weights = optimize_ensemble_weights(
        y_validation,
        [validation_probabilities[name] for name in ensemble_names],
    )
    validation_probabilities["optimized_ensemble"] = blend_probabilities(
        [validation_probabilities[name] for name in ensemble_names], weights
    )
    test_probabilities["optimized_ensemble"] = blend_probabilities(
        [test_probabilities[name] for name in ensemble_names], weights
    )

    validation_report = pd.DataFrame(
        [
            _format_metrics(name, y_validation, proba)
            for name, proba in validation_probabilities.items()
        ]
    ).sort_values("log_loss")
    test_report = pd.DataFrame(
        [_format_metrics(name, y_test, proba) for name, proba in test_probabilities.items()]
    ).sort_values("log_loss")
    promotion_rows = []
    for candidate, baseline in (
        ("hist_gbm_temperature", "hist_gbm"),
        ("optimized_ensemble", "elo_logistic"),
    ):
        validation_decision = select_candidate_model(
            candidate,
            evaluate_probabilities(y_validation, validation_probabilities[candidate]),
            baseline,
            evaluate_probabilities(y_validation, validation_probabilities[baseline]),
        )
        test_decision = select_candidate_model(
            candidate,
            evaluate_probabilities(y_test, test_probabilities[candidate]),
            baseline,
            evaluate_probabilities(y_test, test_probabilities[baseline]),
        )
        promotion_rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "validation_pass": validation_decision.approved,
                "test_pass": test_decision.approved,
                "promoted": validation_decision.approved and test_decision.approved,
            }
        )

    print("Chronological split")
    print(f"rows_train: {len(train)}")
    print(f"rows_validation: {len(validation)}")
    print(f"rows_test: {len(test)}")
    print(f"features: {len(feature_columns)}")
    print(
        f"ensemble_weights: "
        f"{dict(zip(ensemble_names, [float(value) for value in weights.round(4)]))}"
    )
    print(f"temperature: {temperature.temperature:.4f}")
    print("\nValidation metrics")
    print(validation_report.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nTest metrics")
    print(test_report.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nPromotion decisions")
    print(pd.DataFrame(promotion_rows).to_string(index=False))


if __name__ == "__main__":
    main()
