# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from football_forecast.evaluation.baselines import ClassPriorBaseline, EloLogisticBaseline
from football_forecast.evaluation.experiments import (
    date_mask,
    metrics_row,
    recency_sample_weights,
)
from football_forecast.evaluation.rolling_search import run_rolling_logistic_search
from football_forecast.features.advanced import (
    add_advanced_context_features,
    advanced_feature_columns,
    compact_feature_columns,
)
from football_forecast.features.build import default_feature_columns
from football_forecast.models.outcome import OutcomeModel
from football_forecast.models.poisson import TwoPoissonScoreModel
from football_forecast.reporting.tables import prediction_table


def probabilities_from_lambdas(lambda1: np.ndarray, lambda2: np.ndarray) -> np.ndarray:
    rows = []
    for value1, value2 in zip(lambda1, lambda2):
        matrix = TwoPoissonScoreModel.score_matrix(value1, value2, max_goals=8)
        rows.append(
            [
                np.triu(matrix, k=1).sum(),
                np.trace(matrix),
                np.tril(matrix, k=-1).sum(),
            ]
        )
    probabilities = np.asarray(rows)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fit_logistic(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    *,
    c_value: float,
    half_life_years: float | None,
    reference_date: str,
) -> np.ndarray:
    weights = recency_sample_weights(
        train["kickoff_utc"],
        reference_date,
        half_life_years=half_life_years,
    )
    model = OutcomeModel(
        "logistic",
        random_state=42,
        model_params={"C": c_value},
    ).fit(train[columns], train["outcome"], sample_weight=weights)
    return model.predict_proba(validation[columns])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="data/processed/features_1990_2026-06-10.parquet",
    )
    parser.add_argument(
        "--advanced-features",
        default="data/processed/features_advanced_1990_2026-06-10.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/advanced_model_search",
    )
    parser.add_argument("--rebuild-advanced", action="store_true")
    args = parser.parse_args()

    base_path = PROJECT_ROOT / args.features
    advanced_path = PROJECT_ROOT / args.advanced_features
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if advanced_path.exists() and not args.rebuild_advanced:
        frame = pd.read_parquet(advanced_path)
    else:
        frame = add_advanced_context_features(pd.read_parquet(base_path))
        advanced_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(advanced_path, index=False)

    base_columns = default_feature_columns(frame)
    extra_columns = advanced_feature_columns(frame)
    all_columns = list(dict.fromkeys([*base_columns, *extra_columns]))
    compact_columns = compact_feature_columns(frame)
    feature_sets = {
        "base": base_columns,
        "advanced": all_columns,
        "compact": compact_columns,
    }
    folds, aggregate = run_rolling_logistic_search(frame, feature_sets)
    folds.to_csv(output_dir / "rolling_folds.csv", index=False)
    aggregate.to_csv(output_dir / "rolling_aggregate.csv", index=False)

    candidate_rows = aggregate.loc[aggregate["feature_set"] != "baseline"]
    selected = candidate_rows.iloc[0]
    selected_columns = feature_sets[str(selected["feature_set"])]
    train_mask = date_mask(frame["kickoff_utc"], end="2025-01-01")
    test_mask = date_mask(
        frame["kickoff_utc"],
        start="2025-01-01",
        end="2026-06-11",
    )
    train = frame.loc[train_mask]
    test = frame.loc[test_mask]
    y_test = test["outcome"].to_numpy()

    probabilities: dict[str, np.ndarray] = {
        "class_prior": ClassPriorBaseline()
        .fit(train["outcome"])
        .predict_proba(len(test)),
        "elo_logistic": EloLogisticBaseline()
        .fit(train, train["outcome"])
        .predict_proba(test),
        "online_poisson": probabilities_from_lambdas(
            test["online_lambda_team1_pre"].to_numpy(),
            test["online_lambda_team2_pre"].to_numpy(),
        ),
        "base_refit_c1_hl8": fit_logistic(
            train,
            test,
            base_columns,
            c_value=1.0,
            half_life_years=8.0,
            reference_date="2025-01-01",
        ),
        "rolling_selected": fit_logistic(
            train,
            test,
            selected_columns,
            c_value=float(selected["C"]),
            half_life_years=float(selected["half_life_years"]),
            reference_date="2025-01-01",
        ),
    }
    final_results = pd.DataFrame(
        [
            metrics_row(name, "holdout_2025_2026", y_test, proba)
            for name, proba in probabilities.items()
        ]
    ).sort_values(["log_loss", "brier", "rps"], ignore_index=True)
    final_results.to_csv(output_dir / "holdout_results.csv", index=False)

    selected_predictions = prediction_table(test, probabilities["rolling_selected"])
    selected_predictions["outcome"] = y_test
    selected_predictions["team1_goals"] = test["team1_goals"].to_numpy()
    selected_predictions["team2_goals"] = test["team2_goals"].to_numpy()
    selected_predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)

    selection = {
        "validation_years": [2018, 2021, 2022, 2023, 2024],
        "selected_candidate": str(selected["model"]),
        "selected_feature_set": str(selected["feature_set"]),
        "selected_C": float(selected["C"]),
        "selected_half_life_years": float(selected["half_life_years"]),
        "feature_counts": {
            name: len(columns) for name, columns in feature_sets.items()
        },
        "rolling_metrics": {
            key: float(selected[key])
            for key in ("log_loss", "brier", "rps", "accuracy", "calibration_error")
        },
        "holdout_metrics": final_results.loc[
            final_results["model"] == "rolling_selected"
        ].iloc[0].to_dict(),
        "holdout_best_model": str(final_results.iloc[0]["model"]),
        "holdout_was_previously_observed": True,
        "data_cutoff_exclusive": "2026-06-11T00:00:00Z",
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, default=str),
        encoding="utf-8",
    )

    print("Rolling-origin aggregate")
    print(aggregate.head(15).to_string(index=False))
    print("\nFinal holdout")
    print(final_results.to_string(index=False))
    print("\nSelection")
    print(json.dumps(selection, indent=2, default=str))


if __name__ == "__main__":
    main()
