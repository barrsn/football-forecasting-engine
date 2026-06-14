# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from football_forecast.evaluation.experiments import (
    date_mask,
    metrics_row,
    recency_sample_weights,
)
from football_forecast.models.ensemble import (
    blend_probabilities,
    optimize_ensemble_weights,
)
from football_forecast.features.advanced import compact_feature_columns
from football_forecast.features.fifa import fifa_feature_columns
from football_forecast.models.outcome import OutcomeModel
from football_forecast.reporting.tables import prediction_table

VALIDATION_YEARS = (2018, 2021, 2022, 2023, 2024)


def candidates() -> list[dict[str, object]]:
    values: list[dict[str, object]] = [
        {
            "name": "lightgbm_small",
            "model_type": "lightgbm",
            "params": {
                "n_estimators": 350,
                "learning_rate": 0.02,
                "num_leaves": 7,
                "max_depth": 3,
                "min_child_samples": 75,
                "reg_alpha": 1.0,
                "reg_lambda": 10.0,
            },
        },
        {
            "name": "lightgbm_medium",
            "model_type": "lightgbm",
            "params": {
                "n_estimators": 500,
                "learning_rate": 0.02,
                "num_leaves": 15,
                "max_depth": 4,
                "min_child_samples": 75,
                "reg_alpha": 1.0,
                "reg_lambda": 15.0,
            },
        },
        {
            "name": "lightgbm_slow",
            "model_type": "lightgbm",
            "params": {
                "n_estimators": 650,
                "learning_rate": 0.015,
                "num_leaves": 15,
                "max_depth": 5,
                "min_child_samples": 100,
                "reg_alpha": 2.0,
                "reg_lambda": 20.0,
            },
        },
        {
            "name": "hist_gbm_small",
            "model_type": "hist_gbm",
            "params": {
                "max_iter": 350,
                "learning_rate": 0.025,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 75,
                "l2_regularization": 10.0,
            },
        },
        {
            "name": "hist_gbm_medium",
            "model_type": "hist_gbm",
            "params": {
                "max_iter": 450,
                "learning_rate": 0.02,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 75,
                "l2_regularization": 15.0,
            },
        },
    ]
    return [
        {**value, "half_life_years": half_life}
        for value in values
        for half_life in (12.0, 20.0, None)
    ]


def fit_model(
    candidate: dict[str, object],
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    reference_date: str,
) -> tuple[OutcomeModel, np.ndarray]:
    weights = recency_sample_weights(
        train["kickoff_utc"],
        reference_date,
        half_life_years=candidate["half_life_years"],
    )
    model = OutcomeModel(
        str(candidate["model_type"]),
        model_params=dict(candidate["params"]),
    ).fit(train[columns], train["outcome"], sample_weight=weights)
    return model, model.predict_proba(test[columns])


def main() -> None:
    frame = pd.read_parquet(
        PROJECT_ROOT / "data/processed/features_fifa_1990_2026-06-10.parquet"
    )
    columns = list(
        dict.fromkeys([*compact_feature_columns(frame), *fifa_feature_columns(frame)])
    )
    output_dir = PROJECT_ROOT / "reports/fifa_boosting_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_list = candidates()
    logistic_candidate = {
        "name": "logistic_fifa",
        "model_type": "logistic",
        "params": {"C": 3.0},
        "half_life_years": 20.0,
    }
    all_candidates = [logistic_candidate, *candidate_list]
    fold_rows = []
    prediction_rows = []

    for year in VALIDATION_YEARS:
        train = frame.loc[date_mask(frame["kickoff_utc"], end=f"{year}-01-01")]
        validation = frame.loc[
            date_mask(
                frame["kickoff_utc"],
                start=f"{year}-01-01",
                end=f"{year + 1}-01-01",
            )
        ]
        for candidate in all_candidates:
            _, probabilities = fit_model(
                candidate,
                train,
                validation,
                columns,
                f"{year}-01-01",
            )
            name = (
                f"{candidate['name']}__hl"
                f"{candidate['half_life_years']}"
            )
            fold_rows.append(
                metrics_row(
                    name,
                    str(year),
                    validation["outcome"].to_numpy(),
                    probabilities,
                    model_type=candidate["model_type"],
                    parameters=json.dumps(candidate, sort_keys=True),
                )
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "model": name,
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
    for name, rows in predictions.groupby("model", sort=False):
        aggregate_rows.append(
            metrics_row(
                name,
                "pooled_rolling_origin",
                rows["outcome"].to_numpy(),
                rows[["p_team2_win", "p_draw", "p_team1_win"]].to_numpy(),
                kind="single",
            )
        )

    logistic_name = "logistic_fifa__hl20.0"
    logistic_rows = predictions.loc[predictions["model"] == logistic_name].sort_values(
        ["year", "row_index"]
    )
    ensemble_weights: dict[str, list[float]] = {}
    for candidate in candidate_list:
        other_name = f"{candidate['name']}__hl{candidate['half_life_years']}"
        other_rows = predictions.loc[predictions["model"] == other_name].sort_values(
            ["year", "row_index"]
        )
        y_true = logistic_rows["outcome"].to_numpy()
        logistic_proba = logistic_rows[
            ["p_team2_win", "p_draw", "p_team1_win"]
        ].to_numpy()
        other_proba = other_rows[
            ["p_team2_win", "p_draw", "p_team1_win"]
        ].to_numpy()
        weights = optimize_ensemble_weights(
            y_true,
            [logistic_proba, other_proba],
        )
        ensemble_name = f"blend__{other_name}"
        blend = blend_probabilities([logistic_proba, other_proba], weights)
        aggregate_rows.append(
            metrics_row(
                ensemble_name,
                "pooled_rolling_origin",
                y_true,
                blend,
                kind="blend",
            )
        )
        ensemble_weights[ensemble_name] = [float(value) for value in weights]

    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["log_loss", "brier", "rps"],
        ignore_index=True,
    )
    folds.to_csv(output_dir / "rolling_folds.csv", index=False)
    aggregate.to_csv(output_dir / "rolling_aggregate.csv", index=False)
    predictions.to_parquet(output_dir / "rolling_predictions.parquet", index=False)

    selected_name = str(aggregate.iloc[0]["model"])
    train = frame.loc[date_mask(frame["kickoff_utc"], end="2025-01-01")]
    test = frame.loc[
        date_mask(
            frame["kickoff_utc"],
            start="2025-01-01",
            end="2026-06-11",
        )
    ]
    _, logistic_test = fit_model(
        logistic_candidate,
        train,
        test,
        columns,
        "2025-01-01",
    )
    if selected_name == logistic_name:
        selected_test = logistic_test
    elif selected_name.startswith("blend__"):
        other_name = selected_name.removeprefix("blend__")
        other = next(
            candidate
            for candidate in candidate_list
            if other_name
            == f"{candidate['name']}__hl{candidate['half_life_years']}"
        )
        _, other_test = fit_model(other, train, test, columns, "2025-01-01")
        selected_test = blend_probabilities(
            [logistic_test, other_test],
            np.asarray(ensemble_weights[selected_name]),
        )
    else:
        selected = next(
            candidate
            for candidate in candidate_list
            if selected_name
            == f"{candidate['name']}__hl{candidate['half_life_years']}"
        )
        _, selected_test = fit_model(selected, train, test, columns, "2025-01-01")

    holdout = pd.DataFrame(
        [
            metrics_row(
                selected_name,
                "holdout_2025_2026",
                test["outcome"].to_numpy(),
                selected_test,
            ),
            metrics_row(
                logistic_name,
                "holdout_2025_2026",
                test["outcome"].to_numpy(),
                logistic_test,
            ),
        ]
    )
    holdout.to_csv(output_dir / "holdout_results.csv", index=False)
    holdout_predictions = prediction_table(test, selected_test)
    holdout_predictions["outcome"] = test["outcome"].to_numpy()
    holdout_predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)
    selection = {
        "selected_model": selected_name,
        "feature_count": len(columns),
        "rolling_metrics": aggregate.iloc[0].to_dict(),
        "holdout_metrics": holdout.iloc[0].to_dict(),
        "ensemble_weights": ensemble_weights.get(selected_name),
        "holdout_was_previously_observed": True,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, default=str),
        encoding="utf-8",
    )
    print(aggregate.head(20).to_string(index=False))
    print("\nHoldout")
    print(holdout.to_string(index=False))
    print("\nSelection")
    print(json.dumps(selection, indent=2, default=str))


if __name__ == "__main__":
    main()
