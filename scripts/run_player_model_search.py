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
from football_forecast.features.advanced import compact_feature_columns
from football_forecast.features.fifa import fifa_feature_columns
from football_forecast.features.scorers import scorer_feature_columns
from football_forecast.models.ensemble import (
    blend_probabilities,
    optimize_ensemble_weights,
)
from football_forecast.models.outcome import OutcomeModel

VALIDATION_YEARS = (2018, 2021, 2022, 2023, 2024)
PROBABILITY_COLUMNS = ["p_team2_win", "p_draw", "p_team1_win"]


def _fit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    *,
    model_type: str,
    year: int,
) -> np.ndarray:
    model_params: dict[str, object]
    if model_type == "logistic":
        model_params = {"C": 3.0}
    elif model_type == "hist_gbm":
        model_params = {
            "max_iter": 350,
            "learning_rate": 0.025,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 75,
            "l2_regularization": 10.0,
        }
    else:
        model_params = {
            "n_estimators": 350,
            "learning_rate": 0.02,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 75,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
        }
    weights = recency_sample_weights(
        train["kickoff_utc"],
        f"{year}-01-01",
        half_life_years=20.0 if model_type == "logistic" else None,
    )
    model = OutcomeModel(model_type, model_params=model_params).fit(
        train[columns],
        train["outcome"],
        sample_weight=weights,
    )
    return model.predict_proba(validation[columns])


def main() -> None:
    frame = pd.read_parquet(
        PROJECT_ROOT / "data/processed/features_players_1990_2026-06-10.parquet"
    )
    base_columns = list(
        dict.fromkeys(
            [*compact_feature_columns(frame), *fifa_feature_columns(frame)]
        )
    )
    scorer_columns = scorer_feature_columns(frame)
    player_columns = list(dict.fromkeys([*base_columns, *scorer_columns]))
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    for year in VALIDATION_YEARS:
        train = frame.loc[date_mask(frame["kickoff_utc"], end=f"{year}-01-01")]
        validation = frame.loc[
            date_mask(
                frame["kickoff_utc"],
                start=f"{year}-01-01",
                end=f"{year + 1}-01-01",
            )
        ]
        candidates = {
            "logistic_fifa_baseline": _fit(
                train,
                validation,
                base_columns,
                model_type="logistic",
                year=year,
            ),
            "logistic_player_scorer": _fit(
                train,
                validation,
                player_columns,
                model_type="logistic",
                year=year,
            ),
            "hist_gbm_player_scorer": _fit(
                train,
                validation,
                player_columns,
                model_type="hist_gbm",
                year=year,
            ),
            "lightgbm_player_scorer": _fit(
                train,
                validation,
                player_columns,
                model_type="lightgbm",
                year=year,
            ),
        }
        for name, probabilities in candidates.items():
            fold_rows.append(
                metrics_row(
                    name,
                    str(year),
                    validation["outcome"].to_numpy(),
                    probabilities,
                )
            )
            predictions.append(
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

    prediction_frame = pd.concat(predictions, ignore_index=True)
    aggregate_rows = []
    for name, group in prediction_frame.groupby("model", sort=False):
        aggregate_rows.append(
            metrics_row(
                name,
                "pooled_rolling_origin",
                group["outcome"].to_numpy(),
                group[PROBABILITY_COLUMNS].to_numpy(),
                kind="single",
            )
        )

    logistic = prediction_frame.loc[
        prediction_frame["model"] == "logistic_player_scorer"
    ].sort_values(["year", "row_index"])
    blend_weights: dict[str, list[float]] = {}
    for tree_name in (
        "hist_gbm_player_scorer",
        "lightgbm_player_scorer",
    ):
        tree = prediction_frame.loc[
            prediction_frame["model"] == tree_name
        ].sort_values(["year", "row_index"])
        weights = optimize_ensemble_weights(
            logistic["outcome"].to_numpy(),
            [
                logistic[PROBABILITY_COLUMNS].to_numpy(),
                tree[PROBABILITY_COLUMNS].to_numpy(),
            ],
        )
        blend = blend_probabilities(
            [
                logistic[PROBABILITY_COLUMNS].to_numpy(),
                tree[PROBABILITY_COLUMNS].to_numpy(),
            ],
            weights,
        )
        blend_name = f"blend_player_scorer__{tree_name}"
        aggregate_rows.append(
            metrics_row(
                blend_name,
                "pooled_rolling_origin",
                logistic["outcome"].to_numpy(),
                blend,
                kind="blend",
            )
        )
        blend_weights[blend_name] = [float(value) for value in weights]

    output_dir = PROJECT_ROOT / "reports/player_model_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(fold_rows)
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["log_loss", "brier", "rps"],
        ignore_index=True,
    )
    folds.to_csv(output_dir / "rolling_folds.csv", index=False)
    aggregate.to_csv(output_dir / "rolling_aggregate.csv", index=False)
    prediction_frame.to_parquet(
        output_dir / "rolling_predictions.parquet", index=False
    )
    selection = {
        "selected_validation_candidate": str(aggregate.iloc[0]["model"]),
        "base_feature_count": len(base_columns),
        "player_scorer_feature_count": len(scorer_columns),
        "candidate_feature_count": len(player_columns),
        "blend_weights": blend_weights,
        "validation_metrics": aggregate.to_dict(orient="records"),
        "holdout_opened": False,
        "promotion_rule": (
            "Do not promote unless validation improves the current champion "
            "and final holdout checks also pass."
        ),
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
