# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import sklearn

from football_forecast.evaluation.backtesting import evaluate_probabilities
from football_forecast.evaluation.experiments import (
    date_mask,
    metrics_row,
    recency_sample_weights,
)
from football_forecast.evaluation.selection import select_candidate_model
from football_forecast.features.advanced import compact_feature_columns
from football_forecast.features.fifa import fifa_feature_columns
from football_forecast.models.ensemble import (
    WeightedOutcomeEnsemble,
    blend_probabilities,
    optimize_ensemble_weights,
)
from football_forecast.models.outcome import OutcomeModel
from football_forecast.reporting.tables import prediction_table

VALIDATION_YEARS = (2018, 2021, 2022, 2023, 2024)
CHAMPION_NAME = "logistic_fifa_hist_gbm_blend"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_logistic(
    train: pd.DataFrame,
    columns: list[str],
    reference_date: str,
) -> OutcomeModel:
    weights = recency_sample_weights(
        train["kickoff_utc"],
        reference_date,
        half_life_years=20.0,
    )
    return OutcomeModel(
        "logistic",
        model_params={"C": 3.0},
    ).fit(train[columns], train["outcome"], sample_weight=weights)


def fit_hist_gbm(train: pd.DataFrame, columns: list[str]) -> OutcomeModel:
    return OutcomeModel(
        "hist_gbm",
        model_params={
            "max_iter": 350,
            "learning_rate": 0.025,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 75,
            "l2_regularization": 10.0,
        },
    ).fit(train[columns], train["outcome"])


def baseline_oof(
    frame: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    outcomes = []
    probabilities = []
    for year in VALIDATION_YEARS:
        train = frame.loc[date_mask(frame["kickoff_utc"], end=f"{year}-01-01")]
        validation = frame.loc[
            date_mask(
                frame["kickoff_utc"],
                start=f"{year}-01-01",
                end=f"{year + 1}-01-01",
            )
        ]
        weights = recency_sample_weights(
            train["kickoff_utc"],
            f"{year}-01-01",
            half_life_years=20.0,
        )
        model = OutcomeModel(
            "logistic",
            model_params={"C": 3.0},
        ).fit(train[columns], train["outcome"], sample_weight=weights)
        outcomes.append(validation["outcome"].to_numpy())
        probabilities.append(model.predict_proba(validation[columns]))
    return np.concatenate(outcomes), np.vstack(probabilities)


def main() -> None:
    features_path = (
        PROJECT_ROOT / "data/processed/features_fifa_1990_2026-06-10.parquet"
    )
    frame = pd.read_parquet(features_path)
    base_columns = compact_feature_columns(frame)
    champion_columns = list(
        dict.fromkeys([*base_columns, *fifa_feature_columns(frame)])
    )
    oof = pd.read_parquet(
        PROJECT_ROOT / "reports/fifa_boosting_search/rolling_predictions.parquet"
    )
    logistic_oof = oof.loc[
        oof["model"] == "logistic_fifa__hl20.0"
    ].sort_values(["year", "row_index"])
    hist_oof = oof.loc[
        oof["model"] == "hist_gbm_small__hlNone"
    ].sort_values(["year", "row_index"])
    probability_columns = ["p_team2_win", "p_draw", "p_team1_win"]
    weights = optimize_ensemble_weights(
        logistic_oof["outcome"].to_numpy(),
        [
            logistic_oof[probability_columns].to_numpy(),
            hist_oof[probability_columns].to_numpy(),
        ],
    )
    champion_oof = blend_probabilities(
        [
            logistic_oof[probability_columns].to_numpy(),
            hist_oof[probability_columns].to_numpy(),
        ],
        weights,
    )
    baseline_y, baseline_probabilities = baseline_oof(frame, base_columns)
    validation_rows = [
        metrics_row(
            "compact_no_fifa_baseline",
            "pooled_rolling_origin",
            baseline_y,
            baseline_probabilities,
        ),
        metrics_row(
            CHAMPION_NAME,
            "pooled_rolling_origin",
            logistic_oof["outcome"].to_numpy(),
            champion_oof,
        ),
    ]
    validation_results = pd.DataFrame(validation_rows).sort_values("log_loss")
    validation_decision = select_candidate_model(
        CHAMPION_NAME,
        evaluate_probabilities(
            logistic_oof["outcome"].to_numpy(),
            champion_oof,
        ),
        "compact_no_fifa_baseline",
        evaluate_probabilities(baseline_y, baseline_probabilities),
    )

    train = frame.loc[date_mask(frame["kickoff_utc"], end="2025-01-01")]
    holdout = frame.loc[
        date_mask(
            frame["kickoff_utc"],
            start="2025-01-01",
            end="2026-06-11",
        )
    ]
    logistic = fit_logistic(train, champion_columns, "2025-01-01")
    hist_gbm = fit_hist_gbm(train, champion_columns)
    champion = WeightedOutcomeEnsemble([logistic, hist_gbm], weights)
    champion_holdout = champion.predict_proba(holdout[champion_columns])
    baseline_weights = recency_sample_weights(
        train["kickoff_utc"],
        "2025-01-01",
        half_life_years=20.0,
    )
    baseline_model = OutcomeModel(
        "logistic",
        model_params={"C": 3.0},
    ).fit(
        train[base_columns],
        train["outcome"],
        sample_weight=baseline_weights,
    )
    baseline_holdout = baseline_model.predict_proba(holdout[base_columns])
    holdout_results = pd.DataFrame(
        [
            metrics_row(
                "compact_no_fifa_baseline",
                "holdout_2025_2026",
                holdout["outcome"].to_numpy(),
                baseline_holdout,
            ),
            metrics_row(
                CHAMPION_NAME,
                "holdout_2025_2026",
                holdout["outcome"].to_numpy(),
                champion_holdout,
            ),
        ]
    ).sort_values("log_loss")
    holdout_decision = select_candidate_model(
        CHAMPION_NAME,
        evaluate_probabilities(holdout["outcome"].to_numpy(), champion_holdout),
        "compact_no_fifa_baseline",
        evaluate_probabilities(holdout["outcome"].to_numpy(), baseline_holdout),
    )
    promoted = validation_decision.approved and holdout_decision.approved

    report_dir = PROJECT_ROOT / "reports/champion_model"
    report_dir.mkdir(parents=True, exist_ok=True)
    validation_results.to_csv(report_dir / "validation_results.csv", index=False)
    holdout_results.to_csv(report_dir / "holdout_results.csv", index=False)
    predictions = prediction_table(holdout, champion_holdout)
    predictions["outcome"] = holdout["outcome"].to_numpy()
    predictions["team1_goals"] = holdout["team1_goals"].to_numpy()
    predictions["team2_goals"] = holdout["team2_goals"].to_numpy()
    predictions.to_csv(report_dir / "holdout_predictions.csv", index=False)

    full_training = frame.loc[
        date_mask(frame["kickoff_utc"], end="2026-06-11")
    ]
    production_logistic = fit_logistic(
        full_training,
        champion_columns,
        "2026-06-11",
    )
    production_hist = fit_hist_gbm(full_training, champion_columns)
    production_model = WeightedOutcomeEnsemble(
        [production_logistic, production_hist],
        weights,
    )
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "world_cup_2026_champion.joblib"
    joblib.dump(production_model, model_path)

    results_manifest = json.loads(
        (
            PROJECT_ROOT
            / "data/raw/martj42_international_results/source_manifest.json"
        ).read_text(encoding="utf-8")
    )
    fifa_manifest = json.loads(
        (PROJECT_ROOT / "data/raw/fifa_rankings/source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    selection = {
        "model": CHAMPION_NAME,
        "promoted": promoted,
        "validation_decision": validation_decision.reason,
        "holdout_decision": holdout_decision.reason,
        "weights": {
            "logistic_fifa": float(weights[0]),
            "hist_gbm_fifa": float(weights[1]),
        },
        "feature_count": len(champion_columns),
        "base_feature_count": len(base_columns),
        "validation_metrics": validation_results.to_dict(orient="records"),
        "holdout_metrics": holdout_results.to_dict(orient="records"),
        "holdout_was_previously_observed": True,
        "production_training_rows": len(full_training),
        "production_training_cutoff_exclusive": "2026-06-11T00:00:00Z",
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "model_sha256": sha256(model_path),
    }
    (report_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, default=str),
        encoding="utf-8",
    )
    metadata = {
        **selection,
        "feature_columns": champion_columns,
        "python": sys.version,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "match_results_commit": results_manifest["commit"],
        "match_results_sha256": results_manifest["sha256"],
        "fifa_page_build_id": fifa_manifest["page_build_id"],
        "fifa_processed_sha256": fifa_manifest["processed_sha256"],
        "features_sha256": sha256(features_path),
    }
    (models_dir / "world_cup_2026_champion.metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    print(validation_results.to_string(index=False))
    print("\nHoldout")
    print(holdout_results.to_string(index=False))
    print("\nSelection")
    print(json.dumps(selection, indent=2, default=str))


if __name__ == "__main__":
    main()
