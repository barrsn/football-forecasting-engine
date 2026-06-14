# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import pandas as pd
import sklearn

from football_forecast.evaluation.experiments import (
    date_mask,
    metrics_row,
    recency_sample_weights,
)
from football_forecast.features.advanced import compact_feature_columns
from football_forecast.features.fifa import fifa_feature_columns
from football_forecast.features.scorers import scorer_feature_columns
from football_forecast.models.ensemble import WeightedOutcomeEnsemble
from football_forecast.models.outcome import OutcomeModel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_models(
    train: pd.DataFrame,
    columns: list[str],
    *,
    reference_date: str,
) -> tuple[OutcomeModel, OutcomeModel]:
    weights = recency_sample_weights(
        train["kickoff_utc"],
        reference_date,
        half_life_years=20.0,
    )
    logistic = OutcomeModel(
        "logistic",
        model_params={"C": 3.0},
    ).fit(train[columns], train["outcome"], sample_weight=weights)
    lightgbm = OutcomeModel(
        "lightgbm",
        model_params={
            "n_estimators": 350,
            "learning_rate": 0.02,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 75,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
        },
    ).fit(train[columns], train["outcome"])
    return logistic, lightgbm


def main() -> None:
    feature_path = (
        PROJECT_ROOT / "data/processed/features_players_1990_2026-06-10.parquet"
    )
    frame = pd.read_parquet(feature_path)
    core_columns = list(
        dict.fromkeys(
            [*compact_feature_columns(frame), *fifa_feature_columns(frame)]
        )
    )
    scorer_columns = scorer_feature_columns(frame)
    columns = list(dict.fromkeys([*core_columns, *scorer_columns]))

    search_selection = json.loads(
        (
            PROJECT_ROOT / "reports/player_model_search/selection.json"
        ).read_text(encoding="utf-8")
    )
    candidate_name = "blend_player_scorer__lightgbm_player_scorer"
    blend_weights = search_selection["blend_weights"][candidate_name]

    train = frame.loc[date_mask(frame["kickoff_utc"], end="2025-01-01")]
    holdout = frame.loc[
        date_mask(
            frame["kickoff_utc"],
            start="2025-01-01",
            end="2026-06-11",
        )
    ]
    logistic, lightgbm = _fit_models(
        train,
        columns,
        reference_date="2025-01-01",
    )
    candidate = WeightedOutcomeEnsemble(
        [logistic, lightgbm],
        blend_weights,
    )
    probabilities = candidate.predict_proba(holdout[columns])
    holdout_results = pd.DataFrame(
        [
            metrics_row(
                candidate_name,
                "holdout_2025_2026",
                holdout["outcome"].to_numpy(),
                probabilities,
            )
        ]
    )

    champion_holdout = pd.read_csv(
        PROJECT_ROOT / "reports/champion_model/holdout_results.csv"
    ).loc[
        lambda values: values["model"] == "logistic_fifa_hist_gbm_blend"
    ].iloc[0]
    candidate_holdout = holdout_results.iloc[0]
    holdout_not_worse = (
        candidate_holdout["brier"] <= champion_holdout["brier"] * 1.01
        and candidate_holdout["rps"] <= champion_holdout["rps"] * 1.01
    )
    validation_candidate = next(
        row
        for row in search_selection["validation_metrics"]
        if row["model"] == candidate_name
    )
    champion_validation = pd.read_csv(
        PROJECT_ROOT / "reports/champion_model/validation_results.csv"
    ).loc[
        lambda values: values["model"] == "logistic_fifa_hist_gbm_blend"
    ].iloc[0]
    validation_improvement = (
        champion_validation["log_loss"] - validation_candidate["log_loss"]
    ) / champion_validation["log_loss"]
    materially_better = bool(validation_improvement >= 0.005)
    promoted = bool(
        materially_better
        and holdout_not_worse
        and candidate_holdout["log_loss"] < champion_holdout["log_loss"]
    )

    full_training = frame.loc[
        date_mask(frame["kickoff_utc"], end="2026-06-11")
    ]
    production_logistic, production_lightgbm = _fit_models(
        full_training,
        columns,
        reference_date="2026-06-11",
    )
    production_model = WeightedOutcomeEnsemble(
        [production_logistic, production_lightgbm],
        blend_weights,
    )
    model_path = PROJECT_ROOT / "models/world_cup_2026_player_scorer.joblib"
    joblib.dump(production_model, model_path)

    output_dir = PROJECT_ROOT / "reports/player_model_search"
    holdout_results.to_csv(output_dir / "holdout_results.csv", index=False)
    selection = {
        **search_selection,
        "holdout_opened": True,
        "holdout_was_previously_observed_for_repository": True,
        "holdout_metrics": holdout_results.to_dict(orient="records"),
        "current_champion_holdout": champion_holdout.to_dict(),
        "validation_improvement_vs_current_champion": float(
            validation_improvement
        ),
        "materially_better_than_current_champion": materially_better,
        "holdout_brier_rps_not_worse": bool(holdout_not_worse),
        "promoted_to_core_champion": promoted,
        "decision": (
            "Promoted"
            if promoted
            else "Kept as optional candidate; current core champion remains simpler"
        ),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "model_sha256": _sha256(model_path),
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, default=str),
        encoding="utf-8",
    )

    source_manifest = json.loads(
        (
            PROJECT_ROOT
            / "data/raw/martj42_international_results/source_manifest.json"
        ).read_text(encoding="utf-8")
    )
    metadata = {
        **selection,
        "feature_columns": columns,
        "player_scorer_feature_columns": scorer_columns,
        "feature_file": str(feature_path.relative_to(PROJECT_ROOT)),
        "feature_file_sha256": _sha256(feature_path),
        "goalscorers_source": source_manifest["goalscorers"],
        "python": sys.version,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (
        PROJECT_ROOT / "models/world_cup_2026_player_scorer.metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    print(holdout_results.to_string(index=False))
    print(json.dumps(selection, indent=2, default=str))


if __name__ == "__main__":
    main()
