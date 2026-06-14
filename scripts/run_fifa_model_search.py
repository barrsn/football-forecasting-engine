# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from football_forecast.evaluation.experiments import (
    date_mask,
    metrics_row,
    recency_sample_weights,
)
from football_forecast.evaluation.rolling_search import (
    RollingSearchConfig,
    run_rolling_logistic_search,
)
from football_forecast.features.advanced import compact_feature_columns
from football_forecast.features.fifa import (
    add_fifa_ranking_features,
    fifa_feature_columns,
)
from football_forecast.models.outcome import OutcomeModel
from football_forecast.reporting.tables import prediction_table


def fit_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    row: pd.Series,
) -> pd.DataFrame:
    half_life = (
        None if pd.isna(row["half_life_years"]) else float(row["half_life_years"])
    )
    weights = recency_sample_weights(
        train["kickoff_utc"],
        "2025-01-01",
        half_life_years=half_life,
    )
    model = OutcomeModel(
        "logistic",
        model_params={"C": float(row["C"])},
    ).fit(train[columns], train["outcome"], sample_weight=weights)
    return model.predict_proba(test[columns])


def main() -> None:
    advanced_path = (
        PROJECT_ROOT / "data/processed/features_advanced_1990_2026-06-10.parquet"
    )
    rankings_path = (
        PROJECT_ROOT / "data/processed/fifa_rankings_1992_2026-04-01.parquet"
    )
    output_features = (
        PROJECT_ROOT / "data/processed/features_fifa_1990_2026-06-10.parquet"
    )
    output_dir = PROJECT_ROOT / "reports/fifa_model_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = add_fifa_ranking_features(
        pd.read_parquet(advanced_path),
        pd.read_parquet(rankings_path),
    )
    frame.to_parquet(output_features, index=False)
    compact_columns = compact_feature_columns(frame)
    fifa_columns = fifa_feature_columns(frame)
    feature_sets = {
        "compact_v2": compact_columns,
        "compact_fifa": list(dict.fromkeys([*compact_columns, *fifa_columns])),
    }
    folds, aggregate = run_rolling_logistic_search(
        frame,
        feature_sets,
        config=RollingSearchConfig(
            c_values=(0.1, 0.3, 1.0, 3.0),
            half_life_years=(8.0, 12.0, 20.0, None),
        ),
    )
    folds.to_csv(output_dir / "rolling_folds.csv", index=False)
    aggregate.to_csv(output_dir / "rolling_aggregate.csv", index=False)

    selected_by_set = {
        feature_set: aggregate.loc[aggregate["feature_set"] == feature_set].iloc[0]
        for feature_set in feature_sets
    }
    selected = min(selected_by_set.values(), key=lambda row: row["log_loss"])
    train = frame.loc[date_mask(frame["kickoff_utc"], end="2025-01-01")]
    test = frame.loc[
        date_mask(
            frame["kickoff_utc"],
            start="2025-01-01",
            end="2026-06-11",
        )
    ]
    probabilities = {
        feature_set: fit_candidate(
            train,
            test,
            feature_sets[feature_set],
            row,
        )
        for feature_set, row in selected_by_set.items()
    }
    holdout = pd.DataFrame(
        [
            metrics_row(
                feature_set,
                "holdout_2025_2026",
                test["outcome"].to_numpy(),
                proba,
            )
            for feature_set, proba in probabilities.items()
        ]
    ).sort_values(["log_loss", "brier", "rps"], ignore_index=True)
    holdout.to_csv(output_dir / "holdout_results.csv", index=False)

    selected_name = str(selected["feature_set"])
    predictions = prediction_table(test, probabilities[selected_name])
    predictions["outcome"] = test["outcome"].to_numpy()
    predictions["team1_goals"] = test["team1_goals"].to_numpy()
    predictions["team2_goals"] = test["team2_goals"].to_numpy()
    predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)

    coverage = {
        side: float(frame[f"{side}_fifa_points"].notna().mean())
        for side in ("team1", "team2")
    }
    selection = {
        "selected_candidate": str(selected["model"]),
        "selected_feature_set": selected_name,
        "C": float(selected["C"]),
        "half_life_years": (
            None
            if pd.isna(selected["half_life_years"])
            else float(selected["half_life_years"])
        ),
        "feature_counts": {
            name: len(columns) for name, columns in feature_sets.items()
        },
        "fifa_feature_count": len(fifa_columns),
        "fifa_coverage": coverage,
        "rolling_metrics": selected.to_dict(),
        "holdout_metrics": holdout.loc[
            holdout["model"] == selected_name
        ].iloc[0].to_dict(),
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
