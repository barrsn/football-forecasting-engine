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
    run_rolling_structured_search,
)
from football_forecast.features.advanced import (
    add_advanced_context_features,
    compact_feature_columns,
)
from football_forecast.models.outcome import StructuredOutcomeModel
from football_forecast.reporting.tables import prediction_table


def main() -> None:
    base_path = PROJECT_ROOT / "data/processed/features_1990_2026-06-10.parquet"
    advanced_path = (
        PROJECT_ROOT / "data/processed/features_advanced_1990_2026-06-10.parquet"
    )
    output_dir = PROJECT_ROOT / "reports/structured_model_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = add_advanced_context_features(pd.read_parquet(base_path))
    frame.to_parquet(advanced_path, index=False)
    columns = compact_feature_columns(frame)
    folds, aggregate, predictions = run_rolling_structured_search(frame, columns)
    folds.to_csv(output_dir / "rolling_folds.csv", index=False)
    aggregate.to_csv(output_dir / "rolling_aggregate.csv", index=False)
    predictions.to_parquet(output_dir / "rolling_predictions.parquet", index=False)

    selected = aggregate.iloc[0]
    train = frame.loc[date_mask(frame["kickoff_utc"], end="2025-01-01")]
    test = frame.loc[
        date_mask(
            frame["kickoff_utc"],
            start="2025-01-01",
            end="2026-06-11",
        )
    ]
    half_life = (
        None
        if pd.isna(selected["half_life_years"])
        else float(selected["half_life_years"])
    )
    weights = recency_sample_weights(
        train["kickoff_utc"],
        "2025-01-01",
        half_life_years=half_life,
    )
    model = StructuredOutcomeModel(
        draw_c=float(selected["draw_C"]),
        decisive_c=float(selected["decisive_C"]),
    ).fit(train[columns], train["outcome"], sample_weight=weights)
    probabilities = model.predict_proba(test[columns])
    holdout = pd.DataFrame(
        [
            metrics_row(
                "structured_selected",
                "holdout_2025_2026",
                test["outcome"].to_numpy(),
                probabilities,
            )
        ]
    )
    holdout.to_csv(output_dir / "holdout_results.csv", index=False)
    holdout_predictions = prediction_table(test, probabilities)
    holdout_predictions["outcome"] = test["outcome"].to_numpy()
    holdout_predictions["team1_goals"] = test["team1_goals"].to_numpy()
    holdout_predictions["team2_goals"] = test["team2_goals"].to_numpy()
    holdout_predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)

    selection = {
        "selected_candidate": str(selected["model"]),
        "draw_C": float(selected["draw_C"]),
        "decisive_C": float(selected["decisive_C"]),
        "half_life_years": half_life,
        "feature_count": len(columns),
        "rolling_metrics": selected.to_dict(),
        "holdout_metrics": holdout.iloc[0].to_dict(),
        "holdout_was_previously_observed": True,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, default=str),
        encoding="utf-8",
    )
    print(aggregate.head(15).to_string(index=False))
    print("\nHoldout")
    print(holdout.to_string(index=False))
    print("\nSelection")
    print(json.dumps(selection, indent=2, default=str))


if __name__ == "__main__":
    main()
