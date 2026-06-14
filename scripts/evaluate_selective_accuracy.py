# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from football_forecast.evaluation.selective import (
    choose_stable_confidence_threshold,
    evaluate_selective_accuracy,
    selective_predictions,
)

PROBABILITY_COLUMNS = ["p_team2_win", "p_draw", "p_team1_win"]
LOGISTIC_WEIGHT = 0.6785109953795312
HIST_GBM_WEIGHT = 0.32148900462046875


def _champion_validation_predictions() -> tuple[pd.DataFrame, np.ndarray]:
    predictions = pd.read_parquet(
        PROJECT_ROOT / "reports/fifa_boosting_search/rolling_predictions.parquet"
    )
    logistic = predictions.loc[
        predictions["model"] == "logistic_fifa__hl20.0"
    ].sort_values(["year", "row_index"])
    hist_gbm = predictions.loc[
        predictions["model"] == "hist_gbm_small__hlNone"
    ].sort_values(["year", "row_index"])
    probabilities = (
        LOGISTIC_WEIGHT * logistic[PROBABILITY_COLUMNS].to_numpy()
        + HIST_GBM_WEIGHT * hist_gbm[PROBABILITY_COLUMNS].to_numpy()
    )
    return logistic, probabilities


def _report_row(
    split: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    year: int | None = None,
) -> dict[str, object]:
    report = evaluate_selective_accuracy(
        y_true,
        probabilities,
        threshold=threshold,
    )
    return {
        "split": split,
        "year": year,
        **report.__dict__,
    }


def main() -> None:
    validation, validation_probabilities = _champion_validation_predictions()
    threshold = choose_stable_confidence_threshold(
        validation["outcome"].to_numpy(),
        validation_probabilities,
        validation["year"].to_numpy(),
        target_accuracy=0.65,
        min_group_predictions=100,
    )

    rows = [
        _report_row(
            "pooled_rolling_origin",
            validation["outcome"].to_numpy(),
            validation_probabilities,
            threshold,
        )
    ]
    for year in sorted(validation["year"].unique()):
        mask = validation["year"].to_numpy() == year
        rows.append(
            _report_row(
                "rolling_year",
                validation.loc[mask, "outcome"].to_numpy(),
                validation_probabilities[mask],
                threshold,
                year=int(year),
            )
        )

    holdout = pd.read_csv(
        PROJECT_ROOT / "reports/champion_model/holdout_predictions.csv"
    )
    holdout_probabilities = holdout[PROBABILITY_COLUMNS].to_numpy()
    rows.append(
        _report_row(
            "holdout_2025_2026",
            holdout["outcome"].to_numpy(),
            holdout_probabilities,
            threshold,
        )
    )
    policy = selective_predictions(
        holdout_probabilities,
        threshold=threshold,
    )
    output_predictions = pd.concat(
        [holdout.reset_index(drop=True), policy],
        axis=1,
    )

    output_dir = PROJECT_ROOT / "reports/selective_accuracy"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results.csv", index=False)
    output_predictions.to_csv(
        output_dir / "holdout_predictions.csv",
        index=False,
    )
    pooled = results.loc[
        results["split"] == "pooled_rolling_origin"
    ].iloc[0]
    holdout_row = results.loc[
        results["split"] == "holdout_2025_2026"
    ].iloc[0]
    selection = {
        "policy": "high_confidence_hard_pick_with_abstention",
        "threshold": threshold,
        "target_accuracy": 0.65,
        "selection_data": "rolling validation only",
        "probabilities_unchanged": True,
        "validation": {
            "threshold": float(pooled["threshold"]),
            "coverage": float(pooled["coverage"]),
            "selective_accuracy": float(pooled["selective_accuracy"]),
            "full_accuracy": float(pooled["full_accuracy"]),
            "selected_matches": int(pooled["selected_matches"]),
            "total_matches": int(pooled["total_matches"]),
        },
        "holdout": {
            "threshold": float(holdout_row["threshold"]),
            "coverage": float(holdout_row["coverage"]),
            "selective_accuracy": float(holdout_row["selective_accuracy"]),
            "full_accuracy": float(holdout_row["full_accuracy"]),
            "selected_matches": int(holdout_row["selected_matches"]),
            "total_matches": int(holdout_row["total_matches"]),
        },
        "minimum_validation_year_accuracy": float(
            results.loc[
                results["split"] == "rolling_year",
                "selective_accuracy",
            ].min()
        ),
        "notes": (
            "Accuracy is measured only on selected high-confidence matches. "
            "All matches still receive calibrated probabilities."
        ),
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )

    metadata_path = PROJECT_ROOT / "models/world_cup_2026_champion.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["selective_hard_pick_policy"] = selection
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    print(results.to_string(index=False))
    print(json.dumps(selection, indent=2, default=str))


if __name__ == "__main__":
    main()
