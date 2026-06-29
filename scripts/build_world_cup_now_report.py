# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from football_forecast.world_cup.current import (
    add_world_cup_predictions,
    build_world_cup_feature_frame,
    load_world_cup_fixtures,
    merge_actual_results,
    summarize_prediction_results,
)


def main() -> None:
    metadata = json.loads(
        (PROJECT_ROOT / "models/world_cup_2026_champion.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    selective = json.loads(
        (PROJECT_ROOT / "reports/selective_accuracy/selection.json").read_text(
            encoding="utf-8"
        )
    )
    fixtures = load_world_cup_fixtures(
        PROJECT_ROOT / "data/raw/martj42_international_results/results.csv",
        mapping_path=PROJECT_ROOT / "data/mapping/team_names.yaml",
        as_of_date="2026-06-29",
    )
    historical_features = pd.read_parquet(
        PROJECT_ROOT / "data/processed/features_fifa_1990_2026-06-10.parquet"
    )
    feature_frame = build_world_cup_feature_frame(fixtures, historical_features)
    predictions = add_world_cup_predictions(
        feature_frame,
        model_path=PROJECT_ROOT / metadata["model_path"],
        feature_columns=list(metadata["feature_columns"]),
        confidence_threshold=float(selective["threshold"]),
    )
    actual_results_path = PROJECT_ROOT / "data/interim/world_cup_2026_results_current.csv"
    if actual_results_path.exists():
        predictions = merge_actual_results(
            predictions,
            actual_results_path,
            mapping_path=PROJECT_ROOT / "data/mapping/team_names.yaml",
        )

    output_dir = PROJECT_ROOT / "reports/world_cup_now"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary = {
        "competition": "FIFA World Cup 2026",
        "source": "data/raw/martj42_international_results/results.csv",
        "fixture_rows": int(len(predictions)),
        "groups": sorted(predictions["group"].dropna().unique().tolist()),
        "min_date": predictions["date"].min().isoformat(),
        "max_date": predictions["date"].max().isoformat(),
        "results_available": int(predictions["has_result"].sum()),
        "results_missing": int((~predictions["has_result"]).sum()),
        "as_of_date": "2026-06-29",
        "status_counts": predictions["status"].value_counts().to_dict(),
        "actual_results_path": (
            str(actual_results_path.relative_to(PROJECT_ROOT))
            if actual_results_path.exists()
            else None
        ),
        "actual_results_source_url": (
            predictions["source_url"].dropna().iloc[0]
            if "source_url" in predictions and predictions["source_url"].notna().any()
            else None
        ),
        "prediction_results": summarize_prediction_results(predictions),
        "model": metadata["model"],
        "model_path": metadata["model_path"],
        "confidence_threshold": float(selective["threshold"]),
        "high_confidence_picks": int(predictions["is_high_confidence"].sum()),
        "note": (
            "Fixture predictions are generated from a pre-tournament/current local "
            "feature snapshot. Actual group-stage scores are joined from the current "
            "results snapshot when available. Official knockout simulation is not "
            "inferred without an official complete bracket input."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
