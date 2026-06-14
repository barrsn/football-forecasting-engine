# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from football_forecast.features.h2h import add_h2h_features
from football_forecast.features.scorers import (
    add_prior_scorer_features,
    scorer_feature_columns,
)


def main() -> None:
    input_path = (
        PROJECT_ROOT / "data/processed/features_fifa_1990_2026-06-10.parquet"
    )
    goalscorers_path = PROJECT_ROOT / "data/processed/goalscorers.parquet"
    output_path = (
        PROJECT_ROOT / "data/processed/features_players_1990_2026-06-10.parquet"
    )
    report_path = PROJECT_ROOT / "reports/player_features/coverage.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    goalscorers = pd.read_parquet(goalscorers_path)
    frame = add_h2h_features(frame)
    frame = add_prior_scorer_features(frame, goalscorers)
    frame.to_parquet(output_path, index=False)

    scorer_columns = scorer_feature_columns(frame)
    coverage = {
        "rows": len(frame),
        "feature_count": len(scorer_columns),
        "features": scorer_columns,
        "team1_mean_complete_matches": float(
            frame["team1_scorer_complete_matches"].mean()
        ),
        "team2_mean_complete_matches": float(
            frame["team2_scorer_complete_matches"].mean()
        ),
        "team1_mean_source_coverage": float(frame["team1_scorer_coverage"].mean()),
        "team2_mean_source_coverage": float(frame["team2_scorer_coverage"].mean()),
        "rows_with_both_scorer_histories": int(
            (
                (frame["team1_scorer_history_matches"] > 0)
                & (frame["team2_scorer_history_matches"] > 0)
            ).sum()
        ),
        "output": str(output_path.relative_to(PROJECT_ROOT)),
    }
    report_path.write_text(
        json.dumps(coverage, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
