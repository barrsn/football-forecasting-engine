# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from football_forecast.evaluation.model_search import (
    run_chronological_model_search,
    write_search_result,
)
from football_forecast.features.build import default_feature_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="data/processed/features_1990_2026-06-10.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/real_model_search",
    )
    parser.add_argument("--no-lightgbm", action="store_true")
    args = parser.parse_args()

    frame = pd.read_parquet(args.features)
    feature_columns = default_feature_columns(frame)
    result = run_chronological_model_search(
        frame,
        feature_columns,
        include_lightgbm=not args.no_lightgbm,
    )
    write_search_result(result, args.output_dir)

    print("Top tuning models")
    print(
        result.tuning_results.head(15).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print("\nValidation")
    print(
        result.validation_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print("\nFinal holdout")
    print(
        result.test_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print("\nSelection")
    for key, value in result.selection.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
