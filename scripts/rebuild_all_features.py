# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_forecast.data.io import read_matches_csv
from football_forecast.features.advanced import add_advanced_context_features
from football_forecast.features.build import build_feature_table


def main() -> None:
    matches_path = PROJECT_ROOT / "data/processed/international_matches_1990_2026-06-10.csv"
    fifa_path = PROJECT_ROOT / "data/processed/fifa_rankings_1992_2026-04-01.parquet"
    goalscorers_path = PROJECT_ROOT / "data/processed/goalscorers.parquet"
    
    out_base = PROJECT_ROOT / "data/processed/features_1990_2026-06-10.parquet"
    out_advanced = PROJECT_ROOT / "data/processed/features_advanced_1990_2026-06-10.parquet"
    out_fifa = PROJECT_ROOT / "data/processed/features_fifa_1990_2026-06-10.parquet"
    
    print("Reading matches...")
    matches = read_matches_csv(matches_path)
    
    print("Reading FIFA rankings...")
    fifa = pd.read_parquet(fifa_path) if fifa_path.exists() else None
    print("Reading player goalscorer history...")
    goalscorers = (
        pd.read_parquet(goalscorers_path) if goalscorers_path.exists() else None
    )
    
    print("Building base features...")
    base = build_feature_table(
        matches,
        windows=(5, 10, 20),
        fifa_rankings=fifa,
        goalscorers=goalscorers,
    )
    print(f"Base columns: {len(base.columns)}")
    base.to_parquet(out_base, index=False)
    
    print("Building advanced features...")
    advanced = add_advanced_context_features(base)
    advanced.to_parquet(out_advanced, index=False)
    
    advanced.to_parquet(out_fifa, index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
