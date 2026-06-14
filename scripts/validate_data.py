# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from football_forecast.data.schema import coerce_matches, validate_matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    report = validate_matches(df)
    print(report)
    if not report.is_valid:
        raise SystemExit(1)
    canonical = coerce_matches(df)
    print(f"canonical_rows: {len(canonical)}")


if __name__ == "__main__":
    main()
