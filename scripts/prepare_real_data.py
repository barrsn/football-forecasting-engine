# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_forecast.data.prepare import prepare_completed_international_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_csv")
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--start-date", default="1990-01-01")
    parser.add_argument("--cutoff", default="2026-06-11")
    parser.add_argument(
        "--output",
        default="data/processed/international_matches_1990_2026-06-10.csv",
    )
    parser.add_argument(
        "--audit",
        default="reports/data_audit_1990_2026-06-10.json",
    )
    args = parser.parse_args()

    _, audit = prepare_completed_international_results(
        args.source_csv,
        source_version=args.source_version,
        expected_sha256=args.sha256,
        start_date=args.start_date,
        cutoff=args.cutoff,
        output_path=args.output,
        audit_path=args.audit,
    )
    for key, value in audit.__dict__.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
