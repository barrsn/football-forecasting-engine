# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_forecast.data.ingest_international import ingest_international_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--snapshot-at")
    parser.add_argument("--mapping", default="data/mapping/team_names.yaml")
    parser.add_argument("--output", default="data/processed/international_matches.csv")
    args = parser.parse_args()

    matches, report = ingest_international_results(
        args.input_csv,
        mapping_path=args.mapping,
        source_version=args.source_version,
        expected_sha256=args.sha256,
        snapshot_at=args.snapshot_at,
        output_path=args.output,
    )
    print(f"rows: {len(matches)}")
    print(f"output: {args.output}")
    print(f"unresolved_team_names: {len(report.unresolved_names)}")
    for name in report.unresolved_names:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
