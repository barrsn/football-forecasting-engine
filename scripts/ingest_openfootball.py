# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_forecast.data.ingest_openfootball import ingest_openfootball_worldcup
from football_forecast.data.io import write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("worldcup_json")
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--snapshot-at")
    parser.add_argument("--output", default="data/processed/worldcup_fixtures.csv")
    args = parser.parse_args()

    fixtures = ingest_openfootball_worldcup(
        args.worldcup_json,
        source_version=args.source_version,
        expected_sha256=args.sha256,
        snapshot_at=args.snapshot_at,
    )
    write_csv(fixtures, args.output)
    print(f"rows: {len(fixtures)}")
    print(f"unresolved_placeholders: {int(fixtures[['unresolved_team1', 'unresolved_team2']].sum().sum())}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
