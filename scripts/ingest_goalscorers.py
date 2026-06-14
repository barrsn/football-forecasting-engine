# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_forecast.data.ingest_goalscorers import (
    download_pinned_goalscorers,
    ingest_goalscorers,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--snapshot-at", required=True)
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data/raw/martj42_international_results"
    raw_path = raw_dir / "goalscorers.csv"
    manifest_path = raw_dir / "source_manifest.json"
    source = download_pinned_goalscorers(
        commit=args.commit,
        raw_path=raw_path,
        manifest_path=manifest_path,
    )
    output = PROJECT_ROOT / "data/processed/goalscorers.parquet"
    frame, report = ingest_goalscorers(
        raw_path,
        mapping_path=PROJECT_ROOT / "data/mapping/team_names.yaml",
        source_version=args.commit,
        expected_sha256=str(source["sha256"]),
        snapshot_at=args.snapshot_at,
        output_path=output,
    )
    print(f"rows: {len(frame)}")
    print(f"sha256: {source['sha256']}")
    print(f"unresolved_team_names: {len(report.unresolved_names)}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
