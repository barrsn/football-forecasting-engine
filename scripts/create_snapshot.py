# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_forecast.data.io import read_matches_csv, write_csv
from football_forecast.data.snapshot import create_snapshot, write_snapshot_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matches_csv")
    parser.add_argument("--snapshot-at", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    matches = read_matches_csv(args.matches_csv)
    snapshot, manifest = create_snapshot(matches, args.snapshot_at)
    write_csv(snapshot, args.output)
    write_snapshot_manifest(manifest, args.manifest)
    print(f"snapshot_rows: {len(snapshot)}")
    print(f"data_hash: {manifest.data_hash}")


if __name__ == "__main__":
    main()
