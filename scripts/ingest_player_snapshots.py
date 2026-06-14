# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from football_forecast.data.players import coerce_player_snapshots
from football_forecast.data.teams import (
    load_team_aliases,
    normalize_team_series,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument(
        "--output",
        default="data/processed/player_snapshots.parquet",
    )
    parser.add_argument(
        "--report",
        default="reports/player_features/snapshot_quality.json",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    aliases = load_team_aliases(PROJECT_ROOT / "data/mapping/team_names.yaml")
    frame["team"], team_report = normalize_team_series(frame["team"], aliases)
    snapshots, report = coerce_player_snapshots(frame)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(output_path, index=False)
    quality = {
        **report.__dict__,
        "unresolved_team_names": list(team_report.unresolved_names),
        "output": str(output_path.relative_to(PROJECT_ROOT)),
    }
    report_path = PROJECT_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(quality, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
