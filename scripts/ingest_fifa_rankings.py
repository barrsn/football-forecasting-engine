# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from football_forecast.data.fifa_rankings import (
    extract_ranking_schedules,
    parse_ranking_snapshot,
)
from football_forecast.data.teams import load_team_aliases

PAGE_URL = "https://inside.fifa.com/fifa-world-ranking/men?dateId=id14338"
API_TEMPLATE = (
    "https://api.fifa.com/api/v3/fifarankings/rankings/rankingsbyschedule"
    "?rankingScheduleId={schedule_id}&language=en"
)


def download(url: str, *, attempts: int = 4) -> bytes:
    request = Request(url, headers={"User-Agent": "football-forecasting-engine/0.3"})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="1992-01-01")
    parser.add_argument("--end-date", default="2026-06-11")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data/raw/fifa_rankings"
    snapshots_dir = raw_dir / "snapshots"
    processed_path = (
        PROJECT_ROOT / "data/processed/fifa_rankings_1992_2026-04-01.parquet"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    page_bytes = download(PAGE_URL)
    schedules, build_id = extract_ranking_schedules(page_bytes.decode("utf-8"))
    start = pd.Timestamp(args.start_date, tz="UTC")
    end = pd.Timestamp(args.end_date, tz="UTC")
    schedules = [
        schedule
        for schedule in schedules
        if start <= schedule.published_at < end
    ]
    aliases = load_team_aliases(PROJECT_ROOT / "data/mapping/team_names.yaml")
    frames = []
    manifest_snapshots = []
    unresolved: set[str] = set()

    for index, schedule in enumerate(schedules, start=1):
        raw_path = snapshots_dir / f"{schedule.schedule_id}.json.gz"
        if raw_path.exists():
            with gzip.open(raw_path, "rb") as handle:
                payload_bytes = handle.read()
        else:
            payload_bytes = download(
                API_TEMPLATE.format(schedule_id=schedule.schedule_id)
            )
            with gzip.open(raw_path, "wb", compresslevel=9) as handle:
                handle.write(payload_bytes)
            time.sleep(args.sleep_seconds)
        payload = json.loads(payload_bytes.decode("utf-8-sig"))
        frame, snapshot_unresolved = parse_ranking_snapshot(
            payload,
            schedule,
            aliases,
        )
        frames.append(frame)
        unresolved.update(snapshot_unresolved)
        manifest_snapshots.append(
            {
                "schedule_id": schedule.schedule_id,
                "ranking_date": schedule.ranking_date,
                "published_at": schedule.published_at.isoformat(),
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "rows": len(frame),
                "path": str(raw_path.relative_to(PROJECT_ROOT)),
            }
        )
        if index % 25 == 0 or index == len(schedules):
            print(f"Processed {index}/{len(schedules)} FIFA ranking snapshots")

    rankings = pd.concat(frames, ignore_index=True).sort_values(
        ["rating_date", "fifa_rank", "team"]
    )
    match_features = pd.read_parquet(
        PROJECT_ROOT / "data/processed/features_1990_2026-06-10.parquet",
        columns=["team1", "team2"],
    )
    match_teams = set(match_features["team1"]) | set(match_features["team2"])
    unmatched_to_results = sorted(set(rankings["team"]) - match_teams)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_parquet(processed_path, index=False)
    manifest = {
        "source": "FIFA/Coca-Cola Men's World Ranking",
        "page_url": PAGE_URL,
        "api_template": API_TEMPLATE,
        "page_build_id": build_id,
        "page_sha256": hashlib.sha256(page_bytes).hexdigest(),
        "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "start_date": args.start_date,
        "end_date_exclusive": args.end_date,
        "snapshot_count": len(manifest_snapshots),
        "row_count": len(rankings),
        "source_names_without_explicit_alias": sorted(unresolved),
        "unmatched_to_results": unmatched_to_results,
        "processed_path": str(processed_path.relative_to(PROJECT_ROOT)),
        "processed_sha256": hashlib.sha256(processed_path.read_bytes()).hexdigest(),
        "snapshots": manifest_snapshots,
    }
    (raw_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(processed_path)
    print(f"Rows: {len(rankings)}")
    print(f"Unmatched to results: {unmatched_to_results}")


if __name__ == "__main__":
    main()
