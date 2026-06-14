from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_at: str
    data_hash: str
    n_matches: int
    data_start: str | None
    data_end: str | None
    feature_columns: tuple[str, ...]
    model_parameters: dict[str, Any]
    source_versions: tuple[str, ...]


def dataframe_sha256(df: pd.DataFrame) -> str:
    canonical_frame = df.sort_index(axis=1).astype("string").fillna("<NA>")
    canonical_frame = canonical_frame.sort_values(
        list(canonical_frame.columns), kind="mergesort"
    )
    canonical = canonical_frame.to_csv(
        index=False, date_format="%Y-%m-%dT%H:%M:%S.%f%z"
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def create_snapshot(
    matches: pd.DataFrame,
    snapshot_at: str | pd.Timestamp,
    *,
    feature_columns: tuple[str, ...] = (),
    model_parameters: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, SnapshotManifest]:
    cutoff = pd.to_datetime(snapshot_at, utc=True)
    if "available_at" not in matches:
        raise ValueError("Snapshot input must contain available_at")
    available = pd.to_datetime(matches["available_at"], utc=True)
    snapshot = matches.loc[available < cutoff].copy()
    snapshot["snapshot_at"] = cutoff

    dates = pd.to_datetime(snapshot.get("kickoff_utc"), utc=True)
    versions = tuple(sorted(snapshot.get("source_version", pd.Series(dtype=str)).dropna().unique()))
    manifest = SnapshotManifest(
        snapshot_at=cutoff.isoformat(),
        data_hash=dataframe_sha256(snapshot),
        n_matches=len(snapshot),
        data_start=None if snapshot.empty else dates.min().isoformat(),
        data_end=None if snapshot.empty else dates.max().isoformat(),
        feature_columns=tuple(feature_columns),
        model_parameters=dict(model_parameters or {}),
        source_versions=versions,
    )
    return snapshot.reset_index(drop=True), manifest


def write_snapshot_manifest(manifest: SnapshotManifest, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
