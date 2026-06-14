from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import pandas as pd
import yaml


@dataclass(frozen=True)
class TeamNormalizationReport:
    total_values: int
    alias_matches: int
    unresolved_names: tuple[str, ...]


def _normalize_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return re.sub(r"\s+", " ", text).casefold()


def load_team_aliases(path: str | Path) -> dict[str, str]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    aliases: dict[str, str] = {}
    for canonical, source_aliases in raw.items():
        all_names = [canonical, *(source_aliases or [])]
        for alias in all_names:
            key = _normalize_key(alias)
            previous = aliases.get(key)
            if previous is not None and previous != canonical:
                raise ValueError(f"Team alias {alias!r} maps to both {previous!r} and {canonical!r}")
            aliases[key] = str(canonical).strip()
    return aliases


def normalize_team_series(
    series: pd.Series,
    aliases: dict[str, str],
    *,
    strict: bool = False,
) -> tuple[pd.Series, TeamNormalizationReport]:
    if series.isna().any():
        raise ValueError("Team names contain missing values")

    resolved: list[str] = []
    unresolved: set[str] = set()
    alias_matches = 0
    for value in series.astype(str):
        stripped = unicodedata.normalize("NFKC", value).strip()
        canonical = aliases.get(_normalize_key(stripped))
        if canonical is None:
            unresolved.add(stripped)
            canonical = stripped
        else:
            alias_matches += int(canonical != stripped)
        resolved.append(canonical)

    if strict and unresolved:
        raise ValueError(f"Unresolved team names: {sorted(unresolved)}")
    report = TeamNormalizationReport(
        total_values=len(series),
        alias_matches=alias_matches,
        unresolved_names=tuple(sorted(unresolved)),
    )
    return pd.Series(resolved, index=series.index, dtype="object"), report


def normalize_match_teams(
    matches: pd.DataFrame,
    mapping_path: str | Path,
    *,
    strict: bool = False,
) -> tuple[pd.DataFrame, TeamNormalizationReport]:
    aliases = load_team_aliases(mapping_path)
    out = matches.copy()
    combined = pd.concat([out["team1"], out["team2"]], ignore_index=True)
    normalized, report = normalize_team_series(combined, aliases, strict=strict)
    split = len(out)
    out["team1"] = normalized.iloc[:split].to_numpy()
    out["team2"] = normalized.iloc[split:].to_numpy()
    return out, report
