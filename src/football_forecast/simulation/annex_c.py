from __future__ import annotations

from pathlib import Path

import pandas as pd

WINNER_COLUMNS = ("A", "B", "D", "E", "G", "I", "K", "L")


def load_annex_c_allocations(
    path: str | Path,
    *,
    require_complete: bool = True,
) -> dict[tuple[str, ...], dict[str, str]]:
    """Load FIFA Annex C allocations from a pinned local CSV.

    Required columns are `qualified_groups` plus one column per group winner that
    faces a third-placed team: A, B, D, E, G, I, K, and L.
    """
    frame = pd.read_csv(path, dtype=str)
    required = {"qualified_groups", *WINNER_COLUMNS}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"Missing Annex C columns: {missing}")
    allocations: dict[tuple[str, ...], dict[str, str]] = {}
    for row in frame.itertuples(index=False):
        qualified = tuple(sorted(str(row.qualified_groups).replace(" ", "")))
        if len(qualified) != 8 or len(set(qualified)) != 8:
            raise ValueError(f"Invalid qualified group combination: {row.qualified_groups}")
        assignment = {winner: str(getattr(row, winner)).replace("3", "") for winner in WINNER_COLUMNS}
        if set(assignment.values()) != set(qualified):
            raise ValueError(f"Annex C row does not assign each qualified group once: {qualified}")
        allocations[qualified] = assignment
    if len(allocations) != len(frame):
        raise ValueError("Annex C contains duplicate qualified group combinations")
    if require_complete and len(allocations) != 495:
        raise ValueError(f"Expected all 495 Annex C combinations, got {len(allocations)}")
    return allocations
