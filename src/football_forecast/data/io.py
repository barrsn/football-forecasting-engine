from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_forecast.data.schema import coerce_matches


def read_matches_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return coerce_matches(df)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
