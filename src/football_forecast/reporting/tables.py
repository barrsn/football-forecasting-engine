from __future__ import annotations

import pandas as pd

from football_forecast.evaluation.probabilities import probability_frame
from football_forecast.evaluation.selective import selective_predictions


def prediction_table(
    matches: pd.DataFrame,
    proba,
    lambda1=None,
    lambda2=None,
    *,
    confidence_threshold: float | None = None,
) -> pd.DataFrame:
    columns = [column for column in ("match_id", "date", "kickoff_utc", "team1", "team2") if column in matches]
    out = matches[columns].reset_index(drop=True).copy()
    probabilities = probability_frame(proba).reset_index(drop=True)
    out = pd.concat([out, probabilities], axis=1)
    if lambda1 is not None:
        out["lambda_team1"] = lambda1
    if lambda2 is not None:
        out["lambda_team2"] = lambda2
    if confidence_threshold is not None:
        policy = selective_predictions(
            probabilities.to_numpy(),
            threshold=confidence_threshold,
        )
        out = pd.concat([out, policy], axis=1)
    return out
