# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from football_forecast.data.io import read_matches_csv
from football_forecast.evaluation.backtesting import (
    annual_rolling_origin_folds,
    evaluate_probabilities,
    replay_folds,
)
from football_forecast.evaluation.baselines import ClassPriorBaseline, EloLogisticBaseline
from football_forecast.features.build import build_feature_table, default_feature_columns
from football_forecast.models.outcome import OutcomeModel
from football_forecast.models.poisson import TwoPoissonScoreModel


def _evaluate_fold(
    frame: pd.DataFrame,
    train_index,
    test_index,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    train = frame.iloc[train_index]
    test = frame.iloc[test_index]
    y_test = test["outcome"].to_numpy()
    predictions = {}

    prior = ClassPriorBaseline().fit(train["outcome"])
    predictions["class_prior"] = prior.predict_proba(len(test))
    predictions["elo_logistic"] = EloLogisticBaseline().fit(
        train, train["outcome"]
    ).predict_proba(test)
    for model_type in ("logistic", "hist_gbm"):
        predictions[model_type] = (
            OutcomeModel(model_type=model_type)
            .fit(train[feature_columns], train["outcome"])
            .predict_proba(test[feature_columns])
        )
    predictions["poisson"] = (
        TwoPoissonScoreModel()
        .fit(
            train[feature_columns],
            train["team1_goals"],
            train["team2_goals"],
        )
        .predict_outcome_proba(test[feature_columns])
    )
    rows = []
    for model_name, proba in predictions.items():
        metrics = evaluate_probabilities(y_test, proba)
        rows.append({"model": model_name, **metrics.__dict__})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matches_csv")
    parser.add_argument("--first-test-year", type=int, default=2014)
    parser.add_argument("--output", default="reports/backtest_metrics.csv")
    args = parser.parse_args()

    matches = read_matches_csv(args.matches_csv)
    frame = build_feature_table(matches)
    feature_columns = default_feature_columns(frame)
    folds = annual_rolling_origin_folds(
        frame["kickoff_utc"], first_test_year=args.first_test_year
    )
    folds.extend(replay_folds(frame["kickoff_utc"]))
    rows = []
    for fold in folds:
        for result in _evaluate_fold(
            frame, fold.train_index, fold.test_index, feature_columns
        ):
            rows.append({"fold": fold.name, **result})
    if not rows:
        raise SystemExit("No non-empty backtest folds were generated")
    report = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(report.to_string(index=False))
    print(f"\noutput: {output}")


if __name__ == "__main__":
    main()
