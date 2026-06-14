# Evaluation

## Objective

Select calibrated football probability models, not winner classifiers.

## Required baselines

Every report includes:

- smoothed class-prior probabilities
- logistic regression on pre-match Elo/context
- independent Poisson score model
- regularized logistic regression on the full feature set

Candidates include Dixon-Coles, HistGradientBoosting, and optional
CatBoost/LightGBM. Odds are an external benchmark only.

## Chronological protocol

Development uses annual rolling-origin folds and frozen replays:

- World Cup 2014
- World Cup 2018
- World Cup 2022
- major 2024 tournaments

The final untouched holdout is:

```text
2025-01-01 <= kickoff_utc < 2026-06-11
```

The pre-tournament snapshot and daily-refresh mode are evaluated separately.
Random splitting is forbidden.

## Metrics

Primary:

- Log Loss

Required secondary:

- multiclass Brier Score
- Ranked Probability Score
- expected calibration error
- reliability tables
- sharpness
- accuracy

Reports also slice neutral matches, draws, low Elo gaps, and high Elo gaps.

## Calibration and ensemble

Temperature scaling is the default. Sigmoid calibration is a comparison.
Isotonic calibration requires at least 1,000 calibration matches.

Calibration and ensemble weights are fitted only on chronological validation or
out-of-fold probabilities. They are not promoted unless they improve validation
and holdout performance.

## Promotion rule

A candidate must:

1. Improve Log Loss by at least 1% over the strongest non-market baseline.
2. Avoid degrading Brier or RPS by more than 1%.
3. Pass the rule on validation and the untouched holdout.

Models within 0.5% Log Loss of each other are resolved in favor of the simpler
model.
