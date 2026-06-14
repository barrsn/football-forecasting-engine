# Skill: Football Forecasting

## Objective

Build calibrated probabilistic football forecasts for match outcomes and scorelines.

## Correct modeling sequence

1. Define canonical match schema.
2. Generate as-of-date team ratings.
3. Generate prior-only rolling features.
4. Train baseline Elo/logistic model.
5. Train score model using Poisson regressions.
6. Train outcome model for W/D/L probabilities.
7. Calibrate probabilities on chronological validation data.
8. Blend models with validation-optimized weights.
9. Evaluate using proper scoring rules.
10. Simulate tournaments from probabilities.

## Recommended first models

- Elo baseline
- Poisson score model
- Logistic regression baseline
- HistGradientBoosting or CatBoost/LightGBM optional
- Simple convex ensemble

## Avoid

- LSTM/Transformer without high-volume event/tracking data
- random split
- team-name memorization as the main signal
- target leakage through current rankings or post-match stats
