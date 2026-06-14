# Skill: Model Evaluation

## Primary metrics

- Log Loss: main probability metric.
- Multiclass Brier Score: probability calibration quality.
- Ranked Probability Score: football W/D/L ordered outcome metric.
- Accuracy: secondary only.

## Required baselines

Every candidate model must be compared with at least one baseline:

- majority class
- Elo bucket probability model
- logistic regression on Elo difference
- bookmaker implied probabilities if available and legal to use

## Validation

Use chronological splits or walk-forward validation.

Do not use random split.
