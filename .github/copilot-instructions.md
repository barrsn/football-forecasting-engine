# Copilot Instructions

This repo forecasts football outcomes probabilistically. Generate code that is chronological, leakage-safe, and testable.

When implementing feature engineering:

- sort by `date`
- group by team
- use prior matches only
- include tests proving no future data enters features

When implementing models:

- expose `fit`, `predict_proba`, and `predict_score_matrix` where applicable
- add baseline comparison
- optimize probability metrics, not accuracy only

When implementing tournament simulation:

- keep match prediction separate from tournament rules
- make random seed explicit
- return stage probabilities, not only one simulated winner
