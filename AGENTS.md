# Agent Instructions

You are editing a probabilistic football forecasting repository. Optimize for correctness, reproducibility, and leakage prevention.

## Hard rules

1. Never create a feature unless it is available before `match_date`.
2. Never use random train/test split for historical match forecasting.
3. Every rolling feature must use `shift(1)` or an equivalent prior-only mechanism.
4. Every prediction model must be evaluated against a simple baseline.
5. Prefer calibrated probabilities over hard labels.
6. Do not add heavy dependencies unless they are optional.
7. Keep sample tests runnable without internet.

## Primary workflow

1. Validate data schema.
2. Normalize team names.
3. Sort all matches by date.
4. Build as-of-date ratings.
5. Build prior-only rolling features.
6. Train baseline and candidate models.
7. Evaluate with Log Loss, Brier Score, and RPS.
8. Calibrate probabilities.
9. Simulate tournament only from probability outputs.
10. Write assumptions to the model card.

## Skills

Use the following local skill files before changing relevant areas:

- `skills/football-forecasting/SKILL.md`
- `skills/leakage-prevention/SKILL.md`
- `skills/model-evaluation/SKILL.md`
- `skills/tournament-simulation/SKILL.md`
- `skills/repo-quality/SKILL.md`

## Preferred style

- Small pure functions.
- Type hints where useful.
- Tests for every bug fix.
- No hidden global state.
- No notebooks as the only source of truth. Notebooks may call package code, not contain the core logic.
