# Contributing

This repository is optimized for leakage-safe probabilistic forecasting.

Before opening a change:

1. Keep package logic in `src/football_forecast`.
2. Keep scripts as thin wrappers around package code.
3. Add tests for bug fixes and feature-engineering changes.
4. Do not use random train/test splits for historical match forecasting.
5. Compare every candidate model with at least one simple baseline.
6. Prefer calibrated probability outputs over hard labels.
7. Keep optional heavy dependencies behind extras.

Required local checks:

```bash
pytest -q
python scripts/run_sample_pipeline.py
```

Recommended full check:

```bash
ruff check src tests scripts app
pytest -q
python scripts/run_sample_pipeline.py
```

When adding a feature, document when the value becomes available and prove it is
known before `match_date` / `kickoff_utc`. Rolling features must exclude the
target match with `shift(1)` or an equivalent prior-only mechanism.
