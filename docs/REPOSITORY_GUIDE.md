# Repository Guide

## Public Files

Commit source code, tests, small sample data, metadata, model cards, and selected
evaluation outputs that are needed to understand the release.

Do not commit local raw data, processed feature snapshots, binary model files,
cache directories, HTML notebook exports, or ad hoc search outputs.

## Artifact Policy

The public repository should contain:

- `models/*.metadata.json` with feature names, hashes, metrics, and assumptions
- `reports/MODEL_CARD.md` with release evidence and limitations
- compact CSV/JSON report outputs when they are referenced by docs or the app
- sample data that keeps tests runnable without internet

The public repository should not contain:

- `data/raw/`, `data/interim/`, or `data/processed/`
- `*.joblib` model binaries
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`
- `reports/current_model_search/` or other scratch searches
- notebook HTML exports

## Release Checklist

1. `ruff check src tests scripts app`
2. `pytest -q`
3. `python scripts/run_sample_pipeline.py`
4. Update `reports/MODEL_CARD.md` when metrics, data cutoffs, or promotion
   decisions change.
5. Update README files when user-facing commands or headline metrics change.
6. Verify that every new feature is available before kickoff.

## GitHub Presentation

Keep `README.md` as the primary project overview, `README_HE.md` as the Hebrew
summary, and `reports/MODEL_CARD.md` as the evidence record. The GitHub default
branch should stay runnable from sample data without internet access.
