# Implementation Status

## Implemented

- strict canonical schema and boolean/score validation
- centralized team-name normalization with unresolved-name reporting
- pinned local ingestion with commit/version and SHA-256 checks
- strict historical rating joins
- reproducible data snapshots and manifests
- equal-kickoff-safe Elo and rolling features
- time decay, opponent-adjusted form, rest, host, and importance features
- class-prior, Elo logistic, Poisson, Dixon-Coles, logistic, and HistGBM models
- optional CatBoost and LightGBM adapters
- temperature, sigmoid, and guarded isotonic calibration
- chronological folds, tournament replays, holdout selection rules, and slices
- named probability interfaces and validation
- full 48-team tournament engine, extra time, penalties, and Monte Carlo error
- official match-number bracket and Annex C loader
- CI for Python 3.10, 3.11, and 3.12
- pinned real-data preparation and audit through June 10, 2026
- executed chronological model-search notebook
- real holdout report for 2025 through June 10, 2026

## External-data gates

The repository does not bundle mutable internet datasets. Before a production
forecast:

1. Download and pin the historical results commit.
2. Record the file SHA-256 and ingest it.
3. Add historical FIFA ranking snapshots.
4. Verify OpenFootball fixtures against the official FIFA snapshot.
5. Load all 495 official FIFA Annex C allocations.
6. Run annual and tournament replay backtests.
7. Run the untouched 2025-2026 holdout.
8. Approve the model card only if promotion thresholds pass.

## Deferred

- injuries and player market values until timestamped historical coverage exists
- cards and corners models
- live in-match forecasting
- production dashboard expansion
- scheduled refresh, registry, and data-versioning service
