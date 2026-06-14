# Football Forecasting Engine

A leakage-safe probabilistic forecasting pipeline for international football and
the 2026 World Cup.

## Implemented

- strict canonical data validation and team-name normalization
- pinned local ingestion, checksums, and reproducible snapshots
- equal-kickoff-safe Elo and prior-only rolling features
- class-prior, Elo, logistic, Poisson, Dixon-Coles, and HistGBM models
- optional CatBoost and LightGBM
- temperature/sigmoid calibration and validation-optimized ensembles
- rolling-origin backtesting, tournament replays, and promotion rules
- named W/D/L probabilities, expected goals, and score matrices
- 48-team World Cup simulation with group tie-breakers, extra time, penalties,
  stage probabilities, and Monte Carlo error
- Python 3.10-3.12 CI and offline tests
- 346 checksummed official FIFA ranking snapshots with strict as-of joins
- timestamped player/squad/availability/lineup contract with strict as-of joins
- 47,601 pinned scorer events and 27 leakage-safe player-threat features
- promoted core-dependency logistic + HistGBM production ensemble

## Important status

The executed real-data notebook is:

```text
notebooks/world_cup_2026_real_models.ipynb
```

It uses 32,252 completed matches and 70,215 official FIFA ranking rows through
June 10, 2026. The promoted ensemble achieved holdout Log Loss 0.8153, Brier
0.4787, RPS 0.1563, and 61.9% accuracy on 1,308 matches from 2025 through
June 10, 2026. Read
`reports/MODEL_CARD.md` for the release limitations.

The player-aware candidate notebook is
`notebooks/player_features_model_evaluation.ipynb`. Its optional LightGBM blend
improved rolling validation slightly but did not beat the core champion on the
final holdout, so it was not promoted.

For users who need hard picks, the validated high-confidence policy predicts
only when maximum probability is at least 0.50. It achieved 69.55% rolling
accuracy and 73.19% holdout accuracy at roughly 66% coverage. Lower-confidence
matches are explicitly marked `abstain`; all matches still receive W/D/L
probabilities.

Production simulation requires a pinned official FIFA fixture snapshot and all
495 Annex C third-place allocations. It will not silently substitute an
approximate bracket.

## Local environment

The verified local environment is `python`:

```powershell
 python -m pytest -q
 ruff check src tests scripts app
 python scripts/run_sample_pipeline.py
 jupyter nbconvert --execute --inplace \
  notebooks/world_cup_2026_real_models.ipynb
```

Generic setup:

```bash
python -m venv .venv
pip install -e ".[dev]"
pytest -q
python scripts/run_sample_pipeline.py
```

## Real-data workflow

1. Run `scripts/ingest_international.py` for the pinned results source.
2. Run `scripts/ingest_fifa_rankings.py` for official ranking snapshots.
3. Run `scripts/ingest_goalscorers.py` for the matching pinned scorer source.
4. Optionally ingest timestamped squads with `scripts/ingest_player_snapshots.py`.
5. Build strict prior-only features and frozen snapshots.
6. Run the rolling-origin model searches.
7. Run `scripts/finalize_champion_model.py`.
8. Execute the reporting notebooks.
9. Load the official 495-row Annex C CSV before tournament simulation.

Core methodology and contracts are in `docs/`.
