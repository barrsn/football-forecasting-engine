# Codex Task Pack

Use these tasks in order. Do not skip data validation or leakage tests.

## Task 1 - Real data ingestion

Implement `src/football_forecast/data/ingest_international.py`.

Requirements:

- read Kaggle international football results CSV after manual download
- map columns to canonical schema
- normalize team names using `data/mapping/team_names.yaml`
- write `data/processed/international_matches.csv`
- add unit tests with a tiny fixture

## Task 2 - Historical ranking joins

Implement `src/football_forecast/features/asof_join.py`.

Requirements:

- support `pd.merge_asof`
- guarantee `rating_date < match_date`
- test that future ranking values are not joined

## Task 3 - Calibration report

Implement a report script that outputs per-class reliability tables and before/after Log Loss/Brier.

## Task 4 - World Cup fixture ingestion

Implement `src/football_forecast/data/ingest_openfootball.py`.

Requirements:

- parse OpenFootball worldcup.json structure
- output fixtures table with match_id, group/stage, team1, team2, date
- unresolved knockout placeholders must be marked explicitly

## Task 5 - Streamlit dashboard

Add pages:

- model evaluation
- team ratings
- match predictor
- tournament simulation
- data quality
