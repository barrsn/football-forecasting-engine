# Model Card: International Outcome Model 0.4.0

## Release decision

`PROMOTED FOR MATCH FORECASTING`

The W/D/L model passed the repository's validation and holdout promotion
thresholds. Tournament publication still requires a pinned official fixture
snapshot and final official bracket inputs.

## Data

- Match source: Mart Jürisoo international results
- Match commit: `c636851f6e388d7aabd1feabbd4dad94e7e6e266`
- Match SHA-256: `50f17eb331a3d8367184f3314cf41616782f842ebb39e42191260b414b56bc78`
- Official rating source: FIFA/Coca-Cola Men's World Ranking API
- Player scoring source: Mart Jürisoo `goalscorers.csv`
- Player scoring SHA-256: `6a0984888333d3c67ea16966d34d964bac422857d361e8439ea4371e8faff52d`
- FIFA page build: `0-ZXhkYnWdNxG_JfZhYJx`
- FIFA page SHA-256: `469a686a4bd93648bf6312251faea1f63870f8899a4f74be93c9ac16f4919f32`
- FIFA snapshots: 346
- FIFA ranking rows: 70,215
- FIFA names unmatched to results: 0
- Training cutoff: June 11, 2026 00:00 UTC, exclusive
- Completed matches: 32,252
- Teams represented: 326

Every FIFA feature uses the most recent snapshot satisfying
`published_at < kickoff_utc`. Unknown historical publication times become
eligible on the next UTC day.

## Chronological protocol

- Expanding rolling-origin validation years: 2018, 2021, 2022, 2023, 2024
- Pooled validation matches: 5,297
- Fixed holdout: January 1, 2025 through June 10, 2026
- Holdout matches: 1,308
- Production refit: all completed matches before June 11, 2026

No random split was used. The holdout had been observed during earlier
repository iterations; model selection in version 0.4.0 used historical
rolling-origin predictions and records this limitation explicitly.

## Selected model

Core-dependency ensemble:

- 67.85% multinomial logistic regression
- 32.15% `HistGradientBoostingClassifier`
- 114 prior-only features
- 86 Elo, form, attack/defence, rest, venue, and context features
- 28 official FIFA rating and confederation features
- no team-name identity feature
- no post-match, future-ranking, player-value, or odds feature

The ensemble weights were optimized on pooled out-of-fold historical
predictions. LightGBM was evaluated but not required by the production model.

## Player feature evaluation

Version 0.4.0 now includes a timestamped player-snapshot contract for squad
selection, injuries/suspensions, official lineups, expected starts/minutes,
ratings, caps, goals, recent minutes/form, age, club, and positional depth.
Every snapshot must satisfy `available_at < kickoff_utc`.

The pinned historical source contains 47,601 scorer events but does not contain
all appearances, minutes, injuries, or lineup publication timestamps. It was
therefore evaluated as 27 explicitly named scorer-threat and source-coverage
features, not as a complete player rating.

| Split / Model | Log Loss | Brier | RPS | Accuracy |
|---|---:|---:|---:|---:|
| Rolling player + optional LightGBM blend | 0.8662 | 0.5090 | 0.1684 | 60.54% |
| Holdout player + optional LightGBM blend | 0.8169 | 0.4798 | 0.1568 | 62.00% |

The candidate improved rolling Log Loss by only 0.11% versus the current
champion and was worse on holdout Log Loss. It remains available as
`models/world_cup_2026_player_scorer.joblib` but was not promoted. The generic
squad/availability features will enter model selection only when a consistent
historical snapshot source covers the validation years.

## Metrics

| Split / Model | Log Loss | Brier | RPS | Accuracy | ECE |
|---|---:|---:|---:|---:|---:|
| Rolling champion | 0.8671 | 0.5094 | 0.1685 | 60.68% | 0.0090 |
| Rolling no-FIFA baseline | 0.8764 | 0.5153 | 0.1713 | 60.19% | 0.0163 |
| Holdout champion | 0.8153 | 0.4787 | 0.1563 | 61.93% | 0.0254 |
| Holdout no-FIFA baseline | 0.8325 | 0.4887 | 0.1609 | 61.31% | 0.0279 |
| Holdout Elo logistic | 0.8550 | 0.5031 | 0.1669 | 60.09% | 0.0257 |
| Holdout class prior | 1.0445 | 0.6292 | 0.2272 | 48.78% | 0.0024 |

Promotion checks:

- Rolling Log Loss improvement: 1.06%
- Holdout Log Loss improvement: 2.07%
- Holdout Brier improvement: 2.03%
- Holdout RPS improvement: 2.85%

## High-confidence accuracy

The probability model still scores every match. A separate hard-pick policy
uses a confidence threshold selected only from rolling validation and abstains
when `max(probability) < 0.50`.

| Split | Selected accuracy | Coverage | Full accuracy |
|---|---:|---:|---:|
| Rolling validation | 69.55% | 66.02% | 60.68% |
| Holdout 2025-2026 | 73.19% | 65.60% | 61.93% |

The minimum selected accuracy in any validation year was 65.07%. Accuracy above
65% therefore applies to high-confidence hard picks, not to all matches. Log
Loss, Brier, RPS, and tournament simulation continue to use the unchanged
probabilities for every match.

## Artifact

- Model: `models/world_cup_2026_champion.joblib`
- Model SHA-256: `a9aac6e5d7a745c353b5e8b7db88689ff47ce8f29141e918f04098ebe99d1c5f`
- Metadata: `models/world_cup_2026_champion.metadata.json`
- Executed notebook: `notebooks/world_cup_2026_real_models.ipynb`
- Executed player notebook: `notebooks/player_features_model_evaluation.ipynb`
- Evaluation artifacts: `reports/champion_model/`
- Environment: Conda `trade310`, Python 3.10

The package must be installed or `src` must be on `PYTHONPATH` before loading
the joblib artifact.

## Remaining limitations

- Many historical records have date-level rather than exact kickoff times.
- FIFA coverage is approximately 90%; non-FIFA teams rely on other features.
- Player availability and lineup features are implemented but not promoted
  without timestamped historical coverage. Current-only squad data is not
  backfilled into training.
- Odds remain an external benchmark and are not available in this snapshot.
- Final World Cup fixtures and official bracket inputs still need a pinned
  tournament snapshot before publishing tournament probabilities.
