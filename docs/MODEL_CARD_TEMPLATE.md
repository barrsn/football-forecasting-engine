# Model Card

## Identity

- Model name:
- Version:
- Snapshot timestamp:
- Data hash:
- Source versions:

## Intended use

Probabilistic pre-match forecasting and World Cup tournament simulation.

Not intended for guaranteed betting returns or deterministic claims.

## Training and evaluation data

- Date range:
- Number of matches:
- Team universe:
- Exclusions:
- Validation folds:
- Final holdout:

## Features and availability

List every feature with its source timestamp. Confirm:

- all rolling features exclude the target match
- equal-kickoff matches are batched
- all rating joins are strictly backward
- every feature satisfies `available_at < kickoff_utc`

## Metrics

| Model/Split | Log Loss | Brier | RPS | ECE | Accuracy |
|---|---:|---:|---:|---:|---:|
| strongest baseline / validation | | | | | |
| candidate / validation | | | | | |
| strongest baseline / holdout | | | | | |
| candidate / holdout | | | | | |

## Calibration and ensemble

- Calibration method and sample size:
- Before/after metrics:
- Ensemble members and weights:
- Promotion checks:

## Tournament assumptions

- Annex C source/version/hash:
- Extra-time method:
- Penalty probability:
- Conduct-score handling:
- Number of simulations and seed:

## Limitations

- international football has limited samples
- player availability may be unavailable
- partial event data must not be treated as universal coverage
- probabilities depend on the snapshot and model assumptions

## Release decision

`APPROVED`, `REJECTED`, or `NEEDS CHANGES`, with evidence.
