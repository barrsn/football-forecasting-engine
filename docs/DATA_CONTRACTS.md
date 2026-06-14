# Data Contracts

## Canonical match table

Required source columns:

| Column | Type | Rule |
|---|---|---|
| `date` | datetime-like | Used when `kickoff_utc` is absent |
| `team1`, `team2` | string | Canonical non-empty names; must differ |
| `team1_goals`, `team2_goals` | integer | Non-negative final score used by the task |
| `tournament` | string | Non-empty competition name |
| `neutral` | boolean-like | Accepted strings are explicit, for example `true`/`false` and `1`/`0` |

Canonical output adds:

| Column | Rule |
|---|---|
| `match_id` | Stable unique identifier |
| `kickoff_utc` | Timezone-aware UTC timestamp |
| `available_at` | Time at which the completed match became usable |
| `source`, `source_version` | Dataset identity and pinned version |
| `snapshot_at` | Forecast/data snapshot time |
| regulation/extra-time/penalty goal columns | Separate score components |

Validation rejects missing values, negative or fractional goals, duplicate matches,
identical opponents, and a team appearing in two matches at the same kickoff.

## Feature contract

Every feature must satisfy:

```text
feature_available_at < kickoff_utc
```

Equal-kickoff matches are processed as one batch. Neither Elo nor rolling features
may consume another result from the same kickoff. External ratings use strict
backward `merge_asof` joins with exact timestamp matches disabled.

## Outcome and probability contract

Internal outcome values are:

```text
0 = team2 win
1 = draw
2 = team1 win
```

Public outputs never rely on this positional order. They expose:

```text
p_team2_win
p_draw
p_team1_win
```

Every probability must be finite, within `[0, 1]`, and each row must sum to one.

## Snapshot contract

Every forecasting snapshot records:

- UTC cutoff timestamp
- canonical data hash
- date range and number of matches
- source versions
- model parameters
- feature column list

Only rows with `available_at < snapshot_at` enter a snapshot.
