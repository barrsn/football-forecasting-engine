# Player data contract

Player features are accepted only from timestamped snapshots that were available
strictly before kickoff. Current values must never be copied backward into
historical matches.

## Required fields

- `snapshot_id`: one complete team/squad publication.
- `player_id`, `player_name`, `team`.
- `available_at`: when the information became usable by the forecaster.
- `snapshot_at`: source observation time.
- `source`, `source_version`: immutable provenance.

## Availability and lineup

- `availability_status`: `available`, `doubtful`, `injured`, `suspended`,
  `unavailable`, or `unknown`.
- `lineup_status`: `starter`, `bench`, `not_in_squad`, or `unknown`.
- `expected_start_probability`: optional pre-match estimate in `[0, 1]`.
- `expected_minutes`: optional estimate in `[0, 130]`.

An official starting lineup can be used only if its publication time is recorded
and is earlier than the prediction cutoff. A post-match lineup is not a valid
pre-match feature.

## Quality and form

The schema supports:

- timestamped `player_rating` and uncertainty;
- international caps and goals known at the snapshot;
- prior 365-day minutes, starts, goals, assists, and cards;
- position, club, birth date, and age as of the snapshot.

Ratings from different providers must not be mixed without a documented
normalization. Market value, injury status, and club performance require
historical snapshots; a current-only table is inference documentation, not
training data.

## Team aggregates

`football_forecast.features.players` produces:

- squad, availability, and lineup coverage;
- official and expected starter counts;
- mean, top-11, available top-11, and expected-XI ratings;
- rating loss caused by unavailable players;
- caps, goals, recent minutes, starts, goals, and assists;
- age, club diversity, and position depth;
- team1-minus-team2 differences for every numeric measure.

Use `data/templates/player_snapshots.csv` as the import template and run:

```powershell
conda run -n trade310 python scripts/ingest_player_snapshots.py PATH_TO_CSV
```

The generic player snapshot layer is production-ready, but it is not included in
the core champion until a timestamped historical source covers the validation
years. The pinned scorer-event source is evaluated separately because it does not
contain appearances or injury status.
