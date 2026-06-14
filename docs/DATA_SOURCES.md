# Data Sources

Last reviewed: 2026-06-14.

## Authority order

1. FIFA official snapshots for fixtures, rankings, squads, regulations, and Annex C.
2. Pinned open repositories for historical results and fixture mirrors.
3. Other public datasets only after schema, timestamp, and licensing review.

## Historical results

The supported ingestion schema is the Mart Jürisoo international-results dataset.
Download it manually, record the exact Git commit and SHA-256, then run:

```bash
python scripts/ingest_international.py results.csv \
  --source-version <commit-sha> \
  --sha256 <file-sha256>
```

Tests never download this dataset.

## FIFA rankings

Historical snapshots must include `team`, `rating_date`, and a rating/rank field.
Joins are strict: `rating_date < kickoff_utc`. A current ranking must never be
backfilled into historical matches.

## World Cup fixtures

OpenFootball `worldcup.json` is a CC0 mirror and is parsed locally. It is not the
authority. Before use, compare its match IDs, UTC kickoffs, teams, groups, and
stages with a pinned FIFA snapshot.

## FIFA Annex C

Production tournament simulation requires a local 495-row CSV with:

```text
qualified_groups,A,B,D,E,G,I,K,L
```

Each winner column contains the third-place group assigned to that group winner.
The file publication date and SHA-256 belong in the snapshot manifest/model card.

## Richer features

The pinned Mart Jürisoo commit also supplies `goalscorers.csv`. The repository
stores its SHA-256, normalizes participating team names, and derives only
prior-match scorer-threat features. Source completeness is measured explicitly;
missing scorer events are not treated as evidence that a team has no quality.

Full player snapshots use the contract in `docs/PLAYER_DATA.md`. Squad status,
injuries, suspensions, ratings, expected starts/minutes, and official lineups
are accepted only with a recorded `available_at` timestamp before kickoff.
Current-only ratings or availability must not be backfilled into historical
training rows.

The scorer-event source does not contain complete appearances, minutes, injury
history, or lineup publication timestamps. For that reason the optional
player-scorer model is evaluated separately and is not the production champion.
