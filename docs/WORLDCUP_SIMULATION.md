# World Cup 2026 Simulation

## Inputs

- the 72 official group fixtures
- a score-matrix provider for arbitrary team matchups
- latest and progressively older FIFA rankings where required
- optional projected conduct scores
- all 495 FIFA Annex C third-place allocations
- explicit seed and simulation count

## Group ranking

Teams tied on points are ranked by:

1. mini-table points
2. mini-table goal difference
3. mini-table goals scored
4. reapplication to teams still tied
5. overall goal difference
6. overall goals scored
7. conduct score
8. FIFA ranking

Third-placed teams are ranked by overall points, goal difference, goals scored,
conduct score, and FIFA ranking.

## Knockout stage

The bracket uses official match numbers 73 through 104. The eight qualifying
third-place groups are looked up in FIFA Annex C. Production simulation refuses
to run without all 495 mappings. A deterministic valid fallback exists only for
tests and development and is labelled in the output.

Regulation scores come from the match score matrix. A regulation draw receives a
30-minute Poisson extension using one-third of regulation expected goals. If the
match remains tied, the version-one penalty baseline is 50/50.

## Outputs

- probability of each group rank
- probability of reaching R32, R16, QF, SF, final, and winning
- per-probability Monte Carlo standard error
- allocation mode used

The default production run is 100,000 simulations with seed 42.
