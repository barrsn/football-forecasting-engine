# Skill: Tournament Simulation

## Purpose

Convert per-match probability distributions into tournament-stage probabilities.

## Rules

- Match model is separate from tournament rules.
- Random seed is explicit.
- Simulations must return distributions, not only a single bracket.
- Group ranking and knockout handling must be deterministic given sampled scores.
- Penalty shootout probabilities should be explicit. Use 50/50 only as a documented baseline.

## Outputs

- group rank probabilities
- reach R32/R16/QF/SF/final probabilities
- champion probability
- uncertainty diagnostics
