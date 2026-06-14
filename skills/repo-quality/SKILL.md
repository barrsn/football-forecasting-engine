# Skill: Repository Quality

## Required quality gates

```bash
pytest -q
python scripts/run_sample_pipeline.py
```

## Code expectations

- Package logic in `src/football_forecast`.
- Scripts are thin wrappers.
- No notebook-only logic.
- Small functions.
- Explicit random seeds.
- Clear assumptions in docs.

## Documentation expectations

Update relevant docs when behavior changes:

- `docs/DATA_CONTRACTS.md`
- `docs/MODEL_CARD_TEMPLATE.md`
- `docs/EVALUATION.md`
- `docs/IMPLEMENTATION_PLAN.md`
