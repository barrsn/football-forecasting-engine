# Skill: Leakage Prevention

## Leakage checklist

Before adding any feature, answer:

1. Was this value known before kickoff?
2. Is its timestamp earlier than match date?
3. Was it computed with the target match excluded?
4. Was it computed without future matches from the same team?
5. Is the train/test split chronological?

## Required patterns

Use:

```python
df = df.sort_values("date")
feature = groupby_obj["value"].shift(1).rolling(window).mean()
```

Never use:

```python
groupby_obj["value"].rolling(window).mean()
```

unless you have already shifted/excluded the current row.

## Tests

Every rolling feature implementation needs a unit test that manually checks the first two matches of a team.
