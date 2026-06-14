import pandas as pd
import pytest

from football_forecast.data.schema import OutcomeClass, add_outcome, coerce_matches


def _matches(**overrides):
    data = {
        "date": ["2020-01-01", "2020-01-02"],
        "team1": ["A", "C"],
        "team2": ["B", "D"],
        "team1_goals": [1, 0],
        "team2_goals": [0, 1],
        "tournament": ["friendly", "friendly"],
        "neutral": ["false", "true"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_boolean_strings_are_parsed_by_value():
    out = coerce_matches(_matches())
    assert out["neutral"].tolist() == [False, True]


@pytest.mark.parametrize(
    "column,values",
    [
        ("team1_goals", [-1, 0]),
        ("team2_goals", [0.5, 1]),
    ],
)
def test_invalid_scores_are_rejected(column, values):
    with pytest.raises(ValueError, match="non-negative integers"):
        coerce_matches(_matches(**{column: values}))


def test_duplicate_match_is_rejected():
    duplicate = pd.concat([_matches().iloc[[0]], _matches().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        coerce_matches(duplicate)


def test_team_cannot_play_twice_at_same_kickoff():
    frame = _matches(
        date=["2020-01-01", "2020-01-01"],
        team1=["A", "A"],
        team2=["B", "C"],
    )
    with pytest.raises(ValueError, match="appears more than once"):
        coerce_matches(frame)


def test_outcome_class_order_is_repository_wide():
    out = add_outcome(coerce_matches(_matches()))
    assert out["outcome"].tolist() == [
        int(OutcomeClass.TEAM1_WIN),
        int(OutcomeClass.TEAM2_WIN),
    ]
