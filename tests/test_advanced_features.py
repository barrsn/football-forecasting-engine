import pandas as pd

from football_forecast.features.advanced import (
    add_advanced_context_features,
    tournament_category,
)
from football_forecast.features.strength import (
    OnlinePoissonConfig,
    add_online_poisson_strength,
    add_prior_match_counts,
)


def _matches() -> pd.DataFrame:
    kickoff = pd.to_datetime(
        [
            "2020-01-01T12:00:00Z",
            "2020-01-01T12:00:00Z",
            "2020-02-01T12:00:00Z",
        ],
        utc=True,
    )
    return pd.DataFrame(
        {
            "match_id": ["a", "b", "c"],
            "kickoff_utc": kickoff,
            "available_at": kickoff,
            "team1": ["A", "A", "A"],
            "team2": ["B", "C", "B"],
            "team1_goals": [4, 0, 1],
            "team2_goals": [0, 1, 1],
            "neutral": [True, True, True],
            "tournament": ["Friendly", "Friendly", "FIFA World Cup"],
        }
    )


def test_same_kickoff_does_not_change_online_strength_or_counts():
    frame = _matches()
    counts = add_prior_match_counts(frame)
    strength = add_online_poisson_strength(
        frame,
        OnlinePoissonConfig(learning_rate=0.1),
    )

    assert counts.loc[0, "team1_prior_match_count"] == 0
    assert counts.loc[1, "team1_prior_match_count"] == 0
    assert counts.loc[2, "team1_prior_match_count"] == 2
    assert strength.loc[0, "online_attack_team1_pre"] == 0.0
    assert strength.loc[1, "online_attack_team1_pre"] == 0.0
    assert strength.loc[2, "online_attack_team1_pre"] != 0.0


def test_delayed_result_is_not_available_at_equal_cutoff():
    frame = _matches().iloc[[0, 2]].reset_index(drop=True)
    frame.loc[0, "available_at"] = frame.loc[1, "kickoff_utc"]
    counts = add_prior_match_counts(frame)
    strength = add_online_poisson_strength(frame)

    assert counts.loc[1, "team1_prior_match_count"] == 0
    assert strength.loc[1, "online_attack_team1_pre"] == 0.0


def test_advanced_features_are_finite_for_established_teams():
    frame = _matches()
    frame["elo_diff_pre"] = [0.0, 0.0, 20.0]
    frame["neutral_int"] = 1
    frame["tournament_importance"] = [0.5, 0.5, 2.0]
    frame["team1_days_rest"] = [float("nan"), float("nan"), 31.0]
    frame["team2_days_rest"] = [float("nan"), float("nan"), 31.0]
    frame["days_rest_diff"] = [float("nan"), float("nan"), 0.0]
    out = add_advanced_context_features(frame)

    assert out.loc[2, "online_lambda_total_pre"] > 0
    assert out.loc[2, "prior_match_count_min"] == 1
    assert out.loc[2, "tournament_category_world_cup"] == 1


def test_tournament_categories_distinguish_qualifiers_and_finals():
    assert tournament_category("FIFA World Cup qualification") == "world_cup_qualifier"
    assert tournament_category("FIFA World Cup") == "world_cup"
    assert tournament_category("UEFA Euro qualification") == "continental_qualifier"
    assert tournament_category("UEFA Euro") == "continental_final"
