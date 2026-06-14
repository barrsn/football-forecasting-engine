from football_forecast.evaluation.model_search import (
    default_outcome_candidates,
    default_score_candidates,
)


def test_default_search_space_has_baseline_and_boosting_models():
    outcome_types = {candidate["model_type"] for candidate in default_outcome_candidates()}
    assert {"logistic", "hist_gbm", "lightgbm"}.issubset(outcome_types)
    assert {candidate["model_type"] for candidate in default_score_candidates()} == {
        "poisson",
        "dixon_coles",
    }
