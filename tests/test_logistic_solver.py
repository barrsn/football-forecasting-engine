import pandas as pd

from football_forecast.evaluation.baselines import EloLogisticBaseline
from football_forecast.models.outcome import OutcomeModel, StructuredOutcomeModel
from football_forecast.models.sklearn_config import LOGISTIC_SOLVER


def test_logistic_models_use_stable_solver():
    outcome = OutcomeModel("logistic")
    baseline = EloLogisticBaseline()
    structured = StructuredOutcomeModel()

    assert outcome.model[-1].solver == LOGISTIC_SOLVER
    assert baseline.model[-1].solver == LOGISTIC_SOLVER
    assert structured.draw_model[-1].solver == LOGISTIC_SOLVER
    assert structured.decisive_model[-1].solver == LOGISTIC_SOLVER


def test_logistic_outcome_model_fits_and_predicts_probabilities():
    x = pd.DataFrame(
        {
            "strength_diff": [-2.0, -1.0, 0.0, 1.0, 2.0] * 8,
            "neutral_int": [0, 1, 0, 1, 0] * 8,
        }
    )
    y = pd.Series([0, 0, 1, 2, 2] * 8)

    probabilities = OutcomeModel("logistic").fit(x, y).predict_proba(x.iloc[:5])

    assert probabilities.shape == (5, 3)
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()
