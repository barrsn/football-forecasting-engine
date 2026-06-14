import numpy as np

from football_forecast.evaluation.selective import (
    choose_stable_confidence_threshold,
    evaluate_selective_accuracy,
    selective_predictions,
)


def test_selective_predictions_abstain_below_threshold():
    probabilities = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.34, 0.33, 0.33],
        ]
    )
    out = selective_predictions(probabilities, threshold=0.5)

    assert out.loc[0, "predicted_label"] == "team1_win"
    assert out.loc[0, "is_high_confidence"]
    assert out.loc[1, "predicted_label"] == "abstain"
    assert not out.loc[1, "is_high_confidence"]


def test_selective_accuracy_reports_coverage_separately():
    probabilities = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1],
            [0.34, 0.33, 0.33],
        ]
    )
    report = evaluate_selective_accuracy(
        np.array([2, 0, 2]),
        probabilities,
        threshold=0.5,
    )

    assert report.selective_accuracy == 1.0
    assert report.coverage == 2 / 3
    assert report.full_accuracy == 2 / 3


def test_threshold_selection_requires_each_chronological_group_to_pass():
    probabilities = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.4, 0.3, 0.3],
            [0.1, 0.2, 0.7],
            [0.4, 0.3, 0.3],
        ]
    )
    threshold = choose_stable_confidence_threshold(
        np.array([2, 2, 2, 2]),
        probabilities,
        np.array([2023, 2023, 2024, 2024]),
        target_accuracy=1.0,
        thresholds=(0.4, 0.5),
        min_group_predictions=1,
    )
    assert threshold == 0.5
