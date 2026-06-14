from __future__ import annotations

from dataclasses import dataclass

from football_forecast.evaluation.backtesting import ModelMetrics


@dataclass(frozen=True)
class SelectionDecision:
    approved: bool
    selected_model: str
    reason: str


def select_candidate_model(
    candidate_name: str,
    candidate: ModelMetrics,
    baseline_name: str,
    baseline: ModelMetrics,
    *,
    minimum_log_loss_improvement: float = 0.01,
    maximum_secondary_degradation: float = 0.01,
) -> SelectionDecision:
    improvement = (baseline.log_loss - candidate.log_loss) / baseline.log_loss
    brier_degradation = (candidate.brier - baseline.brier) / baseline.brier
    rps_degradation = (candidate.rps - baseline.rps) / baseline.rps
    approved = (
        improvement >= minimum_log_loss_improvement
        and brier_degradation <= maximum_secondary_degradation
        and rps_degradation <= maximum_secondary_degradation
    )
    if approved:
        return SelectionDecision(
            True,
            candidate_name,
            f"Log Loss improved by {improvement:.2%} without material Brier/RPS degradation",
        )
    return SelectionDecision(
        False,
        baseline_name,
        "Candidate did not satisfy the promotion thresholds",
    )
