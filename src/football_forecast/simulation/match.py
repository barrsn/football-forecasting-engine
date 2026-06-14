from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def sample_score_from_matrix(score_matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    matrix = np.asarray(score_matrix, dtype=float)
    matrix = matrix / matrix.sum()
    flat_idx = rng.choice(matrix.size, p=matrix.ravel())
    goals_team1, goals_team2 = np.unravel_index(flat_idx, matrix.shape)
    return int(goals_team1), int(goals_team2)


def sample_outcome(proba: np.ndarray, rng: np.random.Generator) -> int:
    p = np.asarray(proba, dtype=float)
    p = p / p.sum()
    return int(rng.choice([0, 1, 2], p=p))


def expected_goals_from_matrix(score_matrix: np.ndarray) -> tuple[float, float]:
    matrix = np.asarray(score_matrix, dtype=float)
    matrix = matrix / matrix.sum()
    goals1 = np.arange(matrix.shape[0], dtype=float)
    goals2 = np.arange(matrix.shape[1], dtype=float)
    return float((matrix.sum(axis=1) * goals1).sum()), float(
        (matrix.sum(axis=0) * goals2).sum()
    )


def sample_extra_time_score(
    lambda1_regulation: float,
    lambda2_regulation: float,
    rng: np.random.Generator,
    *,
    minutes: int = 30,
) -> tuple[int, int]:
    scale = minutes / 90.0
    return (
        int(poisson.rvs(lambda1_regulation * scale, random_state=rng)),
        int(poisson.rvs(lambda2_regulation * scale, random_state=rng)),
    )


def sample_penalty_winner(
    team1: str,
    team2: str,
    rng: np.random.Generator,
    *,
    p_team1: float = 0.5,
) -> str:
    if not 0.0 <= p_team1 <= 1.0:
        raise ValueError("Penalty probability must be within [0, 1]")
    return team1 if rng.random() < p_team1 else team2
