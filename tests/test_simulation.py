from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from football_forecast.simulation.worldcup import (
    MatchRecord,
    TeamStanding,
    allocate_third_place_groups,
    build_round_of_32,
    rank_group,
    rank_third_placed,
    simulate_knockout_match,
    simulate_world_cup,
)


def test_head_to_head_mini_table_precedes_overall_goal_difference():
    table = {
        "A": TeamStanding("A", points=3, goals_for=10, goals_against=1, fifa_rank=1),
        "B": TeamStanding("B", points=3, goals_for=2, goals_against=2, fifa_rank=20),
    }
    records = [MatchRecord("A", "B", 0, 1)]
    assert rank_group(table, records) == ["B", "A"]


def test_best_third_place_uses_conduct_then_fifa_rank():
    third = [
        ("A", TeamStanding("A", points=4, goals_for=2, goals_against=2, conduct_score=-2, fifa_rank=2)),
        ("B", TeamStanding("B", points=4, goals_for=2, goals_against=2, conduct_score=-1, fifa_rank=20)),
    ]
    assert rank_third_placed(third)[0][0] == "B"


def test_third_place_fallback_is_valid_and_deterministic():
    groups = list("ABCDEFGH")
    first, mode = allocate_third_place_groups(groups)
    second, second_mode = allocate_third_place_groups(groups)
    assert first == second
    assert mode == second_mode == "deterministic_valid_fallback"
    assert set(first.values()) == set(groups)


def test_knockout_penalty_result_is_reproducible():
    matrix = np.array([[1.0]])
    first = simulate_knockout_match("A", "B", matrix, np.random.default_rng(7))
    second = simulate_knockout_match("A", "B", matrix, np.random.default_rng(7))
    assert first == second
    assert first.decided_by_penalties


def test_round_of_32_uses_official_match_numbers():
    rankings = {
        group: [f"{group}{rank}" for rank in range(1, 5)]
        for group in "ABCDEFGHIJKL"
    }
    standings = {
        group: {
            team: TeamStanding(team, points=5 - rank, fifa_rank=rank)
            for rank, team in enumerate(ranking, start=1)
        }
        for group, ranking in rankings.items()
    }
    matchups, _ = build_round_of_32(rankings, standings)
    assert matchups["M73"] == ("A2", "B2")
    assert matchups["M74"] == ("C1", "F2")
    assert matchups["M77"] == ("E2", "I2")
    assert matchups["M83"] == ("H1", "J2")
    assert matchups["M84"] == ("K2", "L2")
    assert matchups["M86"] == ("D2", "G2")
    assert matchups["M87"] == ("J1", "H2")


def _world_cup_fixtures() -> pd.DataFrame:
    rows = []
    for group in "ABCDEFGHIJKL":
        teams = [f"{group}{index}" for index in range(1, 5)]
        for match_number, (team1, team2) in enumerate(combinations(teams, 2), start=1):
            rows.append(
                {
                    "match_id": f"{group}-{match_number}",
                    "group": group,
                    "team1": team1,
                    "team2": team2,
                }
            )
    return pd.DataFrame(rows)


def test_full_simulation_is_reproducible_and_probabilities_sum():
    fixtures = _world_cup_fixtures()
    matrix = np.array([[0.35, 0.15], [0.20, 0.30]])

    def provider(team1, team2, stage):
        return matrix

    rankings = {team: index + 1 for index, team in enumerate(sorted(set(fixtures["team1"]) | set(fixtures["team2"])))}
    first = simulate_world_cup(
        fixtures,
        provider,
        n_simulations=5,
        random_seed=11,
        fifa_rankings=rankings,
        require_official_allocations=False,
    )
    second = simulate_world_cup(
        fixtures,
        provider,
        n_simulations=5,
        random_seed=11,
        fifa_rankings=rankings,
        require_official_allocations=False,
    )
    pd.testing.assert_frame_equal(first.stage_probabilities, second.stage_probabilities)
    assert np.isclose(first.stage_probabilities["champion"].sum(), 1.0)
    group_totals = first.group_rank_probabilities.groupby(["group", "rank"])["probability"].sum()
    assert np.allclose(group_totals.to_numpy(), 1.0)


def test_production_simulation_requires_complete_annex_c():
    fixtures = _world_cup_fixtures()

    def provider(team1, team2, stage):
        return np.array([[1.0]])

    with pytest.raises(ValueError, match="495 official FIFA Annex C"):
        simulate_world_cup(fixtures, provider, n_simulations=1)
