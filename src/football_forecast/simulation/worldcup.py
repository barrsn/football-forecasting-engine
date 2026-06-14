from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from math import sqrt
from typing import Callable

import numpy as np
import pandas as pd

from football_forecast.simulation.match import (
    expected_goals_from_matrix,
    sample_extra_time_score,
    sample_penalty_winner,
    sample_score_from_matrix,
)

ScoreMatrixProvider = Callable[[str, str, str], np.ndarray]

THIRD_PLACE_WINNER_ORDER = ("A", "B", "D", "E", "G", "I", "K", "L")
THIRD_PLACE_CANDIDATES = {
    "A": set("CEFHI"),
    "B": set("EFGIJ"),
    "D": set("BEFIJ"),
    "E": set("ABCDF"),
    "G": set("AEHIJ"),
    "I": set("CDFGH"),
    "K": set("DEIJL"),
    "L": set("EHIJK"),
}

ROUND_OF_16_BRACKET = {
    "M89": ("M73", "M75"),
    "M90": ("M74", "M77"),
    "M91": ("M76", "M78"),
    "M92": ("M79", "M80"),
    "M93": ("M83", "M84"),
    "M94": ("M81", "M82"),
    "M95": ("M86", "M88"),
    "M96": ("M85", "M87"),
}
QUARTERFINAL_BRACKET = {
    "M97": ("M89", "M90"),
    "M98": ("M93", "M94"),
    "M99": ("M91", "M92"),
    "M100": ("M95", "M96"),
}
SEMIFINAL_BRACKET = {
    "M101": ("M97", "M98"),
    "M102": ("M99", "M100"),
}


@dataclass
class TeamStanding:
    team: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    conduct_score: int = 0
    fifa_rank: int = 10_000

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against


@dataclass(frozen=True)
class MatchRecord:
    team1: str
    team2: str
    goals1: int
    goals2: int


@dataclass(frozen=True)
class KnockoutResult:
    winner: str
    loser: str
    regulation_score: tuple[int, int]
    extra_time_score: tuple[int, int] | None
    decided_by_penalties: bool


@dataclass
class TournamentSimulationResult:
    n_simulations: int
    group_rank_probabilities: pd.DataFrame
    stage_probabilities: pd.DataFrame
    monte_carlo_error: pd.DataFrame
    allocation_mode: str


def update_standings(
    table: dict[str, TeamStanding],
    team1: str,
    team2: str,
    g1: int,
    g2: int,
) -> None:
    table.setdefault(team1, TeamStanding(team=team1))
    table.setdefault(team2, TeamStanding(team=team2))
    table[team1].goals_for += g1
    table[team1].goals_against += g2
    table[team2].goals_for += g2
    table[team2].goals_against += g1
    if g1 > g2:
        table[team1].points += 3
    elif g1 < g2:
        table[team2].points += 3
    else:
        table[team1].points += 1
        table[team2].points += 1


def _mini_table_key(team: str, tied: set[str], records: list[MatchRecord]) -> tuple[int, int, int]:
    points = goals_for = goals_against = 0
    for record in records:
        if record.team1 not in tied or record.team2 not in tied:
            continue
        if record.team1 == team:
            goals_for += record.goals1
            goals_against += record.goals2
            points += 3 if record.goals1 > record.goals2 else 1 if record.goals1 == record.goals2 else 0
        elif record.team2 == team:
            goals_for += record.goals2
            goals_against += record.goals1
            points += 3 if record.goals2 > record.goals1 else 1 if record.goals2 == record.goals1 else 0
    return points, goals_for - goals_against, goals_for


def _rank_equal_points(
    teams: list[str],
    table: dict[str, TeamStanding],
    records: list[MatchRecord],
) -> list[str]:
    if len(teams) <= 1:
        return teams
    tied_set = set(teams)
    mini_keys = {team: _mini_table_key(team, tied_set, records) for team in teams}
    distinct_keys = set(mini_keys.values())
    if len(distinct_keys) > 1:
        ranked: list[str] = []
        for key in sorted(distinct_keys, reverse=True):
            subgroup = [team for team in teams if mini_keys[team] == key]
            ranked.extend(_rank_equal_points(subgroup, table, records))
        return ranked

    return sorted(
        teams,
        key=lambda team: (
            table[team].goal_diff,
            table[team].goals_for,
            table[team].conduct_score,
            -table[team].fifa_rank,
            team,
        ),
        reverse=True,
    )


def rank_group(
    table: dict[str, TeamStanding],
    records: list[MatchRecord] | None = None,
) -> list[str]:
    records = records or []
    point_groups: dict[int, list[str]] = defaultdict(list)
    for standing in table.values():
        point_groups[standing.points].append(standing.team)
    ranked: list[str] = []
    for points in sorted(point_groups, reverse=True):
        ranked.extend(_rank_equal_points(point_groups[points], table, records))
    return ranked


def rank_third_placed(
    third_placed: list[tuple[str, TeamStanding]],
) -> list[tuple[str, TeamStanding]]:
    return sorted(
        third_placed,
        key=lambda item: (
            item[1].points,
            item[1].goal_diff,
            item[1].goals_for,
            item[1].conduct_score,
            -item[1].fifa_rank,
            item[0],
        ),
        reverse=True,
    )


def allocate_third_place_groups(
    qualified_groups: list[str],
    official_allocations: dict[tuple[str, ...], dict[str, str]] | None = None,
) -> tuple[dict[str, str], str]:
    """Return winner-group to third-place-group assignments.

    If the official Annex C mapping is supplied, it is authoritative. The fallback
    is a deterministic valid assignment and is explicitly labelled as such.
    """
    key = tuple(sorted(qualified_groups))
    if len(key) != 8 or len(set(key)) != 8:
        raise ValueError("Exactly eight distinct third-place groups are required")
    if official_allocations and key in official_allocations:
        return dict(official_allocations[key]), "official_annex_c"
    return dict(_fallback_third_place_allocation(key)), "deterministic_valid_fallback"


@lru_cache(maxsize=495)
def _fallback_third_place_allocation(key: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    winners = sorted(
        THIRD_PLACE_WINNER_ORDER,
        key=lambda winner: len(THIRD_PLACE_CANDIDATES[winner].intersection(key)),
    )
    for candidate_order in permutations(key):
        assignment = dict(zip(winners, candidate_order))
        if all(
            assignment[winner] in THIRD_PLACE_CANDIDATES[winner]
            for winner in winners
        ):
            return tuple(sorted(assignment.items()))
    raise ValueError(f"No valid third-place allocation for groups {key}")


def simulate_group_stage(
    fixtures: pd.DataFrame,
    score_matrices: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    fifa_rankings: dict[str, int] | None = None,
    conduct_scores: dict[str, int] | None = None,
) -> tuple[dict[str, list[str]], dict[str, dict[str, TeamStanding]]]:
    """Simulate all groups using exact-score matrices and FIFA tie-break order."""
    rankings: dict[str, list[str]] = {}
    standings: dict[str, dict[str, TeamStanding]] = {}
    fifa_rankings = fifa_rankings or {}
    conduct_scores = conduct_scores or {}
    for group, group_matches in fixtures.groupby("group", sort=True):
        table: dict[str, TeamStanding] = {}
        records: list[MatchRecord] = []
        teams = set(group_matches["team1"]).union(group_matches["team2"])
        for team in teams:
            table[team] = TeamStanding(
                team=team,
                conduct_score=conduct_scores.get(team, 0),
                fifa_rank=fifa_rankings.get(team, 10_000),
            )
        for row in group_matches.itertuples(index=False):
            matrix = score_matrices[str(row.match_id)]
            g1, g2 = sample_score_from_matrix(matrix, rng)
            update_standings(table, row.team1, row.team2, g1, g2)
            records.append(MatchRecord(row.team1, row.team2, g1, g2))
        rankings[str(group)] = rank_group(table, records)
        standings[str(group)] = table
    return rankings, standings


def simulate_knockout_match(
    team1: str,
    team2: str,
    score_matrix: np.ndarray,
    rng: np.random.Generator,
    *,
    penalty_probability_team1: float = 0.5,
) -> KnockoutResult:
    g1, g2 = sample_score_from_matrix(score_matrix, rng)
    extra_time_score: tuple[int, int] | None = None
    penalties = False
    if g1 == g2:
        lambda1, lambda2 = expected_goals_from_matrix(score_matrix)
        extra_time_score = sample_extra_time_score(lambda1, lambda2, rng)
        g1 += extra_time_score[0]
        g2 += extra_time_score[1]
    if g1 == g2:
        penalties = True
        winner = sample_penalty_winner(
            team1, team2, rng, p_team1=penalty_probability_team1
        )
    else:
        winner = team1 if g1 > g2 else team2
    loser = team2 if winner == team1 else team1
    return KnockoutResult(
        winner=winner,
        loser=loser,
        regulation_score=(g1 - (extra_time_score or (0, 0))[0], g2 - (extra_time_score or (0, 0))[1]),
        extra_time_score=extra_time_score,
        decided_by_penalties=penalties,
    )


def build_round_of_32(
    group_rankings: dict[str, list[str]],
    standings: dict[str, dict[str, TeamStanding]],
    *,
    official_allocations: dict[tuple[str, ...], dict[str, str]] | None = None,
) -> tuple[dict[str, tuple[str, str]], str]:
    third_placed = [
        (group, standings[group][ranking[2]])
        for group, ranking in group_rankings.items()
    ]
    best_thirds = rank_third_placed(third_placed)[:8]
    third_by_group = {group: standing.team for group, standing in best_thirds}
    assignment, mode = allocate_third_place_groups(
        list(third_by_group), official_allocations=official_allocations
    )
    winner = {group: ranking[0] for group, ranking in group_rankings.items()}
    runner_up = {group: ranking[1] for group, ranking in group_rankings.items()}
    matches = {
        "M73": (runner_up["A"], runner_up["B"]),
        "M74": (winner["C"], runner_up["F"]),
        "M75": (winner["E"], third_by_group[assignment["E"]]),
        "M76": (winner["F"], runner_up["C"]),
        "M77": (runner_up["E"], runner_up["I"]),
        "M78": (winner["I"], third_by_group[assignment["I"]]),
        "M79": (winner["A"], third_by_group[assignment["A"]]),
        "M80": (winner["L"], third_by_group[assignment["L"]]),
        "M81": (winner["G"], third_by_group[assignment["G"]]),
        "M82": (winner["D"], third_by_group[assignment["D"]]),
        "M83": (winner["H"], runner_up["J"]),
        "M84": (runner_up["K"], runner_up["L"]),
        "M85": (winner["B"], third_by_group[assignment["B"]]),
        "M86": (runner_up["D"], runner_up["G"]),
        "M87": (winner["J"], runner_up["H"]),
        "M88": (winner["K"], third_by_group[assignment["K"]]),
    }
    return matches, mode


def _simulate_round(
    matchups: dict[str, tuple[str, str]],
    stage: str,
    provider: ScoreMatrixProvider,
    rng: np.random.Generator,
) -> tuple[dict[str, str], dict[str, str]]:
    winners: dict[str, str] = {}
    losers: dict[str, str] = {}
    for match_id, (team1, team2) in matchups.items():
        result = simulate_knockout_match(team1, team2, provider(team1, team2, stage), rng)
        winners[match_id] = result.winner
        losers[match_id] = result.loser
    return winners, losers


def _advance_matchups(
    bracket: dict[str, tuple[str, str]],
    previous_winners: dict[str, str],
) -> dict[str, tuple[str, str]]:
    return {
        match_id: (previous_winners[source1], previous_winners[source2])
        for match_id, (source1, source2) in bracket.items()
    }


def simulate_world_cup(
    group_fixtures: pd.DataFrame,
    score_provider: ScoreMatrixProvider,
    *,
    n_simulations: int = 100_000,
    random_seed: int = 42,
    fifa_rankings: dict[str, int] | None = None,
    conduct_scores: dict[str, int] | None = None,
    official_allocations: dict[tuple[str, ...], dict[str, str]] | None = None,
    require_official_allocations: bool = True,
) -> TournamentSimulationResult:
    required = {"match_id", "group", "team1", "team2"}
    if missing := sorted(required.difference(group_fixtures.columns)):
        raise ValueError(f"Missing group fixture columns: {missing}")
    groups = sorted(group_fixtures["group"].astype(str).unique())
    if groups != list("ABCDEFGHIJKL"):
        raise ValueError("World Cup simulation requires groups A through L")
    if require_official_allocations and (
        official_allocations is None or len(official_allocations) != 495
    ):
        raise ValueError(
            "Production simulation requires all 495 official FIFA Annex C allocations"
        )

    rng = np.random.default_rng(random_seed)
    teams = sorted(set(group_fixtures["team1"]).union(group_fixtures["team2"]))
    group_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    stage_counts = {team: defaultdict(int) for team in teams}
    allocation_modes: dict[str, int] = defaultdict(int)

    group_score_matrices = {
        str(row.match_id): score_provider(row.team1, row.team2, "group")
        for row in group_fixtures.itertuples(index=False)
    }
    for _ in range(n_simulations):
        rankings, standings = simulate_group_stage(
            group_fixtures,
            group_score_matrices,
            rng,
            fifa_rankings=fifa_rankings,
            conduct_scores=conduct_scores,
        )
        for group, ranking in rankings.items():
            for rank, team in enumerate(ranking, start=1):
                group_counts[(group, team, rank)] += 1

        round32, allocation_mode = build_round_of_32(
            rankings, standings, official_allocations=official_allocations
        )
        allocation_modes[allocation_mode] += 1
        for team1, team2 in round32.values():
            stage_counts[team1]["r32"] += 1
            stage_counts[team2]["r32"] += 1
        winners32, _ = _simulate_round(round32, "r32", score_provider, rng)

        round16 = _advance_matchups(ROUND_OF_16_BRACKET, winners32)
        for team1, team2 in round16.values():
            stage_counts[team1]["r16"] += 1
            stage_counts[team2]["r16"] += 1
        winners16, _ = _simulate_round(round16, "r16", score_provider, rng)

        quarterfinals = _advance_matchups(QUARTERFINAL_BRACKET, winners16)
        for team1, team2 in quarterfinals.values():
            stage_counts[team1]["qf"] += 1
            stage_counts[team2]["qf"] += 1
        winners_qf, _ = _simulate_round(quarterfinals, "qf", score_provider, rng)

        semifinals = _advance_matchups(SEMIFINAL_BRACKET, winners_qf)
        for team1, team2 in semifinals.values():
            stage_counts[team1]["sf"] += 1
            stage_counts[team2]["sf"] += 1
        winners_sf, losers_sf = _simulate_round(semifinals, "sf", score_provider, rng)

        finalists = (winners_sf["M101"], winners_sf["M102"])
        for team in finalists:
            stage_counts[team]["final"] += 1
        final_result = simulate_knockout_match(
            finalists[0],
            finalists[1],
            score_provider(finalists[0], finalists[1], "final"),
            rng,
        )
        stage_counts[final_result.winner]["champion"] += 1

        third_place = (losers_sf["M101"], losers_sf["M102"])
        simulate_knockout_match(
            third_place[0],
            third_place[1],
            score_provider(third_place[0], third_place[1], "third_place"),
            rng,
        )

    group_rows = [
        {
            "group": group,
            "team": team,
            "rank": rank,
            "probability": count / n_simulations,
        }
        for (group, team, rank), count in sorted(group_counts.items())
    ]
    stage_names = ("r32", "r16", "qf", "sf", "final", "champion")
    stage_rows = []
    error_rows = []
    for team in teams:
        row = {"team": team}
        error = {"team": team}
        for stage in stage_names:
            probability = stage_counts[team][stage] / n_simulations
            row[stage] = probability
            error[stage] = sqrt(probability * (1.0 - probability) / n_simulations)
        stage_rows.append(row)
        error_rows.append(error)

    dominant_mode = max(allocation_modes, key=allocation_modes.get)
    return TournamentSimulationResult(
        n_simulations=n_simulations,
        group_rank_probabilities=pd.DataFrame(group_rows),
        stage_probabilities=pd.DataFrame(stage_rows),
        monte_carlo_error=pd.DataFrame(error_rows),
        allocation_mode=dominant_mode,
    )
