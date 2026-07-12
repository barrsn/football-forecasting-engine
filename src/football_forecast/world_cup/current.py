from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from football_forecast.data.teams import normalize_match_teams
from football_forecast.evaluation.metrics import normalize_probabilities
from football_forecast.features.elo import expected_score, tournament_importance
from football_forecast.features.advanced import tournament_category

PROBABILITY_COLUMNS = ["p_team2_win", "p_draw", "p_team1_win"]
OUTCOME_LABELS = {
    0: "team2_win",
    1: "draw",
    2: "team1_win",
}


def load_world_cup_fixtures(
    raw_results_path: str | Path,
    *,
    mapping_path: str | Path,
    start_date: str = "2026-06-11",
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    raw = pd.read_csv(raw_results_path)
    dates = pd.to_datetime(raw["date"], errors="raise")
    fixtures = raw.loc[
        raw["tournament"].eq("FIFA World Cup")
        & dates.ge(pd.Timestamp(start_date))
    ].copy()
    fixtures["_source_order"] = np.arange(len(fixtures))
    fixtures = fixtures.rename(
        columns={
            "home_team": "team1",
            "away_team": "team2",
            "home_score": "team1_goals",
            "away_score": "team2_goals",
        }
    )
    fixtures, _ = normalize_match_teams(fixtures, mapping_path)
    fixtures["date"] = pd.to_datetime(fixtures["date"], utc=True)
    fixtures["has_result"] = (
        fixtures["team1_goals"].notna() & fixtures["team2_goals"].notna()
    )
    fixtures = fixtures.sort_values(["_source_order"]).reset_index(drop=True)
    as_of = pd.Timestamp(as_of_date, tz="UTC") if as_of_date is not None else pd.Timestamp.now(tz="UTC")
    fixture_days = fixtures["date"].dt.normalize()
    as_of_day = as_of.normalize()
    fixtures["status"] = np.select(
        [
            fixtures["has_result"],
            fixture_days.lt(as_of_day),
            fixture_days.eq(as_of_day),
        ],
        ["completed", "result_missing", "today"],
        default="upcoming",
    )
    fixtures["match_id"] = [f"WC2026-G{index + 1:02d}" for index in range(len(fixtures))]
    fixtures["group"] = infer_fixture_groups(fixtures)
    return fixtures.drop(columns="_source_order")


def infer_fixture_groups(fixtures: pd.DataFrame) -> list[str]:
    """Infer group labels from connected components in the group-stage fixtures."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    first_seen: dict[str, int] = {}
    for index, row in enumerate(fixtures.itertuples(index=False)):
        team1 = str(row.team1)
        team2 = str(row.team2)
        adjacency[team1].add(team2)
        adjacency[team2].add(team1)
        first_seen.setdefault(team1, index)
        first_seen.setdefault(team2, index)

    components: list[list[str]] = []
    seen: set[str] = set()
    for team in sorted(first_seen, key=first_seen.get):
        if team in seen:
            continue
        queue: deque[str] = deque([team])
        seen.add(team)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    labels = list("ABCDEFGHIJKL")
    team_to_group: dict[str, str] = {}
    for label, component in zip(labels, components):
        for team in component:
            team_to_group[team] = label
    return [
        team_to_group.get(str(row.team1), "")
        for row in fixtures.itertuples(index=False)
    ]


def latest_team_state(feature_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = feature_frame.sort_values(["kickoff_utc", "match_id"])
    for side in ("team1", "team2"):
        side_rows = pd.DataFrame(
            {
                "team": ordered[side],
                "last_match_date": pd.to_datetime(ordered["kickoff_utc"], utc=True),
                "elo_pre": ordered[f"elo_{side}_pre"],
                "online_attack_pre": ordered[f"online_attack_{side}_pre"],
                "online_defence_pre": ordered[f"online_defence_{side}_pre"],
                "prior_match_count": ordered[f"{side}_prior_match_count"],
                "online_global_goals_pre": ordered["online_global_goals_pre"],
            }
        )
        for column in ordered.columns:
            if column.startswith(f"{side}_") and column not in side_rows:
                side_rows[column.removeprefix(f"{side}_")] = ordered[column]
        rows.append(side_rows)
    state = pd.concat(rows, ignore_index=True)
    return state.sort_values("last_match_date").groupby("team", as_index=False).tail(1)


def _state_lookup(state: pd.DataFrame) -> dict[str, dict[str, object]]:
    return state.set_index("team").to_dict(orient="index")


def _value(
    states: dict[str, dict[str, object]],
    team: str,
    key: str,
    default: float = np.nan,
) -> object:
    return states.get(team, {}).get(key, default)


def _numeric(
    states: dict[str, dict[str, object]],
    team: str,
    key: str,
    default: float = np.nan,
) -> float:
    value = _value(states, team, key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _days_rest(states: dict[str, dict[str, object]], team: str, kickoff: pd.Timestamp) -> float:
    last_match = _value(states, team, "last_match_date", pd.NaT)
    if pd.isna(last_match):
        return float("nan")
    return float((kickoff - pd.Timestamp(last_match)).total_seconds() / 86400.0)


def build_world_cup_feature_frame(
    fixtures: pd.DataFrame,
    historical_features: pd.DataFrame,
) -> pd.DataFrame:
    states = _state_lookup(latest_team_state(historical_features))
    rows = []
    for fixture in fixtures.itertuples(index=False):
        team1 = str(fixture.team1)
        team2 = str(fixture.team2)
        kickoff = pd.Timestamp(fixture.date)
        neutral = bool(fixture.neutral)
        elo1 = _numeric(states, team1, "elo_pre", 1500.0)
        elo2 = _numeric(states, team2, "elo_pre", 1500.0)
        home_adjustment = 0.0 if neutral else 60.0
        category = tournament_category(fixture.tournament)
        row: dict[str, object] = {
            "match_id": fixture.match_id,
            "group": fixture.group,
            "date": kickoff,
            "team1": team1,
            "team2": team2,
            "city": getattr(fixture, "city", ""),
            "country": getattr(fixture, "country", ""),
            "neutral": neutral,
            "has_result": bool(fixture.has_result),
            "status": fixture.status,
            "elo_diff_pre": elo1 - elo2,
            "elo_exp_team1": expected_score(elo1 + home_adjustment, elo2),
            "neutral_int": int(neutral),
            "tournament_importance": tournament_importance(str(fixture.tournament)),
        }
        row.update(_calendar_features(kickoff, category))
        row.update(_rolling_features(states, team1, team2))
        row.update(_rest_and_count_features(states, team1, team2, kickoff))
        row.update(_online_strength_features(states, team1, team2, neutral))
        row.update(_fifa_features(states, team1, team2))
        rows.append(row)
    return pd.DataFrame(rows)


def _calendar_features(kickoff: pd.Timestamp, category: str) -> dict[str, float | int]:
    year_fraction = kickoff.year + (kickoff.dayofyear - 1) / 365.25
    features: dict[str, float | int] = {
        "calendar_year_scaled": (year_fraction - 2000.0) / 10.0,
        "month_sin": float(np.sin(2.0 * np.pi * kickoff.month / 12.0)),
        "month_cos": float(np.cos(2.0 * np.pi * kickoff.month / 12.0)),
    }
    for name in (
        "friendly",
        "nations_league",
        "world_cup_qualifier",
        "continental_qualifier",
        "world_cup",
        "continental_final",
    ):
        features[f"tournament_category_{name}"] = int(category == name)
    return features


def _rolling_features(
    states: dict[str, dict[str, object]],
    team1: str,
    team2: str,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for suffix in ("roll5", "roll10", "roll20", "ewm"):
        for prefix in ("goal_diff", "goals_for", "goals_against", "opponent_adjusted_goal_diff"):
            left = _numeric(states, team1, f"{prefix}_{suffix}")
            right = _numeric(states, team2, f"{prefix}_{suffix}")
            features[f"{prefix}_{suffix}_diff"] = left - right
            if prefix in {"goal_diff", "opponent_adjusted_goal_diff"}:
                features[f"{prefix}_{suffix}_diff_abs"] = abs(left - right)

        team1_matchup = (
            _numeric(states, team1, f"goals_for_{suffix}")
            + _numeric(states, team2, f"goals_against_{suffix}")
        ) / 2.0
        team2_matchup = (
            _numeric(states, team2, f"goals_for_{suffix}")
            + _numeric(states, team1, f"goals_against_{suffix}")
        ) / 2.0
        features[f"matchup_goals_{suffix}_diff"] = team1_matchup - team2_matchup
        features[f"matchup_goals_{suffix}_diff_abs"] = abs(team1_matchup - team2_matchup)
        features[f"matchup_goals_{suffix}_total"] = team1_matchup + team2_matchup
        features[f"win_form_{suffix}_diff"] = (
            _numeric(states, team1, f"win_{suffix}")
            - _numeric(states, team2, f"win_{suffix}")
        )
        features[f"draw_form_{suffix}_diff"] = (
            _numeric(states, team1, f"draw_{suffix}")
            - _numeric(states, team2, f"draw_{suffix}")
        )
        features[f"draw_form_{suffix}_mean"] = (
            _numeric(states, team1, f"draw_{suffix}")
            + _numeric(states, team2, f"draw_{suffix}")
        ) / 2.0

    for prefix in ("goal_diff", "goals_for", "opponent_adjusted_goal_diff"):
        value = features[f"{prefix}_roll5_diff"] - features[f"{prefix}_roll20_diff"]
        features[f"{prefix}_momentum_diff"] = value
        features[f"{prefix}_momentum_abs"] = abs(value)
    return features


def _rest_and_count_features(
    states: dict[str, dict[str, object]],
    team1: str,
    team2: str,
    kickoff: pd.Timestamp,
) -> dict[str, float]:
    days1 = _days_rest(states, team1, kickoff)
    days2 = _days_rest(states, team2, kickoff)
    count1 = _numeric(states, team1, "prior_match_count", 0.0)
    count2 = _numeric(states, team2, "prior_match_count", 0.0)
    diff = days1 - days2
    return {
        "days_rest_diff": diff,
        "days_rest_diff_clipped": float(np.clip(diff, -60.0, 60.0)),
        "prior_match_count_diff": count1 - count2,
        "prior_match_count_min_log1p": float(np.log1p(max(min(count1, count2), 0.0))),
    }


def _online_strength_features(
    states: dict[str, dict[str, object]],
    team1: str,
    team2: str,
    neutral: bool,
) -> dict[str, float]:
    attack1 = _numeric(states, team1, "online_attack_pre", 0.0)
    defence1 = _numeric(states, team1, "online_defence_pre", 0.0)
    attack2 = _numeric(states, team2, "online_attack_pre", 0.0)
    defence2 = _numeric(states, team2, "online_defence_pre", 0.0)
    global_goals = np.nanmean(
        [
            _numeric(states, team1, "online_global_goals_pre", 1.25),
            _numeric(states, team2, "online_global_goals_pre", 1.25),
        ]
    )
    home_adjustment = 0.0 if neutral else 0.16
    lambda1 = float(np.clip(global_goals * np.exp(attack1 - defence2 + home_adjustment), 0.15, 5.0))
    lambda2 = float(np.clip(global_goals * np.exp(attack2 - defence1), 0.15, 5.0))
    ratio = float(np.log(lambda1 / lambda2))
    return {
        "online_lambda_team1_pre": lambda1,
        "online_lambda_team2_pre": lambda2,
        "online_lambda_diff_pre": lambda1 - lambda2,
        "online_lambda_total_pre": lambda1 + lambda2,
        "online_log_lambda_ratio_pre": ratio,
        "online_attack_diff_pre": attack1 - attack2,
        "online_defence_diff_pre": defence1 - defence2,
        "online_lambda_diff_abs_pre": abs(lambda1 - lambda2),
        "online_log_lambda_ratio_abs_pre": abs(ratio),
    }


def _fifa_features(
    states: dict[str, dict[str, object]],
    team1: str,
    team2: str,
) -> dict[str, float | int]:
    features: dict[str, float | int] = {}
    for column in ("rank", "points", "rated_matches", "rank_percentile", "points_z"):
        left = _numeric(states, team1, f"fifa_{column}")
        right = _numeric(states, team2, f"fifa_{column}")
        features[f"team1_fifa_{column}"] = left
        features[f"team2_fifa_{column}"] = right
        features[f"fifa_{column}_diff"] = left - right
    conf1 = str(_value(states, team1, "fifa_confederation", "")).upper()
    conf2 = str(_value(states, team2, "fifa_confederation", "")).upper()
    features["fifa_same_confederation_int"] = int(conf1 == conf2 and conf1 != "")
    for confederation in ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA"):
        suffix = confederation.casefold()
        features[f"team1_fifa_confederation_{suffix}"] = int(conf1 == confederation)
        features[f"team2_fifa_confederation_{suffix}"] = int(conf2 == confederation)
    return features


def add_world_cup_predictions(
    feature_frame: pd.DataFrame,
    *,
    model_path: str | Path,
    feature_columns: list[str],
    confidence_threshold: float = 0.65,
) -> pd.DataFrame:
    model = joblib.load(model_path)
    features = feature_frame.copy()
    for column in feature_columns:
        if column not in features:
            features[column] = np.nan
    probabilities = normalize_probabilities(
        model.predict_proba(features.loc[:, feature_columns])
    )
    output = feature_frame.copy()
    output[PROBABILITY_COLUMNS] = probabilities
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    output["predicted_outcome"] = prediction
    output["predicted_label"] = [OUTCOME_LABELS[int(value)] for value in prediction]
    output["confidence"] = confidence
    output["is_high_confidence"] = confidence >= confidence_threshold
    output["policy_pick"] = output["predicted_label"].where(
        output["is_high_confidence"],
        "abstain",
    )
    return output


def merge_actual_results(
    predictions: pd.DataFrame,
    actual_results_path: str | Path,
    *,
    mapping_path: str | Path,
) -> pd.DataFrame:
    actual = pd.read_csv(actual_results_path)
    actual, _ = normalize_match_teams(actual, mapping_path)
    actual["date"] = pd.to_datetime(actual["date"], utc=True).dt.normalize()
    actual["team1_goals"] = pd.to_numeric(actual["team1_goals"], errors="raise").astype(int)
    actual["team2_goals"] = pd.to_numeric(actual["team2_goals"], errors="raise").astype(int)
    actual["actual_outcome"] = 1
    actual.loc[actual["team1_goals"] > actual["team2_goals"], "actual_outcome"] = 2
    actual.loc[actual["team1_goals"] < actual["team2_goals"], "actual_outcome"] = 0
    actual["actual_label"] = actual["actual_outcome"].map(OUTCOME_LABELS)
    actual["actual_score"] = (
        actual["team1_goals"].astype(str) + "-" + actual["team2_goals"].astype(str)
    )
    actual_columns = [
        "date",
        "team1",
        "team2",
        "team1_goals",
        "team2_goals",
        "actual_outcome",
        "actual_label",
        "actual_score",
        "source_url",
    ]
    base = predictions.copy()
    base["_match_day"] = pd.to_datetime(base["date"], utc=True).dt.normalize()
    merged = base.merge(
        actual[actual_columns],
        left_on=["_match_day", "team1", "team2"],
        right_on=["date", "team1", "team2"],
        how="left",
        suffixes=("", "_actual"),
    ).drop(columns=["_match_day", "date_actual"], errors="ignore")
    merged["has_result"] = merged["actual_outcome"].notna()
    merged["prediction_correct"] = (
        merged["predicted_outcome"].eq(merged["actual_outcome"])
        & merged["has_result"]
    )
    merged["policy_correct"] = (
        merged["prediction_correct"]
        & merged["is_high_confidence"].astype(bool)
    )
    merged["status"] = np.where(merged["has_result"], "completed", merged["status"])
    return merged


def summarize_prediction_results(predictions: pd.DataFrame) -> dict[str, object]:
    completed = predictions.loc[predictions["has_result"].astype(bool)]
    selected = completed.loc[completed["is_high_confidence"].astype(bool)]
    return {
        "completed_matches": int(len(completed)),
        "prediction_accuracy": (
            float(completed["prediction_correct"].mean()) if len(completed) else None
        ),
        "high_confidence_completed": int(len(selected)),
        "high_confidence_accuracy": (
            float(selected["prediction_correct"].mean()) if len(selected) else None
        ),
        "correct_predictions": int(completed["prediction_correct"].sum()),
        "wrong_predictions": int((~completed["prediction_correct"]).sum()) if len(completed) else 0,
        "high_confidence_correct": int(selected["prediction_correct"].sum()),
        "high_confidence_wrong": int((~selected["prediction_correct"]).sum()) if len(selected) else 0,
    }
