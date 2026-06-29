from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTCOME_LABELS = {
    0: "Team 2 win",
    1: "Draw",
    2: "Team 1 win",
}
PROBABILITY_COLUMNS = ["p_team2_win", "p_draw", "p_team1_win"]


def available_report_dirs() -> list[Path]:
    candidates = [
        PROJECT_ROOT / "reports" / "champion_model",
        PROJECT_ROOT / "reports" / "current_model_search",
        PROJECT_ROOT / "reports" / "real_model_search",
        PROJECT_ROOT / "reports" / "fifa_boosting_search",
        PROJECT_ROOT / "reports" / "fifa_model_search",
    ]
    return [
        path
        for path in candidates
        if (path / "selection.json").exists()
        and (
            (path / "holdout_predictions.csv").exists()
            or (path / "test_predictions.csv").exists()
        )
    ]


@st.cache_data
def _load_json_cached(path: str, modified_at: float) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def _load_csv_cached(path: str, modified_at: float) -> pd.DataFrame:
    return pd.read_csv(path)


def load_json(path: str) -> dict[str, object]:
    file_path = Path(path)
    return _load_json_cached(str(file_path), file_path.stat().st_mtime)


def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    return _load_csv_cached(str(file_path), file_path.stat().st_mtime)


def predictions_path(report_dir: Path) -> Path:
    for name in ("test_predictions.csv", "holdout_predictions.csv"):
        path = report_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No predictions CSV found in {report_dir}")


def metrics_path(report_dir: Path) -> Path:
    for name in ("test_results.csv", "holdout_results.csv"):
        path = report_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No metrics CSV found in {report_dir}")


def prepare_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce")
    probabilities = data[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    data["predicted_outcome"] = probabilities.argmax(axis=1)
    data["confidence"] = probabilities.max(axis=1)
    data["prediction"] = data["predicted_outcome"].map(OUTCOME_LABELS)
    if "outcome" in data:
        data["actual"] = data["outcome"].map(OUTCOME_LABELS)
        data["correct"] = data["predicted_outcome"].eq(data["outcome"])
    if {"team1_goals", "team2_goals"}.issubset(data.columns):
        data["score"] = (
            data["team1_goals"].astype(str) + "-" + data["team2_goals"].astype(str)
        )
    return data


def selected_model(selection: dict[str, object]) -> str:
    for key in ("deployment_model", "selected_model", "selected_candidate"):
        value = selection.get(key)
        if value:
            return str(value)
    metrics = selection.get("deployment_holdout_metrics")
    if isinstance(metrics, dict) and metrics.get("model"):
        return str(metrics["model"])
    return "Unknown"


def load_selective_policy() -> tuple[dict[str, object], pd.DataFrame] | None:
    report_dir = PROJECT_ROOT / "reports" / "selective_accuracy"
    selection_path = report_dir / "selection.json"
    predictions_file = report_dir / "holdout_predictions.csv"
    if not selection_path.exists() or not predictions_file.exists():
        return None
    return load_json(str(selection_path)), prepare_predictions(load_csv(str(predictions_file)))


def load_world_cup_now() -> tuple[dict[str, object], pd.DataFrame] | None:
    report_dir = PROJECT_ROOT / "reports" / "world_cup_now"
    summary_path = report_dir / "summary.json"
    predictions_file = report_dir / "predictions.csv"
    if not summary_path.exists() or not predictions_file.exists():
        return None
    frame = prepare_predictions(load_csv(str(predictions_file)))
    if "policy_pick" not in frame and "predicted_label" in frame:
        frame["policy_pick"] = frame["predicted_label"].where(
            frame.get("is_high_confidence", False),
            "abstain",
        )
    return load_json(str(summary_path)), frame


def existing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


st.set_page_config(page_title="Football Forecasting Engine", layout="wide")
st.title("Football Forecasting Engine")
st.caption("Model reports and W/D/L probability forecasts from chronological evaluation")

report_dirs = available_report_dirs()
if not report_dirs:
    st.error("No model report with predictions was found under reports/.")
    st.stop()

with st.sidebar:
    selected_report = st.selectbox(
        "Report",
        report_dirs,
        format_func=lambda path: path.relative_to(PROJECT_ROOT).as_posix(),
    )

selection = load_json(str(selected_report / "selection.json"))
metrics = load_csv(str(metrics_path(selected_report)))
predictions = prepare_predictions(load_csv(str(predictions_path(selected_report))))
audit_path = PROJECT_ROOT / "reports" / "current_data_audit.json"
audit = load_json(str(audit_path)) if audit_path.exists() else {}
selective_policy = load_selective_policy()
selective_selection = selective_policy[0] if selective_policy else {}
selective_predictions_frame = selective_policy[1] if selective_policy else pd.DataFrame()
can_use_selective = selected_report.name == "champion_model" and selective_policy is not None
policy_threshold = float(selective_selection.get("threshold", 0.0)) if can_use_selective else 0.0
world_cup_now = load_world_cup_now()
world_cup_summary = world_cup_now[0] if world_cup_now else {}
world_cup_predictions = world_cup_now[1] if world_cup_now else pd.DataFrame()

with st.sidebar:
    use_selective = st.checkbox(
        "75%+ high-confidence policy",
        value=can_use_selective,
        disabled=not can_use_selective,
    )
    min_confidence = st.slider(
        "Minimum confidence",
        0.0,
        1.0,
        policy_threshold if use_selective else 0.0,
        0.01,
    )
    team_query = st.text_input("Team filter", "")
    only_correct = st.checkbox("Only correct predictions", value=False)

model_name = selected_model(selection)
deployment_metrics = metrics.loc[metrics["model"].astype(str).eq(model_name)]
if deployment_metrics.empty:
    deployment_metrics = metrics.sort_values(["log_loss", "brier", "rps"]).head(1)
metric_row = deployment_metrics.iloc[0]

left, middle, right = st.columns([1.2, 1.0, 1.0])
left.metric("Model", model_name)
middle.metric("Log Loss", f"{metric_row['log_loss']:.4f}")
if use_selective:
    holdout_policy = selective_selection["holdout"]
    right.metric("High-confidence accuracy", f"{holdout_policy['selective_accuracy']:.2%}")
else:
    right.metric("Accuracy", f"{metric_row['accuracy']:.2%}")

metric_cols = st.columns(4)
metric_cols[0].metric("Brier", f"{metric_row['brier']:.4f}")
metric_cols[1].metric("RPS", f"{metric_row['rps']:.4f}")
metric_cols[2].metric("Calibration error", f"{metric_row['calibration_error']:.4f}")
if use_selective:
    metric_cols[3].metric(
        "Coverage",
        f"{holdout_policy['coverage']:.2%}",
        f"{holdout_policy['selected_matches']:,}/{holdout_policy['total_matches']:,}",
    )
else:
    metric_cols[3].metric("Matches", f"{int(metric_row['n_matches']):,}")

if can_use_selective:
    holdout_policy = selective_selection["holdout"]
    validation_policy = selective_selection["validation"]
    st.success(
        "Improved high-confidence policy: "
        f"{holdout_policy['selective_accuracy']:.2%} holdout accuracy at "
        f"{holdout_policy['coverage']:.2%} coverage. "
        f"Threshold {selective_selection['threshold']:.2f} was selected on rolling "
        f"validation only, where accuracy was {validation_policy['selective_accuracy']:.2%}."
    )

if audit:
    st.info(
        "Current completed-match data ends at "
        f"{audit.get('output_max_date')}. "
        f"{audit.get('incomplete_rows')} raw rows are incomplete and were excluded."
    )

tabs = st.tabs(["World Cup 2026", "Forecasts", "Model Comparison", "Data"])

with tabs[1]:
    display_predictions = (
        selective_predictions_frame
        if use_selective and not selective_predictions_frame.empty
        else predictions
    )
    filtered = display_predictions.loc[
        display_predictions["confidence"].ge(min_confidence)
    ].copy()
    if use_selective and "is_high_confidence" in filtered:
        filtered = filtered.loc[filtered["is_high_confidence"].astype(bool)]
    if team_query.strip():
        query = team_query.strip().casefold()
        filtered = filtered.loc[
            filtered["team1"].astype(str).str.casefold().str.contains(query, na=False)
            | filtered["team2"].astype(str).str.casefold().str.contains(query, na=False)
        ]
    if only_correct and "correct" in filtered:
        filtered = filtered.loc[filtered["correct"]]

    st.subheader("Forecasts")
    forecast_columns = [
        "date",
        "team1",
        "team2",
        "prediction",
        "confidence",
        "p_team1_win",
        "p_draw",
        "p_team2_win",
        "score",
        "actual",
        "correct",
    ]
    if use_selective and "predicted_label" in filtered:
        filtered["policy_pick"] = filtered["predicted_label"]
        forecast_columns.insert(3, "policy_pick")
    st.dataframe(
        filtered.sort_values(["date", "confidence"], ascending=[False, False])[
            existing_columns(filtered, forecast_columns)
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Highest confidence")
    st.dataframe(
        predictions.sort_values("confidence", ascending=False)
        .head(25)[
            existing_columns(
                predictions,
                [
                "date",
                "team1",
                "team2",
                "prediction",
                "confidence",
                "p_team1_win",
                "p_draw",
                "p_team2_win",
                "score",
                "actual",
                ],
            )
        ],
        use_container_width=True,
        hide_index=True,
    )

with tabs[0]:
    st.subheader("World Cup 2026 Now")
    if world_cup_predictions.empty:
        st.warning(
            "No World Cup report was found. Run "
            "`python scripts/build_world_cup_now_report.py`."
        )
    else:
        wc_cols = st.columns(4)
        wc_cols[0].metric("Fixtures", f"{world_cup_summary['fixture_rows']:,}")
        wc_cols[1].metric("Groups", f"{len(world_cup_summary['groups'])}")
        wc_cols[2].metric(
            "High-confidence picks",
            f"{world_cup_summary['high_confidence_picks']:,}",
        )
        wc_cols[3].metric(
            "Actual scores joined",
            f"{world_cup_summary['results_available']:,}",
            f"{world_cup_summary['results_missing']:,} missing",
        )
        result_summary = world_cup_summary.get("prediction_results", {})
        if result_summary:
            result_cols = st.columns(4)
            result_cols[0].metric(
                "Prediction accuracy",
                f"{result_summary['prediction_accuracy']:.2%}",
                f"{result_summary['correct_predictions']} right / {result_summary['wrong_predictions']} wrong",
            )
            result_cols[1].metric(
                "High-confidence accuracy",
                f"{result_summary['high_confidence_accuracy']:.2%}",
                (
                    f"{result_summary['high_confidence_correct']} right / "
                    f"{result_summary['high_confidence_wrong']} wrong"
                ),
            )
            result_cols[2].metric(
                "Completed compared",
                f"{result_summary['completed_matches']:,}",
            )
            result_cols[3].metric(
                "High-confidence compared",
                f"{result_summary['high_confidence_completed']:,}",
            )
        st.caption(f"As of: {world_cup_summary.get('as_of_date', 'local snapshot')}")
        st.info(str(world_cup_summary["note"]))
        if world_cup_summary.get("actual_results_source_url"):
            st.caption(
                "Actual results source: "
                f"{world_cup_summary['actual_results_source_url']}"
            )

        with st.sidebar:
            world_cup_only_high = st.checkbox(
                "World Cup high-confidence only",
                value=False,
            )
            world_cup_group = st.selectbox(
                "World Cup group",
                ["All", *world_cup_summary["groups"]],
            )

        wc_frame = world_cup_predictions.copy()
        if world_cup_only_high:
            wc_frame = wc_frame.loc[wc_frame["is_high_confidence"].astype(bool)]
        if world_cup_group != "All":
            wc_frame = wc_frame.loc[wc_frame["group"].eq(world_cup_group)]
        if team_query.strip():
            query = team_query.strip().casefold()
            wc_frame = wc_frame.loc[
                wc_frame["team1"].astype(str).str.casefold().str.contains(query, na=False)
                | wc_frame["team2"].astype(str).str.casefold().str.contains(query, na=False)
            ]

        st.subheader("Fixtures and Predictions")
        st.dataframe(
            wc_frame.sort_values(["date", "group", "match_id"])[
                [
                    "group",
                    "status",
                    "date",
                    "team1",
                    "team2",
                    "city",
                    "country",
                    "policy_pick",
                    "actual_score",
                    "actual_label",
                    "prediction_correct",
                    "prediction",
                    "confidence",
                    "p_team1_win",
                    "p_draw",
                    "p_team2_win",
                    "is_high_confidence",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Strongest World Cup Picks")
        st.dataframe(
            world_cup_predictions.sort_values("confidence", ascending=False)
            .head(20)[
                [
                    "group",
                    "status",
                    "date",
                    "team1",
                    "team2",
                    "policy_pick",
                    "actual_score",
                    "actual_label",
                    "prediction_correct",
                    "confidence",
                    "p_team1_win",
                    "p_draw",
                    "p_team2_win",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Group Fixture Counts")
        group_counts = (
            world_cup_predictions.groupby("group", as_index=False)
            .agg(fixtures=("match_id", "count"), high_confidence=("is_high_confidence", "sum"))
            .sort_values("group")
        )
        st.dataframe(group_counts, use_container_width=True, hide_index=True)

        st.subheader("Wrong Predictions")
        wrong = world_cup_predictions.loc[
            world_cup_predictions["has_result"].astype(bool)
            & ~world_cup_predictions["prediction_correct"].astype(bool)
        ]
        st.dataframe(
            wrong.sort_values(["date", "group"])[
                [
                    "group",
                    "date",
                    "team1",
                    "team2",
                    "actual_score",
                    "actual_label",
                    "prediction",
                    "confidence",
                    "is_high_confidence",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

with tabs[2]:
    st.subheader("Model comparison")
    st.dataframe(
        metrics.sort_values(["log_loss", "brier", "rps"]),
        use_container_width=True,
        hide_index=True,
    )
    st.json(selection)

with tabs[3]:
    st.subheader("Prediction data")
    st.write(f"Report directory: `{selected_report.relative_to(PROJECT_ROOT).as_posix()}`")
    st.write(f"Prediction rows: `{len(predictions):,}`")
    if audit:
        st.json(audit)
    st.dataframe(predictions.tail(50), use_container_width=True, hide_index=True)
