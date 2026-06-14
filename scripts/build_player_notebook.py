from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (trade310)",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.10"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Player features and model evaluation

## tl;dr

- 47,601 pinned scorer events produced 27 prior-only player-threat features.
- The optional player + LightGBM blend reached rolling Log Loss **0.8662**.
- Final holdout Log Loss was **0.8169**, versus **0.8153** for the simpler core champion.
- The candidate was correctly **not promoted**.
- Full squad, injury, availability, lineup, expected-minutes, and player-rating
  features are implemented, but require timestamped historical snapshots before
  they can enter training."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

All model choices use expanding chronological folds for 2018, 2021, 2022,
2023, and 2024. No random split is used. Player information must be available
strictly before kickoff. Scorer-event coverage is modeled explicitly because
the source is incomplete."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

coverage = json.loads((ROOT / "reports/player_features/coverage.json").read_text())
selection = json.loads((ROOT / "reports/player_model_search/selection.json").read_text())
validation = pd.read_csv(ROOT / "reports/player_model_search/rolling_aggregate.csv")
holdout = pd.read_csv(ROOT / "reports/player_model_search/holdout_results.csv")
champion_validation = pd.read_csv(ROOT / "reports/champion_model/validation_results.csv")
champion_holdout = pd.read_csv(ROOT / "reports/champion_model/holdout_results.csv")"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            """pd.DataFrame({
    "measure": [
        "model rows",
        "scorer-derived features",
        "rows with both team histories",
        "team1 mean source coverage",
        "team2 mean source coverage",
    ],
    "value": [
        coverage["rows"],
        coverage["feature_count"],
        coverage["rows_with_both_scorer_histories"],
        coverage["team1_mean_source_coverage"],
        coverage["team2_mean_source_coverage"],
    ],
})"""
        ),
        nbf.v4.new_markdown_cell(
            """The roughly 42.7% event completeness is a material limitation.
The model receives coverage and history-count columns so incomplete source data
is not silently interpreted as weak players."""
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            """validation[[
    "model", "log_loss", "brier", "rps", "accuracy", "calibration_error"
]].sort_values("log_loss")"""
        ),
        nbf.v4.new_code_cell(
            """comparison = pd.concat([
    champion_holdout.assign(source="core champion"),
    holdout.assign(source="player candidate"),
], ignore_index=True)
comparison[[
    "source", "model", "log_loss", "brier", "rps", "accuracy",
    "calibration_error"
]].sort_values("log_loss")"""
        ),
        nbf.v4.new_markdown_cell("## Snapshot feature contract"),
        nbf.v4.new_code_cell(
            """template = pd.read_csv(ROOT / "data/templates/player_snapshots.csv")
pd.DataFrame({"supported_column": template.columns})"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. Player-derived information can improve historical validation, but the
   improvement was too small and did not survive the final holdout.
2. The core champion remains the production model because it is simpler and
   has better holdout Log Loss, Brier, and RPS.
3. Official squad and lineup snapshots can now be ingested without leakage.
   They must not be promoted until matching historical snapshots cover the
   evaluation period.
4. `models/world_cup_2026_player_scorer.joblib` is an optional research artifact,
   not the default production model."""
        ),
    ]
    output = PROJECT_ROOT / "notebooks/player_features_model_evaluation.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()
