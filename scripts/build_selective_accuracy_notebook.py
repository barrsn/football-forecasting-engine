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
            """# High-confidence accuracy policy

## tl;dr

The core model's full-match holdout accuracy is 61.93%. A validation-selected
confidence threshold of 0.50 produces **73.19% holdout accuracy** on **65.60%
coverage**. Uncertain matches are marked `abstain`; their probabilities remain
available for simulation and probabilistic scoring."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

The threshold is selected on rolling-origin predictions only. It must achieve
at least 65% accuracy separately in every validation year, with at least 100
selected predictions per year. The fixed 2025-2026 holdout is used only for the
final audit."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

results = pd.read_csv(ROOT / "reports/selective_accuracy/results.csv")
selection = json.loads(
    (ROOT / "reports/selective_accuracy/selection.json").read_text()
)
predictions = pd.read_csv(
    ROOT / "reports/selective_accuracy/holdout_predictions.csv"
)"""
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            """results[[
    "split", "year", "threshold", "coverage", "selective_accuracy",
    "full_accuracy", "selected_matches", "total_matches"
]]"""
        ),
        nbf.v4.new_code_cell(
            """summary = pd.DataFrame([
    {
        "metric": "Rolling selected accuracy",
        "value": selection["validation"]["selective_accuracy"],
    },
    {
        "metric": "Rolling coverage",
        "value": selection["validation"]["coverage"],
    },
    {
        "metric": "Holdout selected accuracy",
        "value": selection["holdout"]["selective_accuracy"],
    },
    {
        "metric": "Holdout coverage",
        "value": selection["holdout"]["coverage"],
    },
    {
        "metric": "Holdout full accuracy",
        "value": selection["holdout"]["full_accuracy"],
    },
])
summary"""
        ),
        nbf.v4.new_markdown_cell("## Prediction output"),
        nbf.v4.new_code_cell(
            """predictions[[
    "team1", "team2", "p_team1_win", "p_draw", "p_team2_win",
    "predicted_label", "confidence", "is_high_confidence"
]].head(20)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. The requested 65% target is exceeded for the operational high-confidence
   subset, including every rolling validation year.
2. Coverage is part of the contract: approximately one third of matches are
   deliberately not assigned a hard pick.
3. Probability metrics and tournament simulation remain based on all matches.
4. Reporting selected accuracy without coverage would be misleading, so both
   are emitted in every artifact."""
        ),
    ]
    output = PROJECT_ROOT / "notebooks/high_confidence_accuracy.ipynb"
    nbf.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()
