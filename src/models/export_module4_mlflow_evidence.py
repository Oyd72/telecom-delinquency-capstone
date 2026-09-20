"""Export a compact, repository-safe record of Module 4 MLflow experiments.

The MLflow SQLite database stays local and is not committed. This script reads the
local tracking database and exports only aggregate run metadata, parameters and metrics
needed to demonstrate experiment tracking for the assignment.

Outputs:
- reports/tables/module4_mlflow_runs.csv
- reports/tables/module4_mlflow_experiments.json
- reports/figures/module4/mlflow_experiment_run_counts.png
- reports/module4_mlflow_evidence.md
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

TABLE_PATH = PROJECT_ROOT / "reports/tables/module4_mlflow_runs.csv"
JSON_PATH = PROJECT_ROOT / "reports/tables/module4_mlflow_experiments.json"
FIGURE_PATH = PROJECT_ROOT / "reports/figures/module4/mlflow_experiment_run_counts.png"
REPORT_PATH = PROJECT_ROOT / "reports/module4_mlflow_evidence.md"

EXPECTED_EXPERIMENTS = [
    "module4_candidate_model_benchmark",
    "module4_model_tuning",
    "module4_final_holdout_evaluation",
    "module4_development_calibration",
    "module4_calibrated_holdout_sensitivity",
]


def flatten_run(run, experiment_name: str) -> dict:
    row = {
        "experiment": experiment_name,
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "status": run.info.status,
        "start_time": pd.to_datetime(run.info.start_time, unit="ms", utc=True),
        "end_time": (
            pd.to_datetime(run.info.end_time, unit="ms", utc=True)
            if run.info.end_time is not None
            else pd.NaT
        ),
    }

    for key, value in sorted(run.data.params.items()):
        row[f"param__{key}"] = value

    for key, value in sorted(run.data.metrics.items()):
        row[f"metric__{key}"] = value

    return row


def main() -> None:
    if not MLFLOW_DB.exists():
        raise FileNotFoundError(
            f"MLflow database not found: {MLFLOW_DB}. "
            "Run the Module 4 benchmark/tuning/evaluation scripts first."
        )

    tracking_uri = f"sqlite:///{MLFLOW_DB.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiments = {
        exp.name: exp
        for exp in client.search_experiments()
        if exp.name in EXPECTED_EXPERIMENTS
    }

    missing = [name for name in EXPECTED_EXPERIMENTS if name not in experiments]
    if missing:
        raise ValueError(
            "Expected MLflow experiments are missing from the local database: "
            + ", ".join(missing)
        )

    run_rows = []
    experiment_rows = []

    for name in EXPECTED_EXPERIMENTS:
        exp = experiments[name]
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time ASC"],
        )

        experiment_rows.append(
            {
                "experiment": name,
                "experiment_id": exp.experiment_id,
                "run_count": len(runs),
                "artifact_location": exp.artifact_location,
            }
        )

        for run in runs:
            run_rows.append(flatten_run(run, name))

    runs_df = pd.DataFrame(run_rows)
    experiments_df = pd.DataFrame(experiment_rows)

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    runs_df.to_csv(TABLE_PATH, index=False)

    payload = {
        "tracking_backend": "sqlite",
        "database_committed_to_git": False,
        "expected_experiments": EXPECTED_EXPERIMENTS,
        "experiments": experiment_rows,
        "total_runs": int(len(runs_df)),
        "note": (
            "The SQLite tracking database remains local. This export contains only "
            "aggregate experiment metadata, parameters and metrics suitable for repository evidence."
        ),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(experiments_df["experiment"], experiments_df["run_count"])
    ax.set_ylabel("Tracked runs")
    ax.set_title("Module 4 MLflow experiment coverage")
    ax.tick_params(axis="x", rotation=28)
    for i, value in enumerate(experiments_df["run_count"]):
        ax.text(i, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report_lines = [
        "# Module 4 MLflow evidence",
        "",
        "The Module 4 modelling work was tracked locally with MLflow using a SQLite backend.",
        "The database itself is intentionally excluded from GitHub, while this report and the",
        "aggregate run export provide a repository-safe record of the experiments.",
        "",
        "## Experiment coverage",
        "",
        "| Experiment | Runs |",
        "|---|---:|",
    ]

    for row in experiment_rows:
        report_lines.append(f"| {row['experiment']} | {row['run_count']} |")

    report_lines += [
        "",
        f"**Total tracked runs:** {len(runs_df)}",
        "",
        "![MLflow experiment run counts](figures/module4/mlflow_experiment_run_counts.png)",
        "",
        "## What was tracked",
        "",
        "The runs record model and calibration choices, feature counts, chronology controls,",
        "and the evaluation metrics produced by each stage. The exact fields vary by experiment",
        "because benchmark, tuning, calibration and holdout evaluation answer different questions.",
        "",
        "The full repository-safe export is available in:",
        "",
        "- `reports/tables/module4_mlflow_runs.csv`",
        "- `reports/tables/module4_mlflow_experiments.json`",
        "",
        "For an assignment screenshot, the local MLflow UI can be opened with:",
        "",
        "```powershell",
        "mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000",
        "```",
        "",
        "Then open `http://127.0.0.1:5000` and select a Module 4 experiment.",
        "A screenshot should show the experiment name and the visible run/metric table.",
        "",
        "## Interpretation",
        "",
        "MLflow is used here for traceability rather than as a source of model truth. The formal",
        "results remain the versioned tables, figures, model card and experiment record in GitHub.",
    ]

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Module 4 MLflow evidence export completed.")
    print(f"Experiments found: {len(experiment_rows)}")
    print(f"Total tracked runs: {len(runs_df)}")
    print()
    print(experiments_df[["experiment", "run_count"]].to_string(index=False))
    print()
    print(f"Run export: {TABLE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Experiment summary: {JSON_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Coverage figure: {FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Evidence report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
