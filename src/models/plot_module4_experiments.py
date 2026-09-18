"""Generate reproducible visuals for the running Module 4 experiment record."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "reports/figures/module4"
TABLE_DIR = PROJECT_ROOT / "reports/tables"


def save_bar(series: pd.Series, title: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def split_figure() -> None:
    path = TABLE_DIR / "module4_split_summary.json"
    if not path.exists():
        return
    data = pd.read_json(path)
    if "splits" not in data.columns:
        return


def candidate_figures() -> None:
    path = TABLE_DIR / "module4_candidate_model_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).set_index("model")
    save_bar(
        df["mean_roc_auc"].sort_values(ascending=False),
        "Module 4 candidate models: mean ROC-AUC",
        "Mean ROC-AUC",
        "candidate_models_mean_roc_auc.png",
    )
    save_bar(
        (df["mean_top20_capture"] * 100).sort_values(ascending=False),
        "Module 4 candidate models: top-20% delinquency capture",
        "Mean capture (%)",
        "candidate_models_top20_capture.png",
    )


def tuning_figures() -> None:
    path = TABLE_DIR / "module4_tuning_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    best = (
        df.sort_values(["model", "mean_roc_auc"], ascending=[True, False])
        .groupby("model", as_index=False)
        .first()
        .set_index("model")
    )
    save_bar(
        best["mean_roc_auc"].sort_values(ascending=False),
        "Best tuned nonlinear models: mean ROC-AUC",
        "Mean ROC-AUC",
        "tuned_models_mean_roc_auc.png",
    )
    save_bar(
        (best["mean_top20_capture"] * 100).sort_values(ascending=False),
        "Best tuned nonlinear models: top-20% delinquency capture",
        "Mean capture (%)",
        "tuned_models_top20_capture.png",
    )


def final_holdout_figures() -> None:
    path = TABLE_DIR / "module4_final_holdout_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).set_index("model")
    save_bar(
        df["roc_auc"].sort_values(ascending=False),
        "Final holdout: ROC-AUC",
        "ROC-AUC",
        "final_holdout_roc_auc.png",
    )
    save_bar(
        (df["top20_capture"] * 100).sort_values(ascending=False),
        "Final holdout: top-20% delinquency capture",
        "Capture (%)",
        "final_holdout_top20_capture.png",
    )


def main() -> None:
    candidate_figures()
    tuning_figures()
    final_holdout_figures()
    print(f"Module 4 figures written to: {FIGURE_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
