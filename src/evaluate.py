from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from utils import FIGURES_DIR, models_path, project_directories


matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def load_predictions(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path else models_path("model_predictions.csv")
    df = pd.read_csv(source, parse_dates=["timestamp"])
    return df


def mean_absolute_percentage_error(actual: pd.Series, predicted: pd.Series) -> float:
    safe_actual = actual.replace(0, np.nan)
    return float((np.abs((actual - predicted) / safe_actual)).mean() * 100)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in df.groupby("model"):
        rmse = root_mean_squared_error(group["actual"], group["prediction"])
        mae = mean_absolute_error(group["actual"], group["prediction"])
        mape = mean_absolute_percentage_error(group["actual"], group["prediction"])
        rows.append({"model": model_name, "rmse": rmse, "mae": mae, "mape": mape})

    metrics = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return metrics


def compute_location_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, location_id), group in df.groupby(["model", "location_id"]):
        rows.append(
            {
                "model": model_name,
                "location_id": location_id,
                "rmse": root_mean_squared_error(group["actual"], group["prediction"]),
                "mae": mean_absolute_error(group["actual"], group["prediction"]),
                "mape": mean_absolute_percentage_error(group["actual"], group["prediction"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "location_id"]).reset_index(drop=True)


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_names = ["rmse", "mae", "mape"]
    titles = ["RMSE", "MAE", "MAPE"]

    for ax, metric_name, title in zip(axes, metric_names, titles):
        sns.barplot(data=metrics, x="model", y=metric_name, hue="model", legend=False, ax=ax)
        ax.set_title(f"Model Comparison: {title}")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=25)

    destination = FIGURES_DIR / "model_comparison_metrics.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Urban Noise Analytics model predictions.")
    parser.add_argument("--input", type=str, default=None, help="Optional override for model_predictions.csv.")
    args = parser.parse_args()

    project_directories()
    predictions = load_predictions(args.input)
    metrics = compute_metrics(predictions)
    location_metrics = compute_location_metrics(predictions)

    metrics.to_csv(models_path("model_metrics.csv"), index=False)
    location_metrics.to_csv(models_path("model_metrics_by_location.csv"), index=False)
    plot_model_comparison(metrics)

    print("Evaluation complete.")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
