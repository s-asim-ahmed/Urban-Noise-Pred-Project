from __future__ import annotations

import argparse
from pathlib import Path

import folium
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import FIGURES_DIR, processed_path, project_directories


matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def load_processed_dataset(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path else processed_path("urban_noise_processed.parquet")
    if source.suffix == ".csv":
        return pd.read_csv(source, parse_dates=["timestamp"])
    return pd.read_parquet(source)


def save_figure(fig: plt.Figure, file_name: str) -> None:
    destination = FIGURES_DIR / file_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    borough_order = (
        df.groupby("borough", observed=True)["noise_level_db"]
        .median()
        .sort_values(ascending=False)
        .index
    )
    sns.boxplot(data=df, x="borough", y="noise_level_db", order=borough_order, ax=ax)
    ax.set_title("Noise Distribution by Borough")
    ax.set_xlabel("Borough")
    ax.set_ylabel("Noise Level dB(A)")
    ax.tick_params(axis="x", rotation=30)
    save_figure(fig, "noise_distribution_by_borough.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    land_use_order = (
        df.groupby("land_use_type", observed=True)["noise_level_db"]
        .median()
        .sort_values(ascending=False)
        .index
    )
    sns.boxplot(data=df, x="land_use_type", y="noise_level_db", order=land_use_order, ax=ax)
    ax.set_title("Noise Distribution by Land Use")
    ax.set_xlabel("Land Use Type")
    ax.set_ylabel("Noise Level dB(A)")
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, "noise_distribution_by_land_use.png")


def plot_temporal_profiles(df: pd.DataFrame) -> None:
    hourly = (
        df.groupby(["hour", "land_use_type"], observed=True)["noise_level_db"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=hourly, x="hour", y="noise_level_db", hue="land_use_type", marker="o", ax=ax)
    ax.set_title("Hourly Noise Profile by Land Use")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Noise dB(A)")
    save_figure(fig, "hourly_noise_profile.png")

    daily = (
        df.assign(date=df["timestamp"].dt.date)
        .groupby(["date", "borough"], observed=True)["noise_level_db"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=daily, x="date", y="noise_level_db", hue="borough", marker="o", ax=ax)
    ax.set_title("Daily Mean Noise by Borough")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Noise dB(A)")
    ax.tick_params(axis="x", rotation=30)
    save_figure(fig, "daily_noise_profile.png")

    weekend_profile = (
        df.groupby(["hour", "is_weekend"], observed=True)["noise_level_db"]
        .mean()
        .reset_index()
    )
    weekend_profile["day_type"] = weekend_profile["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=weekend_profile, x="hour", y="noise_level_db", hue="day_type", marker="o", ax=ax)
    ax.set_title("Weekday vs Weekend Hourly Noise")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Noise dB(A)")
    save_figure(fig, "weekday_vs_weekend_profile.png")

    heatmap_df = (
        df.groupby(["day_of_week", "hour"], observed=True)["noise_level_db"]
        .mean()
        .reset_index()
        .pivot(index="day_of_week", columns="hour", values="noise_level_db")
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(heatmap_df, cmap="magma", ax=ax)
    ax.set_title("Noise Heatmap by Day of Week and Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    save_figure(fig, "noise_heatmap_day_hour.png")


def build_map(df: pd.DataFrame) -> None:
    location_summary = (
        df.groupby(["location_id", "location_name", "borough", "latitude", "longitude"], observed=True)["noise_level_db"]
        .mean()
        .reset_index()
        .rename(columns={"noise_level_db": "avg_noise_level_db"})
    )

    center = [location_summary["latitude"].mean(), location_summary["longitude"].mean()]
    noise_map = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    for row in location_summary.itertuples(index=False):
        color = "green"
        if row.avg_noise_level_db >= 75:
            color = "red"
        elif row.avg_noise_level_db >= 65:
            color = "orange"

        popup = (
            f"<b>{row.location_name}</b><br>"
            f"Borough: {row.borough}<br>"
            f"Average Noise: {row.avg_noise_level_db:.1f} dB(A)"
        )
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.85,
            popup=popup,
        ).add_to(noise_map)

    noise_map.save(FIGURES_DIR / "average_noise_map.html")


def plot_correlations(df: pd.DataFrame) -> None:
    correlation_columns = [
        "noise_level_db",
        "traffic_volume_veh_hr",
        "avg_speed_kmh",
        "temperature_c",
        "humidity_pct",
        "precipitation_mm",
        "wind_speed_kmh",
    ]
    corr = df[correlation_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation of Noise, Traffic, and Weather Variables")
    save_figure(fig, "correlation_heatmap.png")


def write_summary(df: pd.DataFrame) -> None:
    borough_means = (
        df.groupby("borough", observed=True)["noise_level_db"]
        .mean()
        .sort_values(ascending=False)
    )
    land_use_means = (
        df.groupby("land_use_type", observed=True)["noise_level_db"]
        .mean()
        .sort_values(ascending=False)
    )
    hourly_means = df.groupby("hour", observed=True)["noise_level_db"].mean()
    peak_hour = int(hourly_means.idxmax())
    quiet_hour = int(hourly_means.idxmin())
    corr = df[["noise_level_db", "traffic_volume_veh_hr", "temperature_c", "humidity_pct"]].corr(numeric_only=True)

    summary = f"""# EDA Summary

- Highest average borough noise: **{borough_means.index[0]}** at **{borough_means.iloc[0]:.2f} dB(A)**.
- Quietest average borough noise: **{borough_means.index[-1]}** at **{borough_means.iloc[-1]:.2f} dB(A)**.
- Loudest land use category: **{land_use_means.index[0]}** at **{land_use_means.iloc[0]:.2f} dB(A)**.
- Quietest land use category: **{land_use_means.index[-1]}** at **{land_use_means.iloc[-1]:.2f} dB(A)**.
- Peak average hour: **{peak_hour:02d}:00** with **{hourly_means.loc[peak_hour]:.2f} dB(A)**.
- Quietest average hour: **{quiet_hour:02d}:00** with **{hourly_means.loc[quiet_hour]:.2f} dB(A)**.
- Noise and traffic volume correlation: **{corr.loc['noise_level_db', 'traffic_volume_veh_hr']:.2f}**.
- Noise and temperature correlation: **{corr.loc['noise_level_db', 'temperature_c']:.2f}**.
- Noise and humidity correlation: **{corr.loc['noise_level_db', 'humidity_pct']:.2f}**.

Interpretation:
The synthetic sample behaves like a plausible urban network: noisier commercial corridors remain consistently louder than park or residential contexts, and the strongest temporal peaks align with high-traffic commuting and daytime activity windows. Traffic volume is the most directly associated predictor among the available covariates, which supports using lagged traffic and short rolling windows in the forecasting stage. Weather variables show weaker associations, so they are useful as contextual features rather than primary drivers.
"""

    destination = FIGURES_DIR / "eda_summary.md"
    destination.write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis for Urban Noise Analytics.")
    parser.add_argument("--input", type=str, default=None, help="Optional override for the processed dataset path.")
    args = parser.parse_args()

    project_directories()
    df = load_processed_dataset(args.input)
    plot_distributions(df)
    plot_temporal_profiles(df)
    build_map(df)
    plot_correlations(df)
    write_summary(df)
    print("EDA outputs saved to outputs/figures.")


if __name__ == "__main__":
    main()
