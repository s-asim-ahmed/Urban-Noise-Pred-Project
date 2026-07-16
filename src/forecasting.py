from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from utils import figures_path, models_path, processed_path, project_directories, save_json


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message="Too few observations to estimate starting parameters for seasonal ARMA.*",
    category=UserWarning,
)

MAX_FORECAST_DATASET_ROWS = 1000
MIN_REAL_ROWS_FOR_FORECAST = 24
TARGET_TRAIN_ROWS = 96
DEFAULT_VALIDATION_HOLDOUT_HOURS = 12
HOURLY_FREQUENCY = pd.Timedelta(hours=1)

STATIC_COLUMNS = [
    "location_id",
    "location_name",
    "borough",
    "latitude",
    "longitude",
    "land_use_type",
]

SYNTHETIC_NUMERIC_COLUMNS = [
    "noise_level_db",
    "traffic_volume_veh_hr",
    "avg_speed_kmh",
    "temperature_c",
    "humidity_pct",
    "precipitation_mm",
    "wind_speed_kmh",
]


def load_analysis_dataset(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path else processed_path("urban_noise_processed.parquet")
    df = pd.read_parquet(source)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)


def infer_frequency(df: pd.DataFrame) -> pd.Timedelta:
    diffs = df["timestamp"].sort_values().diff().dropna()
    if diffs.empty:
        return pd.Timedelta(hours=1)
    inferred = diffs.mode().iloc[0]
    if inferred <= pd.Timedelta(0):
        return pd.Timedelta(hours=1)
    return inferred


def build_regularized_history(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    ordered = history.sort_values("timestamp").reset_index(drop=True).copy()
    full_index = pd.date_range(ordered["timestamp"].min(), ordered["timestamp"].max(), freq="h")
    regularized = pd.DataFrame(index=full_index)
    regularized.index.name = "timestamp"
    regularized["noise_level_db"] = ordered.drop_duplicates("timestamp").set_index("timestamp")["noise_level_db"].reindex(full_index)
    regularized["hour"] = regularized.index.hour
    regularized["day_of_week"] = regularized.index.dayofweek
    regularized["is_weekend"] = (regularized["day_of_week"] >= 5).astype(int)

    observed = regularized["noise_level_db"].copy()
    dow_hour_profile = observed.groupby([regularized["day_of_week"], regularized["hour"]]).median()
    hour_profile = observed.groupby(regularized["hour"]).median()
    overall_median = float(observed.median())

    regularized["profile_fill"] = [
        dow_hour_profile.get((day, hour), hour_profile.get(hour, overall_median))
        for day, hour in zip(regularized["day_of_week"], regularized["hour"])
    ]
    regularized["noise_filled"] = regularized["noise_level_db"].fillna(regularized["profile_fill"])
    regularized["noise_filled"] = regularized["noise_filled"].interpolate(method="linear", limit_direction="both")
    regularized["noise_filled"] = regularized["noise_filled"].fillna(overall_median)

    coverage_ratio = float(observed.notna().mean())
    return regularized.reset_index(), {"observed_coverage_ratio": coverage_ratio, "regularized_rows": float(len(regularized))}


def forecast_with_hourly_profile(history: pd.DataFrame, future_index: pd.DatetimeIndex) -> tuple[pd.Series, dict[str, Any]]:
    regularized, regularization_meta = build_regularized_history(history)
    observed = history.sort_values("timestamp").copy()
    overall_level = float(observed["noise_level_db"].median())
    hour_profile = observed.groupby("hour", observed=True)["noise_level_db"].median()
    dow_hour_profile = observed.groupby(["day_of_week", "hour"], observed=True)["noise_level_db"].median()
    month_profile = observed.groupby(observed["timestamp"].dt.month, observed=True)["noise_level_db"].median()
    month_counts = observed.groupby(observed["timestamp"].dt.month, observed=True)["noise_level_db"].size()

    recent_window = observed.tail(min(12, len(observed))).copy()
    expected_recent = pd.Series(
        [
            dow_hour_profile.get((row.day_of_week, row.hour), hour_profile.get(row.hour, overall_level))
            for row in recent_window.itertuples(index=False)
        ],
        index=recent_window.index,
        dtype="float64",
    )
    recent_adjustment = float(recent_window["noise_level_db"].mean() - expected_recent.mean())

    if regularization_meta["observed_coverage_ratio"] >= 0.35 and len(recent_window) >= 6:
        trend_slope = float(np.polyfit(np.arange(len(recent_window), dtype=float), recent_window["noise_level_db"].to_numpy(), 1)[0])
        trend_slope = float(np.clip(trend_slope, -0.1, 0.1))
    else:
        trend_slope = 0.0

    lower_bound = float(history["noise_level_db"].quantile(0.05))
    upper_bound = float(history["noise_level_db"].quantile(0.95))

    predictions = []
    for step, timestamp in enumerate(future_index, start=1):
        base_value = dow_hour_profile.get((timestamp.dayofweek, timestamp.hour), hour_profile.get(timestamp.hour, overall_level))
        month_median = month_profile.get(timestamp.month, overall_level)
        month_weight = min(float(month_counts.get(timestamp.month, 0)) / 4.0, 1.0)
        month_adjustment = (float(month_median) - overall_level) * month_weight * 0.6
        damped_adjustment = recent_adjustment * (0.97 ** ((step - 1) / 24))
        damped_trend = trend_slope * min(step, 24) * 0.15
        predicted = float(
            np.clip(base_value + month_adjustment + damped_adjustment + damped_trend, lower_bound - 3.0, upper_bound + 3.0)
        )
        predictions.append(predicted)

    metadata = {
        "selected_model": "Hourly Seasonal Profile",
        "observed_coverage_ratio": regularization_meta["observed_coverage_ratio"],
        "recent_level_adjustment": recent_adjustment,
        "damped_trend_slope": trend_slope,
        "month_seasonality_enabled": True,
    }
    return pd.Series(predictions, index=future_index, dtype="float64"), metadata


def _to_timestamp(value: str | pd.Timestamp, bound: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    raw = str(value)
    if bound == "end" and ":" not in raw and "T" not in raw:
        timestamp = timestamp + pd.Timedelta(hours=23)
    return timestamp


def _autocorrelation(series: pd.Series, lag: int) -> float:
    if len(series) <= lag:
        return float("nan")
    return float(series.autocorr(lag=lag))


def summarize_forecasting_patterns(df: pd.DataFrame) -> dict[str, Any]:
    ordered = df.sort_values("timestamp").copy()
    hourly_means = ordered.groupby("hour", observed=True)["noise_level_db"].mean()
    day_means = ordered.groupby("day_of_week", observed=True)["noise_level_db"].mean()
    per_location_autocorr = (
        ordered.groupby("location_id", observed=True)["noise_level_db"]
        .apply(lambda values: _autocorrelation(values.reset_index(drop=True), 24))
        .dropna()
    )
    traffic_corr = ordered["noise_level_db"].corr(ordered["traffic_volume_veh_hr"])
    temp_corr = ordered["noise_level_db"].corr(ordered["temperature_c"])
    weather_corr = ordered["noise_level_db"].corr(ordered["wind_speed_kmh"])

    timeline = np.arange(len(ordered), dtype=float)
    trend_slope = float(np.polyfit(timeline, ordered["noise_level_db"].to_numpy(), 1)[0]) if len(ordered) > 1 else 0.0

    return {
        "row_count": int(len(ordered)),
        "location_count": int(ordered["location_id"].nunique()),
        "time_start": str(ordered["timestamp"].min()),
        "time_end": str(ordered["timestamp"].max()),
        "peak_hour": int(hourly_means.idxmax()),
        "quiet_hour": int(hourly_means.idxmin()),
        "peak_day_of_week": int(day_means.idxmax()),
        "weekday_weekend_gap_db": float(
            ordered.loc[ordered["is_weekend"] == 0, "noise_level_db"].mean()
            - ordered.loc[ordered["is_weekend"] == 1, "noise_level_db"].mean()
        ),
        "mean_daily_seasonality_autocorr_lag24": float(per_location_autocorr.mean()) if not per_location_autocorr.empty else float("nan"),
        "noise_traffic_correlation": float(traffic_corr),
        "noise_temperature_correlation": float(temp_corr),
        "noise_wind_correlation": float(weather_corr),
        "linear_trend_slope_per_step": trend_slope,
        "recommended_forecast_model": "Seasonal SARIMA with 24-hour seasonality",
    }


def write_forecasting_eda_summary(summary: dict[str, Any]) -> None:
    save_json(summary, models_path("forecasting_eda_summary.json"))
    markdown = f"""# Forecasting EDA Summary

- Rows analysed: **{summary['row_count']}**
- Locations analysed: **{summary['location_count']}**
- Coverage window: **{summary['time_start']}** to **{summary['time_end']}**
- Peak hour-of-day: **{summary['peak_hour']:02d}:00**
- Quietest hour-of-day: **{summary['quiet_hour']:02d}:00**
- Strongest average day-of-week: **{summary['peak_day_of_week']}**
- Weekday minus weekend mean noise gap: **{summary['weekday_weekend_gap_db']:.2f} dB(A)**
- Mean lag-24 autocorrelation: **{summary['mean_daily_seasonality_autocorr_lag24']:.2f}**
- Noise to traffic correlation: **{summary['noise_traffic_correlation']:.2f}**
- Noise to temperature correlation: **{summary['noise_temperature_correlation']:.2f}**
- Noise to wind-speed correlation: **{summary['noise_wind_correlation']:.2f}**

Model design note:
The dataset shows strong intra-day structure and traffic-linked variation, so the forecasting feature uses a seasonal SARIMA candidate with a 24-hour cycle and falls back to a simpler ARIMA candidate when the seasonal fit is unstable. This keeps the prediction pathway date-driven and aligned with the hourly resolution of the source data.
"""
    figures_path("forecasting_eda_summary.md").write_text(markdown, encoding="utf-8")


def _build_hourly_stats(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.groupby("hour", observed=True)[column].agg(["mean", "std"]).reset_index()


def _sample_value(
    timestamp: pd.Timestamp,
    stats_df: pd.DataFrame,
    fallback_mean: float,
    fallback_std: float,
    lower: float,
    upper: float,
    rng: np.random.Generator,
) -> float:
    hour_stats = stats_df.loc[stats_df["hour"] == timestamp.hour]
    mean = float(hour_stats["mean"].iloc[0]) if not hour_stats.empty else fallback_mean
    std = float(hour_stats["std"].iloc[0]) if not hour_stats.empty else fallback_std
    if not np.isfinite(std) or std <= 0:
        std = max(fallback_std, 0.5)
    sampled = rng.normal(mean, std)
    return float(np.clip(sampled, lower, upper))


def synthetic_alignment_report(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict[str, float | bool]:
    if synthetic_df.empty:
        return {
            "synthetic_rows_added": 0,
            "mean_difference_pct": 0.0,
            "std_difference_pct": 0.0,
            "hourly_profile_correlation": 1.0,
            "passes_alignment_check": True,
        }

    real_noise = real_df["noise_level_db"]
    synthetic_noise = synthetic_df["noise_level_db"]
    real_mean = float(real_noise.mean())
    synthetic_mean = float(synthetic_noise.mean())
    real_std = float(real_noise.std(ddof=0))
    synthetic_std = float(synthetic_noise.std(ddof=0))

    real_hourly = real_df.groupby("hour", observed=True)["noise_level_db"].mean()
    synthetic_hourly = synthetic_df.groupby("hour", observed=True)["noise_level_db"].mean()
    aligned_hourly = pd.concat([real_hourly, synthetic_hourly], axis=1, keys=["real", "synthetic"]).ffill().bfill()
    if aligned_hourly["real"].std(ddof=0) == 0 or aligned_hourly["synthetic"].std(ddof=0) == 0:
        hourly_corr = 0.0
    else:
        hourly_corr_raw = aligned_hourly["real"].corr(aligned_hourly["synthetic"])
        hourly_corr = float(0.0 if pd.isna(hourly_corr_raw) else hourly_corr_raw)

    mean_diff_pct = abs(synthetic_mean - real_mean) / max(abs(real_mean), 1.0)
    std_diff_pct = abs(synthetic_std - real_std) / max(abs(real_std), 1.0)
    passes = mean_diff_pct <= 0.2 and std_diff_pct <= 0.35 and hourly_corr >= 0.6

    return {
        "synthetic_rows_added": int(len(synthetic_df)),
        "mean_difference_pct": float(mean_diff_pct),
        "std_difference_pct": float(std_diff_pct),
        "hourly_profile_correlation": hourly_corr,
        "passes_alignment_check": bool(passes),
    }


def prepare_training_history(
    location_history: pd.DataFrame,
    target_rows: int = TARGET_TRAIN_ROWS,
    max_total_rows: int = MAX_FORECAST_DATASET_ROWS,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    history = location_history.sort_values("timestamp").reset_index(drop=True).copy()
    if len(history) < MIN_REAL_ROWS_FOR_FORECAST:
        raise ValueError(
            f"Insufficient baseline data to run predictions: at least {MIN_REAL_ROWS_FOR_FORECAST} real rows are required."
        )

    history = history.tail(max_total_rows).reset_index(drop=True)
    needed_rows = min(max(target_rows - len(history), 0), max_total_rows - len(history))
    if needed_rows <= 0:
        report = synthetic_alignment_report(history, pd.DataFrame(columns=history.columns))
        report.update(
            {
                "final_rows": int(len(history)),
                "real_rows_used": int(len(history)),
                "synthetic_rows_added": 0,
                "max_total_rows": int(max_total_rows),
            }
        )
        return history, pd.DataFrame(columns=history.columns), report

    rng = np.random.default_rng(random_state)
    frequency = infer_frequency(history)
    first_timestamp = history["timestamp"].min()
    synthetic_timestamps = pd.date_range(end=first_timestamp - frequency, periods=needed_rows, freq=frequency)

    base_row = history.iloc[0]
    hourly_stats = {column: _build_hourly_stats(history, column) for column in SYNTHETIC_NUMERIC_COLUMNS if column in history.columns}
    overall_stats = {
        column: {
            "mean": float(history[column].mean()),
            "std": float(history[column].std(ddof=0)) if np.isfinite(history[column].std(ddof=0)) else 1.0,
            "min": float(history[column].min()),
            "max": float(history[column].max()),
        }
        for column in SYNTHETIC_NUMERIC_COLUMNS
        if column in history.columns
    }

    synthetic_rows: list[dict[str, Any]] = []
    for timestamp in synthetic_timestamps:
        row: dict[str, Any] = {column: base_row[column] for column in STATIC_COLUMNS if column in history.columns}
        row["timestamp"] = timestamp
        row["hour"] = int(timestamp.hour)
        row["day_of_week"] = int(timestamp.dayofweek)
        row["is_weekend"] = int(timestamp.dayofweek >= 5)
        for column, stats_df in hourly_stats.items():
            stats = overall_stats[column]
            row[column] = _sample_value(
                timestamp,
                stats_df,
                stats["mean"],
                stats["std"],
                stats["min"],
                stats["max"],
                rng,
            )
        synthetic_rows.append(row)

    synthetic_df = pd.DataFrame(synthetic_rows).sort_values("timestamp").reset_index(drop=True)
    combined = pd.concat([synthetic_df, history], ignore_index=True).sort_values("timestamp").tail(max_total_rows).reset_index(drop=True)

    report = synthetic_alignment_report(history, synthetic_df)
    if not report["passes_alignment_check"]:
        report.update(
            {
                "final_rows": int(len(history)),
                "real_rows_used": int(len(history)),
                "synthetic_rows_added": 0,
                "max_total_rows": int(max_total_rows),
                "synthetic_fallback_applied": True,
            }
        )
        return history, pd.DataFrame(columns=history.columns), report

    report.update(
        {
            "final_rows": int(len(combined)),
            "real_rows_used": int(len(history)),
            "synthetic_rows_added": int(len(synthetic_df)),
            "max_total_rows": int(max_total_rows),
            "synthetic_fallback_applied": False,
        }
    )
    return combined, synthetic_df, report


def select_and_fit_forecast_model(history: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    ordered = history.sort_values("timestamp").copy()
    regularized, regularization_meta = build_regularized_history(ordered)
    series = regularized.set_index("timestamp")["noise_filled"].asfreq("h")

    if regularization_meta["observed_coverage_ratio"] < 0.35 or len(ordered) < 120:
        return None, {
            "selected_model": "Hourly Seasonal Profile",
            "aic": float("nan"),
            "training_rows": int(len(series)),
            "observed_coverage_ratio": regularization_meta["observed_coverage_ratio"],
        }

    candidates = [
        {
            "label": "Seasonal SARIMA (1,0,1)x(1,0,1,24)",
            "order": (1, 0, 1),
            "seasonal_order": (1, 0, 1, 24),
        },
        {
            "label": "ARIMA (1,0,1)",
            "order": (1, 0, 1),
            "seasonal_order": (0, 0, 0, 0),
        },
    ]

    successful_models: list[tuple[Any, dict[str, Any]]] = []
    for candidate in candidates:
        try:
            fitted = SARIMAX(
                series,
                order=candidate["order"],
                seasonal_order=candidate["seasonal_order"],
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            successful_models.append(
                (
                    fitted,
                    {
                        "selected_model": candidate["label"],
                        "order": candidate["order"],
                        "seasonal_order": candidate["seasonal_order"],
                        "aic": float(fitted.aic),
                        "training_rows": int(len(series)),
                        "observed_coverage_ratio": regularization_meta["observed_coverage_ratio"],
                    },
                )
            )
        except Exception:
            continue

    if not successful_models:
        return None, {
            "selected_model": "Hourly Seasonal Profile",
            "aic": float("nan"),
            "training_rows": int(len(series)),
            "observed_coverage_ratio": regularization_meta["observed_coverage_ratio"],
        }

    fitted_model, metadata = sorted(successful_models, key=lambda item: item[1]["aic"])[0]
    return fitted_model, metadata


def _forecast_from_history(
    history: pd.DataFrame,
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
    max_total_rows: int = MAX_FORECAST_DATASET_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = history.sort_values("timestamp").reset_index(drop=True).copy()
    if ordered.empty:
        raise ValueError("No history is available for the requested location.")

    last_timestamp = ordered["timestamp"].max()
    if start_timestamp <= last_timestamp:
        raise ValueError("Forecast start must be strictly later than the latest available timestamp.")
    if end_timestamp < start_timestamp:
        raise ValueError("Forecast end must be greater than or equal to the start timestamp.")

    training_history, synthetic_df, synthetic_report = prepare_training_history(ordered, max_total_rows=max_total_rows)
    fitted_model, model_metadata = select_and_fit_forecast_model(training_history)
    full_forecast_index = pd.date_range(start=last_timestamp.floor("h") + HOURLY_FREQUENCY, end=end_timestamp, freq="h")
    if full_forecast_index.empty:
        raise ValueError("The requested future date range does not produce any forecast timestamps.")

    if fitted_model is None or model_metadata["selected_model"] == "Hourly Seasonal Profile":
        predicted_series, profile_metadata = forecast_with_hourly_profile(training_history, full_forecast_index)
        model_metadata.update(profile_metadata)
    else:
        predicted = fitted_model.get_forecast(steps=len(full_forecast_index)).predicted_mean.to_numpy()
        predicted_series = pd.Series(predicted, index=full_forecast_index, dtype="float64")
    predicted_series = predicted_series.replace([np.inf, -np.inf], np.nan)
    predicted_series = predicted_series.ffill().bfill().fillna(float(ordered["noise_level_db"].iloc[-1]))
    forecast_df = pd.DataFrame(
        {
            "timestamp": full_forecast_index,
            "predicted_noise_db": predicted_series.to_numpy(),
            "location_id": ordered["location_id"].iloc[0],
            "location_name": ordered["location_name"].iloc[0],
            "borough": ordered["borough"].iloc[0],
            "land_use_type": ordered["land_use_type"].iloc[0],
        }
    )
    forecast_df = forecast_df[forecast_df["timestamp"] >= start_timestamp].reset_index(drop=True)

    metadata = {
        **model_metadata,
        **synthetic_report,
        "forecast_rows": int(len(forecast_df)),
        "forecast_start": str(forecast_df["timestamp"].min()) if not forecast_df.empty else None,
        "forecast_end": str(forecast_df["timestamp"].max()) if not forecast_df.empty else None,
        "last_observed_timestamp": str(last_timestamp),
    }
    return forecast_df, metadata


def forecast_future_dates(
    analysis_df: pd.DataFrame,
    location_id: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    max_total_rows: int = MAX_FORECAST_DATASET_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    location_history = (
        analysis_df.loc[analysis_df["location_id"] == location_id]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if location_history.empty:
        raise ValueError(f"Location '{location_id}' was not found in the dataset.")

    start_timestamp = _to_timestamp(start, "start")
    end_timestamp = _to_timestamp(end, "end")
    return _forecast_from_history(
        location_history,
        start_timestamp,
        end_timestamp,
        max_total_rows=max_total_rows,
    )


def validate_forecasting_model(
    analysis_df: pd.DataFrame,
    holdout_hours: int = DEFAULT_VALIDATION_HOLDOUT_HOURS,
    max_total_rows: int = MAX_FORECAST_DATASET_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reports: list[dict[str, Any]] = []

    for location_id, location_history in analysis_df.groupby("location_id", observed=True):
        ordered = location_history.sort_values("timestamp").reset_index(drop=True)
        location_holdout = min(holdout_hours, max(6, len(ordered) // 4))
        if len(ordered) <= location_holdout + MIN_REAL_ROWS_FOR_FORECAST:
            continue

        train_real = ordered.iloc[:-location_holdout].copy()
        test_real = ordered.iloc[-location_holdout:].copy()
        forecast_df, metadata = _forecast_from_history(
            train_real,
            start_timestamp=test_real["timestamp"].min(),
            end_timestamp=test_real["timestamp"].max(),
            max_total_rows=max_total_rows,
        )

        merged = test_real[["timestamp", "noise_level_db"]].merge(forecast_df[["timestamp", "predicted_noise_db"]], on="timestamp", how="left")
        merged = merged.dropna(subset=["noise_level_db", "predicted_noise_db"])
        if merged.empty:
            continue
        rmse = root_mean_squared_error(merged["noise_level_db"], merged["predicted_noise_db"])
        mae = mean_absolute_error(merged["noise_level_db"], merged["predicted_noise_db"])
        mape = float((np.abs((merged["noise_level_db"] - merged["predicted_noise_db"]) / merged["noise_level_db"].replace(0, np.nan))).mean() * 100)

        reports.append(
            {
                "location_id": location_id,
                "location_name": ordered["location_name"].iloc[0],
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "model": metadata["selected_model"],
                "rows_used_for_training": int(metadata["final_rows"]),
                "synthetic_rows_added": int(metadata["synthetic_rows_added"]),
                "passes_alignment_check": bool(metadata["passes_alignment_check"]),
            }
        )

    if reports:
        validation_df = pd.DataFrame(reports).sort_values("rmse").reset_index(drop=True)
    else:
        validation_df = pd.DataFrame(
            columns=[
                "location_id",
                "location_name",
                "rmse",
                "mae",
                "mape",
                "model",
                "rows_used_for_training",
                "synthetic_rows_added",
                "passes_alignment_check",
            ]
        )
    summary = {
        "locations_validated": int(len(validation_df)),
        "mean_rmse": float(validation_df["rmse"].mean()) if not validation_df.empty else float("nan"),
        "mean_mae": float(validation_df["mae"].mean()) if not validation_df.empty else float("nan"),
        "mean_mape": float(validation_df["mape"].mean()) if not validation_df.empty else float("nan"),
        "max_rows_used_for_training": int(validation_df["rows_used_for_training"].max()) if not validation_df.empty else 0,
        "total_synthetic_rows_added": int(validation_df["synthetic_rows_added"].sum()) if not validation_df.empty else 0,
    }
    return validation_df, summary


def save_validation_outputs(validation_df: pd.DataFrame, summary: dict[str, Any]) -> None:
    validation_df.to_csv(models_path("future_forecast_validation.csv"), index=False)
    save_json(summary, models_path("future_forecast_validation_summary.json"))


def save_forecast_output(forecast_df: pd.DataFrame, metadata: dict[str, Any]) -> None:
    forecast_df.to_csv(models_path("future_noise_forecasts.csv"), index=False)
    save_json(metadata, models_path("future_noise_forecasts_metadata.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate future date forecasts for Urban Noise Analytics.")
    parser.add_argument("--analysis-input", type=str, default=None, help="Optional override for the processed analysis dataset.")
    parser.add_argument("--location-id", type=str, default=None, help="Location ID to forecast, for example SITE-031.")
    parser.add_argument("--start", type=str, default=None, help="Future forecast start timestamp or date.")
    parser.add_argument("--end", type=str, default=None, help="Future forecast end timestamp or date.")
    parser.add_argument("--max-total-rows", type=int, default=MAX_FORECAST_DATASET_ROWS, help="Maximum real plus synthetic rows used for training.")
    args = parser.parse_args()

    project_directories()
    analysis_df = load_analysis_dataset(args.analysis_input)
    eda_summary = summarize_forecasting_patterns(analysis_df)
    write_forecasting_eda_summary(eda_summary)

    validation_df, validation_summary = validate_forecasting_model(analysis_df, max_total_rows=args.max_total_rows)
    save_validation_outputs(validation_df, validation_summary)

    if args.location_id and args.start and args.end:
        forecast_df, metadata = forecast_future_dates(
            analysis_df,
            args.location_id,
            args.start,
            args.end,
            max_total_rows=args.max_total_rows,
        )
        save_forecast_output(forecast_df, metadata)
        print("Future forecast generated successfully.")
        print(forecast_df.to_string(index=False))
    else:
        print("Forecasting validation artifacts generated. Supply --location-id, --start, and --end to produce a future forecast.")


if __name__ == "__main__":
    main()
