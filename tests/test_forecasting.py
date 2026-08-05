from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from forecasting import (  # noqa: E402
    forecast_future_dates,
    prepare_training_history,
    synthetic_alignment_report,
    validate_forecasting_model,
)


def make_location_history(periods: int = 96, location_id: str = "SITE-TEST") -> pd.DataFrame:
    timestamps = pd.date_range("2026-03-01 00:00:00", periods=periods, freq="h")
    hours = timestamps.hour
    days = timestamps.dayofweek
    base_noise = 58 + 8 * ((hours >= 7) & (hours <= 9)) + 10 * ((hours >= 17) & (hours <= 20))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "location_id": [location_id] * periods,
            "location_name": ["Test Site"] * periods,
            "borough": ["North West"] * periods,
            "latitude": [40.75] * periods,
            "longitude": [-74.02] * periods,
            "land_use_type": ["Commercial"] * periods,
            "noise_level_db": base_noise + (days >= 5) * 1.5,
            "traffic_volume_veh_hr": 100 + hours * 12,
            "avg_speed_kmh": 35 - (hours >= 7) * 4,
            "temperature_c": 10 + (hours >= 12) * 3,
            "humidity_pct": 70 - (hours >= 12) * 5,
            "precipitation_mm": [0.0] * periods,
            "wind_speed_kmh": 10 + (hours % 5),
            "hour": hours,
            "day_of_week": days,
            "is_weekend": (days >= 5).astype(int),
        }
    )


def test_prepare_training_history_adds_synthetic_rows_without_exceeding_cap() -> None:
    history = make_location_history(periods=72)
    combined, synthetic, report = prepare_training_history(history, target_rows=168, max_total_rows=1000)

    assert len(combined) == 168
    assert len(synthetic) == 96
    assert report["final_rows"] <= 1000
    assert report["synthetic_rows_added"] == 96


def test_synthetic_alignment_report_flags_reasonable_match() -> None:
    history = make_location_history(periods=72)
    _, synthetic, _ = prepare_training_history(history, target_rows=168, max_total_rows=1000)
    report = synthetic_alignment_report(history, synthetic)

    assert report["synthetic_rows_added"] == len(synthetic)
    assert report["hourly_profile_correlation"] >= 0.6
    assert report["passes_alignment_check"] is True


def test_forecast_future_dates_returns_requested_hourly_range() -> None:
    analysis_df = make_location_history(periods=200)
    last_timestamp = analysis_df["timestamp"].max()
    start = last_timestamp + pd.Timedelta(hours=1)
    end = start + pd.Timedelta(hours=23)

    forecast_df, metadata = forecast_future_dates(analysis_df, "SITE-TEST", start, end)

    assert len(forecast_df) == 24
    assert forecast_df["timestamp"].min() == start
    assert forecast_df["timestamp"].max() == end
    assert metadata["final_rows"] <= 1000


def test_forecast_future_dates_preserves_hourly_pattern_variation() -> None:
    analysis_df = make_location_history(periods=200)
    last_timestamp = analysis_df["timestamp"].max()
    start = last_timestamp + pd.Timedelta(hours=1)
    end = start + pd.Timedelta(hours=47)

    forecast_df, metadata = forecast_future_dates(analysis_df, "SITE-TEST", start, end)

    assert metadata["selected_model"] in {"Hourly Seasonal Profile", "Seasonal SARIMA (1,0,1)x(1,0,1,24)", "ARIMA (1,0,1)"}
    assert forecast_df["predicted_noise_db"].nunique() > 5
    assert forecast_df.groupby(forecast_df["timestamp"].dt.hour)["predicted_noise_db"].mean().max() > forecast_df.groupby(
        forecast_df["timestamp"].dt.hour
    )["predicted_noise_db"].mean().min()


def test_forecast_future_dates_rejects_non_future_ranges() -> None:
    analysis_df = make_location_history(periods=120)
    with pytest.raises(ValueError, match="strictly later"):
        forecast_future_dates(
            analysis_df,
            "SITE-TEST",
            pd.Timestamp("2026-03-05 00:00:00"),
            pd.Timestamp("2026-03-05 23:00:00"),
        )


def test_validate_forecasting_model_respects_row_cap() -> None:
    analysis_df = pd.concat(
        [
            make_location_history(periods=140, location_id="SITE-001"),
            make_location_history(periods=140, location_id="SITE-002"),
        ],
        ignore_index=True,
    )
    validation_df, summary = validate_forecasting_model(analysis_df, holdout_hours=24, max_total_rows=1000)

    assert not validation_df.empty
    assert summary["max_rows_used_for_training"] <= 1000
    assert {"rmse", "mae", "mape", "rows_used_for_training"}.issubset(validation_df.columns)
