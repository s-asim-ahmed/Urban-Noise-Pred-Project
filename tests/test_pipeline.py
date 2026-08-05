from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline import add_temporal_features, build_processed_dataset, impute_core_measurements  # noqa: E402
from data_pipeline import normalize_raw_dataset  # noqa: E402


def make_sample_df() -> pd.DataFrame:
    timestamps = pd.date_range("2026-03-01 00:00:00", periods=30, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "location_id": ["SITE-TEST"] * len(timestamps),
            "location_name": ["Test Site"] * len(timestamps),
            "borough": pd.Categorical(["North West"] * len(timestamps)),
            "latitude": [40.75] * len(timestamps),
            "longitude": [-74.02] * len(timestamps),
            "land_use_type": pd.Categorical(["Commercial"] * len(timestamps)),
            "noise_level_db": [60.0, None, 62.0] + [63.0] * 27,
            "traffic_volume_veh_hr": [100.0, None, 120.0] + [130.0] * 27,
            "avg_speed_kmh": [35.0] * len(timestamps),
            "temperature_c": [10.0, None, 12.0] + [11.0] * 27,
            "humidity_pct": [70.0] * len(timestamps),
            "precipitation_mm": [0.0] * len(timestamps),
            "wind_speed_kmh": [10.0] * len(timestamps),
            "hour": timestamps.hour,
            "day_of_week": timestamps.dayofweek,
            "is_weekend": (timestamps.dayofweek >= 5).astype(int),
        }
    )


def test_imputation_fills_expected_columns() -> None:
    sample = make_sample_df()
    cleaned, summary = impute_core_measurements(sample)

    assert cleaned["traffic_volume_veh_hr"].isna().sum() == 0
    assert cleaned["temperature_c"].isna().sum() == 0
    assert cleaned["noise_level_db"].isna().sum() == 0
    assert summary["traffic_imputed_rows"] == 1
    assert summary["temperature_imputed_rows"] == 1
    assert summary["noise_imputed_rows"] == 1
    assert cleaned.loc[1, "noise_missing_original"] == 1


def test_temporal_features_create_expected_lags() -> None:
    sample = make_sample_df()
    cleaned, _ = impute_core_measurements(sample)
    featured = add_temporal_features(cleaned)

    assert pd.isna(featured.loc[0, "noise_lag_1"])
    assert featured.loc[1, "noise_lag_1"] == featured.loc[0, "noise_level_db"]
    assert featured.loc[24, "noise_lag_24"] == featured.loc[0, "noise_level_db"]
    assert pd.notna(featured.loc[5, "noise_roll_mean_3h"])
    assert pd.notna(featured.loc[29, "noise_roll_mean_24h"])


def test_build_processed_dataset_returns_encoded_columns() -> None:
    featured, encoded, summary = build_processed_dataset(
        PROJECT_ROOT / "data" / "raw" / "urban_noise_levels.csv"
    )

    assert len(featured) == len(encoded)
    assert "hour_sin" in featured.columns
    assert any(column.startswith("land_use_") for column in encoded.columns)
    assert summary["rows_after_cleaning"] == len(featured)


def test_normalize_raw_dataset_supports_alternate_schema() -> None:
    alternate = pd.DataFrame(
        {
            "id": [1, 2],
            "latitude": [40.7, 40.7],
            "longitude": [-74.0, -74.0],
            "datetime": ["2023-04-01 18:50:00", "2023-04-01 18:55:00"],
            "decibel_level": [78.2, 79.1],
            "hour": [18, 18],
            "day_of_week": [5, 5],
            "is_weekend": [1, 1],
            "temperature_c": [16.7, 16.9],
            "humidity_%": [43.0, 42.5],
            "wind_speed_kmh": [28.7, 29.0],
            "precipitation_mm": [0.5, 0.6],
            "traffic_density": [4, 4],
            "near_airport": [0, 0],
            "near_highway": [1, 1],
            "near_construction": [0, 0],
            "population_density": [28000, 28000],
            "park_proximity": [0, 0],
            "industrial_zone": [0, 0],
            "vehicle_count": [24, 30],
            "honking_events": [1, 2],
            "public_event": [0, 0],
            "holiday": [0, 0],
            "school_zone": [1, 1],
            "noise_complaints": [1, 1],
            "sensor_id": [31, 31],
        }
    )

    normalized = normalize_raw_dataset(alternate)

    assert list(normalized.columns) == [
        "timestamp",
        "location_id",
        "location_name",
        "borough",
        "latitude",
        "longitude",
        "land_use_type",
        "noise_level_db",
        "traffic_volume_veh_hr",
        "avg_speed_kmh",
        "temperature_c",
        "humidity_pct",
        "precipitation_mm",
        "wind_speed_kmh",
        "hour",
        "day_of_week",
        "is_weekend",
    ]
    assert len(normalized) == 1
    assert normalized.loc[0, "location_id"] == "SITE-031"
    assert normalized.loc[0, "hour"] == 18
