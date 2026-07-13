from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils import RAW_DIR, processed_path, project_directories, save_json


RAW_DATASET_NAME = "urban_noise_levels.csv"

STATIC_COLUMNS = [
    "location_id",
    "location_name",
    "borough",
    "latitude",
    "longitude",
    "land_use_type",
]

TIME_COLUMNS = [
    "timestamp",
    "hour",
    "day_of_week",
    "is_weekend",
]

NOISE_COLUMNS = ["timestamp", "location_id", "noise_level_db"]
TRAFFIC_COLUMNS = ["timestamp", "location_id", "traffic_volume_veh_hr", "avg_speed_kmh"]
WEATHER_COLUMNS = [
    "timestamp",
    "location_id",
    "temperature_c",
    "humidity_pct",
    "precipitation_mm",
    "wind_speed_kmh",
]

NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
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

STANDARD_COLUMNS = [
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

ALTERNATE_SCHEMA_COLUMNS = {
    "datetime",
    "decibel_level",
    "humidity_%",
    "vehicle_count",
    "sensor_id",
}


def resolve_source_path(csv_path: str | Path | None = None) -> Path:
    if csv_path:
        return Path(csv_path)

    preferred_candidates = [
        RAW_DIR / RAW_DATASET_NAME,
        Path(__file__).resolve().parents[1] / RAW_DATASET_NAME,
    ]
    for candidate in preferred_candidates:
        if candidate.exists():
            return candidate

    return RAW_DIR / RAW_DATASET_NAME


def assign_borough(latitude: float, longitude: float, lat_mid: float, lon_mid: float) -> str:
    north_south = "North" if latitude >= lat_mid else "South"
    east_west = "East" if longitude >= lon_mid else "West"
    return f"{north_south} {east_west}"


def infer_land_use_type(row: pd.Series) -> str:
    if row.get("industrial_zone", 0) == 1 or row.get("near_construction", 0) == 1:
        return "Industrial"
    if row.get("park_proximity", 0) == 1 and row.get("near_highway", 0) == 0:
        return "Park"
    if row.get("near_highway", 0) == 1 or row.get("near_airport", 0) == 1 or row.get("traffic_density", 0) >= 4:
        return "Commercial"
    if row.get("school_zone", 0) == 1 or row.get("population_density", 0) >= row.get("_population_density_median", 0):
        return "Residential"
    return "Mixed"


def estimate_avg_speed_kmh(traffic_density: pd.Series, vehicle_count: pd.Series, near_highway: pd.Series) -> pd.Series:
    speed = 60 - (traffic_density.fillna(0) * 7) - (vehicle_count.fillna(0) * 0.35) + (near_highway.fillna(0) * 4)
    return speed.clip(lower=8, upper=70)


def normalize_alternate_schema(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["datetime"]).dt.floor("h")
    normalized["sensor_id"] = pd.to_numeric(normalized["sensor_id"], errors="coerce").astype("Int64")
    normalized["location_id"] = normalized["sensor_id"].apply(lambda value: f"SITE-{int(value):03d}" if pd.notna(value) else "SITE-UNK")
    normalized["location_name"] = normalized["sensor_id"].apply(lambda value: f"Sensor {int(value)}" if pd.notna(value) else "Unknown Sensor")

    lat_mid = float(normalized["latitude"].median())
    lon_mid = float(normalized["longitude"].median())
    normalized["borough"] = normalized.apply(
        lambda row: assign_borough(float(row["latitude"]), float(row["longitude"]), lat_mid, lon_mid),
        axis=1,
    )
    normalized["_population_density_median"] = normalized["population_density"].median()
    normalized["land_use_type"] = normalized.apply(infer_land_use_type, axis=1)

    normalized["noise_level_db"] = pd.to_numeric(normalized["decibel_level"], errors="coerce")
    normalized["temperature_c"] = pd.to_numeric(normalized["temperature_c"], errors="coerce")
    normalized["humidity_pct"] = pd.to_numeric(normalized["humidity_%"], errors="coerce")
    normalized["wind_speed_kmh"] = pd.to_numeric(normalized["wind_speed_kmh"], errors="coerce")
    normalized["precipitation_mm"] = pd.to_numeric(normalized["precipitation_mm"], errors="coerce")
    normalized["traffic_volume_veh_hr"] = pd.to_numeric(normalized["vehicle_count"], errors="coerce") * 12
    normalized["avg_speed_kmh"] = estimate_avg_speed_kmh(
        pd.to_numeric(normalized["traffic_density"], errors="coerce"),
        pd.to_numeric(normalized["vehicle_count"], errors="coerce"),
        pd.to_numeric(normalized["near_highway"], errors="coerce"),
    )

    aggregated = (
        normalized.groupby(["location_id", "timestamp"], observed=True)
        .agg(
            location_name=("location_name", "first"),
            borough=("borough", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            land_use_type=("land_use_type", lambda values: values.mode().iloc[0] if not values.mode().empty else "Mixed"),
            noise_level_db=("noise_level_db", "mean"),
            traffic_volume_veh_hr=("traffic_volume_veh_hr", "mean"),
            avg_speed_kmh=("avg_speed_kmh", "mean"),
            temperature_c=("temperature_c", "mean"),
            humidity_pct=("humidity_pct", "mean"),
            precipitation_mm=("precipitation_mm", "mean"),
            wind_speed_kmh=("wind_speed_kmh", "mean"),
        )
        .reset_index()
    )
    aggregated["hour"] = aggregated["timestamp"].dt.hour
    aggregated["day_of_week"] = aggregated["timestamp"].dt.dayofweek
    aggregated["is_weekend"] = (aggregated["day_of_week"] >= 5).astype(int)
    aggregated["location_id"] = aggregated["location_id"].astype("string")
    aggregated["location_name"] = aggregated["location_name"].astype("string")
    aggregated["borough"] = aggregated["borough"].astype("category")
    aggregated["land_use_type"] = aggregated["land_use_type"].astype("category")
    return aggregated[STANDARD_COLUMNS].sort_values(["location_id", "timestamp"]).reset_index(drop=True)


def normalize_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if {"timestamp", "location_id", "noise_level_db"}.issubset(df.columns):
        normalized = df.copy()
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
        return normalized

    if ALTERNATE_SCHEMA_COLUMNS.issubset(df.columns):
        return normalize_alternate_schema(df)

    missing = sorted({"timestamp", "location_id", "noise_level_db"} - set(df.columns))
    raise ValueError(f"Unsupported raw dataset schema. Missing required fields: {missing}")


def load_raw_dataset(csv_path: str | Path | None = None) -> pd.DataFrame:
    project_directories()
    source = resolve_source_path(csv_path)
    df = pd.read_csv(source)
    df = normalize_raw_dataset(df)
    df["location_id"] = df["location_id"].astype("string")
    df["location_name"] = df["location_name"].astype("string")
    df["borough"] = df["borough"].astype("category")
    df["land_use_type"] = df["land_use_type"].astype("category")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
    return df


def prepare_source_feeds(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Simulate the common case where static site metadata and time-varying feeds arrive separately.
    spatial = df[STATIC_COLUMNS].drop_duplicates("location_id").reset_index(drop=True)
    time_index = df[["timestamp", "location_id", "hour", "day_of_week", "is_weekend"]].copy()
    noise = df[NOISE_COLUMNS].copy()
    traffic = df[TRAFFIC_COLUMNS].copy()
    weather = df[WEATHER_COLUMNS].copy()

    return {
        "spatial": spatial,
        "time_index": time_index,
        "noise": noise,
        "traffic": traffic,
        "weather": weather,
    }


def merge_source_feeds(feeds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = feeds["time_index"].merge(feeds["spatial"], on="location_id", how="left")
    merged = merged.merge(feeds["noise"], on=["timestamp", "location_id"], how="left")
    merged = merged.merge(feeds["traffic"], on=["timestamp", "location_id"], how="left")
    merged = merged.merge(feeds["weather"], on=["timestamp", "location_id"], how="left")
    merged = merged.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    derived_hour = merged["timestamp"].dt.hour
    derived_day = merged["timestamp"].dt.dayofweek
    derived_weekend = (derived_day >= 5).astype(int)
    merged["hour"] = merged["hour"].fillna(derived_hour).astype(int)
    merged["day_of_week"] = merged["day_of_week"].fillna(derived_day).astype(int)
    merged["is_weekend"] = merged["is_weekend"].fillna(derived_weekend).astype(int)

    return merged


def interpolate_by_location(df: pd.DataFrame, column: str) -> pd.Series:
    interpolated_parts: list[pd.Series] = []
    for _, group in df.groupby("location_id", observed=True):
        indexed = group.sort_values("timestamp").set_index("timestamp")[column]
        filled = indexed.interpolate(method="time", limit_direction="both")
        interpolated_parts.append(pd.Series(filled.to_numpy(), index=group.index, name=column))

    return pd.concat(interpolated_parts).sort_index()


def impute_core_measurements(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned = df.copy()
    summary: dict[str, int] = {}

    cleaned["traffic_missing_original"] = cleaned["traffic_volume_veh_hr"].isna().astype(int)
    cleaned["temperature_missing_original"] = cleaned["temperature_c"].isna().astype(int)
    cleaned["noise_missing_original"] = cleaned["noise_level_db"].isna().astype(int)

    cleaned["traffic_volume_veh_hr"] = interpolate_by_location(cleaned, "traffic_volume_veh_hr")
    cleaned["temperature_c"] = interpolate_by_location(cleaned, "temperature_c")
    noise_time_interp = interpolate_by_location(cleaned, "noise_level_db")
    cleaned["noise_level_db"] = cleaned["noise_level_db"].fillna(noise_time_interp)

    noise_hour_medians = cleaned.groupby(["location_id", "hour"], observed=True)["noise_level_db"].transform("median")
    borough_hour_medians = cleaned.groupby(["borough", "hour"], observed=True)["noise_level_db"].transform("median")
    land_use_hour_medians = cleaned.groupby(["land_use_type", "hour"], observed=True)["noise_level_db"].transform("median")
    global_noise_median = cleaned["noise_level_db"].median()

    cleaned["noise_level_db"] = cleaned["noise_level_db"].fillna(noise_hour_medians)
    cleaned["noise_level_db"] = cleaned["noise_level_db"].fillna(borough_hour_medians)
    cleaned["noise_level_db"] = cleaned["noise_level_db"].fillna(land_use_hour_medians)
    cleaned["noise_level_db"] = cleaned["noise_level_db"].fillna(global_noise_median)

    summary["traffic_imputed_rows"] = int(cleaned["traffic_missing_original"].sum())
    summary["temperature_imputed_rows"] = int(cleaned["temperature_missing_original"].sum())
    summary["noise_imputed_rows"] = int(cleaned["noise_missing_original"].sum())
    summary["rows_after_cleaning"] = int(len(cleaned))

    return cleaned, summary


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy().sort_values(["location_id", "timestamp"]).reset_index(drop=True)
    location_groups = featured.groupby("location_id", observed=True)

    featured["noise_lag_1"] = location_groups["noise_level_db"].shift(1)
    featured["noise_lag_24"] = location_groups["noise_level_db"].shift(24)
    featured["traffic_lag_1"] = location_groups["traffic_volume_veh_hr"].shift(1)
    featured["traffic_lag_24"] = location_groups["traffic_volume_veh_hr"].shift(24)

    shifted_noise = location_groups["noise_level_db"].shift(1)
    shifted_traffic = location_groups["traffic_volume_veh_hr"].shift(1)
    featured["noise_roll_mean_3h"] = shifted_noise.groupby(featured["location_id"]).transform(
        lambda values: values.rolling(window=3, min_periods=1).mean()
    )
    featured["noise_roll_mean_24h"] = shifted_noise.groupby(featured["location_id"]).transform(
        lambda values: values.rolling(window=24, min_periods=1).mean()
    )
    featured["traffic_roll_mean_3h"] = shifted_traffic.groupby(featured["location_id"]).transform(
        lambda values: values.rolling(window=3, min_periods=1).mean()
    )
    featured["traffic_roll_mean_24h"] = shifted_traffic.groupby(featured["location_id"]).transform(
        lambda values: values.rolling(window=24, min_periods=1).mean()
    )

    featured["hour_sin"] = np.sin(2 * np.pi * featured["hour"] / 24)
    featured["hour_cos"] = np.cos(2 * np.pi * featured["hour"] / 24)
    featured["day_of_week_sin"] = np.sin(2 * np.pi * featured["day_of_week"] / 7)
    featured["day_of_week_cos"] = np.cos(2 * np.pi * featured["day_of_week"] / 7)

    return featured


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.get_dummies(
        df,
        columns=["land_use_type", "borough", "location_id"],
        prefix=["land_use", "borough", "site"],
        dtype=int,
    )
    return encoded


def build_processed_dataset(csv_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    raw = load_raw_dataset(csv_path)
    feeds = prepare_source_feeds(raw)
    merged = merge_source_feeds(feeds)
    cleaned, summary = impute_core_measurements(merged)
    featured = add_temporal_features(cleaned)
    encoded = encode_categorical_features(featured)
    return featured, encoded, summary


def save_processed_outputs(featured: pd.DataFrame, encoded: pd.DataFrame, summary: dict[str, int]) -> None:
    featured_csv = processed_path("urban_noise_processed.csv")
    featured_parquet = processed_path("urban_noise_processed.parquet")
    model_csv = processed_path("urban_noise_model_ready.csv")
    model_parquet = processed_path("urban_noise_model_ready.parquet")
    summary_json = processed_path("pipeline_summary.json")

    featured.to_csv(featured_csv, index=False)
    featured.to_parquet(featured_parquet, index=False)
    encoded.to_csv(model_csv, index=False)
    encoded.to_parquet(model_parquet, index=False)
    save_json(summary, summary_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Urban Noise Analytics processed datasets.")
    parser.add_argument("--input", type=str, default=None, help="Optional override for the raw CSV path.")
    args = parser.parse_args()

    featured, encoded, summary = build_processed_dataset(args.input)
    save_processed_outputs(featured, encoded, summary)

    print("Processed dataset written successfully.")
    print(f"Rows: {len(featured)}")
    print(f"Columns (analysis): {featured.shape[1]}")
    print(f"Columns (model-ready): {encoded.shape[1]}")
    print(f"Noise values imputed: {summary['noise_imputed_rows']}")


if __name__ == "__main__":
    main()
