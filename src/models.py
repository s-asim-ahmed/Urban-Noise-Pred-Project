from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import models_path, processed_path, project_directories, save_json


warnings.filterwarnings("ignore")

TEST_HOURS = 72


def load_datasets(
    analysis_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_source = Path(analysis_path) if analysis_path else processed_path("urban_noise_processed.parquet")
    model_source = Path(model_path) if model_path else processed_path("urban_noise_model_ready.parquet")
    analysis_df = pd.read_parquet(analysis_source)
    model_df = pd.read_parquet(model_source)
    return analysis_df, model_df


def chronological_split(df: pd.DataFrame, hours: int = TEST_HOURS) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_timestamps = np.sort(df["timestamp"].unique())
    split_timestamp = pd.Timestamp(unique_timestamps[-hours])
    train_df = df[df["timestamp"] < split_timestamp].copy()
    test_df = df[df["timestamp"] >= split_timestamp].copy()
    return train_df, test_df, split_timestamp


def metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "timestamp",
            "location_id",
            "location_name",
            "borough",
            "land_use_type",
            "noise_level_db",
            "noise_lag_1",
            "noise_roll_mean_3h",
        ]
    ].copy()


def baseline_predictions(test_df: pd.DataFrame) -> pd.DataFrame:
    predictions = []

    persistence = metadata_frame(test_df)
    persistence["prediction"] = persistence["noise_lag_1"]
    persistence["model"] = "Naive Persistence"
    predictions.append(persistence)

    moving_average = metadata_frame(test_df)
    moving_average["prediction"] = moving_average["noise_roll_mean_3h"]
    moving_average["model"] = "Moving Average (3h)"
    predictions.append(moving_average)

    return pd.concat(predictions, ignore_index=True)


def get_feature_columns(model_df: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "location_name", "noise_level_db"}
    return [column for column in model_df.columns if column not in excluded]


def fit_feature_models(
    analysis_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    train_analysis, test_analysis, split_timestamp = chronological_split(analysis_df)
    train_model = model_df.loc[train_analysis.index].copy()
    test_model = model_df.loc[test_analysis.index].copy()

    feature_columns = get_feature_columns(train_model)
    X_train = train_model[feature_columns]
    y_train = train_model["noise_level_db"]
    X_test = test_model[feature_columns]

    model_specs = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    prediction_frames = []
    persisted_models: dict[str, str] = {}

    for model_name, estimator in model_specs.items():
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        frame = metadata_frame(test_analysis)
        frame["prediction"] = preds
        frame["model"] = model_name
        prediction_frames.append(frame)

        model_file = models_path(f"{model_name.lower().replace(' ', '_')}.joblib")
        joblib.dump(pipeline, model_file)
        persisted_models[model_name] = str(model_file)

    split_info = {
        "split_timestamp": str(split_timestamp),
        "train_rows": int(len(train_analysis)),
        "test_rows": int(len(test_analysis)),
        "feature_count": int(len(feature_columns)),
    }
    save_json(split_info, models_path("train_test_split.json"))
    save_json({"feature_columns": feature_columns}, models_path("feature_columns.json"))

    return pd.concat(prediction_frames, ignore_index=True), persisted_models, split_info


def fit_sarima_per_location(analysis_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    train_df, test_df, _ = chronological_split(analysis_df)
    prediction_frames = []
    model_notes: dict[str, str] = {}

    for location_id, location_train in train_df.groupby("location_id", observed=True):
        location_test = test_df[test_df["location_id"] == location_id].copy()
        train_series = location_train.set_index("timestamp")["noise_level_db"]

        try:
            model = SARIMAX(
                train_series,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 24),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            forecast = fitted.get_forecast(steps=len(location_test)).predicted_mean.to_numpy()
            note = "seasonal_sarima"
        except Exception as exc:  # pragma: no cover - defensive fallback
            forecast = np.repeat(train_series.iloc[-1], len(location_test))
            note = f"fallback_last_value:{exc.__class__.__name__}"

        frame = metadata_frame(location_test)
        frame["prediction"] = forecast
        frame["model"] = "SARIMA"
        prediction_frames.append(frame)
        model_notes[str(location_id)] = note

    save_json(model_notes, models_path("sarima_training_notes.json"))
    return pd.concat(prediction_frames, ignore_index=True), model_notes


def save_predictions(predictions: pd.DataFrame) -> None:
    destination = models_path("model_predictions.csv")
    predictions = predictions.rename(columns={"noise_level_db": "actual"})
    predictions.to_csv(destination, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist Urban Noise Analytics models.")
    parser.add_argument("--analysis-input", type=str, default=None, help="Optional processed analysis dataset path.")
    parser.add_argument("--model-input", type=str, default=None, help="Optional processed model-ready dataset path.")
    args = parser.parse_args()

    project_directories()
    analysis_df, model_df = load_datasets(args.analysis_input, args.model_input)
    baseline_df = baseline_predictions(chronological_split(analysis_df)[1])
    feature_model_df, persisted_models, split_info = fit_feature_models(analysis_df, model_df)
    sarima_df, sarima_notes = fit_sarima_per_location(analysis_df)

    all_predictions = pd.concat([baseline_df, feature_model_df, sarima_df], ignore_index=True)
    save_predictions(all_predictions)
    save_json(persisted_models, models_path("trained_models.json"))
    save_json(
        {
            "split_info": split_info,
            "sarima_locations": sarima_notes,
            "available_prediction_models": sorted(all_predictions["model"].unique().tolist()),
        },
        models_path("model_run_summary.json"),
    )

    print("Model training complete.")
    print(f"Prediction rows written: {len(all_predictions)}")
    print(f"Models: {', '.join(sorted(all_predictions['model'].unique()))}")


if __name__ == "__main__":
    main()
