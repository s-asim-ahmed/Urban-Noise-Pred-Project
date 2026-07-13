from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from forecasting import forecast_future_dates  # noqa: E402
from utils import models_path, processed_path  # noqa: E402


st.set_page_config(page_title="Urban Noise Analytics", layout="wide")


@st.cache_data
def load_processed() -> pd.DataFrame:
    return pd.read_parquet(processed_path("urban_noise_processed.parquet"))


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(models_path("model_predictions.csv"), parse_dates=["timestamp"])


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(models_path("model_metrics.csv"))


@st.cache_data
def load_forecast_validation() -> pd.DataFrame:
    validation_path = models_path("future_forecast_validation.csv")
    if validation_path.exists():
        return pd.read_csv(validation_path)
    return pd.DataFrame()


def filter_dataframe(
    df: pd.DataFrame,
    selected_boroughs: list[str],
    selected_land_uses: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    mask = (
        df["borough"].isin(selected_boroughs)
        & df["land_use_type"].isin(selected_land_uses)
        & (df["timestamp"].dt.date >= start_date.date())
        & (df["timestamp"].dt.date <= end_date.date())
    )
    return df.loc[mask].copy()


def render_map(df: pd.DataFrame, mode: str) -> None:
    if mode == "Current":
        map_df = (
            df.sort_values("timestamp")
            .groupby(["location_id", "location_name", "borough", "latitude", "longitude"], observed=True)
            .tail(1)
            .rename(columns={"noise_level_db": "noise_value"})
        )
        title = "Current Noise by Site"
    else:
        map_df = (
            df.groupby(["location_id", "location_name", "borough", "latitude", "longitude"], observed=True)["noise_level_db"]
            .mean()
            .reset_index()
            .rename(columns={"noise_level_db": "noise_value"})
        )
        title = "Average Noise by Site"

    fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="noise_value",
        size="noise_value",
        hover_name="location_name",
        hover_data={"borough": True, "noise_value": ":.1f", "latitude": False, "longitude": False},
        color_continuous_scale="Turbo",
        size_max=20,
        zoom=10,
        height=500,
        title=title,
    )
    fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_time_series(df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    locations = sorted(df["location_name"].unique().tolist())
    selected_location = st.selectbox("Location", locations)
    prediction_models = sorted(predictions["model"].unique().tolist())
    selected_models = st.multiselect(
        "Prediction Models",
        prediction_models,
        default=["Gradient Boosting", "Random Forest", "SARIMA"] if "Gradient Boosting" in prediction_models else prediction_models[:3],
    )

    history_df = df[df["location_name"] == selected_location].copy()
    location_predictions = predictions[
        (predictions["location_name"] == selected_location) & (predictions["model"].isin(selected_models))
    ].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["timestamp"],
            y=history_df["noise_level_db"],
            mode="lines",
            name="Observed Noise",
            line=dict(color="#1f77b4", width=2),
        )
    )

    for model_name, group in location_predictions.groupby("model"):
        fig.add_trace(
            go.Scatter(
                x=group["timestamp"],
                y=group["prediction"],
                mode="lines",
                name=model_name,
                line=dict(width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title=f"Historical Noise and Forecast Overlay: {selected_location}",
        xaxis_title="Timestamp",
        yaxis_title="Noise Level dB(A)",
        height=450,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def summarize_forecast_for_display(
    observed_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if forecast_df.empty:
        return observed_df, forecast_df, "hourly"

    horizon_hours = max(int((forecast_df["timestamp"].max() - forecast_df["timestamp"].min()).total_seconds() // 3600) + 1, 1)

    if horizon_hours <= 24 * 14:
        return observed_df, forecast_df, "hourly"

    rule = "D"
    label = "daily"
    observed_window = observed_df[observed_df["timestamp"] >= observed_df["timestamp"].max() - pd.Timedelta(days=90)].copy()

    observed_display = (
        observed_window.set_index("timestamp")["noise_level_db"].resample(rule).mean().dropna().reset_index()
    )
    forecast_display = (
        forecast_df.set_index("timestamp")["predicted_noise_db"].resample(rule).agg(["mean", "min", "max"]).dropna().reset_index()
    )
    return observed_display, forecast_display, label


def render_future_forecast(df: pd.DataFrame) -> None:
    st.subheader("Future Noise Forecast")
    locations_df = (
        df[["location_id", "location_name", "borough"]]
        .drop_duplicates()
        .sort_values(["location_name", "location_id"])
        .reset_index(drop=True)
    )
    location_labels = [f"{row.location_name} ({row.location_id})" for row in locations_df.itertuples(index=False)]
    selected_label = st.selectbox("Forecast Location", location_labels, key="forecast_location")
    selected_row = locations_df.iloc[location_labels.index(selected_label)]
    location_id = selected_row["location_id"]

    location_history = df[df["location_id"] == location_id].sort_values("timestamp").copy()
    last_timestamp = location_history["timestamp"].max()
    default_start = (last_timestamp + pd.Timedelta(days=1)).date()
    default_end = (last_timestamp + pd.Timedelta(days=3)).date()

    forecast_columns = st.columns(3)
    start_date = forecast_columns[0].date_input("Future Start Date", value=default_start, min_value=default_start, key="forecast_start")
    end_date = forecast_columns[1].date_input("Future End Date", value=default_end, min_value=default_start, key="forecast_end")
    generate = forecast_columns[2].button("Generate Forecast", use_container_width=True)

    if not generate:
        st.caption("Choose a location and future date range, then generate an hourly forecast.")
        return

    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date) + pd.Timedelta(hours=23)

    try:
        forecast_df, metadata = forecast_future_dates(
            df,
            location_id,
            start_timestamp,
            end_timestamp,
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - Streamlit runtime guard
        st.error(f"Forecast generation failed: {exc}")
        return

    context_history = location_history.tail(72)
    observed_display, forecast_display, display_mode = summarize_forecast_for_display(location_history, forecast_df)

    fig = go.Figure()
    if display_mode == "hourly":
        fig.add_trace(
            go.Scatter(
                x=context_history["timestamp"],
                y=context_history["noise_level_db"],
                mode="lines",
                name="Recent Observed Noise",
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["timestamp"],
                y=forecast_df["predicted_noise_db"],
                mode="lines",
                name="Future Forecast",
                line=dict(color="#d62728", width=2),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=observed_display["timestamp"],
                y=observed_display["noise_level_db"],
                mode="lines+markers",
                name=f"Observed Noise ({display_mode.title()} Mean)",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_display["timestamp"],
                y=forecast_display["max"],
                mode="lines",
                name=f"Forecast Range ({display_mode.title()})",
                line=dict(color="rgba(214,39,40,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_display["timestamp"],
                y=forecast_display["min"],
                mode="lines",
                name=f"Forecast Range ({display_mode.title()})",
                line=dict(color="rgba(214,39,40,0)"),
                fill="tonexty",
                fillcolor="rgba(214,39,40,0.15)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_display["timestamp"],
                y=forecast_display["mean"],
                mode="lines",
                name=f"Future Forecast ({display_mode.title()} Mean)",
                line=dict(color="#d62728", width=2),
            )
        )
    fig.update_layout(
        title=f"Future {display_mode.title()} Forecast for {selected_row['location_name']}",
        xaxis_title="Timestamp",
        yaxis_title="Predicted Noise dB(A)",
        height=450,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    if display_mode != "hourly":
        st.caption(
            f"Showing {display_mode} forecast summaries for readability because the selected future range contains many hourly points."
        )

    summary_cols = st.columns(4)
    summary_cols[0].metric("Forecast Rows", f"{metadata['forecast_rows']:,}")
    summary_cols[1].metric("Model", metadata["selected_model"])
    summary_cols[2].metric("Rows Used", metadata["final_rows"])
    summary_cols[3].metric("Synthetic Rows Added", metadata["synthetic_rows_added"])

    st.dataframe(forecast_df, use_container_width=True, hide_index=True)


def render_metrics(metrics: pd.DataFrame) -> None:
    best_row = metrics.sort_values("rmse").iloc[0]
    metric_columns = st.columns(3)
    metric_columns[0].metric("Best Model", best_row["model"])
    metric_columns[1].metric("Best RMSE", f"{best_row['rmse']:.2f}")
    metric_columns[2].metric("Best MAE", f"{best_row['mae']:.2f}")

    fig = px.bar(
        metrics.sort_values("rmse"),
        x="model",
        y=["rmse", "mae", "mape"],
        barmode="group",
        title="Model Comparison Metrics",
        height=420,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(metrics, use_container_width=True, hide_index=True)


def render_forecast_validation(validation_df: pd.DataFrame) -> None:
    st.subheader("Forecast Validation")
    if validation_df.empty:
        st.info("Run `python src/forecasting.py` to generate future-forecast validation artifacts.")
        return

    best_row = validation_df.sort_values("rmse").iloc[0]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Validated Sites", f"{len(validation_df)}")
    metric_columns[1].metric("Best Site RMSE", f"{best_row['rmse']:.2f}")
    metric_columns[2].metric("Mean RMSE", f"{validation_df['rmse'].mean():.2f}")
    metric_columns[3].metric("Synthetic Rows Used", f"{int(validation_df['synthetic_rows_added'].sum())}")

    st.dataframe(validation_df, use_container_width=True, hide_index=True)


def render_live_data_summary(df: pd.DataFrame) -> None:
    date_min = df["timestamp"].min().strftime("%Y-%m-%d")
    date_max = df["timestamp"].max().strftime("%Y-%m-%d")
    peak_hour = (
        df.groupby("hour", observed=True)["noise_level_db"].mean().sort_values(ascending=False).index[0]
    )
    borough_means = (
        df.groupby("borough", observed=True)["noise_level_db"].mean().sort_values(ascending=False)
    )
    land_use_means = (
        df.groupby("land_use_type", observed=True)["noise_level_db"].mean().sort_values(ascending=False)
    )

    summary_cols = st.columns(4)
    summary_cols[0].metric("Date Window", f"{date_min} to {date_max}")
    summary_cols[1].metric("Peak Noise Hour", f"{int(peak_hour):02d}:00")
    summary_cols[2].metric("Noisiest Borough", borough_means.index[0])
    summary_cols[3].metric("Loudest Land Use", land_use_means.index[0])

    with st.expander("Current data summary", expanded=False):
        st.markdown(
            "\n".join(
                [
                    f"- Filtered observations: `{len(df):,}` across `{df['location_id'].nunique()}` sites.",
                    f"- Mean filtered noise level: `{df['noise_level_db'].mean():.2f} dB(A)`.",
                    f"- Highest average borough noise: `{borough_means.index[0]}` at `{borough_means.iloc[0]:.2f} dB(A)`.",
                    f"- Lowest average borough noise: `{borough_means.index[-1]}` at `{borough_means.iloc[-1]:.2f} dB(A)`.",
                    f"- Highest average land-use noise: `{land_use_means.index[0]}` at `{land_use_means.iloc[0]:.2f} dB(A)`.",
                ]
            )
        )


def render_saved_figures(df: pd.DataFrame, metrics: pd.DataFrame) -> None:
    st.subheader("Current Data Figures")
    st.caption("These figures are regenerated from the current dashboard filters and selected date range.")
    render_live_data_summary(df)

    chart_tabs = st.tabs(
        [
            "Distributions",
            "Temporal Profiles",
            "Heatmap",
            "Correlations",
            "Metrics",
        ]
    )

    with chart_tabs[0]:
        distribution_columns = st.columns(2)
        with distribution_columns[0]:
            borough_order = (
                df.groupby("borough", observed=True)["noise_level_db"]
                .median()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
            borough_fig = px.box(
                df,
                x="borough",
                y="noise_level_db",
                category_orders={"borough": borough_order},
                title="Noise Distribution by Borough (Current Filters)",
            )
            borough_fig.update_xaxes(tickangle=30)
            st.plotly_chart(borough_fig, use_container_width=True)
        with distribution_columns[1]:
            land_use_order = (
                df.groupby("land_use_type", observed=True)["noise_level_db"]
                .median()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
            land_use_fig = px.box(
                df,
                x="land_use_type",
                y="noise_level_db",
                category_orders={"land_use_type": land_use_order},
                title="Noise Distribution by Land Use (Current Filters)",
            )
            land_use_fig.update_xaxes(tickangle=20)
            st.plotly_chart(land_use_fig, use_container_width=True)

    with chart_tabs[1]:
        hourly = (
            df.groupby(["hour", "land_use_type"], observed=True)["noise_level_db"]
            .mean()
            .reset_index()
        )
        hourly_fig = px.line(
            hourly,
            x="hour",
            y="noise_level_db",
            color="land_use_type",
            markers=True,
            title="Hourly Noise Profile by Land Use (Current Filters)",
        )
        st.plotly_chart(hourly_fig, use_container_width=True)

        daily = (
            df.assign(date=df["timestamp"].dt.date)
            .groupby(["date", "borough"], observed=True)["noise_level_db"]
            .mean()
            .reset_index()
        )
        daily_fig = px.line(
            daily,
            x="date",
            y="noise_level_db",
            color="borough",
            markers=True,
            title="Daily Mean Noise by Borough (Current Filters)",
        )
        st.plotly_chart(daily_fig, use_container_width=True)

        weekend_profile = (
            df.groupby(["hour", "is_weekend"], observed=True)["noise_level_db"]
            .mean()
            .reset_index()
        )
        weekend_profile["day_type"] = weekend_profile["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        weekend_fig = px.line(
            weekend_profile,
            x="hour",
            y="noise_level_db",
            color="day_type",
            markers=True,
            title="Weekday vs Weekend Hourly Noise (Current Filters)",
        )
        st.plotly_chart(weekend_fig, use_container_width=True)

    with chart_tabs[2]:
        heatmap_df = (
            df.groupby(["day_of_week", "hour"], observed=True)["noise_level_db"]
            .mean()
            .reset_index()
            .pivot(index="day_of_week", columns="hour", values="noise_level_db")
        )
        heatmap_fig = px.imshow(
            heatmap_df,
            aspect="auto",
            color_continuous_scale="Magma",
            labels={"x": "Hour of Day", "y": "Day of Week", "color": "Noise dB(A)"},
            title="Noise Heatmap by Day of Week and Hour (Current Filters)",
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

    with chart_tabs[3]:
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
        corr_fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation of Noise, Traffic, and Weather Variables (Current Filters)",
        )
        st.plotly_chart(corr_fig, use_container_width=True)

    with chart_tabs[4]:
        metrics_fig = px.bar(
            metrics.sort_values("rmse"),
            x="model",
            y=["rmse", "mae", "mape"],
            barmode="group",
            title="Latest Trained Model Comparison Metrics",
        )
        st.plotly_chart(metrics_fig, use_container_width=True)
        st.caption("Model metrics come from the latest trained artifacts. The other tabs above use the current filtered observations.")


def main() -> None:
    st.title("Urban Noise Analytics")
    st.caption("Spatio-temporal analysis and short-term urban noise forecasting from open city-style data.")

    processed_df = load_processed()
    predictions_df = load_predictions()
    metrics_df = load_metrics()
    forecast_validation_df = load_forecast_validation()

    min_date = processed_df["timestamp"].min().date()
    max_date = processed_df["timestamp"].max().date()

    st.sidebar.header("Filters")
    selected_boroughs = st.sidebar.multiselect(
        "Borough",
        sorted(processed_df["borough"].unique().tolist()),
        default=sorted(processed_df["borough"].unique().tolist()),
    )
    selected_land_uses = st.sidebar.multiselect(
        "Land Use Type",
        sorted(processed_df["land_use_type"].unique().tolist()),
        default=sorted(processed_df["land_use_type"].unique().tolist()),
    )
    date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start_date = pd.Timestamp(min_date)
        end_date = pd.Timestamp(max_date)

    filtered_df = filter_dataframe(processed_df, selected_boroughs, selected_land_uses, start_date, end_date)
    filtered_predictions = predictions_df[
        (predictions_df["borough"].isin(selected_boroughs))
        & (predictions_df["land_use_type"].isin(selected_land_uses))
        & (predictions_df["timestamp"].dt.date >= start_date.date())
        & (predictions_df["timestamp"].dt.date <= end_date.date())
    ].copy()

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Filtered Observations", f"{len(filtered_df):,}")
    summary_cols[1].metric("Sites", filtered_df["location_id"].nunique())
    summary_cols[2].metric("Mean Noise", f"{filtered_df['noise_level_db'].mean():.2f} dB(A)")
    summary_cols[3].metric("Mean Traffic", f"{filtered_df['traffic_volume_veh_hr'].mean():.0f} veh/hr")

    map_mode = st.radio("Map Metric", ["Current", "Average"], horizontal=True)
    render_map(filtered_df, map_mode)

    st.subheader("Time Series and Predictions")
    render_time_series(filtered_df, filtered_predictions if not filtered_predictions.empty else predictions_df)

    render_future_forecast(processed_df)

    st.subheader("Model Comparison")
    render_metrics(metrics_df)

    render_forecast_validation(forecast_validation_df)
    render_saved_figures(filtered_df, metrics_df)


if __name__ == "__main__":
    main()
