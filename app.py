from __future__ import annotations

import pandas as pd
import streamlit as st

from retail_forecasting.models import forecast_months, load_artifact
from retail_forecasting.paths import METRICS_CSV

st.set_page_config(page_title="Retail Revenue Forecast", layout="wide")
st.title("Retail Revenue Forecast")
st.caption(
    "Academic prototype using UCI Online Retail II. Forecasts are estimates, "
    "not financial commitments; review them before business use."
)

try:
    artifact = load_artifact()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.info("Acquire and prepare the data, then run `python scripts/train_model.py`.")
    st.stop()

history = pd.DataFrame(artifact["history"])
history["Month"] = pd.to_datetime(history["Month"])
horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=24, value=6)
forecast = forecast_months(artifact, horizon)
forecast["Month"] = pd.to_datetime(forecast["Month"])

metric_columns = st.columns(3)
metric_columns[0].metric("Selected model", artifact["selected_model"])
metric_columns[1].metric("Training cutoff", artifact["training_cutoff"])
metric_columns[2].metric("Holdout periods", artifact["holdout_test_periods"])

chart_history = history.rename(columns={"Revenue": "Historical revenue"}).set_index("Month")
chart_forecast = forecast.rename(columns={"ForecastRevenue": "Forecast revenue"}).set_index("Month")
st.subheader("Monthly revenue and recursive forecast")
st.line_chart(chart_history[["Historical revenue"]].join(chart_forecast, how="outer"))
st.dataframe(forecast.rename(columns={"Month": "Month", "ForecastRevenue": "Forecast revenue"}), use_container_width=True)

if METRICS_CSV.is_file():
    metrics = pd.read_csv(METRICS_CSV)
    st.subheader("Chronological holdout model comparison")
    st.caption(
        f"Training periods: {artifact['holdout_training_periods']}; "
        f"test periods: {artifact['holdout_test_periods']}. "
        "The short history makes scores unstable; lower MAE/RMSE is better and R² can be negative."
    )
    st.dataframe(metrics, hide_index=True, use_container_width=True)
else:
    st.warning("Holdout metrics are missing. Run `python scripts/train_model.py` again.")

st.caption(
    "Returns treatment: cancellation invoices are excluded and are not netted against sales."
)
