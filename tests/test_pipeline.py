import pandas as pd
import pytest

from retail_forecasting.data import clean_and_aggregate
from retail_forecasting.features import FEATURE_COLUMNS, make_supervised
from retail_forecasting.models import forecast_months, train_and_save


def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Invoice": ["1", "C2", "3", "4", "5"],
            "StockCode": ["A"] * 5,
            "Description": ["item"] * 5,
            "Quantity": [2, 1, 0, 3, 1],
            "InvoiceDate": [
                "2010-01-02",
                "2010-01-03",
                "2010-01-04",
                None,
                "2010-02-01",
            ],
            "Price": [10, 10, 10, 10, 5],
            "Customer ID": [1] * 5,
            "Country": ["United Kingdom"] * 5,
        }
    )


def test_cleaning_excludes_cancellations_and_invalid_rows():
    monthly, report = clean_and_aggregate(sample_transactions())

    assert monthly.to_dict("records") == [
        {"Month": "2010-01", "Revenue": 20.0},
        {"Month": "2010-02", "Revenue": 5.0},
    ]
    assert report["retained_rows"] == 2
    assert report["cancellation_rows_excluded"] == 1
    assert report["missing_or_invalid_dates_excluded"] == 1


def test_feature_values_only_use_previous_months():
    monthly = pd.DataFrame(
        {
            "Month": pd.period_range("2020-01", periods=5, freq="M").astype(str),
            "Revenue": [10, 20, 30, 40, 50],
        }
    )

    supervised = make_supervised(monthly)

    assert supervised.iloc[0]["Month"] == pd.Period("2020-04", freq="M")
    assert supervised.iloc[0][FEATURE_COLUMNS[2:]].tolist() == [30, 20, 10, 20]


def test_recursive_forecast_uses_predicted_months_and_checks_horizon():
    class LastValueModel:
        def predict(self, features):
            return features["lag_1"].to_numpy()

    artifact = {
        "model": LastValueModel(),
        "feature_schema": FEATURE_COLUMNS,
        "history": [
            {"Month": "2020-01", "Revenue": 10.0},
            {"Month": "2020-02", "Revenue": 20.0},
            {"Month": "2020-03", "Revenue": 30.0},
        ],
    }

    forecast = forecast_months(artifact, 2)

    assert forecast["Month"].tolist() == ["2020-04", "2020-05"]
    assert forecast["ForecastRevenue"].tolist() == [30.0, 30.0]
    with pytest.raises(ValueError, match="between 1 and 24"):
        forecast_months(artifact, 25)


def test_seasonal_naive_can_be_selected_and_forecast_recursively(tmp_path, monkeypatch):
    import retail_forecasting.models as models

    monthly = pd.DataFrame(
        {
            "Month": pd.period_range("2020-01", periods=30, freq="M").astype(str),
            "Revenue": range(100, 130),
        }
    )
    metrics = pd.DataFrame(
        [{"model": "Seasonal-Naive Baseline", "mae": 1.0, "rmse": 1.0, "r2": 0.5}]
    )
    monkeypatch.setattr(
        models,
        "evaluate_models",
        lambda supervised: (
            metrics,
            {"training_periods": 15, "test_periods": 3},
        ),
    )

    artifact, _ = train_and_save(
        monthly,
        artifact_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.csv",
    )

    forecast = forecast_months(artifact, 2)

    assert artifact["selected_model"] == "Seasonal-Naive Baseline"
    assert forecast["ForecastRevenue"].tolist() == [118.0, 119.0]


def test_api_serves_local_forecast_and_rejects_invalid_horizon(monkeypatch):
    from fastapi.testclient import TestClient

    import retail_forecasting.api as api

    class LastValueModel:
        def predict(self, features):
            return features["lag_1"].to_numpy()

    artifact = {
        "model": LastValueModel(),
        "selected_model": "Ridge Regression",
        "feature_schema": FEATURE_COLUMNS,
        "training_cutoff": "2020-03",
        "history": [
            {"Month": "2020-01", "Revenue": 10.0},
            {"Month": "2020-02", "Revenue": 20.0},
            {"Month": "2020-03", "Revenue": 30.0},
        ],
    }
    monkeypatch.setattr(api, "load_artifact", lambda: artifact)

    response = TestClient(api.app).get("/forecast?months=2")
    invalid_response = TestClient(api.app).get("/forecast?months=0")

    assert response.status_code == 200
    assert response.json()["training_cutoff"] == "2020-03"
    assert len(response.json()["forecasts"]) == 2
    assert invalid_response.status_code == 422
