from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import FEATURE_COLUMNS, make_supervised, recursive_features
from .paths import METRICS_CSV, MODEL_ARTIFACT

MODEL_FACTORIES = {
    "Ridge Regression": lambda: Ridge(alpha=1.0),
    "Random Forest Regressor": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=3, min_samples_leaf=2, random_state=42, n_jobs=1
    ),
    "Gradient Boosting Regressor": lambda: GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.03, max_depth=2, loss="huber", random_state=42
    ),
}


class SeasonalNaiveRegressor:
    """Persistable seasonal-naive estimator using the prior-year target value."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "SeasonalNaiveRegressor":
        if len(features) != len(target):
            raise ValueError("Features and targets must have matching lengths.")
        self.is_fitted_ = True
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not getattr(self, "is_fitted_", False):
            raise ValueError("Seasonal-naive estimator has not been fitted.")
        if "lag_12" not in features:
            raise ValueError("Seasonal-naive prediction requires a 12-month lag.")
        return features["lag_12"].to_numpy(dtype=float)


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    observed = dict(zip(train["Month"], train["target"]))
    predictions: list[float] = []
    for row in test.itertuples(index=False):
        prior_year = row.Month - 12
        if prior_year not in observed:
            raise ValueError(
                "Seasonal-naive evaluation requires a 12-month historical value "
                "for each test month."
            )
        predictions.append(float(observed[prior_year]))
    return np.asarray(predictions)


def evaluate_models(supervised: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if len(supervised) < 15:
        raise ValueError(
            "At least 15 labeled periods are required for chronological holdout "
            "and model comparison."
        )
    test_periods = min(6, max(3, int(np.ceil(len(supervised) * 0.2))))
    train = supervised.iloc[:-test_periods]
    test = supervised.iloc[-test_periods:]
    if len(train) < 12:
        raise ValueError("Holdout would leave fewer than 12 training periods.")

    predictions = {
        "Seasonal-Naive Baseline": seasonal_naive(train, test),
    }
    for name, factory in MODEL_FACTORIES.items():
        model = factory()
        model.fit(train[FEATURE_COLUMNS], train["target"])
        predictions[name] = model.predict(test[FEATURE_COLUMNS])

    rows = []
    for name, values in predictions.items():
        rows.append(
            {
                "model": name,
                "mae": mean_absolute_error(test["target"], values),
                "rmse": float(np.sqrt(mean_squared_error(test["target"], values))),
                "r2": r2_score(test["target"], values),
                "training_periods": len(train),
                "test_periods": len(test),
            }
        )
    metrics = pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)
    return metrics, {
        "training_periods": len(train),
        "test_periods": len(test),
        "train": train,
        "test": test,
    }


def train_and_save(
    monthly: pd.DataFrame,
    artifact_path: Path = MODEL_ARTIFACT,
    metrics_path: Path = METRICS_CSV,
) -> tuple[dict[str, object], pd.DataFrame]:
    supervised = make_supervised(monthly)
    metrics, split = evaluate_models(supervised)
    selected_name = str(metrics.iloc[0]["model"])
    if selected_name == "Seasonal-Naive Baseline":
        selected_model = SeasonalNaiveRegressor()
        selected_model.fit(supervised[FEATURE_COLUMNS], supervised["target"])
    else:
        selected_model = MODEL_FACTORIES[selected_name]()
        selected_model.fit(supervised[FEATURE_COLUMNS], supervised["target"])
    history = monthly.copy()
    history["Month"] = pd.PeriodIndex(history["Month"], freq="M").astype(str)
    artifact: dict[str, object] = {
        "model": selected_model,
        "selected_model": selected_name,
        "feature_schema": FEATURE_COLUMNS.copy(),
        "training_cutoff": str(history["Month"].iloc[-1]),
        "history": history.to_dict(orient="records"),
        "holdout_training_periods": split["training_periods"],
        "holdout_test_periods": split["test_periods"],
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    metrics.to_csv(metrics_path, index=False)
    return artifact, metrics


def load_artifact(path: Path = MODEL_ARTIFACT) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python scripts/train_model.py`."
        )
    artifact = joblib.load(path)
    required = {
        "model",
        "selected_model",
        "feature_schema",
        "training_cutoff",
        "history",
    }
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError(f"Model artifact at {path} has an invalid structure.")
    if artifact["feature_schema"] != FEATURE_COLUMNS:
        raise ValueError(f"Model artifact at {path} has an incompatible feature schema.")
    return artifact


def forecast_months(artifact: dict[str, object], months: int) -> pd.DataFrame:
    if not 1 <= months <= 24:
        raise ValueError("months must be between 1 and 24.")
    if artifact["feature_schema"] != FEATURE_COLUMNS:
        raise ValueError("Model artifact has an incompatible feature schema.")

    historical = pd.DataFrame(artifact["history"])
    if historical.empty or not {"Month", "Revenue"}.issubset(historical.columns):
        raise ValueError("Model artifact does not contain valid monthly history.")
    historical["Month"] = pd.PeriodIndex(historical["Month"], freq="M")
    historical = historical.sort_values("Month").reset_index(drop=True)
    history_values = historical["Revenue"].astype(float).tolist()
    last_month = historical["Month"].iloc[-1]
    is_seasonal_naive = (
        artifact.get("selected_model") == "Seasonal-Naive Baseline"
    )
    forecasts = []

    for offset in range(1, months + 1):
        target_month = last_month + offset
        features = recursive_features(
            history_values,
            target_month,
            len(historical) + offset - 1,
        )
        if is_seasonal_naive:
            if len(history_values) < 12:
                raise ValueError(
                    "Seasonal-naive forecasting requires at least 12 historical months."
                )
            features["lag_12"] = history_values[-12]
        row = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        if is_seasonal_naive:
            row["lag_12"] = features["lag_12"]
        prediction = float(artifact["model"].predict(row)[0])
        forecasts.append({"Month": str(target_month), "ForecastRevenue": prediction})
        history_values.append(prediction)
    return pd.DataFrame(forecasts)
