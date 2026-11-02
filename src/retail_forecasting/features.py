from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "month",
    "year_trend",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
]


def make_supervised(monthly: pd.DataFrame) -> pd.DataFrame:
    """Build target-month features using only revenue observed before that month."""
    if not {"Month", "Revenue"}.issubset(monthly.columns):
        raise ValueError("Monthly data must contain Month and Revenue columns.")
    data = monthly[["Month", "Revenue"]].copy()
    data["Month"] = pd.PeriodIndex(data["Month"], freq="M")
    data = data.sort_values("Month").reset_index(drop=True)
    if data.empty:
        raise ValueError("Monthly data must contain at least one period.")
    if data["Month"].duplicated().any():
        raise ValueError("Monthly data contains duplicate periods.")
    if data["Month"].isna().any() or data["Revenue"].isna().any():
        raise ValueError("Month and Revenue values must not be missing.")
    if not data["Month"].equals(
        pd.Series(
            pd.period_range(data["Month"].iloc[0], data["Month"].iloc[-1], freq="M")
        )
    ):
        raise ValueError("Monthly data must contain every calendar month in sequence.")
    if not np.isfinite(pd.to_numeric(data["Revenue"], errors="raise")).all():
        raise ValueError("Revenue values must be finite numbers.")

    result = pd.DataFrame(
        {
            "Month": data["Month"],
            "target": pd.to_numeric(data["Revenue"], errors="raise"),
        }
    )
    result["month"] = data["Month"].dt.month
    result["year_trend"] = range(len(data))
    result["lag_1"] = result["target"].shift(1)
    result["lag_2"] = result["target"].shift(2)
    result["lag_3"] = result["target"].shift(3)
    result["rolling_mean_3"] = result["target"].shift(1).rolling(3).mean()
    return result.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)


def recursive_features(history: list[float], month: pd.Period, year_trend: int) -> dict:
    if len(history) < 3:
        raise ValueError("At least three observed monthly revenues are required.")
    return {
        "month": month.month,
        "year_trend": year_trend,
        "lag_1": history[-1],
        "lag_2": history[-2],
        "lag_3": history[-3],
        "rolling_mean_3": sum(history[-3:]) / 3,
    }
