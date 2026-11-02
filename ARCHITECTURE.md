# Architecture and workflow

```text
UCI Online Retail II
        |
        v
scripts/acquire_data.py --> data/raw/online_retail_ii.xlsx
        |
        v
retail_forecasting.data
  schema validation, date parsing, exclusions, monthly aggregation
        |                         \
        v                          v
data/processed/monthly_revenue.csv reports/cleaning_report.json
        |
        v
retail_forecasting.features
  calendar + prior-only lags/rolling mean
        |
        v
retail_forecasting.models
  chronological holdout -> baseline/Ridge/Random Forest/Gradient Boosting
        |                         \
        v                          v
models/revenue_forecaster.joblib reports/model_metrics.csv
        |                         |
        +--------+----------------+
                 v
       FastAPI API / Streamlit dashboard
```

## Boundaries and limitations

- Initial deployment is local. There is no authentication, transaction ingestion, inventory decision, or financial commitment functionality.
- Forecasting is supervised regression on monthly totals. Features for a target month include its month number, a sequential trend index, three prior monthly revenues, and a three-month mean computed from those prior months.
- Evaluation preserves chronology: the latest three to six labeled periods are held out. The model is selected by holdout MAE and then refit on all labeled periods.
- Recursive forecasts append each predicted revenue to the history used for later forecast lags. Long-range forecasts can accumulate errors.
- Cancellation invoices are excluded; returns are not netted. This target definition must accompany any reported results.
- Two years of historical data do not support robust annual seasonality conclusions. Sparse months/outliers and the small holdout can make metrics unstable.
- Joblib loading is restricted to the configured local artifact path, but joblib is pickle-based: never load an untrusted artifact.
- Actual run evidence and screenshots should be captured after acquisition, training, and local UI/API execution in the intended submission environment.
