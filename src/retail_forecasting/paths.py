from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
RAW_WORKBOOK = RAW_DATA_DIR / "online_retail_II.xlsx"
MONTHLY_CSV = PROCESSED_DATA_DIR / "monthly_revenue.csv"
CLEANING_REPORT = REPORT_DIR / "cleaning_report.json"
METRICS_CSV = REPORT_DIR / "model_metrics.csv"
MODEL_ARTIFACT = MODEL_DIR / "revenue_forecaster.joblib"
