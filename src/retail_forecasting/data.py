from __future__ import annotations

import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from .paths import CLEANING_REPORT, MONTHLY_CSV, RAW_WORKBOOK

DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DATASET_CITATION = (
    "Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5CG6D"
)
REQUIRED_COLUMNS = {
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
}


def acquire_dataset(
    destination: Path = RAW_WORKBOOK, source_url: str = DATASET_URL
) -> Path:
    """Download and extract the UCI workbook; an existing workbook is reused."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    archive_path = destination.parent / "online_retail_ii.zip"
    with urlopen(source_url, timeout=60) as response, archive_path.open("wb") as output:
        output.write(response.read())

    try:
        with zipfile.ZipFile(archive_path) as archive:
            workbook_names = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".xlsx", ".xls"))
            ]
            if not workbook_names:
                raise ValueError("The UCI archive did not contain an Excel workbook.")
            workbook_name = next(
                (name for name in workbook_names if "online retail ii" in name.lower()),
                workbook_names[0],
            )
            with archive.open(workbook_name) as source, destination.open("wb") as target:
                target.write(source.read())
    finally:
        archive_path.unlink(missing_ok=True)
    return destination


def load_transactions(workbook: Path = RAW_WORKBOOK) -> pd.DataFrame:
    if not workbook.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {workbook}. Run `python scripts/acquire_data.py`."
        )

    sheets = pd.read_excel(workbook, sheet_name=None)
    if not sheets:
        raise ValueError(f"No worksheets were found in {workbook}.")
    transactions = pd.concat(sheets.values(), ignore_index=True)
    transactions.columns = [str(column).strip() for column in transactions.columns]
    missing = REQUIRED_COLUMNS.difference(transactions.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return transactions


def clean_and_aggregate(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Apply the documented exclusions and return a gap-filled monthly series."""
    missing = REQUIRED_COLUMNS.difference(transactions.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    original_rows = len(transactions)
    invoice = transactions["Invoice"].astype("string").str.strip()
    cancellation = invoice.str.upper().str.startswith("C", na=False)
    dates = pd.to_datetime(transactions["InvoiceDate"], errors="coerce")
    quantity = pd.to_numeric(transactions["Quantity"], errors="coerce")
    price = pd.to_numeric(transactions["Price"], errors="coerce")

    remaining = ~cancellation
    invalid_date = remaining & dates.isna()
    remaining &= dates.notna()
    invalid_quantity_or_price = remaining & (
        quantity.isna() | price.isna() | quantity.le(0) | price.le(0)
    )
    remaining &= quantity.notna() & price.notna() & quantity.gt(0) & price.gt(0)
    missing_invoice = remaining & (invoice.isna() | invoice.eq(""))
    valid = remaining & invoice.notna() & invoice.ne("")
    cleaned = pd.DataFrame(
        {
            "InvoiceDate": dates.loc[valid],
            "Revenue": (quantity.loc[valid] * price.loc[valid]),
        }
    )
    if cleaned.empty:
        raise ValueError("No valid transaction rows remain after cleaning.")

    monthly = (
        cleaned.assign(Month=cleaned["InvoiceDate"].dt.to_period("M"))
        .groupby("Month", as_index=False)["Revenue"]
        .sum()
        .sort_values("Month")
    )
    month_range = pd.period_range(monthly["Month"].min(), monthly["Month"].max(), freq="M")
    monthly = (
        monthly.set_index("Month")
        .reindex(month_range, fill_value=0.0)
        .rename_axis("Month")
        .reset_index()
    )
    monthly["Month"] = monthly["Month"].astype(str)
    monthly["Revenue"] = monthly["Revenue"].astype(float)

    report: dict[str, int | str] = {
        "source_citation": DATASET_CITATION,
        "input_rows": original_rows,
        "cancellation_rows_excluded": int(cancellation.sum()),
        "missing_or_invalid_dates_excluded": int(invalid_date.sum()),
        "invalid_quantity_or_price_rows_excluded": int(invalid_quantity_or_price.sum()),
        "missing_invoice_rows_excluded": int(missing_invoice.sum()),
        "retained_rows": int(valid.sum()),
        "monthly_periods": len(monthly),
        "first_month": str(monthly["Month"].iloc[0]),
        "last_month": str(monthly["Month"].iloc[-1]),
        "returns_treatment": "Cancellation invoices are excluded; returns are not netted.",
    }
    return monthly, report


def build_monthly_dataset(
    workbook: Path = RAW_WORKBOOK,
    output_csv: Path = MONTHLY_CSV,
    report_path: Path = CLEANING_REPORT,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    monthly, report = clean_and_aggregate(load_transactions(workbook))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_csv, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return monthly, report
