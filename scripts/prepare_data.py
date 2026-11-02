from retail_forecasting.data import build_monthly_dataset


if __name__ == "__main__":
    monthly, report = build_monthly_dataset()
    print(f"Saved {len(monthly)} monthly periods.")
    print(f"Rows retained: {report['retained_rows']} / {report['input_rows']}")
