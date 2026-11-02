from retail_forecasting.data import acquire_dataset


if __name__ == "__main__":
    path = acquire_dataset()
    print(f"Dataset workbook available at: {path}")
