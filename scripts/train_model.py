from retail_forecasting.data import build_monthly_dataset
from retail_forecasting.models import train_and_save
from retail_forecasting.paths import MONTHLY_CSV


if __name__ == "__main__":
    if MONTHLY_CSV.exists():
        import pandas as pd

        monthly = pd.read_csv(MONTHLY_CSV)
    else:
        monthly, _ = build_monthly_dataset()
    artifact, metrics = train_and_save(monthly)
    print(f"Selected model: {artifact['selected_model']}")
    print(f"Training cutoff: {artifact['training_cutoff']}")
    print(metrics.to_string(index=False))
