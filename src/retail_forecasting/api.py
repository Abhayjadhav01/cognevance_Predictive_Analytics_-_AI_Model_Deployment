from fastapi import FastAPI, HTTPException, Query

from .models import forecast_months, load_artifact
from .paths import MODEL_ARTIFACT

app = FastAPI(
    title="Retail Revenue Forecast API",
    description="Local monthly revenue forecasts from the UCI Online Retail II dataset.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    status = "ready" if MODEL_ARTIFACT.is_file() else "model_missing"
    return {"status": status}


@app.get("/forecast")
def forecast(
    months: int = Query(default=6, ge=1, le=24, description="Forecast horizon (1-24 months)."),
) -> dict[str, object]:
    try:
        artifact = load_artifact()
        forecast_data = forecast_months(artifact, months)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return {
        "selected_model": artifact["selected_model"],
        "training_cutoff": artifact["training_cutoff"],
        "forecasts": forecast_data.to_dict(orient="records"),
    }
