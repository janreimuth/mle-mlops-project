"""Minimal prediction API for the reference solution."""

import csv
import json
import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
MODEL_METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
REQUEST_LOG_PATH = ROOT_DIR / "monitoring" / "requests.csv"

PREDICTION_REQUESTS = Counter(
    "api_prediction_requests_total",
    "Total number of prediction requests handled by the API.",
)
PREDICTION_LATENCY = Histogram(
    "api_prediction_latency_seconds",
    "Latency of prediction requests in seconds.",
)

REQUEST_LOG_LOCK = threading.Lock()

app = FastAPI(title="MLOps Reference Solution API", version="1.0.0")


class PredictionRequest(BaseModel):
    """Input schema expected by the trained Iris model."""

    sepal_length_cm: float = Field(..., gt=0, allow_inf_nan=False, examples=[5.1])
    sepal_width_cm: float = Field(..., gt=0, allow_inf_nan=False, examples=[3.5])
    petal_length_cm: float = Field(..., gt=0, allow_inf_nan=False, examples=[1.4])
    petal_width_cm: float = Field(..., gt=0, allow_inf_nan=False, examples=[0.2])


def _load_model() -> tuple[Any, dict[str, Any]]:
    """Load the trained model and its metadata from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model is missing. Run `uv run dvc repro` inside `solution/` first."
        )
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            "The model metadata is missing. Run `uv run dvc repro` inside `solution/` first."
        )

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(MODEL_METADATA_PATH.read_text())
    return model, metadata


MODEL, MODEL_METADATA = _load_model()
FEATURE_COLUMNS = MODEL_METADATA["feature_columns"]


def _append_request_log(payload: PredictionRequest, prediction: str) -> None:
    """Append a row to the request log used by the Evidently report."""
    REQUEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "sepal_length_cm": payload.sepal_length_cm,
        "sepal_width_cm": payload.sepal_width_cm,
        "petal_length_cm": payload.petal_length_cm,
        "petal_width_cm": payload.petal_width_cm,
        "prediction": prediction,
    }

    with REQUEST_LOG_LOCK:
        file_exists = REQUEST_LOG_PATH.exists()
        with REQUEST_LOG_PATH.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness check for the API."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics in the expected text format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    """Run a prediction, return the class label, and log the request."""
    try:
        start_time = perf_counter()

        input_row = {
            "sepal_length_cm": payload.sepal_length_cm,
            "sepal_width_cm": payload.sepal_width_cm,
            "petal_length_cm": payload.petal_length_cm,
            "petal_width_cm": payload.petal_width_cm,
        }
        ordered_features = pd.DataFrame(
            [[input_row[column] for column in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )
        prediction = str(MODEL.predict(ordered_features)[0])

        _append_request_log(payload, prediction)
        PREDICTION_REQUESTS.inc()
        PREDICTION_LATENCY.observe(perf_counter() - start_time)

        return {"prediction": prediction}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
