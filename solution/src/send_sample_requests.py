"""Send a small, varied Iris batch to the local reference API."""

from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "iris.csv"
PREDICTION_URL = "http://127.0.0.1:8000/predict"


def send_sample_requests() -> None:
    """Replay two observations per class, excluding the ground-truth columns."""
    dataframe = pd.read_csv(DATA_PATH)
    samples = dataframe.groupby("target_name", sort=False).head(2)
    payloads = samples.drop(columns=["target_id", "target_name"]).to_dict(
        orient="records"
    )

    for index, payload in enumerate(payloads, start=1):
        response = requests.post(PREDICTION_URL, json=payload, timeout=5)
        response.raise_for_status()
        print(f"Request {index}: {response.json()['prediction']}")


if __name__ == "__main__":
    send_sample_requests()
