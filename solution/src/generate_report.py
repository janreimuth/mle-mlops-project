"""Generate a static input-drift report from the API request history."""

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DATA_PATH = ROOT_DIR / "data" / "raw" / "iris.csv"
REQUEST_LOG_PATH = ROOT_DIR / "monitoring" / "requests.csv"
REPORT_PATH = ROOT_DIR / "monitoring" / "reports" / "data_drift_report.html"

FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]


def generate_report() -> Path:
    """Compare the full Iris reference with all logged request inputs."""
    if not REQUEST_LOG_PATH.exists():
        raise FileNotFoundError(
            "The request log is missing. Send a few requests to the API before "
            "running the Evidently report."
        )

    # Use all stored rows, keeping only inputs: exclude labels and predictions.
    reference_data = pd.read_csv(REFERENCE_DATA_PATH)[FEATURE_COLUMNS]
    current_data = pd.read_csv(REQUEST_LOG_PATH)[FEATURE_COLUMNS]

    # This demo guard is not a statistical reliability threshold.
    if len(current_data) < 5:
        raise ValueError(
            "The request log needs at least five rows for this demonstration. "
            "Send more prediction requests first."
        )

    # The preset defines drift checks; run computes them for these two datasets.
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data, current_data=current_data)

    # Export a static snapshot; new requests require running this script again.
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_PATH))
    return REPORT_PATH


if __name__ == "__main__":
    output_path = generate_report()
    print(f"Wrote Evidently report to {output_path}")
