"""Train the minimal reference model and log the result with MLflow."""

import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "iris.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_PATH = REPORTS_DIR / "train_metrics.json"
MLFLOW_DB_PATH = ROOT_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = ROOT_DIR / "mlartifacts"

FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
TARGET_COLUMN = "target_name"


def _get_or_create_experiment() -> str:
    """Create the local MLflow experiment on first run and reuse it afterwards."""
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("mlops-reference-solution")
    if experiment is not None:
        return experiment.experiment_id

    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return client.create_experiment(
        "mlops-reference-solution",
        artifact_location=MLFLOW_ARTIFACTS_DIR.resolve().as_uri(),
    )


def train_model() -> dict[str, float | int | str]:
    """Train a small classification pipeline and persist the artifacts."""
    dataframe = pd.read_csv(DATA_PATH)

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "scale_numeric_features",
                StandardScaler(),
                FEATURE_COLUMNS,
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=300,
                    random_state=42,
                ),
            ),
        ]
    )

    experiment_id = _get_or_create_experiment()

    with mlflow.start_run(
        experiment_id=experiment_id, run_name="logistic-regression-iris"
    ):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, MODEL_PATH)

        metrics = {
            "accuracy": round(float(accuracy), 4),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "model_type": "logistic_regression",
        }
        metadata = {
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "class_labels": sorted(y.unique().tolist()),
        }

        MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))

        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "test_size": 0.2,
                "random_state": 42,
            }
        )
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(str(MODEL_METADATA_PATH), artifact_path="model")
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")
        mlflow.log_artifact(str(DATA_PATH), artifact_path="data")

    return metrics


if __name__ == "__main__":
    training_metrics = train_model()
    print(json.dumps(training_metrics, indent=2))
