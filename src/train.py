import logging
import os
from pathlib import Path
from urllib.request import urlretrieve

import json
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Reduce utility log noise without hiding warnings about model packaging.
logging.getLogger("mlflow.utils").setLevel(logging.WARNING)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("green-taxi-duration")

def downloadData(year=2024, month=1, color ="green"):
    # Download the data
    data_path = Path(f"./data/{color}_tripdata_{year}-{month:02d}.parquet")
    data_path.parent.mkdir(exist_ok=True)

    if not data_path.exists():
        urlretrieve(
            f"https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet",
            data_path,
        )

def loadData(year=2024, month=1, color ="green"):
    df = pd.read_parquet(f"./data/{color}_tripdata_{year}-{month:02d}.parquet")
    return df

def initMlFlow():
    # Load environment variables from .env file
    load_dotenv()

    # Get the MLflow tracking URI from environment variables
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    # Set up the connection to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment_name = "green-taxi-monitoring"
    exp = mlflow.get_experiment_by_name(experiment_name)

    if exp is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            tags={"developer": "<your-name>"},
        )
        print(f"Created experiment: {experiment_name} (ID: {experiment_id})")
    else:
        print(f"Using existing experiment: {experiment_name} (ID: {exp.experiment_id})")

    mlflow.set_experiment(experiment_name)

features = ["PULocationID", "DOLocationID", "trip_distance"]
target = "duration"

# Calculate duration, remove outliers, and treat the location IDs as categories
def preprocess(df):
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[df["trip_distance"].between(0.1, 100)]
    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(
        str
    )
    df = df[features + [target]]
    return df

def splitData(df_processed):
    y = df_processed["duration"]
    X = df_processed.drop(columns=["duration"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2);
    return X_train, X_test, y_train, y_test



def train_model(X_train, X_test, y_train, y_test, color, year, month, features, target):
    with mlflow.start_run():
        tags = {
            "model": "linear regression",
            "developer": "<your-name>",
            "dataset": f"{color}-taxi",
            "year": year,
            "month": month,
            "features": features,
            "target": target,
        }
        mlflow.set_tags(tags)

        pipeline = make_pipeline(DictVectorizer(), LinearRegression())
        pipeline.fit(X_train.to_dict(orient="records"), y_train)

        y_pred = pipeline.predict(X_test.to_dict(orient="records"))
        rmse = root_mean_squared_error(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline, name="mlflow-model", serialization_format="skops"
        )
        # metrics = {
        #     "accuracy": round(float(accuracy), 4),
        #     "train_rows": len(X_train),
        #     "test_rows": len(X_test),
        #     "model_type": "logistic_regression",
        # }
        # metadata = {
        #     "feature_columns": FEATURE_COLUMNS,
        #     "target_column": TARGET_COLUMN,
        #     "class_labels": sorted(y_train.unique().tolist()),
        # }

        # MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        # METRICS_PATH.write_text(json.dumps(metrics, indent=2))

        # mlflow.log_params(
        #     {
        #         "model_type": "logistic_regression",
        #         "test_size": 0.2,
        #         "random_state": 42,
        #     }
        # )
        # mlflow.log_metric("accuracy", float(accuracy))
        # mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        # mlflow.log_artifact(str(MODEL_METADATA_PATH), artifact_path="model")
        # mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")
        # mlflow.log_artifact(str(DATA_PATH), artifact_path="data")

        active_run = mlflow.active_run()
        assert active_run is not None
        run_id = active_run.info.run_id

        client = MlflowClient()
        model_name = "green-taxi-ride-duration"

        try:
            client.create_registered_model(
                name=model_name,
                tags={"framework": "scikit-learn"},
                description="Model to predict ride duration in NYC",
            )
        except MlflowException as exc:
            # Reuse the model only if it already exists; re-raise any other MLflow error.
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
            print(f"Registered model {model_name} already exists. Reusing it.")

        mv = client.create_model_version(
            name=model_name,
            source=model_info.model_uri,
            run_id=run_id,
        )

        client.set_registered_model_alias(
            name=model_name,
            version=mv.version,
            alias="production",
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="mlflow-model",
            serialization_format="skops",
            registered_model_name="green-taxi-ride-duration",
        )

def main():
    year=2024
    month=1
    color ="green"
    features = "undefined"
    target = "undefined"

    downloadData()
    df = loadData()
    df_preprocessed = preprocess(df)

    X_train, X_test, y_train, y_test = splitData(df_preprocessed)
    train_model(X_train, X_test, y_train, y_test, color, year, month, features, target)

if __name__ == "__main__":
    main()