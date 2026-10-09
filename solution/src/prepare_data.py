"""Create the reference dataset used by the DVC pipeline.

This script keeps the repo self-contained by exporting the built-in Iris dataset
to a CSV file inside the solution folder. That gives us a reproducible dataset
artifact without depending on an external download step.
"""

from pathlib import Path

from sklearn.datasets import load_iris

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
DATA_PATH = DATA_DIR / "iris.csv"


def prepare_data() -> Path:
    """Export Iris data to a beginner-friendly CSV file."""
    iris = load_iris(as_frame=True)
    dataframe = iris.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length_cm",
            "sepal width (cm)": "sepal_width_cm",
            "petal length (cm)": "petal_length_cm",
            "petal width (cm)": "petal_width_cm",
            "target": "target_id",
        }
    )
    dataframe["target_name"] = dataframe["target_id"].map(
        {index: name for index, name in enumerate(iris.target_names)}
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(DATA_PATH, index=False)
    return DATA_PATH


if __name__ == "__main__":
    output_path = prepare_data()
    print(f"Created reference dataset at {output_path}")
