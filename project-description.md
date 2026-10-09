# MLOps Project

This project is intentionally small, but it mirrors the workflow of a real production ML system. The emphasis is not on building a more complex model, but on showing that the full path from training to deployment to monitoring is reproducible, automated, and easy to inspect.

## Main Goal

Build a small but complete MLOps workflow around the model you trained during the Evidently introduction. A copy of the training notebook is supplied in [01-train-ml-model.ipynb](01-train-ml-model.ipynb), so you do not need to retrieve it from another repository.

The supplied model uses completed taxi trips for historical replay. `trip_distance` is the taximeter distance measured during the trip, not an estimate available at pickup. Keep that scope explicit when presenting your API and monitoring results.

If you want a runnable baseline to compare your own work against, see the reference implementation in [solution/README.md](solution/README.md). It uses Iris to demonstrate the same operational steps with a small local dataset; your project should refactor the supplied taxi model.

Your final project should show that you can move from experimentation to operations:

- Training code is reusable outside a notebook.
- Training data is versioned with DVC.
- Model runs and artifacts are tracked with MLflow.
- Retraining is automated with GitHub Actions.
- The trained model is served through an API.
- The API and model behavior are monitored with Evidently, Prometheus and Grafana.

## Suggested Workflow

### 1. Refactor the training code

**Objective:** Move the training logic out of the original notebook into Python files that can be executed repeatedly.

Suggested steps:

- Start from [01-train-ml-model.ipynb](01-train-ml-model.ipynb) and verify its baseline before refactoring.
- Create a small training entry point such as `src/train.py`.
- Move preprocessing, training and evaluation into reusable functions.
- Save the trained artifact in a predictable location.

Deliverables:

- Training code outside the notebook.
- A repeatable training command.
- A saved model artifact.

### 2. Track data and experiments

**Objective:** Make the training inputs and outputs reproducible.

Suggested steps:

- Version the training dataset with DVC.
- Configure an MLflow experiment.
- Log parameters, metrics and model artifacts for each training run.

Deliverables:

- DVC metadata committed to the repository.
- An MLflow run showing at least one successful training execution.

Hints:

- Initialize DVC at the repository root for your taxi pipeline. `solution/.dvc/` belongs to the separate Iris example.
- Keep the first DVC setup simple.
- Track only the data needed to reproduce training.

### 3. Automate training with GitHub Actions

**Objective:** Run the training pipeline automatically from CI.

The supplied [reference workflow](.github/workflows/reference-solution.yaml) validates the Iris solution only. Create or adapt a workflow for your taxi pipeline; a successful reference run does not validate your own training code.

Suggested steps:

- Create a workflow that installs dependencies and runs the training script.
- Use Python 3.13, `uv sync --locked`, and `uv run` so CI uses the same locked dependencies as your local environment.
- Make sure the workflow can access the data required for training.
- Log the training run to MLflow as part of the workflow.

Deliverables:

- A working GitHub Actions workflow.
- Evidence that the workflow completes successfully.

### 4. Serve the model through an API

**Objective:** Expose the latest trained model through an inference endpoint.

Suggested steps:

- Reuse or adapt the API you built previously.
- Load the trained artifact on startup.
- Add a prediction endpoint with a clear request and response schema.

Deliverables:

- Runnable API code.
- An example request that returns a prediction.

### 5. Add monitoring

**Objective:** Track both system behavior and model behavior.

Suggested steps:

- Expose API metrics for Prometheus.
- Create at least one Grafana dashboard.
- Send a batch of requests to the API to populate the metrics and monitoring data.
- Configure Evidently to monitor predictions, input data, or drift indicators.

Deliverables:

- Visible Prometheus metrics.
- A Grafana dashboard with at least one useful panel.
- Evidently output or reports connected to your prediction flow.

## Suggested Architecture

You do not need to match this structure exactly, but a project like the following is usually easy to review:

```text
.
├── .github/workflows/
├── data/
├── src/
│   ├── train.py
│   ├── preprocess.py
│   └── api.py
├── monitoring/
├── models/
├── dvc.yaml
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## Bonus Ideas

- Compare a baseline request batch with a deliberately shifted batch and interpret the drift report.
- Trigger retraining when you detect meaningful data drift.
- Add a small smoke test for the API or training entry point.
