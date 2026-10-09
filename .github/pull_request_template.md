# Checklist for the MLOps Project

<!-- If you are done with a topic mark the checkboxes with an `x` (like `[x]`) -->

- [ ] I read and understood the tasks.
- [ ] I refactored the training notebook into reusable Python code.
- [ ] I can run model training outside the notebook.
- [ ] I wrote a GitHub Action that trains the model.
- [ ] I use DVC to version the training data.
- [ ] I track training runs with MLflow.
- [ ] I updated or created an API that serves the trained model.
- [ ] I added monitoring with Evidently, Prometheus and Grafana.
- [ ] I sent prediction requests and inspected the monitoring output.
- [ ] I documented how to run the project locally.

Bonus:

- [ ] I compared baseline and deliberately shifted request batches and interpreted the drift report.
- [ ] I retrain the model with new data if I see data drift.
