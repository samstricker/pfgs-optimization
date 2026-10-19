"""Helper functions for the Pareto front visualization app (v2 - minimal)."""

import mlflow


def get_client_and_run(
    mlflow_port: int, mlflow_experiment: str, mlflow_run: str
) -> tuple[mlflow.tracking.MlflowClient | None, mlflow.entities.Run | None]:
    """Get the MLflow client and run information."""
    if mlflow_port is None or mlflow_experiment is None or mlflow_run is None:
        return None, None
    client = mlflow.tracking.MlflowClient(
        tracking_uri=f"http://127.0.0.1:{mlflow_port}"
    )
    experiment = client.get_experiment_by_name(mlflow_experiment)
    if experiment is None:
        return client, None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName='{mlflow_run}'",
    )
    return client, runs[0] if runs else None
