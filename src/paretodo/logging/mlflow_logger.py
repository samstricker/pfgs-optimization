"""Logger class for logging to mlflow."""

import datetime as dt
import os
from pathlib import Path
import tempfile
from typing import Optional

import mlflow as mlf
import polars as pl

from paretodo import InvestigationSpace, OptimizationProblem


class MlFlowLogger:
    """
    Logger class for logging to mlflow.

    Parameters
    ----------
    experiment_name : str
        Name of the mlflow experiment.
    """

    def __init__(self, experiment_name: str):
        mlf.set_tracking_uri("sqlite:///mlflow.db")
        mlf.set_experiment(experiment_name)
        run_name = f"Run_at_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        mlf.start_run(run_name=run_name)

    def log_investigation_space(self, investigation_space: InvestigationSpace):
        """
        Log the investigation space to mlflow.

        Parameters
        ----------
        investigation_space : InvestigationSpace
            Investigation space to log.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "investigation_space.json")
            investigation_space.to_json(file_path=Path(temp_file_path))
            self.log_artifact(file_path=temp_file_path, artifact_path="general")

    def log_optimization_problem(
        self,
        optimization_problem: OptimizationProblem,
        optimization_type: str = "optimization",
    ):
        """
        Log the optimization problem to mlflow.

        Parameters
        ----------
        optimization_problem : OptimizationProblem
            Optimization problem to log.
        optimization_type : str
            Type of the optimization problem.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            if optimization_type.lower() == "manufacturing":
                temp_file_name = "manufacturing_optimization_problem.json"
            else:
                temp_file_name = "optimization_problem.json"
            temp_file_path = os.path.join(temp_dir, temp_file_name)
            optimization_problem.to_json(file_path=Path(temp_file_path))
            self.log_artifact(file_path=temp_file_path, artifact_path="general")

    def log_dataframe(
        self,
        df: pl.DataFrame,
        datatype: str,
        iteration: int,
    ):
        """Log a Polars DataFrame as a CSV artifact to mlflow.

        Parameters
        ----------
        df : pl.DataFrame
            Polars DataFrame to log.
        datatype : str
            Type of the datatype.
            Supported types are: experiments, meshgrid, pareto_front, selected_recipes.
        iteration : int
            Iteration number to log the artifact under.
        """
        if datatype.lower() == "experiments":
            artifact_name = "experiments.csv"
        elif datatype.lower() == "meshgrid":
            artifact_name = "meshgrid_predictions.csv"
        elif datatype.lower() == "pareto_front":
            artifact_name = "pareto_front_predictions.csv"
        elif datatype.lower() == "selected_recipes":
            artifact_name = "selected_recipes.csv"
        elif datatype.lower() == "manufacturing_front":
            artifact_name = "manufacturing_front_predictions.csv"
        else:
            raise ValueError(
                f"Unsupported datatype: {datatype}. Supported types are:"
                "experiments, meshgrid, pareto_front, selected_recipes, manufacturing_front."
            )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_csv_path = os.path.join(temp_dir, artifact_name)
            df.write_csv(temp_csv_path)
            self.log_artifact(
                file_path=temp_csv_path, artifact_path=f"Iteration_{iteration:03d}"
            )

    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """Log an artifact to mlflow.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing artifacts.
        artifact_path : Optional[str]
            Path to the artifact. Default is None.
        """
        mlf.log_artifact(local_path=file_path, artifact_path=artifact_path)

    def add_tag(self, key: str, value: str):
        """Add a tag to the mlflow run.

        Parameters
        ----------
        key : str
            Tag key.
        value : str
            Tag value.
        """
        mlf.set_tag(key, value)

    def add_parameter(self, key: str, value: str):
        """Add a parameter to the mlflow run.

        Parameters
        ----------
        key : str
            Parameter key.
        value : str
            Parameter value.
        """
        mlf.log_param(key, value)

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Parameters
        ----------
        name : str
            Name of the variable
        value : float
            Value to log
        step : Optional[int]
            Timestep to log it to
        """
        mlf.log_metric(key=name, value=value, step=step)

    def end_run(self):
        """End the mlflow run."""
        mlf.end_run()
