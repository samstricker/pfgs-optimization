"""This module provides a factory function to retrieve the appropriate simulator"""

from typing import Dict, Union

from paretodo import MlFlowLogger

from helpers.problems.hb_butterot_simulator import HimmelblauButterotSimulator
from helpers.problems.digital_biolab_simulator import DigitalBioLabSimulator


def get_simulator(
    config: Dict, mlflow_logger: MlFlowLogger
) -> Union[HimmelblauButterotSimulator, DigitalBioLabSimulator]:
    """Return the appropriate simulator based on the problem name.

    Parameters
    ----------
    config: Dict
        The configuration dictionary for the simulator.
    mlflow_logger: MlFlowLogger
        The MLflow logger for tracking experiments.

    Returns
    -------
    Simulator
        An instance of the simulator corresponding to the specified problem.
    """
    problem = config["problem"]
    if problem in ["hb_butterot", "himmelblau"]:
        return HimmelblauButterotSimulator()
    elif problem == "bio":
        if config is None:
            raise ValueError("Config must be provided for the bio problem.")
        return DigitalBioLabSimulator(
            config=config["dbiolab"], mlflow_logger=mlflow_logger
        )
    else:
        raise ValueError(f"Unsupported problem: {problem}")
