""" This file runs one iteration of the optimization process using Provided Data. """

import logging as log
from pathlib import Path

from paretodo import (
    MlFlowLogger,
    InvestigationSpace,
    OptimizationProblem,
)

from helpers.logger import set_logger
from helpers.config import load_config
from helpers.data.meshgrid import get_meshgrid
from helpers.botorch_gp import BoTorchGP
from helpers.sampling.sampling_pfgs import sample_pfgs

from helpers.data.dataset import Dataset



def main():
    """Main function to run the optimization."""
    log.info("################################################")
    log.info("### Starting Optimization Script  ###")
    log.info("################################################")

    log.info("Loading data and configuration...")
    config, files = load_config()
    iteration = config["iteration"]

    ml_logger = MlFlowLogger(experiment_name=config["name"])
    investigation_space = InvestigationSpace.from_json(
        Path(files["investigation_space"])
    )
    ml_logger.log_investigation_space(investigation_space)
    optimization_problem = OptimizationProblem.from_json(
        Path(files["optimization_problem"])
    )
    ml_logger.log_optimization_problem(
        optimization_problem, optimization_type="optimization"
    )
    experiment_dataset = Dataset.from_json(files["dataset"])
    ml_logger.log_dataframe(
        experiment_dataset.to_dataframe(),
        datatype="experiments",
        iteration=iteration,
    )

    meshgrid_recipes = get_meshgrid(
        investigation_space=investigation_space,
        x_labels=experiment_dataset.x_labels.copy(),
        y_labels=experiment_dataset.y_labels.copy(),
        n_points_per_dim=config["meshgrid_points_per_dim"],
    )

    model_dataset = experiment_dataset.copy()
    
    log.info("Training surrogate model ...")
    model = BoTorchGP()
    model.train(
        dataset=model_dataset,
    )
    current_meshgrid_recipes = meshgrid_recipes.copy()
    current_meshgrid_recipes.set_normalization_params(
        model_dataset.get_normalization_params()
    )
    log.info("Visualizing surrogate model ...")
    meshgrid_dataset = model.predict(current_meshgrid_recipes)
    ml_logger.log_dataframe(
        meshgrid_dataset.to_dataframe(),
        datatype="meshgrid",
        iteration=iteration,
    )

    log.info("Running PFGS optimization ...")
    pareto_df = sample_pfgs(
        nsga2_config=config["nsga2_config"],
        investigation_space=investigation_space,
        model_dataset=model_dataset,
        model=model,
        optimization_problem=optimization_problem,
        problem_type=config["problem"],
    )
    ml_logger.log_dataframe(
        pareto_df,
        datatype="pareto_front",
        iteration=iteration,
    )
    
if __name__ == "__main__":
    set_logger()
    main()