"""This module contains the main function to run the PFGS."""

import logging as log

import polars as pl

from paretodo import InvestigationSpace, ParetoFrontGenerator

from helpers.data.dataset import Dataset
from helpers.logger import set_logger
from helpers.botorch_gp import BoTorchGP
from helpers.sampling.pfgs_problem import GenericProblem

set_logger()


def sample_pfgs(
    nsga2_config: dict,
    investigation_space: InvestigationSpace,
    model_dataset: Dataset,
    model: BoTorchGP,
    optimization_problem,
    problem_type: str,
) -> pl.DataFrame:

    if problem_type in ["himmelblau", "hb_butterot", "bio"]:
        problem = GenericProblem(
            investigation_space=investigation_space,
            dataset=model_dataset.copy(),
            model=model,
            optimization_problem=optimization_problem,
            robustness_confidence=nsga2_config["robustness_confidence"],
        )
    else:
        log.error("Unsupported problem type: %s", problem_type)
        raise ValueError(f"Unsupported problem type: {problem_type}")

    pareto_generator = ParetoFrontGenerator(
        seed=nsga2_config["seed"],
        problem=problem,
        pop_size_per_dim=nsga2_config["pop_size_per_dim"],
        n_max_gen=nsga2_config["n_max_gen"],
    )
    _, pareto_df = pareto_generator.generate_pareto_front()
    return pareto_df
