""""""

import logging as log
from typing import Dict, List

import polars as pl

from paretodo import InvestigationSpace, OptimizationProblem, ParetoSelector

from helpers.logger import set_logger


set_logger()


def select_hitl_recipes(pareto_df: pl.DataFrame, iteration: int) -> pl.DataFrame:
    """Select recipes from the Pareto front based on user input in the dashboard.

    Parameters
    ----------
    pareto_df : pl.DataFrame
        DataFrame containing the Pareto front recipes with a "RecipeID" column.
    iteration : int
        The current iteration number.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the selected recipes based on user input.
    """
    log.info("Selecting new recipes using the dashboard.")
    if iteration == 0:
        log.info(
            "Start the dashboard using the command: `uv run python -m paretodo.dashboard.start`"
        )
    user_input = input(
        "Enter the selected recipe nrs. as a comma-separated list of values (e.g., '1.0, 2.0') or 'quit' to exit:"
    )
    if user_input.lower() == "quit" or user_input.lower() == "q":
        log.info("Exiting the optimization loop.")
        return pl.DataFrame()  # Return an empty DataFrame to signal exit
    recipe_list = [int(idx.strip()) for idx in user_input.split(",")]
    if not all(val in pareto_df["RecipeID"].to_list() for val in recipe_list):
        log.error("Invalid input. Please enter valid RecipeIDs from the Pareto front.")
    selected_recipes = pareto_df.filter(pl.col("RecipeID").is_in(recipe_list))
    return selected_recipes


def select_automatic_recipes(
    pareto_df: pl.DataFrame,
    iteration: int,
    discard_values: Dict[str, List[str]],
    investigation_space: InvestigationSpace,
    optimization_problem: OptimizationProblem,
    n_samples: int,
) -> pl.DataFrame:
    selector = ParetoSelector()
    restricted_pareto = discard_regions(pareto_df, discard_values, iteration)
    if restricted_pareto.height == 0:
        return pl.DataFrame()
    selected_points = selector.select_recipes(
        pareto_front=restricted_pareto,
        investigation_space=investigation_space,
        optimization_problem=optimization_problem,
        n_samples=n_samples,
    )
    return selected_points


def discard_regions(
    pareto_df: pl.DataFrame, discard_values: Dict[str, List[str]], iteration: int
) -> pl.DataFrame:
    restricted_pareto = pareto_df

    for column, value_list in discard_values.items():
        values = (
            value_list[iteration] if iteration < len(value_list) else value_list[-1]
        )
        min_val, max_val = values.strip("()").split(",")
        if min_val.lower() not in ["none", "null", " none", " null"]:
            min_value = float(min_val)
            restricted_pareto = restricted_pareto.filter(pl.col(column) >= min_value)
        if max_val.lower() not in ["none", "null", " none", " null"]:
            max_value = float(max_val)
            restricted_pareto = restricted_pareto.filter(pl.col(column) <= max_value)
    return restricted_pareto
