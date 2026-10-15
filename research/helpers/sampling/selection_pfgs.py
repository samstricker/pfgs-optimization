""""""

import logging as log
from typing import Dict, List, Union

import polars as pl

from paretodo import InvestigationSpace, OptimizationProblem, ParetoSelector

from helpers.logger import set_logger
from ..problems.digital_biolab_simulator import DigitalBioLabSimulator
from ..problems.hb_butterot_simulator import HimmelblauButterotSimulator
from ..data.dataset import Dataset

set_logger()



def df_to_dataset(df: pl.DataFrame, investigation_space: InvestigationSpace) -> Dataset:
    """Convert a Polars DataFrame to a Dataset object.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame containing the recipe data.

    Returns
    -------
    Dataset
        A Dataset object containing the recipe data.
    """
    if df.is_empty():
        raise ValueError("The input DataFrame is empty.")
    x_data = df.select(investigation_space.labels).to_numpy()
    recipe_dataset = Dataset(
        x_data=x_data,
        x_labels=investigation_space.labels,
        dataset_type="recipes",
    )
    return recipe_dataset


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


def select_pareto_expert(
        pareto_df: pl.DataFrame,
        iteration: int,
        discard_values: Dict[str, List[str]],
        investigation_space: InvestigationSpace,
        optimization_problem: OptimizationProblem,
        n_samples: int,
        simulator: Union[DigitalBioLabSimulator, HimmelblauButterotSimulator],
        expert_type: str = "expert" # "expert" or "noob"
) -> pl.DataFrame:
    # Selection of Recipes with the best true performance in the Pareto Front
    # Data Leakage is expected here, as the true performance is used to select the best recipes
    # this will only work for the first standard_deviation optimization axis in the optimization problem
    pareto_df = discard_regions(pareto_df, discard_values, iteration)

    pareto_recipe_dataset = df_to_dataset(pareto_df, investigation_space)
    pareto_experiment_dataset = simulator.run_simulation(pareto_recipe_dataset)
    for axis in optimization_problem.problem:
        if axis.optimization_type == "objective":
            obj_idx = pareto_experiment_dataset.y_labels.index(axis.var_name)
            if pareto_experiment_dataset.y_data is None:
                raise ValueError("The experiment dataset has no y_data.")
            pareto_df = pareto_df.with_columns(
                pl.Series(
                    name=f"true_{axis.var_name}",
                    values=pareto_experiment_dataset.y_data[:, obj_idx],
                )
            )
            if axis.minimize:
                pareto_df = pareto_df.sort(f"true_{axis.var_name}", descending=False)
                if expert_type == "noob":
                    pareto_df = pareto_df.sort(f"true_{axis.var_name}", descending=True)
            else:
                pareto_df = pareto_df.sort(f"true_{axis.var_name}", descending=True)
                if expert_type == "noob":
                    pareto_df = pareto_df.sort(f"true_{axis.var_name}", descending=False)
            if n_samples > pareto_df.height:
                raise ValueError(
                    f"Requested number of samples ({n_samples}) exceeds the number of available recipes in the Pareto front ({pareto_df.height})."
                )
            return pareto_df.head(n_samples)


def select_pareto_explore(
    pareto_df: pl.DataFrame,
    iteration: int,
    discard_values: Dict[str, List[str]],
    optimization_problem: OptimizationProblem,
    n_samples: int,
) -> pl.DataFrame:
    # Selection of Recipes with the Highest Variance in the Pareto Front
    # this will only work for the first standard_deviation optimization axis in the optimization problem
    pareto_df = discard_regions(pareto_df, discard_values, iteration)

    for axis in optimization_problem.problem:
        name = "obj:"
        if axis.optimization_type == "standard_deviation":
            if axis.minimize:
                name += "min("
            else:
                name += "max("
            name += f"σ({axis.var_name}))"
            if axis.minimize:
                pareto_df = pareto_df.sort(name, descending=False)
            else:
                pareto_df = pareto_df.sort(name, descending=True)
            if n_samples > pareto_df.height:
                raise ValueError(
                    f"Requested number of samples ({n_samples}) exceeds the number of available recipes in the Pareto front ({pareto_df.height})."
                )
            return pareto_df.head(n_samples)


def select_pareto_exploit(
    pareto_df: pl.DataFrame,
    iteration: int,
    discard_values: Dict[str, List[str]],
    optimization_problem: OptimizationProblem,
    n_samples: int,
) -> pl.DataFrame:
    # Selection of the Highest Performing Recipe (Mean Value) from the Pareto Front
    # this will only work for the first objective optimization axis in the optimization problem
    
    pareto_df = discard_regions(pareto_df, discard_values, iteration)
    for axis in optimization_problem.problem:
        name = "obj:"
        if axis.optimization_type == "objective":
            if axis.minimize:
                name += "min("
            else:
                name += "max("
            name += f"μ({axis.var_name}))"
            if axis.minimize:
                pareto_df = pareto_df.sort(name, descending=False)
            else:
                pareto_df = pareto_df.sort(name, descending=True)
            if n_samples > pareto_df.height:
                raise ValueError(
                    f"Requested number of samples ({n_samples}) exceeds the number of available recipes in the Pareto front ({pareto_df.height})."
                )
            return pareto_df.head(n_samples)

