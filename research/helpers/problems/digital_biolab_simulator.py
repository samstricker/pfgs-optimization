"""This module defines the class, which simulates experiments using the dbiolab package."""

from copy import deepcopy
from typing import Tuple, Optional

import numpy as np
import os
from tempfile import TemporaryDirectory as tempfolder

from dbiolab.mammalian_cellculture_simulation import (
    ReactorType,
    MammalianCellcultureReactor,
    MammalianRecipe,
    ReactionEvolutionTimeseries,
)
from dbiolab.mammalian_cellculture_simulation.data_models import MammalianDataset
from dbiolab.mammalian_cellculture_simulation.visualization import (
    plot_reaction_evolution_timeseries,
)
from paretodo import MlFlowLogger
from helpers.data.dataset import Dataset


class DigitalBioLabSimulator:

    def __init__(self, config: dict, mlflow_logger: Optional[MlFlowLogger] = None):
        self.config = config
        self.mlflow_logger = mlflow_logger

    def run_simulation(
        self, recipes: Dataset, iteration: Optional[int] = None
    ) -> Dataset:
        if recipes.dataset_type != "recipes":
            raise ValueError("Input dataset must be of type 'recipes'.")
        recipes.denormalize()
        experiment_dataset = recipes.copy()
        simulator, mammalian_dataset = self._dataset_to_mamallian_inputs(recipes)
        exp_data = simulator.run_dataset(mammalian_dataset, seed=self.config["seed"])
        if self.config["visualize"].lower() == "true" and iteration is not None:
            self._plot_and_log_time_series(exp_data, iteration)
        experiment_dataset = self._mammalian_dataset_to_experiment_dataset(
            exp_data, experiment_dataset
        )
        return experiment_dataset

    def _dataset_to_mamallian_inputs(
        self, recipes: Dataset
    ) -> Tuple[MammalianCellcultureReactor, MammalianDataset]:
        simulator, mammalian_recipe = self._get_input_data()
        all_recipes = []
        if recipes.x_data is None:
            raise ValueError("The recipes dataset has no x_data.")
        for recipe in recipes.x_data:
            new_mammalian_recipe = deepcopy(mammalian_recipe)
            for j, var in enumerate(recipes.x_labels):
                if var.lower() == "ph_value":
                    new_mammalian_recipe.ph_value = float(recipe[j])
                elif var.lower() == "temperature":
                    new_mammalian_recipe.temperature = float(recipe[j])
                elif var.lower() == "initial_glc":
                    new_mammalian_recipe.initial_conditions.c_glc = float(recipe[j])
                elif var.lower() == "initial_gln":
                    new_mammalian_recipe.initial_conditions.c_gln = float(recipe[j])
                elif var.lower() == "feed_1_glc":
                    new_mammalian_recipe.feeding_strategy[
                        0
                    ].feed_concentrations.c_glc = float(recipe[j])
                elif var.lower() == "feed_1_gln":
                    new_mammalian_recipe.feeding_strategy[
                        0
                    ].feed_concentrations.c_gln = float(recipe[j])
                elif var.lower() == "feed_2_glc":
                    new_mammalian_recipe.feeding_strategy[
                        1
                    ].feed_concentrations.c_glc = float(recipe[j])
                elif var.lower() == "feed_2_gln":
                    new_mammalian_recipe.feeding_strategy[
                        1
                    ].feed_concentrations.c_gln = float(recipe[j])
                else:
                    raise ValueError(f"Unsupported variable in recipes dataset: {var}")
            all_recipes.append(new_mammalian_recipe)
        mammalian_dataset = MammalianDataset(all_recipes)

        return (
            simulator,
            mammalian_dataset,
        )

    def _get_input_data(self) -> Tuple[MammalianCellcultureReactor, MammalianRecipe]:
        reactor = ReactorType.from_json(self.config["reactor"])
        simulator = MammalianCellcultureReactor(reactor_type=reactor)
        mammalian_recipe = MammalianRecipe.from_json(self.config["mammalian_recipe"])
        return simulator, mammalian_recipe

    def _mammalian_dataset_to_experiment_dataset(
        self, mammalian_dataset: MammalianDataset, experiment_dataset: Dataset
    ) -> Dataset:
        # if not isinstance(all(mammalian_dataset.data), ReactionEvolutionTimeseries):
        #     raise ValueError(
        #         f"Mammalian dataset must contain ReactionEvolutionTimeseries data, got {type(mammalian_dataset.data[0])}."
        #     )
        if self.config["objective_time"] == "end_of_batch":
            y_labels = ["product_concentration", "aggregate_concentration", "purity"]
            c_product = [result.c_product[-1] for result in mammalian_dataset.data]
            c_agg = [result.c_aggregates[-1] for result in mammalian_dataset.data]
            purity = [result.purity[-1] for result in mammalian_dataset.data]
            experiment_dataset.y_data = np.array([c_product, c_agg, purity]).T
            experiment_dataset.y_labels = y_labels
            experiment_dataset.dataset_type = "experiments"
        else:
            raise ValueError(
                f"Unsupported objective_time in config: {self.config['objective_time']}"
            )
        return experiment_dataset

    def _plot_and_log_time_series(
        self, mammalian_dataset: MammalianDataset, iteration: int
    ):
        if self.mlflow_logger is None:
            raise ValueError(
                "MLflow logger must be provided for logging time series plots."
            )
        for i, result in enumerate(mammalian_dataset.data):
            (
                cell_density_figure,
                metabolite_concentration_figure,
                product_concentration_figure,
                purity_figure,
            ) = plot_reaction_evolution_timeseries(result)
            filenames = [
                f"experiment_{i}_cell_density.png",
                f"experiment_{i}_metabolite_concentration.png",
                f"experiment_{i}_product_concentration.png",
                f"experiment_{i}_purity.png",
            ]
            with tempfolder() as tmpdir:
                paths = [os.path.join(tmpdir, fn) for fn in filenames]
                cell_density_figure.write_image(paths[0])
                metabolite_concentration_figure.write_image(paths[1])
                product_concentration_figure.write_image(paths[2])
                purity_figure.write_image(paths[3])
                self.mlflow_logger.log_artifact(
                    paths[0], f"Iteration_{iteration:03d}/experiment_{i}"
                )
                self.mlflow_logger.log_artifact(
                    paths[1], f"Iteration_{iteration:03d}/experiment_{i}"
                )
                self.mlflow_logger.log_artifact(
                    paths[2], f"Iteration_{iteration:03d}/experiment_{i}"
                )
                self.mlflow_logger.log_artifact(
                    paths[3], f"Iteration_{iteration:03d}/experiment_{i}"
                )
