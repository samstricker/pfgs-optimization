""""""

from typing import List

import numpy as np
import polars as pl

from paretodo import InvestigationSpace

from helpers.data.dataset import Dataset


def add_recipes_to_experiments(
    experiments: Dataset,
    selected_recipes: pl.DataFrame,
    investigation_space: InvestigationSpace,
) -> Dataset:
    if selected_recipes.is_empty():
        return experiments
    new_experiment_recipes = experiments.copy()
    new_experiment_recipes.denormalize()
    new_experiment_recipes.y_data = None
    new_experiment_recipes.y_errors = None
    for label in (
        new_experiment_recipes.y_labels
        if new_experiment_recipes.y_labels is not None
        else []
    ):
        new_experiment_recipes.normalizer_mean.pop(label, None)
        new_experiment_recipes.normalizer_std.pop(label, None)
    new_experiment_recipes.x_data = np.vstack(
        [
            np.asarray(new_experiment_recipes.x_data),
            selected_recipes.select(investigation_space.labels).to_numpy(),
        ]
    )
    for i, label in enumerate(investigation_space.labels):
        new_experiment_recipes.add_normalization_params(
            label,
            min=investigation_space.variables[i].min,
            max=investigation_space.variables[i].max,
        )
    new_experiment_recipes.normalized = False
    new_experiment_recipes.normalize()
    new_experiment_recipes.denormalize()
    new_experiment_recipes.dataset_type = "recipes"
    return new_experiment_recipes


def concatenate_datasets(dataset1: Dataset, dataset2: Dataset) -> Dataset:
    dataset1.denormalize()
    dataset2.denormalize()
    if dataset1.dataset_type != dataset2.dataset_type:
        raise ValueError("dataset_type of both datasets must be the same")
    if dataset1.x_labels != dataset2.x_labels:
        raise ValueError("x_labels of both datasets must be the same")
    if dataset1.y_labels != dataset2.y_labels:
        raise ValueError("y_labels of both datasets must be the same")
    if dataset1.x_data is None:
        return dataset2 
    if dataset2.x_data is None:
        return dataset1
    x_data = np.vstack([dataset1.x_data, dataset2.x_data])
    if dataset1.y_data is not None and dataset2.y_data is not None:
        y_data = np.vstack([dataset1.y_data, dataset2.y_data])
    else:
        y_data = None
    if dataset1.y_errors is not None and dataset2.y_errors is not None:
        y_errors = np.vstack([dataset1.y_errors, dataset2.y_errors])
    else:
        y_errors = None
    new_dataset = Dataset(
        dataset_type=dataset1.dataset_type,
        x_labels=dataset1.x_labels,
        y_labels=dataset1.y_labels,
        x_data=x_data,
        y_data=y_data,
        y_errors=y_errors,
    )
    return new_dataset


def dataframe_to_recipe_dataset(df: pl.DataFrame, x_labels: List) -> Dataset:
    x_data = pl.DataFrame()
    for label in x_labels:
        if label in df.columns:
            x_data = x_data.with_columns(df[label])
    return Dataset(
        dataset_type="recipes",
        x_labels=x_labels,
        y_labels=None,
        x_data=x_data.to_numpy(),
        y_data=None,
        y_errors=None,
    )
