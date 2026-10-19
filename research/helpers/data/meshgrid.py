"""Module to create a meshgrid dataset for visualization."""

from typing import List, Optional

from paretodo import InvestigationSpace

from helpers.data.dataset import Dataset


def get_meshgrid(
    investigation_space: InvestigationSpace,
    x_labels: List,
    y_labels: Optional[List],
    n_points_per_dim: int = 10,
) -> Dataset:
    """Create a meshgrid dataset for visualization.

    Parameters
    -----------
    investigation_space: InvestigationSpace
        The investigation space defining the bounds and dimensions.
    x_labels: List
        The labels for the x dimensions.
    y_labels: List
        The labels for the y dimensions.
    """
    if y_labels is None:
        raise ValueError("y_labels must be provided for meshgrid creation.")
    meshgrid, _ = investigation_space.create_meshgrid(points_per_dim=n_points_per_dim)
    meshgrid_recipes = Dataset(
        dataset_type="recipes",
        x_labels=x_labels,
        y_labels=y_labels,
        x_data=meshgrid,
    )
    for var in investigation_space.variables:
        meshgrid_recipes.add_normalization_params(var.name, var.min, var.max)
    return meshgrid_recipes
