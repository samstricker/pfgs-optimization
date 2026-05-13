"""Module for selection of Recipes based on the calculated Pareto front."""

from typing import List

import numpy as np
from numpy.typing import NDArray
import polars as pl

from paretodo.data_models import InvestigationSpace, OptimizationProblem


class ParetoSelector:
    """Selector for Recipes based on Pareto front."""

    def select_closest_recipe_to_utopia(
        self,
        pareto_front: pl.DataFrame,
        utopia_point: pl.DataFrame,
        bounds: pl.DataFrame,
    ) -> pl.DataFrame:
        """Select the recipe from the Pareto front closest to the utopia point.

        Parameters
        ----------
        pareto_front : pl.DataFrame
            DataFrame containing the Pareto front recipes.
        utopia_point : pl.DataFrame
            DataFrame containing the utopia point.
        bounds : pl.DataFrame
            DataFrame containing the bounds for normalization.

        Returns
        -------
        pl.DataFrame
            DataFrame containing the selected recipe.
        """

        objective_cols = [obj for obj in pareto_front.columns if obj.startswith("obj")]
        objective_df = pareto_front.select(objective_cols)
        if pareto_front.height != 1 and utopia_point.height != 1:
            raise ValueError("Utopia point must be a single recipe (1D data).")
        if bounds.height != 2 or bounds.width != utopia_point.width:
            raise ValueError(
                "Bounds must have shape (2, n_dimensions) matching Pareto front."
            )
        if utopia_point.columns != objective_df.columns:
            raise ValueError(
                "Utopia point labels must match the Pareto front objective labels."
            )
        utopia_point, objective_df, bounds = self._drop_columns_with_none(
            utopia_point, objective_df, bounds
        )
        norm_pareto_front = self._normalize_array_with_bounds(
            objective_df.to_numpy(), bounds.to_numpy()
        )
        norm_utopia = self._normalize_array_with_bounds(
            utopia_point.to_numpy(), bounds.to_numpy()
        )

        distances = np.linalg.norm(norm_pareto_front - norm_utopia, axis=1)
        closest_idx = np.argmin(distances)

        return pareto_front.slice(int(closest_idx), 1)

    def select_additional_recipes_according_to_spread(
        self,
        pareto_front: pl.DataFrame,
        reference_point: pl.DataFrame,
        n_additional: int,
        investigation_space: List[str],
        experiments_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Select additional recipes from the Pareto front to maximize spread.

        Parameters
        ----------
        pareto_front : pl.DataFrame
            DataFrame containing the Pareto front recipes.
        reference_point : pl.DataFrame
            DataFrame containing the already selected recipes.
        n_additional : int
            Number of additional recipes to select.
        investigation_space : List[str]
            List of investigation space variable names to consider for spread calculation.
        experiments_df : pl.DataFrame | None
            DataFrame containing the experiments data to be far from.

        Returns
        -------
        pl.DataFrame
            DataFrame containing the selected additional recipes.
        """
        # Keep original pareto front for bounds calculation
        pareto_front_original = pareto_front.clone()

        for row in reference_point.iter_rows(named=True):
            pareto_front = pareto_front.filter(pl.col("RecipeID") != row["RecipeID"])

        # Calculate bounds once from the original full pareto front (in investigation space)
        available_cols = [
            col for col in investigation_space if col in pareto_front_original.columns
        ]

        if not available_cols:
            return reference_point

        pareto_space_original = pareto_front_original.select(available_cols)
        min_bounds = pareto_space_original.min().to_numpy().flatten()
        max_bounds = pareto_space_original.max().to_numpy().flatten()
        space_bounds = np.array([min_bounds, max_bounds])

        for iteration in range(n_additional):
            # Use only available columns that exist in remaining pareto front
            available_cols_current = [
                col for col in available_cols if col in pareto_front.columns
            ]

            if not available_cols_current:
                break

            pareto_space_df = pareto_front.select(available_cols_current)
            reference_space_df = reference_point.select(available_cols_current)

            # Use the fixed bounds (computed once from original pareto front)
            # but filter to match current available columns
            cols_indices = [available_cols.index(col) for col in available_cols_current]
            space_bounds_current = space_bounds[:, cols_indices]

            # Normalize pareto front in investigation space
            norm_pareto_space = self._normalize_array_with_bounds(
                pareto_space_df.to_numpy(), space_bounds_current
            )

            # Calculate distances to each reference point
            distances = np.zeros((pareto_space_df.height, reference_space_df.height))
            for j in range(reference_space_df.height):
                ref_row_normalized = self._normalize_array_with_bounds(
                    reference_space_df.slice(j, 1).to_numpy(), space_bounds_current
                )
                distances[:, j] = np.linalg.norm(
                    norm_pareto_space - ref_row_normalized, axis=1
                )

            min_distances = distances.min(axis=1)

            # Also consider experiments data if provided
            if experiments_df is not None:
                available_exp_cols = [
                    col
                    for col in available_cols_current
                    if col in experiments_df.columns
                ]
                if available_exp_cols:
                    experiments_space_df = experiments_df.select(available_exp_cols)

                    # Normalize experiments in investigation space (using same bounds as pareto)
                    norm_experiments_space = self._normalize_array_with_bounds(
                        experiments_space_df.to_numpy(), space_bounds_current
                    )

                    # Calculate distances to all experiment points
                    for i in range(norm_experiments_space.shape[0]):
                        distances_to_exp = np.linalg.norm(
                            norm_pareto_space - norm_experiments_space[i : i + 1],
                            axis=1,
                        )
                        min_distances = np.minimum(min_distances, distances_to_exp)

            if len(min_distances) == 0:
                break
            next_idx = np.argmax(min_distances)

            # Get original row from pareto_front
            original_recipe = pareto_front.slice(int(next_idx), 1)
            pareto_front = pareto_front.slice(0, int(next_idx)).vstack(
                pareto_front.slice(int(next_idx) + 1)
            )

            reference_point = pl.concat(
                [reference_point, original_recipe], how="vertical"
            )
        return reference_point

    def select_recipes(
        self,
        pareto_front: pl.DataFrame,
        investigation_space: InvestigationSpace,
        optimization_problem: OptimizationProblem,
        n_samples: int,
        experiments_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Select recipes from the Pareto front.

        Parameters
        ----------
        pareto_front : pl.DataFrame
            DataFrame containing the Pareto front recipes.
        n_samples : int
            Number of recipes to select.
        experiments_df : pl.DataFrame | None
            DataFrame containing the experiments data.

        Returns
        -------
        pl.DataFrame
            DataFrame containing the selected recipes.
        """
        utopia_point = self._utopia_point_from_optimization_problem(
            optimization_problem
        )
        bounds = self._bounds_for_objectives(pareto_front)
        utopia_recipe = self.select_closest_recipe_to_utopia(
            pareto_front=pareto_front,
            utopia_point=utopia_point,
            bounds=bounds,
        )
        selected_recipes = self.select_additional_recipes_according_to_spread(
            pareto_front=pareto_front,
            reference_point=utopia_recipe,
            n_additional=n_samples - 1,
            investigation_space=investigation_space.labels,
            experiments_df=experiments_df,
        )
        return selected_recipes

    def _bounds_for_objectives(self, pareto_front: pl.DataFrame) -> pl.DataFrame:
        objective_cols = [col for col in pareto_front.columns if col.startswith("obj:")]
        objective_df = pareto_front.select(objective_cols)
        min_bounds = objective_df.min().to_numpy().flatten()
        max_bounds = objective_df.max().to_numpy().flatten()
        bounds_data = {
            col: [min_bounds[i], max_bounds[i]] for i, col in enumerate(objective_cols)
        }
        bounds = pl.DataFrame(bounds_data)
        return bounds

    def _utopia_point_from_optimization_problem(
        self,
        optimization_problem: OptimizationProblem,
    ) -> pl.DataFrame:
        utopia_data = {}
        objective_labels = optimization_problem.objective_labels
        for i, obj in enumerate(optimization_problem.problem):
            if obj.minimize:
                value = 0.0
            else:
                value = 1.0
            utopia_data[objective_labels[i]] = [value]
        utopia_point = pl.DataFrame(utopia_data)
        return utopia_point

    def _drop_columns_with_none(
        self,
        utopia_point: pl.DataFrame,
        pareto_front: pl.DataFrame,
        bounds: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        cols_to_drop = []
        for col in utopia_point.columns:
            first_val = utopia_point[col][0]
            if isinstance(first_val, str) and first_val.lower() == "none":
                cols_to_drop.append(col)
            else:
                utopia_point = utopia_point.with_columns(pl.col(col).cast(pl.Float64))
        if cols_to_drop:
            utopia_point = utopia_point.drop(cols_to_drop)
            pareto_front = pareto_front.drop(cols_to_drop)
            bounds = bounds.drop(cols_to_drop)
        return utopia_point, pareto_front, bounds

    def _normalize_array_with_bounds(
        self, data: NDArray[np.float64], bounds: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Normalize an array of data according to given bounds.

        Parameters
        ----------
        data : NDArray[np.float64]
            The data to be normalized.
        bounds : NDArray[np.float64]
            The bounds for normalization, shape (n_dimensions, 2).

        Returns
        -------
        NDArray[np.float64]
            The normalized data.
        """
        norm_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            min_bound = bounds[0, i]
            max_bound = bounds[1, i]
            range_val = max_bound - min_bound
            if range_val != 0:
                norm_data[:, i] = (data[:, i] - min_bound) / range_val
            else:
                norm_data[:, i] = 0.0
        return norm_data
