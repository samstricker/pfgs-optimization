"""Data filtering and validation utilities."""

from pathlib import Path
import pandas as pd
import numpy as np

from paretodo.data_models import InvestigationSpace, OptimizationProblem
from .types_and_constants import LOCAL_TEMP_DIR, BoundsDict


def load_investigation_space(mlflow_run: str) -> InvestigationSpace:
    """Load investigation space from JSON file."""
    path = Path(f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/investigation_space.json")
    return InvestigationSpace.from_json(path)


def load_optimization_problem(
    mlflow_run: str, mode: str = "Optimization"
) -> OptimizationProblem:
    """Load optimization problem from JSON file.

    Parameters
    ----------
    mlflow_run : str
        The MLflow run name.
    mode : str, optional
        The mode to load ('Optimization' or 'Manufacturing'). Default is 'Optimization'.

    Returns
    -------
    OptimizationProblem
        Optimization problem loaded from the appropriate JSON file.
    """
    filename = (
        "manufacturing_optimization_problem.json"
        if mode == "Manufacturing"
        else "optimization_problem.json"
    )
    path = Path(f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/{filename}")
    return OptimizationProblem.from_json(path)


def load_experiments_df(mlflow_run: str, iteration: str) -> pd.DataFrame:
    """Load experiments data from CSV file."""
    csv_path = f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/experiments.csv"
    return pd.read_csv(csv_path)


def load_pareto_df(mlflow_run: str, iteration: str) -> pd.DataFrame:
    """Load Pareto front data from CSV file."""
    csv_path = f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/pareto_front_predictions.csv"
    return pd.read_csv(csv_path)


def load_meshgrid_df(mlflow_run: str, iteration: str) -> pd.DataFrame:
    """Load meshgrid predictions data from CSV file."""
    csv_path = f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/meshgrid_predictions.csv"
    return pd.read_csv(csv_path)


def load_manufacturing_df(mlflow_run: str, iteration: str) -> pd.DataFrame:
    """Load manufacturing Pareto front data from CSV file.

    Parameters
    ----------
    mlflow_run : str
        The MLflow run name.
    iteration : str
        The iteration name.

    Returns
    -------
    pd.DataFrame
        Manufacturing Pareto front data loaded from CSV file.
        Returns empty dataframe if file doesn't exist.
    """
    csv_path = (
        f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/manufacturing_front_predictions.csv"
    )
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        # Return empty dataframe if manufacturing file doesn't exist
        return pd.DataFrame()


def get_measurement_columns(
    experiments_df: pd.DataFrame, var_names: list[str]
) -> list[str]:
    """Extract measurement columns (numeric columns that aren't investigation variables)."""
    numeric_cols = experiments_df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in var_names]


def get_objective_columns(df: pd.DataFrame) -> list[str]:
    """Extract objective columns (those starting with 'obj:')."""
    return [col for col in df.columns if col.startswith("obj:")]


def get_variable_pairs(variables: list) -> list[tuple[str, str]]:
    """Create variable pairs (var1 vs var2, var3 vs var4, etc.)."""
    pairs = []
    for i in range(0, len(variables), 2):
        if i + 1 < len(variables):
            pairs.append((variables[i].name, variables[i + 1].name))
    return pairs


def get_objective_pairs(num_objectives: int) -> list[tuple[int, int]]:
    """Generate all unique pairs of objectives without repetition."""
    pairs = []
    for i in range(num_objectives):
        for j in range(i + 1, num_objectives):
            pairs.append((i, j))
    return pairs


def filter_dataframe_by_variables(
    df: pd.DataFrame, var_to_filter: dict[str, float]
) -> pd.DataFrame:
    """Filter dataframe by variable values."""
    filtered = df.copy()
    for var_name, filter_val in var_to_filter.items():
        if var_name in filtered.columns:
            filtered = filtered[filtered[var_name] == filter_val]
    return filtered


def get_points_outside_bounds(
    df: pd.DataFrame,
    objective_columns: list[str],
    bounds: BoundsDict,
) -> list[int]:
    """Get indices of points that are outside specified bounds."""
    outside = []
    for idx in df.index:
        for obj_col in objective_columns:
            if obj_col not in bounds:
                continue
            bound = bounds[obj_col]
            value = df.loc[idx, obj_col]

            try:
                # Handle complex numbers by skipping them
                if isinstance(value, complex):
                    continue
                value_float = float(value)  # type: ignore
            except (ValueError, TypeError):
                continue

            min_bound = bound.get("min")
            max_bound = bound.get("max")

            if min_bound is not None:
                try:
                    if value_float < float(min_bound):
                        outside.append(idx)
                        break
                except (ValueError, TypeError):
                    pass

            if max_bound is not None:
                try:
                    if value_float > float(max_bound):
                        outside.append(idx)
                        break
                except (ValueError, TypeError):
                    pass
    return outside


def is_point_within_bounds(
    df: pd.DataFrame, row_idx: int, obj_col: str, bounds: BoundsDict
) -> bool:
    """Check if a point is within specified bounds for given objective."""
    if obj_col not in bounds or obj_col not in df.columns:
        return True

    bound = bounds[obj_col]
    value = df.loc[row_idx, obj_col]

    try:
        # Handle complex numbers by skipping them
        if isinstance(value, complex):
            return True
        value_float = float(value)  # type: ignore
    except (ValueError, TypeError):
        return True

    min_bound = bound.get("min")
    max_bound = bound.get("max")

    if min_bound is not None:
        try:
            if value_float < float(min_bound):
                return False
        except (ValueError, TypeError):
            pass

    if max_bound is not None:
        try:
            if value_float > float(max_bound):
                return False
        except (ValueError, TypeError):
            pass

    return True


def normalize_objectives(df: pd.DataFrame, objective_cols: list[str]) -> pd.DataFrame:
    """Normalize objective columns to [0, 1] range."""
    normalized = df[objective_cols].copy()
    for col in objective_cols:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        if col_max > col_min:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
    return normalized


def get_theme() -> str:
    """Get the Plotly template name for consistent theming."""
    return "plotly_white"


def create_objective_index_map(
    optimization_problem: OptimizationProblem,
) -> dict[int, str]:
    """Create mapping from objective index to objective variable name."""
    return {idx: axis.var_name for idx, axis in enumerate(optimization_problem.problem)}
