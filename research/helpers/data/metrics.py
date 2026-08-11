"""This module provides functionality to compute benchmarking metrics."""

import numpy as np
from numpy.typing import NDArray


def calculate_rmse(data: NDArray, test_data: NDArray) -> float:
    """Calculate the Root Mean Square Error (RMSE) between predicted and actual values.

    Parameters
    ----------
    data : NDArray
        The predicted data.
    test_data : NDArray
        The actual test data.

    Returns
    -------
    float
        The RMSE value.
    """
    # Ensure inputs are 1-D vectors of the same length
    if data.ndim != 1 or test_data.ndim != 1:
        raise ValueError("data and test_data must be 1-D vectors")
    if data.shape != test_data.shape:
        raise ValueError("data and test_data must have the same shape")
    rrmse = float(np.sqrt(np.mean((data - test_data) ** 2)))
    return rrmse


def calculate_rrmse(data: NDArray, test_data: NDArray) -> float:
    """Calculate the Relative Root Mean Square Error (RRMSE) between predicted and actual values.

    Parameters
    ----------
    data : NDArray
        The predicted data.
    test_data : NDArray
        The actual test data.

    Returns
    -------
    float
        The RRMSE value.
    """
    std = float(np.std(test_data))
    rmse = calculate_rmse(data.copy(), test_data.copy())
    if std == 0.0:
        raise ValueError("Standard deviation of test_data is zero, RRMSE undefined")
    return float(rmse / std)
