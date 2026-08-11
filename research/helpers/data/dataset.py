"""Dataset dataclass for storing and normalizing experimental data."""

import logging as log
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
import polars as pl


@dataclass
class Dataset:
    """A dataclass for storing and normalizing experimental data.

    Parameters
    ----------
    dataset_type : str
        The type of dataset, e.g. "recipes", "experiments", "predictions".
    x_labels : List[str]
        The labels for the input features.
    x_data : Optional[NDArray[np.float64]], optional
        The input feature data, by default None.
    y_labels : Optional[List[str]], optional
        The labels for the target variables, by default None.
    y_data : Optional[NDArray[np.float64]], optional
        The target variable data, by default None.
    y_errors : Optional[NDArray[np.float64]], optional
        The errors associated with the target variable data, by default None.
    normalized : bool, optional
        Whether the dataset has been normalized, by default False.
    normalizer_mean : Dict[str, float], optional
        The mean values used for normalization, by default an empty dictionary.
    normalizer_std : Dict[str, float], optional
        The standard deviation values used for normalization, by default an empty dictionary.
    """

    dataset_type: str
    x_labels: List[str]
    x_data: Optional[NDArray[np.float64]] = None
    y_labels: Optional[List[str]] = None
    y_data: Optional[NDArray[np.float64]] = None
    y_errors: Optional[NDArray[np.float64]] = None
    normalized: bool = False

    normalizer_min: Dict[str, float] = field(default_factory=dict)
    normalizer_max: Dict[str, float] = field(default_factory=dict)
    normalizer_mean: Dict[str, float] = field(default_factory=dict)
    normalizer_std: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.dataset_type not in [
            "recipes",
            "experiments",
            "predictions",
            "testset",
        ]:
            raise ValueError(
                f"dataset_type must be one of ['recipes', 'experiments', 'predictions'], got {self.dataset_type}."
            )
        if not isinstance(self.x_labels, list) or not all(
            isinstance(label, str) for label in self.x_labels
        ):
            raise ValueError(
                f"x_labels must be a list of strings, got {self.x_labels}."
            )
        if self.x_data is not None:
            self.x_data = np.array(self.x_data, dtype=np.float64, copy=True)
        if self.y_data is not None:
            self.y_data = np.array(self.y_data, dtype=np.float64, copy=True)

        if self.x_data is not None and self.x_labels is None:
            raise ValueError("x_labels cannot be None if x_data is provided.")
        if self.y_data is not None and self.y_labels is None:
            raise ValueError("y_labels cannot be None if y_data is provided.")

    def normalize(self):
        """Normalize the dataset using mean and standard deviation normalization."""
        if self.normalized:
            return
        if self.x_data is None:
            raise ValueError("x_data is None, cannot normalize.")
        for i, label in enumerate(self.x_labels):
            self.x_data[:, i] = self._min_max_scaling(self.x_data[:, i], label)

        if self.y_labels is not None and self.y_data is not None:
            for i, label in enumerate(self.y_labels):
                self.y_data[:, i] = self._standardize(
                    self.y_data[:, i],
                    label,
                )
                if self.y_errors is not None:
                    # ensure y_errors is writable before in-place assignment
                    self.y_errors = np.array(self.y_errors, dtype=np.float64, copy=True)
                    self.y_errors[:, i] = self._normalize_error(
                        self.y_errors[:, i], label
                    )
        self.normalized = True

    def denormalize(self):
        """Denormalize the dataset using the stored mean and standard deviation."""
        if not self.normalized:
            return
        if self.x_data is None:
            raise ValueError("x_data is None, cannot denormalize.")
        for i, label in enumerate(self.x_labels):
            self.x_data[:, i] = self._min_max_destcaling(self.x_data[:, i], label)
        if self.y_labels is not None and self.y_data is not None:
            for i, label in enumerate(self.y_labels):
                self.y_data[:, i] = self._destandardize(self.y_data[:, i], label)
                if self.y_errors is not None:
                    self.y_errors[:, i] = self._denormalize_error(
                        self.y_errors[:, i], label
                    )
        self.normalized = False

    def add_normalization_params(self, name: str, min: float, max: float):
        """Add normalization parameters for a given feature or target variable.

        Parameters
        ----------
        name : str
            The name of the feature or target variable.
        min : float
            The minimum value of the feature or target variable.
        max : float
            The maximum value of the feature or target variable.
        """
        if self.normalized:
            log.warning("Adding normalization params to a normalized dataset.")
        self.normalizer_min[name] = min
        self.normalizer_max[name] = max

    def get_normalization_params(self) -> Dict[str, Dict[str, float]]:
        """Get the normalization parameters for the dataset.

        Returns
        -------
        Dict[str, Dict[str, float]]
            A dictionary containing the mean and standard deviation for each feature and target variable.
        """
        return {
            "mean": self.normalizer_mean.copy(),
            "std": self.normalizer_std.copy(),
            "min": self.normalizer_min.copy(),
            "max": self.normalizer_max.copy(),
        }

    def set_normalization_params(self, params: Dict[str, Dict[str, float]]):
        """Set the normalization parameters for the dataset.

        Parameters        ----------
        params : Dict[str, Dict[str, float]]
            A dictionary containing the mean and standard deviation for each feature and target variable, with keys "mean" and "std".
        """
        if self.normalized:
            raise ValueError("Cannot set normalization params on a normalized dataset.")
        self.normalizer_mean = params.get("mean", {}).copy()
        self.normalizer_std = params.get("std", {}).copy()
        self.normalizer_min = params.get("min", {}).copy()
        self.normalizer_max = params.get("max", {}).copy()

    def copy(self) -> "Dataset":
        """Create a deep copy of the Dataset instance.

        Return
        -------
        Dataset
            A deep copy of the current Dataset instance.
        """
        return Dataset(
            dataset_type=self.dataset_type,
            x_labels=self.x_labels.copy(),
            x_data=self.x_data.copy() if self.x_data is not None else None,
            y_labels=self.y_labels.copy() if self.y_labels is not None else None,
            y_data=self.y_data.copy() if self.y_data is not None else None,
            y_errors=self.y_errors.copy() if self.y_errors is not None else None,
            normalized=True if self.normalized else False,
            normalizer_mean=self.normalizer_mean.copy(),
            normalizer_std=self.normalizer_std.copy(),
            normalizer_min=self.normalizer_min.copy(),
            normalizer_max=self.normalizer_max.copy(),
        )

    def get_y_labels(self) -> List[str]:
        """Get the y_labels of the dataset.

        Returns
        -------
        List[str]
            The y_labels of the dataset.

        Raises
        ------
        ValueError
            If y_labels is None.
        """
        if self.y_labels is None:
            raise ValueError("y_labels is None.")
        return self.y_labels

    def _min_max_scaling(self, data: NDArray, name: str) -> NDArray:
        if name not in self.normalizer_min or name not in self.normalizer_max:
            min_val = float(np.nanmin(data))
            max_val = float(np.nanmax(data))
            if max_val - min_val == 0:
                raise ValueError(
                    f"Cannot min-max scale feature '{name}' with zero range."
                )
            self.normalizer_min[name] = min_val
            self.normalizer_max[name] = max_val
        return (data - self.normalizer_min[name]) / (
            self.normalizer_max[name] - self.normalizer_min[name]
        )

    def _min_max_destcaling(self, data: NDArray, name: str) -> NDArray:
        if name in self.normalizer_min and name in self.normalizer_max:
            return (
                data * (self.normalizer_max[name] - self.normalizer_min[name])
                + self.normalizer_min[name]
            )
        else:
            raise ValueError(f"Data for '{name}' has not been min-max scaled yet.")

    def _standardize(
        self, data: NDArray, name: str, norm_type: str = "mean_std"
    ) -> NDArray:
        if norm_type == "mean_std":
            # use NaN-aware statistics so missing values don't poison the params
            if name not in self.normalizer_mean or name not in self.normalizer_std:
                mean_val = float(np.nanmean(data))
                std_val = float(np.nanstd(data))
                # guard against zero std (constant feature)
                if std_val == 0 or np.isnan(std_val):
                    std_val = 1.0
                self.normalizer_mean[name] = mean_val
                self.normalizer_std[name] = std_val
            return (data - self.normalizer_mean[name]) / self.normalizer_std[name]
        else:
            raise ValueError(f"Normalization type '{norm_type}' is not supported.")

    def _destandardize(self, data: NDArray, name: str) -> NDArray:
        if name in self.normalizer_mean and name in self.normalizer_std:
            return data * self.normalizer_std[name] + self.normalizer_mean[name]
        else:
            raise ValueError(f"Data for '{name}' has not been standardized yet.")

    def _normalize_error(self, error: NDArray, name: str) -> NDArray:
        """Normalize error (standard deviation) by dividing by std only.

        Errors are magnitudes and should NOT have the mean subtracted.
        Only scale by the standard deviation used for the corresponding data.

        Parameters
        ----------
        error : NDArray
            The error values to normalize.
        name : str
            The name of the corresponding feature or target variable.

        Returns
        -------
        NDArray
            The normalized error values.
        """
        if name not in self.normalizer_std:
            raise ValueError(
                f"Data for '{name}' has not been normalized yet. "
                "Cannot normalize errors without normalization parameters."
            )
        return error / self.normalizer_std[name]

    def _denormalize_error(self, error: NDArray, name: str) -> NDArray:
        """Denormalize error (standard deviation) by multiplying by std only.

        Errors are magnitudes and should NOT have the mean added back.
        Only scale by the standard deviation used for the corresponding data.

        Parameters
        ----------
        error : NDArray
            The normalized error values.
        name : str
            The name of the corresponding feature or target variable.

        Returns
        -------
        NDArray
            The denormalized error values.
        """
        if name not in self.normalizer_std:
            raise ValueError(
                f"Data for '{name}' has not been normalized yet. "
                "Cannot denormalize errors without normalization parameters."
            )
        return error * self.normalizer_std[name]

    def to_dataframe(self) -> pl.DataFrame:
        """Convert the dataset to a Polars DataFrame."""
        data_dict = {}
        if self.x_data is not None:
            for i, label in enumerate(self.x_labels):
                data_dict[label] = self.x_data[:, i]
        if self.y_data is not None and self.y_labels is not None:
            for i, label in enumerate(self.y_labels):
                data_dict[label] = self.y_data[:, i]
        return pl.DataFrame(data_dict)

    @classmethod
    def from_dataframe(
        cls, df: pl.DataFrame, x_labels: List[str], y_labels: Optional[List[str]] = None
    ):
        """Load dataset from a CSV file into the Dataset instance."""
        instance = cls(
            dataset_type="recipes",
            x_labels=x_labels,
            x_data=df.select(pl.col(x_labels)).to_numpy(),
        )
        if y_labels is not None:
            instance.dataset_type = "experiments"
            instance.y_labels = y_labels
            instance.y_data = df.select(pl.col(y_labels)).to_numpy()
        return instance

    def normalize_output_value(self, name: str, value: float) -> float:
        value = self._standardize(np.array([value]), name)[0]
        return value

    def to_json(self, path: str) -> None:
        """Save the dataset to a JSON file."""
        data_dict = {
            "dataset_type": self.dataset_type,
            "x_labels": self.x_labels,
            "x_data": self.x_data.tolist() if self.x_data is not None else None,
            "y_labels": self.y_labels,
            "y_data": self.y_data.tolist() if self.y_data is not None else None,
            "y_errors": self.y_errors.tolist() if self.y_errors is not None else None,
            "normalized": self.normalized,
            "normalizer_mean": self.normalizer_mean,
            "normalizer_std": self.normalizer_std,
            "normalizer_min": self.normalizer_min,
            "normalizer_max": self.normalizer_max,
        }
        with open(path, "w") as f:
            json.dump(data_dict, f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> "Dataset":
        """Load the dataset from a JSON file."""
        with open(path, "r") as f:
            data_dict = json.load(f)
        return cls(
            dataset_type=data_dict["dataset_type"],
            x_labels=data_dict["x_labels"],
            x_data=np.array(data_dict["x_data"], dtype=np.float64)
            if data_dict["x_data"] is not None
            else None,
            y_labels=data_dict.get("y_labels"),
            y_data=np.array(data_dict["y_data"], dtype=np.float64)
            if data_dict.get("y_data") is not None
            else None,
            y_errors=np.array(data_dict["y_errors"], dtype=np.float64)
            if data_dict.get("y_errors") is not None
            else None,
            normalized=data_dict.get("normalized", False),
            normalizer_mean=data_dict.get("normalizer_mean", {}),
            normalizer_std=data_dict.get("normalizer_std", {}),
            normalizer_min=data_dict.get("normalizer_min", {}),
            normalizer_max=data_dict.get("normalizer_max", {}),
        )