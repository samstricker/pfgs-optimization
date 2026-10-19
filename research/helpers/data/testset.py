"""This module provides functions to load the test set for evaluation."""

from typing import Tuple

import polars as pl
from helpers.data.dataset import Dataset


def load_testset(
    testset_path: str, reference_dataset: Dataset
) -> Tuple[Dataset, Dataset]:
    testset_df = pl.read_csv(testset_path)
    test_recipes = Dataset.from_dataframe(
        testset_df,
        x_labels=reference_dataset.x_labels,
    )
    test_recipes.y_labels = reference_dataset.y_labels
    test_dataset = Dataset.from_dataframe(
        testset_df,
        x_labels=reference_dataset.x_labels,
        y_labels=reference_dataset.y_labels,
    )
    return test_dataset, test_recipes
