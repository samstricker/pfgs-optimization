"""This module provides functionality to generate Latin Hypercube Samples (LHS) within a defined parameter space."""

from pyDOE3 import lhs

from paretodo import InvestigationSpace

from helpers.data.dataset import Dataset


def lhs_sampling(
    investigation_space: InvestigationSpace, n_samples: int, seed: int = 11
) -> Dataset:
    """Generate Latin Hypercube Samples within the given space parameters.

    Parameters
    ----------
    investigation_space : InvestigationSpace
        The investigation space defining the variables and their ranges.
    n_samples : int
        The number of samples to generate.

    Returns
    -------
    Dataset
        A Dataset containing the generated samples.
    """
    dim = len(investigation_space.variables)
    sample = lhs(dim, samples=n_samples, criterion="centermaximin", random_state=seed)
    dataset = Dataset(
        dataset_type="recipes",
        x_labels=investigation_space.labels,
        x_data=sample,
        normalized=True,
    )
    for var in investigation_space.variables:
        dataset.add_normalization_params(var.name, var.min, var.max)
    dataset.denormalize()
    return dataset
