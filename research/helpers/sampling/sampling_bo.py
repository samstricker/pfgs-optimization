"""This module contains the main function for sampling new recipes using BO."""

import polars as pl
import numpy as np

from helpers.botorch_gp import BoTorchGP
from helpers.sampling.bayesian_optimisor import BayesianOptimizer

from paretodo import OptimizationProblem


def get_relevant_models_and_utopia(
    model: BoTorchGP, optimization_problem: OptimizationProblem
):
    if model.train_dataset is None:
        raise ValueError("Model must be trained before sampling.")
    if model.train_dataset.y_labels is None:
        raise ValueError("Model's train dataset must have y_labels defined.")
    objective_labels = []
    objective_type = []
    utopia_point = []
    for obj in optimization_problem.problem:
        objective_labels.append(obj.var_name)
        objective_type.append(obj.optimization_type)
        if obj.optimization_type not in ["robustness", "standard_deviation"]:
            if obj.minimize:
                utopia_point.append(0.0)
            else:
                utopia_point.append(1.0)
    models_idx = []
    for label in objective_labels:
        if label not in model.train_dataset.y_labels:
            raise ValueError(
                f"Objective label '{label}' not found in model's train dataset: {model.train_dataset.y_labels}."
            )
        idx = model.train_dataset.y_labels.index(label)
        models_idx.append(idx)
    model_idx = list(dict.fromkeys(models_idx))
    models = [model.models[idx] for idx in model_idx]
    return models, utopia_point, len(model.train_dataset.x_labels), objective_type


def sample_bo(
    bo_config: dict,
    model: BoTorchGP,
    n_samples: int,
    optimization_problem: OptimizationProblem,
) -> pl.DataFrame:
    """ """
    models, _, input_dim, objective_type = get_relevant_models_and_utopia(
        model, optimization_problem
    )

    optimizer = BayesianOptimizer(
        bo_config=bo_config,
        n_dim=input_dim,
    )
    candidate = None
    acq_value = None
    if "objective" in objective_type or "standard_deviation" in objective_type:
        if bo_config["acquisition_function"].lower() == "ucb":
            candidate, acq_value = optimizer.sample_qUCB(
                models[0],
                beta=bo_config["beta"],
                batch_size=n_samples,
            )
        elif bo_config["acquisition_function"].lower() == "ei":
            if bo_config["best_f"].lower() != "auto":
                best_f = bo_config["best_f"]
            else:
                best_f = models[0].train_targets.max().item()
            candidate, acq_value = optimizer.sample_qLogEI(
                models[0],
                best_f=best_f,
                batch_size=n_samples,
            )
        else:
            raise ValueError(
                f"Unsupported acquisition function for single-objective optimization: {bo_config['acquisition_function']}"
            )
    if "constraint" in objective_type:
        constraint_idx = objective_type.index("constraint")
        low_bound = optimization_problem.problem[constraint_idx].lower_bound
        high_bound = optimization_problem.problem[constraint_idx].upper_bound
        if low_bound is not None and high_bound is None:
            constraint_threshold = low_bound
            constraint_greater_than = True
        elif low_bound is None and high_bound is not None:
            constraint_threshold = high_bound
            constraint_greater_than = False
        else:
            raise ValueError(
                "Constraint variable must have either a lower bound or an upper bound defined, but not both."
            )
        constraint_threshold = model.train_dataset.normalize_output_value(
            name=optimization_problem.problem[constraint_idx].var_name,
            value=constraint_threshold,
        )
        if bo_config["acquisition_function"].lower() == "ucb":
            candidate, acq_value = optimizer.sample_qUCB(
                models[0],
                beta=bo_config["beta"],
                batch_size=n_samples,
                constraint_model=models[1],
                constraint_threshold=constraint_threshold,
                constraint_greater_than=constraint_greater_than,
            )
        elif bo_config["acquisition_function"].lower() == "ei":
            if bo_config["best_f"].lower() != "auto":
                best_f = bo_config["best_f"]
            else:
                best_f = models[0].train_targets.max().item()
            candidate, acq_value = optimizer.sample_qLogEI(
                models[0],
                best_f=best_f,
                batch_size=n_samples,
                constraint_model=models[1],
                constraint_threshold=constraint_threshold,
                constraint_greater_than=constraint_greater_than,
            )
    if candidate is None or acq_value is None:
        raise ValueError("Bayesian optimization failed to produce candidates.")
    candidate = candidate.cpu().numpy()
    acq_value = np.atleast_1d(acq_value.cpu().numpy()).reshape(-1)
    selection_dataset = model.train_dataset.copy()
    selection_dataset.x_data = candidate
    selection_dataset.y_data = None
    selection_dataset.dataset_type = "recipe"
    selection_dataset.normalized = True
    selection_dataset.normalizer_mean = {}
    selection_dataset.denormalize()
    df = pl.DataFrame(selection_dataset.x_data, schema=selection_dataset.x_labels)
    if acq_value.size == 1 and df.height > 1:
        acq_value = np.repeat(acq_value, df.height)
    if acq_value.size != df.height:
        raise ValueError(
            "Acquisition value length does not match candidate count: "
            f"{acq_value.size} != {df.height}."
        )
    df = df.with_columns(pl.Series("acquisition_value", acq_value))
    return df
