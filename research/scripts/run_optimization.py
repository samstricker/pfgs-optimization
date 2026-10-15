"""Main script to run the Himmelblau optimization problem."""

import logging as log
from pathlib import Path


from helpers.data.dataset_manipulation import (
    concatenate_datasets,
    dataframe_to_recipe_dataset,
)
from helpers.config import load_config
from helpers.data.meshgrid import get_meshgrid
from helpers.botorch_gp import BoTorchGP
from helpers.logger import set_logger
from helpers.problems.simulation import get_simulator
from helpers.sampling.sampling_lhs import lhs_sampling
from helpers.sampling.selection_pfgs import (
    select_automatic_recipes,
    select_hitl_recipes,
    select_pareto_exploit,
    select_pareto_explore,
    select_pareto_expert,
)
from helpers.sampling.sampling_pfgs import sample_pfgs
from helpers.sampling.sampling_bo import sample_bo
from helpers.data.metrics import calculate_rrmse
from helpers.data.testset import load_testset


from paretodo import (
    MlFlowLogger,
    InvestigationSpace,
    OptimizationProblem,
)


def main():
    """Main function to run the Himmelblau optimization."""
    log.info("################################################")
    log.info("### Starting Himmelblau Optimization Script  ###")
    log.info("################################################")

    log.info("Loading data and configuration...")
    config, files = load_config()
    investigation_space = InvestigationSpace.from_json(
        Path(files["investigation_space"])
    )
    n_experiments = config["initial_samples"]

    log.info("Logging configuration and files to MLflow...")
    ml_logger = MlFlowLogger(experiment_name=config["name"])
    for var_name, var_value in config.items():
        ml_logger.add_parameter(var_name, var_value)
    for var_name, var_value in files.items():
        ml_logger.add_parameter(var_name, var_value)
    ml_logger.log_investigation_space(investigation_space)
    ml_logger.add_tag("Algorithm", config["algorithm"])
    ml_logger.add_tag("Problem", config["problem"])

    log.info("Generating initial samples using Latin Hypercube Sampling...")
    initial_samples = lhs_sampling(
        investigation_space=investigation_space,
        n_samples=n_experiments,
        seed=config["seed"],
    )
    simulator = get_simulator(config=config, mlflow_logger=ml_logger)
    experiment_dataset = simulator.run_simulation(initial_samples, iteration=0)
    meshgrid_recipes = get_meshgrid(
        investigation_space=investigation_space,
        x_labels=experiment_dataset.x_labels.copy(),
        y_labels=experiment_dataset.y_labels.copy(),
        n_points_per_dim=config["meshgrid_points_per_dim"],
    )

    for iteration in range(1, config["n_iterations"] + 1):
        optimization_problem = OptimizationProblem.from_json(
            Path(files["optimization_problem"])
        )
        manufacturing_problem = OptimizationProblem.from_json(
            Path(files["optimization_problem"])
        )
        log.info("Starting iteration %d", iteration)
        ml_logger.log_dataframe(
            experiment_dataset.to_dataframe(),
            datatype="experiments",
            iteration=iteration,
        )

        best_objectives = experiment_dataset.get_best_objective_values(
            optimization_problem
        )
        for obj_name, obj_value in best_objectives.items():
            ml_logger.log_metric(f"best_{obj_name}", obj_value, step=iteration)

        constraint_violations = experiment_dataset.get_constraint_violations(
            optimization_problem
        )
        for con_name, con_value in constraint_violations.items():
            ml_logger.log_metric(
                f"constraint_violations_{con_name}", con_value, step=iteration
            )

        # TODO: Maybe something for robustness

        model_dataset = experiment_dataset.copy()

        log.info("Training surrogate model ...")
        model = BoTorchGP()
        model.train(
            dataset=model_dataset,
        )
        current_meshgrid_recipes = meshgrid_recipes.copy()
        current_meshgrid_recipes.set_normalization_params(
            model_dataset.get_normalization_params()
        )
        meshgrid_dataset = model.predict(current_meshgrid_recipes)
        ml_logger.log_dataframe(
            meshgrid_dataset.to_dataframe(),
            datatype="meshgrid",
            iteration=iteration,
        )

        log.info("Testing surrogate model using testset ...")
        n_experiments = len(experiment_dataset.x_data)
        for file in files["testsets"]:
            testset_name = Path(file).stem
            testset, test_recipies = load_testset(file, experiment_dataset.copy())
            test_recipies.set_normalization_params(
                model.train_dataset.get_normalization_params()
            )
            testset_predictions = model.predict(test_recipies.copy())
            if testset_predictions.x_data.all() != testset.x_data.all():
                raise ValueError("Testset recipes do not match testset data.")
            testset_predictions.denormalize()
            testset.denormalize()

            rrmse = calculate_rrmse(
                data=testset_predictions.y_data[:, 0],
                test_data=testset.y_data[:, 0],
            )
            ml_logger.log_metric(f"rrmse_{testset_name}", rrmse, step=n_experiments)

        for obj in manufacturing_problem.problem:
            if obj.optimization_type == "standard_deviation":
                obj.set_minimize(True)

        ml_logger.log_optimization_problem(
            optimization_problem, optimization_type="optimization"
        )
        ml_logger.log_optimization_problem(
            manufacturing_problem, optimization_type="manufacturing"
        )
        if config["algorithm"] == "pfgs":
            log.info("Running PFGS optimization ...")
            manufacturing_df, manufacturing_convergence_df = sample_pfgs(
                nsga2_config=config["nsga2_config"],
                investigation_space=investigation_space,
                model_dataset=model_dataset,
                model=model,
                optimization_problem=manufacturing_problem,
                problem_type=config["problem"],
            )
            ml_logger.log_dataframe(
                manufacturing_df,
                datatype="manufacturing_front",
                iteration=iteration,
            )
            ml_logger.log_dataframe(
                manufacturing_convergence_df,
                datatype="manufacturing_convergence",
                iteration=iteration,
            )
            pareto_df, pareto_convergence_df = sample_pfgs(
                nsga2_config=config["nsga2_config"],
                investigation_space=investigation_space,
                model_dataset=model_dataset,
                model=model,
                optimization_problem=optimization_problem,
                problem_type=config["problem"],
            )
            ml_logger.log_dataframe(
                pareto_df,
                datatype="pareto_front",
                iteration=iteration,
            )
            ml_logger.log_dataframe(
                pareto_convergence_df,
                datatype="pareto_convergence",
                iteration=iteration,
            )
            if config["sampling_mode"] == "hitl":
                selected_recipes = select_hitl_recipes(pareto_df, iteration)
                if selected_recipes.is_empty():
                    break
            elif config["sampling_mode"] == "automatic":
                selected_recipes = select_automatic_recipes(
                    pareto_df=pareto_df,
                    iteration=iteration,
                    discard_values=config["discard_values"],
                    investigation_space=investigation_space,
                    optimization_problem=optimization_problem,
                    n_samples=config["batch_size"],
                )
                if selected_recipes.is_empty():
                    log.info("No more recipes to select. Ending optimization loop.")
                    break
            elif config["sampling_mode"] == "pareto_expert":
                log.info(
                    "Selecting recipes from Pareto front using expert knowledge ..."
                )
                selected_recipes = select_pareto_expert(
                    pareto_df=pareto_df,
                    iteration=iteration,
                    discard_values=config["discard_values"],
                    investigation_space=investigation_space,
                    optimization_problem=optimization_problem,
                    n_samples=config["batch_size"],
                    simulator=simulator,
                    expert_type="expert",
                )
                if selected_recipes.is_empty():
                    log.info("No more recipes to select. Ending optimization loop.")
                    break
            elif config["sampling_mode"] == "pareto_noob":
                log.info("Selecting recipes from Pareto front using noob knowledge ...")
                selected_recipes = select_pareto_expert(
                    pareto_df=pareto_df,
                    iteration=iteration,
                    discard_values=config["discard_values"],
                    investigation_space=investigation_space,
                    optimization_problem=optimization_problem,
                    n_samples=config["batch_size"],
                    simulator=simulator,
                    expert_type="noob",
                )
                if selected_recipes.is_empty():
                    log.info("No more recipes to select. Ending optimization loop.")
                    break
            elif config["sampling_mode"] == "pareto_explore":
                log.info(
                    "Selecting recipes from Pareto front using explore knowledge ..."
                )
                selected_recipes = select_pareto_explore(
                    pareto_df=pareto_df,
                    iteration=iteration,
                    discard_values=config["discard_values"],
                    optimization_problem=optimization_problem,
                    n_samples=config["batch_size"],
                )
                if selected_recipes.is_empty():
                    log.info("No more recipes to select. Ending optimization loop.")
                    break
            elif config["sampling_mode"] == "pareto_exploit":
                log.info(
                    "Selecting recipes from Pareto front using exploit knowledge ..."
                )
                selected_recipes = select_pareto_exploit(
                    pareto_df=pareto_df,
                    iteration=iteration,
                    discard_values=config["discard_values"],
                    optimization_problem=optimization_problem,
                    n_samples=config["batch_size"],
                )
                if selected_recipes.is_empty():
                    log.info("No more recipes to select. Ending optimization loop.")
                    break
            else:
                log.error("Unsupported sampling mode: %s", config["sampling_mode"])
                return ValueError(
                    f"Unsupported sampling mode: {config['sampling_mode']}"
                )
        elif config["algorithm"] == "bo":
            log.info("Running Bayesian Optimization sampling ...")
            selected_recipes = sample_bo(
                bo_config=config["bo_config"],
                model=model,
                n_samples=config["batch_size"],
                optimization_problem=optimization_problem,
            )
        elif config["algorithm"] == "lhs":
            log.info("Running Latin Hypercube Sampling ...")
            recipes_dataset = lhs_sampling(
                investigation_space=investigation_space,
                n_samples=config["batch_size"] * iteration + config["initial_samples"],
                seed=config["seed"] + iteration,
            )
            selected_recipes = recipes_dataset.to_dataframe()
            # Delete experiments that have already been run as LHS is non iterative
            experiment_dataset.x_data = None
            experiment_dataset.y_data = None
        else:
            log.error("Unsupported algorithm: %s", config["algorithm"])
            return ValueError(f"Unsupported algorithm: {config['algorithm']}")

        ml_logger.log_dataframe(
            selected_recipes,
            datatype="selected_recipes",
            iteration=iteration,
        )
        new_recipes = dataframe_to_recipe_dataset(
            selected_recipes, x_labels=experiment_dataset.x_labels
        )
        if model.train_dataset is None:
            raise ValueError(
                "Model train dataset is None. Cannot set normalization parameters."
            )
        new_recipes.set_normalization_params(
            model.train_dataset.get_normalization_params()
        )
        new_experiments = simulator.run_simulation(new_recipes, iteration=iteration)
        experiment_dataset = concatenate_datasets(experiment_dataset, new_experiments)
        experiment_dataset.denormalize()


if __name__ == "__main__":
    set_logger()
    main()
