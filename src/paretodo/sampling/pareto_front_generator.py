"""Pareto Front Generator Module which is the core of the project."""

import os
from pathlib import Path
from typing import Tuple, Optional, Any
import random

import polars as pl
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from tqdm import tqdm

from paretodo.sampling import BaseProblem


class TqdmCallback(Callback):
    """Progress callback using tqdm."""

    def __init__(self, n_max_gen: int):
        super().__init__()
        self.pbar = tqdm(total=n_max_gen, desc="Optimization", unit="gen", leave=True)

    def notify(self, algorithm):
        self.pbar.update(1)

    def finalize(self):
        """Finalize the progress bar by closing it."""
        try:
            self.pbar.close()
        except (AttributeError, ValueError):
            pass


class ConvergenceCallback(TqdmCallback):
    """Progress callback that also records convergence metrics per generation.

    Parameters
    ----------
    n_max_gen : int
        Maximum number of generations (for the progress bar).
    ref_point : Optional[np.ndarray], optional
        Reference point for the hypervolume (minimization space). If None, it is
        set once from the first generation as max(f_pop) + 10% of its range.
    """

    def __init__(self, n_max_gen: int, ref_point: Optional[np.ndarray] = None):
        super().__init__(n_max_gen)
        self.ref_point = (
            None if ref_point is None else np.asarray(ref_point, dtype=float)
        )
        self._hv = None if self.ref_point is None else HV(ref_point=self.ref_point)
        self.n_gen: list[int] = []
        self.n_evals: list[int] = []
        self.hypervolume: list[float] = []
        self.best_f: list[np.ndarray] = []

    def notify(self, algorithm):
        super().notify(algorithm)
        f_pop = algorithm.pop.get("F")
        if self._hv is None:
            f_min, f_max = f_pop.min(axis=0), f_pop.max(axis=0)
            self.ref_point = f_max + 0.1 * np.maximum(f_max - f_min, 1e-12)
            self._hv = HV(ref_point=self.ref_point)

        self.n_gen.append(algorithm.n_gen)
        self.n_evals.append(algorithm.evaluator.n_eval)
        hv = self._hv(f_pop)
        self.hypervolume.append(float(hv) if hv is not None else np.nan)
        self.best_f.append(f_pop.min(axis=0))

    def to_dataframe(self) -> pl.DataFrame:
        """Return the convergence history as a DataFrame (one row per generation)."""
        best_f = np.vstack(self.best_f) if self.best_f else np.empty((0, 0))
        return pl.DataFrame(
            {
                "n_gen": self.n_gen,
                "n_evals": self.n_evals,
                "hypervolume": self.hypervolume,
                **{f"best_f{i + 1}": best_f[:, i] for i in range(best_f.shape[1])},
            }
        )


class ParetoFrontGenerator:
    """Class to generate Pareto front using multi-objective optimization.

    Parameters
    ----------
    problem : BaseProblem
        The optimization problem to solve.
    seed : Optional[int], optional
        Random seed for reproducibility, by default 11.
    """

    def __init__(
        self,
        problem: BaseProblem,
        seed: Optional[int] = 11,
        pop_size_per_dim: int = 10,
        n_max_gen: int = 50,
        ref_point: Optional[np.ndarray] = None,
    ):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.problem = problem
        self.pop_size_per_dim = pop_size_per_dim
        self.n_max_gen = n_max_gen
        self.ref_point = ref_point
        self.convergence: Optional[ConvergenceCallback] = None
        self._log_problem()

    def generate_pareto_front(self) -> Tuple[Any, pl.DataFrame]:
        """Generate the Pareto front using multi-objective optimization.

        Returns
        -------
        Tuple[Any, pl.DataFrame]
            The return of the model and DataFrame containing the Pareto front solutions.

        """
        algorithm, termination = self._get_algorithm_inputs()
        self.convergence = ConvergenceCallback(self.n_max_gen, self.ref_point)
        res = minimize(
            self.problem,
            algorithm,
            termination,
            seed=self.seed,
            verbose=False,
            callback=self.convergence,
        )
        pareto_front = self._unpack_results(res)
        return pareto_front

    def _get_algorithm_inputs(
        self,
    ) -> Tuple[NSGA2, DefaultMultiObjectiveTermination]:
        """Get the inputs for the optimization algorithm."""
        algorithm = NSGA2(
            pop_size=(self.problem.n_var * self.pop_size_per_dim),
            seed=self.seed,
        )
        termination = DefaultMultiObjectiveTermination(
            xtol=1e-8, ftol=1e-8, period=50, n_max_gen=self.n_max_gen
        )
        return algorithm, termination

    def _unpack_results(
        self,
        res,
    ) -> Tuple[Any, pl.DataFrame]:
        """Unpack the results from the optimization."""
        predictions, pred_df = self.problem.predict(res.X)
        objectives = self.problem.create_objective_list(
            predictions, for_minimization=False
        )
        pareto_df = pl.DataFrame({"RecipeID": np.arange(len(res.X))})
        pareto_df = pl.concat([pareto_df, pred_df], how="horizontal")
        objective_labels = self.problem.optimization_problem.objective_labels
        pareto_df = pareto_df.with_columns(
            [
                pl.Series(name=objective_labels[i], values=objectives[i])
                for i in range(len(objective_labels))
            ]
        )
        return predictions, pareto_df

    def _log_problem(self):
        """Log the problem configuration to mlflow."""
        temp_folder = Path(".tmp/")
        os.makedirs(temp_folder, exist_ok=True)
        self.problem.investigation_space.to_json(
            temp_folder / "investigation_space.json"
        )
        self.problem.optimization_problem.to_json(
            temp_folder / "optimization_problem.json"
        )
        os.remove(temp_folder / "investigation_space.json")
        os.remove(temp_folder / "optimization_problem.json")
        os.rmdir(temp_folder)
