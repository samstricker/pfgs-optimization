# Pareto Front Guided Sampling (PFGS)

Pareto Front Guided Sampling (PFGS) is a human-in-the-loop Bayesian optimization framework that presents candidate experiments as a Pareto front of predicted performance versus model uncertainty, making the exploration–exploitation trade-off explicit and interpretable for domain experts. Rather than returning a single automated recommendation, PFGS allows practitioners to inspect trade-offs, apply domain knowledge as selection requirements, and choose the next experimental batch interactively. The framework is extended here to constrained optimization, where probabilistic CQA satisfaction is incorporated as an additional Pareto objective computed from the Gaussian process posterior, and to robust optimization, where Monte Carlo sampling over expected CPP deviations quantifies recipe resilience to implementation variability. Candidate recipes are visualized through pairwise Pareto projections on an interactive dashboard, enabling selection criteria to evolve iteratively as process understanding improves. 

---

## Installation

```bash
uv sync
```

Requires Python >= 3.11.

To run the research code (paper experiments, tutorial), also install the `research` group. This pulls the [Digital BioLab simulator](https://github.com/samstricker/digital_biolab) (`dbiolab`) from GitHub:

```bash
uv sync --group research
```

---

## File Structure

```
pfgs-optimization/
├── src/paretodo/
│   ├── data_models/
│   │   ├── investigation_space_var.py   # InvestigationSpaceVariable
│   │   ├── investigation_space.py       # InvestigationSpace
│   │   ├── optimization_axis.py         # OptimizationAxis
│   │   └── optimization_problem.py      # OptimizationProblem
│   ├── sampling/
│   │   ├── base_problem.py              # Abstract base class — subclass this
│   │   └── pareto_front_generator.py    # NSGA2-based Pareto front generation
│   ├── selection/
│   │   └── selector.py                  # Recipe selection from Pareto front
│   ├── logging/
│   │   └── mlflow_logger.py             # MLflow logging utility
│   └── dashboard/
│       ├── start.py                     # Dashboard entry point
│       ├── server.py                    # Dash/Flask server setup
│       ├── html_layout.py               # UI layout
│       ├── theme/                       # Plotly theme
│       ├── callbacks/                   # Dash callback handlers
│       └── utils/                       # Plotting helpers
├── research/                            # Paper code + tutorial, see research/Readme.md
├── tests/
│   ├── data_models/
│   ├── sampling/                        # Includes SimpleProblem example
│   └── selection/
├── pyproject.toml
└── uv.lock
```

---

## How to Use

### 1. Define the Investigation Space

```python
from paretodo import InvestigationSpace, InvestigationSpaceVariable

space = InvestigationSpace(variables=[
    InvestigationSpaceVariable(name="temperature", min=30.0, max=40.0, discretization=0.1, expected_error=0.2),
    InvestigationSpaceVariable(name="ph", min=6.0, max=8.0, discretization=0.01, expected_error=0.05),
])
```

`expected_error` is required for `robustness` optimization axes.

### 2. Define the Optimization Problem

```python
from paretodo import OptimizationProblem, OptimizationAxis

problem = OptimizationProblem(problem=[
    OptimizationAxis(var_name="yield",    minimize=False, optimization_type="objective",
                     lower_bound=None, upper_bound=None),
    OptimizationAxis(var_name="yield",    minimize=False, optimization_type="standard_deviation",
                     lower_bound=None, upper_bound=None),
    OptimizationAxis(var_name="impurity", minimize=True,  optimization_type="constraint",
                     lower_bound=None, upper_bound=5.0),
])
```

All five fields are required; bounds must be floats or `None`.

Supported `optimization_type` values:

| Type | Description |
|------|-------------|
| `"objective"` | Direct objective value |
| `"constraint"` | Probability of meeting the bound(s) |
| `"standard_deviation"` | Prediction uncertainty |
| `"robustness"` | Mean over perturbed inputs (requires `expected_error`) |

### 3. Implement BaseProblem

Subclass `BaseProblem` and implement the three abstract methods:

```python
import numpy as np
import polars as pl
from numpy.typing import NDArray
from paretodo import BaseProblem, OptimizationAxis

class MyProblem(BaseProblem):

    def predict(self, x: NDArray[np.float64], normalized: bool = True):
        """
        Called during optimization with a batch of candidates.
        x: shape (n_candidates, n_vars). During optimization normalized=True and
           values are in [0, 1]; for robustness sampling it is called with
           normalized=False and real (unscaled) values.
        Returns: (predictions, pl.DataFrame)
          - predictions: any structure your other methods consume
          - DataFrame: one row per candidate, columns for each output
        """
        # call your surrogate model / GP / lookup here
        model_return_class = my_model.predict(x)
        df = pl.DataFrame({"yield_mean": model_return_class.mean, "yield_std": model_return_class.sigma})
        return model_return_class, df

    def create_objective_list(self, model_return_class, for_minimization: bool = True):
        """
        Convert predictions into a list of objective arrays for NSGA2.
        One list entry per OptimizationAxis in your OptimizationProblem.
        Use self._get_objective_value() for standard types.
        x_vals (real, unscaled inputs) is only used for "robustness" axes.
        """
        objectives = []
        for axis in self.optimization_problem.problem:
            obj = self._get_objective_value(axis, x_vals=model_return_class.x, mean=model_return_class.mean,
                                            sigma=model_return_class.sigma, for_minimization=for_minimization)
            objectives.append(obj)
        return objectives

    def prediction_to_robust_objective_values(self, predictions, opt_axis: OptimizationAxis):
        """
        Required only when using optimization_type="robustness".
        Called with perturbed samples; return a 1-D numpy array of scalar values.
        """
        mean, _ = predictions
        return mean
```

### 4. Generate the Pareto Front

```python
from paretodo import ParetoFrontGenerator

generator = ParetoFrontGenerator(
    problem=MyProblem(space, problem),
    seed=11,
    pop_size_per_dim=20,   # population = pop_size_per_dim * n_variables
    n_max_gen=100,
)
predictions, pareto_df = generator.generate_pareto_front()
```

`pareto_df` is a Polars DataFrame with one row per Pareto-optimal candidate.

### 5. Select Recipes

```python
from paretodo import ParetoSelector

selector = ParetoSelector()
selected = selector.select_recipes(
    pareto_front=pareto_df,
    investigation_space=space,
    optimization_problem=problem,
    n_samples=5,
    experiments_df=None,   # optional: past experiments to keep away from
)
```

The first selected recipe is the one closest to the utopia point; the rest are chosen to spread across the front.

### 6. Log Everything to MLflow

```python
from paretodo import MlFlowLogger

logger = MlFlowLogger(experiment_name="my_experiment")

logger.log_investigation_space(space)
logger.log_optimization_problem(problem)
logger.log_dataframe(pareto_df,  datatype="pareto_front",      iteration=1)
logger.log_dataframe(selected,   datatype="selected_recipes",  iteration=1)
logger.log_dataframe(meshgrid_df, datatype="meshgrid",         iteration=1)  # needed by dashboard
logger.add_tag("iteration", "1")
logger.log_metric("n_pareto_points", len(pareto_df))
logger.end_run()
```

Supported `datatype` values for `log_dataframe`: `"experiments"`, `"meshgrid"`, `"pareto_front"`, `"selected_recipes"`, `"manufacturing_front"`, `"pareto_convergence"`, `"manufacturing_convergence"`.

`MlFlowLogger` writes to `sqlite:///mlflow.db` in the current working directory.

---

## Starting the MLflow UI

```bash
uv run mlflow ui
```

Opens at [http://localhost:5000](http://localhost:5000). Run this from the project root (where `mlflow.db` is created).

---

## Starting the Dashboard

```bash
uv run python -m paretodo.dashboard.start
```

Opens at [http://localhost:8055](http://localhost:8055). The dashboard reads results through the MLflow tracking server, so start the MLflow UI first (see above). The MLflow port is set in the dashboard and defaults to 5000.

For the contour/GP plots to work, log a `"meshgrid"` DataFrame per iteration containing the full prediction grid over the investigation space.

---

## Research

The [`research/`](research/) folder contains the code behind the PFGS paper, plus a tutorial for trying PFGS on your own dataset — see [research/Readme.md](research/Readme.md).

## Publications

- S. Stricker et al., **“A Human-in-the-Loop Bayesian Optimization Framework for Constraint-Aware Bioprocess Development”**, Jun. 17, 2026, doi: 10.48550/arXiv.2606.19230. | [Paper](https://arxiv.org/abs/2606.19230)

- S. Stricker et al., **“Pareto Front Guided Sampling for Efficient Bioprocess Experimentation”** in Systems and Control Transactions, Sheffield, United Kingdom, Jun. 2026, pp. 692–701. doi: 10.69997/sct.137011. | [Paper](https://doi.org/10.69997/sct.137011)

- S. Stricker, **“Optimal Design of Experiments for Bio-Processes using Hybrid Models”**, MSc., ETH Zurich, Zurich, 2024.

