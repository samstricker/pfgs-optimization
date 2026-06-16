# Pareto Front Guided Sampling (PFGS)

Pareto Front Guided Sampling (PFGS) is a human-in-the-loop Bayesian optimization framework that presents candidate experiments as a Pareto front of predicted performance versus model uncertainty, making the exploration–exploitation trade-off explicit and interpretable for domain experts. Rather than returning a single automated recommendation, PFGS allows practitioners to inspect trade-offs, apply domain knowledge as selection requirements, and choose the next experimental batch interactively. The framework is extended here to constrained optimization, where probabilistic CQA satisfaction is incorporated as an additional Pareto objective computed from the Gaussian process posterior, and to robust optimization, where Monte Carlo sampling over expected CPP deviations quantifies recipe resilience to implementation variability. Candidate recipes are visualized through pairwise Pareto projections on an interactive dashboard, enabling selection criteria to evolve iteratively as process understanding improves. 

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
│       ├── callbacks/                   # Dash callback handlers
│       └── utils/                       # Plotting helpers
├── tests/
│   ├── data_models/
│   ├── sampling/                        # Includes SimpleProblem example
│   └── selection/
└── pyproject.toml
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
    OptimizationAxis(var_name="yield",      minimize=False, optimization_type="objective"),
    OptimizationAxis(var_name="yield",      minimize=False, optimization_type="standard_deviation"),
    OptimizationAxis(var_name="impurity",   minimize=True,  optimization_type="constraint",
                     lower_bound=None, upper_bound=5.0),
    
])
```

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
        x: shape (n_candidates, n_vars), values normalized to [0, 1]
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
        """
        
        objectives = []
        for axis in self.optimization_problem.problem:
            obj = self._get_objective_value(axis, x_vals=None, mean=model_return_class.mean, sigma=model_return_class.sigma,
                                            for_minimization=for_minimization)
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

selector = ParetoSelector(pareto_df=pareto_df, optimization_problem=problem)
selected = selector.select_recipes(n=5)
closest  = selector.select_closest_recipe_to_utopia()
```

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

Supported `datatype` values for `log_dataframe`: `"experiments"`, `"meshgrid"`, `"pareto_front"`, `"selected_recipes"`, `"manufacturing_front"`.

---

## Starting the MLflow UI

```bash
mlflow ui
```
or 
```bash
uv run mlflow ui 
```

Opens at [http://localhost:5000](http://localhost:5000). Run this from the project root (where `mlflow.db` is created).

---

## Starting the Dashboard

```bash
python src/paretodo/dashboard/start.py
```
or
```bash
uv run python -m paretodo.dashboard.start
```

Opens at [http://localhost:8055](http://localhost:8055). The dashboard loads results directly from the MLflow SQLite database (`mlflow.db`) in the project root.

For the contour/GP plots to work, log a `"meshgrid"` DataFrame per iteration containing the full prediction grid over the investigation space.

---


## Installation

```bash
pip install -e .
```

Requires Python >= 3.11.