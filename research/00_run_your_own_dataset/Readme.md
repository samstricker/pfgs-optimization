# Run PFGS on Your Own Dataset

A minimal tutorial for trying PFGS on your own experimental data.

## 1. Prepare your data

Put your experiments in `.data/` as a CSV with your input columns (recipe/process variables) and output columns (measured results) — see `.data/example_data.csv` for the format.

## 2. Convert your data to PFGS inputs

Open `setup.ipynb` and adapt it to your dataset:

- Load your CSV and set `X_COLUMNS` / `Y_COLUMNS`.
- Define your `InvestigationSpace` (the min/max/discretization/expected error for each input variable).
- Define your `OptimizationProblem` (which outputs to optimize, and how — see the main [README](../../README.md) for the available `optimization_type` values).

Running the notebook writes three JSON files to `.data/`: the dataset, investigation space, and optimization problem.

## 3. Configure the run

Edit `config_pfgs.toml` — update the `[files]` section to point at the JSON files from step 2, and adjust `nsga2_config` (population size, generations, seed) if needed.

## 4. Run PFGS

```bash
uv run research/scripts/run_one_iteration_with_data.py --config="research/00_run_your_own_dataset/config_pfgs.toml"
```

This logs the results (Pareto front, selected recipes, etc.) to MLflow.

## 5. View the results

```bash
uv run mlflow ui
```
Opens the MLflow UI at [http://localhost:5000](http://localhost:5000).

```bash
uv run python -m paretodo.dashboard.start
```
Opens the interactive Pareto dashboard at [http://localhost:8055](http://localhost:8055).
