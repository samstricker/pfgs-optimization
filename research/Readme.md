# Research

This folder contains the research code behind the PFGS paper.

- **`00_run_your_own_dataset/`** — a self-contained tutorial for trying PFGS on your own data.
- **`paper_results/`** — the MLflow run(s) used to produce the results shown in the paper.
- **`scripts/`**, **`helpers/`**, **`config/`**, **`setup/`** — supporting code used to generate the paper's experiments.

## Run Experiments again 

To run the experiments again, you can use the `research/scripts/run_optimization.py` script. with the `--config` argument, you can specify a TOML file that contains the configuration for the optimization run (examples are in the **`config/`** folder). For example:

```bash
uv run research/scripts/run_optimization.py --config="research/config/config_bio_pfgs.toml"
```

The results are logged to MLflow, and you can view them with:

```bash
uv run mlflow ui
``` 
Opens the MLflow UI at [http://localhost:5000](http://localhost:5000).

The interactive Pareto dashboard can be started with:

```bash
uv run python -m paretodo.dashboard.start
```
Opens the interactive Pareto dashboard at [http://localhost:8055](http://localhost:8055).