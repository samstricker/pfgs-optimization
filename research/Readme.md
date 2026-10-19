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

### Sampling modes (simulated personas)

For PFGS runs (`algorithm = "pfgs"`), `sampling_mode` in the config sets how the next batch is picked from the Pareto front. The persona modes simulate different kinds of users:

| `sampling_mode` | Persona | Selection |
|---|---|---|
| `pareto_expert` | Expert | Best *true* objective values (uses the simulator as an oracle) |
| `pareto_noob` | Adversarial | Worst *true* objective values |
| `pareto_explore` | Explorer | Highest predicted uncertainty (first `standard_deviation` axis) |
| `pareto_exploit` | Exploiter | Best predicted mean (first `objective` axis) |
| `automatic` | — | `ParetoSelector`: closest to utopia, then spread across the front |
| `hitl` | Human | Recipes are selected manually |

All persona modes first remove the regions given in `discard_values` for that iteration. The selection logic is in `helpers/sampling/selection_pfgs.py`.

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