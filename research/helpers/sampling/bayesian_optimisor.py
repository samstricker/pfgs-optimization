from typing import Dict, List
import logging

from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
import torch
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def _make_nonlinear_ineq_constraint_from_model(
    model: SingleTaskGP, out_idx: int, thr: float, greater: bool
):
    def _constraint(X: torch.Tensor) -> torch.Tensor:
        # Accept inputs of shape (d,), (q, d), or (b, q, d).
        if X.ndim == 1:
            # single point
            x = X.unsqueeze(0)
            post = model.posterior(x)
            mean = post.mean
            val = mean.squeeze(0)
            if val.ndim == 0:
                v = val
            else:
                v = val[..., out_idx]
            v = v.to(X.dtype).to(X.device)
            return (v - thr) if greater else (thr - v)

        elif X.ndim == 2:
            # q x d
            flat = X
            post = model.posterior(flat)
            mean = post.mean
            if mean.ndim == 1:
                vals = mean
            else:
                vals = mean[..., out_idx]
            vals = vals.to(X.dtype).to(X.device)
            return (vals - thr) if greater else (thr - vals)

        elif X.ndim == 3:
            # b x q x d
            b, q, d = X.shape
            flat = X.reshape(-1, d)
            post = model.posterior(flat)
            mean = post.mean
            if mean.ndim == 1:
                vals = mean
            else:
                vals = mean[..., out_idx]
            vals = vals.to(X.dtype).to(X.device)
            out = (vals - thr) if greater else (thr - vals)
            return out.view(b, q)

        else:
            raise ValueError(f"Unsupported input shape for constraint: {X.shape}")

    return _constraint


class BayesianOptimizer:
    def __init__(
        self,
        bo_config: Dict,
        n_dim: int,
    ):
        self.bo_config = bo_config
        self.n_dim = n_dim

    def sample_qUCB(
        self,
        model: SingleTaskGP,
        beta: float,
        batch_size: int,
        *,
        constraint_model: SingleTaskGP | None = None,
        constraint_output_index: int = 0,
        constraint_threshold: float = 0.0,
        constraint_greater_than: bool = True,
        constraint_margin: float = 0.0,
    ):
        acq_func = qUpperConfidenceBound(model=model, beta=beta)
        return self._optimize_acquisition_function(
            acq_func,
            batch_size,
            constraint_model=constraint_model,
            constraint_output_index=constraint_output_index,
            constraint_threshold=constraint_threshold,
            constraint_greater_than=constraint_greater_than,
            constraint_margin=constraint_margin,
        )

    def sample_qLogEI(
        self,
        model: SingleTaskGP,
        best_f: float,
        batch_size: int,
        *,
        constraint_model: SingleTaskGP | None = None,
        constraint_output_index: int = 0,
        constraint_threshold: float = 0.0,
        constraint_greater_than: bool = True,
        constraint_margin: float = 0.0,
    ):
        acq_func = qLogExpectedImprovement(model=model, best_f=best_f)
        return self._optimize_acquisition_function(
            acq_func,
            batch_size,
            constraint_model=constraint_model,
            constraint_output_index=constraint_output_index,
            constraint_threshold=constraint_threshold,
            constraint_greater_than=constraint_greater_than,
            constraint_margin=constraint_margin,
        )

    def sample_qEHVI(
        self,
        models: List[SingleTaskGP],
        batch_size: int,
        reference_point: List[float],
    ):
        model = ModelListGP(*models)
        with torch.no_grad():
            train_x = model.train_inputs[0][0]
            pred = model.posterior(train_x).mean
        partitioning = NondominatedPartitioning(
            ref_point=torch.tensor(reference_point),
            Y=pred,
        )
        acq_func = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=torch.tensor(reference_point),
            partitioning=partitioning,
        )
        candidate, acq_value = self._optimize_acquisition_function(
            acq_func, batch_size, sequential=True
        )
        return candidate, acq_value

    def _optimize_acquisition_function(
        self,
        acq_func,
        batch_size: int,
        sequential: bool = False,
        *,
        constraint_model: SingleTaskGP | None = None,
        constraint_output_index: int = 0,
        constraint_threshold: float = 0.0,
        constraint_greater_than: bool = True,
        constraint_margin: float = 0.0,
    ):
        bounds = torch.stack([torch.zeros(self.n_dim), torch.ones(self.n_dim)]).to(
            torch.double
        )
        nonlinear_ineq_constraints = None
        if constraint_model is not None:
            # Apply margin to constraint threshold if specified (relaxes the constraint)
            adjusted_threshold = constraint_threshold
            if constraint_margin != 0.0:
                if constraint_greater_than:
                    # For "greater than" constraints, reduce threshold by margin
                    adjusted_threshold = constraint_threshold - constraint_margin
                else:
                    # For "less than" constraints, increase threshold by margin
                    adjusted_threshold = constraint_threshold + constraint_margin
                logger.info(
                    f"Applying constraint margin: threshold adjusted from {constraint_threshold:.4f} to {adjusted_threshold:.4f}"
                )

            nonlinear_ineq_constraints = [
                (
                    _make_nonlinear_ineq_constraint_from_model(
                        constraint_model,
                        constraint_output_index,
                        adjusted_threshold,
                        constraint_greater_than,
                    ),
                    True,
                )
            ]

        if nonlinear_ineq_constraints is not None:
            try:
                candidate, acq_value = optimize_acqf(
                    acq_func,
                    bounds=bounds,
                    q=batch_size,
                    num_restarts=self.bo_config["num_restarts"],
                    raw_samples=self.bo_config["raw_samples"],
                    sequential=sequential,
                    nonlinear_inequality_constraints=nonlinear_ineq_constraints,
                    ic_generator=gen_batch_initial_conditions,
                )
            except ValueError as e:
                msg = str(e)
                if "batch_initial_conditions" in msg:
                    logger.warning(
                        f"Failed to generate feasible initial conditions. Starting fallback procedure..."
                    )
                    # Try to generate feasible initial conditions and pass them
                    # directly to optimize_acqf. Retry a few times before
                    # failing with a clearer message.
                    max_attempts = 10  # Increased from 5
                    valid_initials = None
                    constraint_fn = nonlinear_ineq_constraints[0][0]
                    for attempt in range(max_attempts):
                        batch_ics = gen_batch_initial_conditions(
                            acq_func,
                            bounds=bounds,
                            q=batch_size,
                            num_restarts=self.bo_config["num_restarts"],
                            raw_samples=self.bo_config["raw_samples"],
                        )
                        # batch_ics shape: (num_restarts, q, d)
                        valid_idxs = []
                        for i in range(batch_ics.shape[0]):
                            ic = batch_ics[i]
                            try:
                                res = constraint_fn(ic)
                            except Exception:
                                res = None
                            if res is None:
                                continue
                            # res expected shape (q,) or scalar; require all >= 0
                            try:
                                ok = torch.all(res >= 0)
                            except Exception:
                                ok = False
                            if ok:
                                valid_idxs.append(i)
                        if len(valid_idxs) > 0:
                            valid_initials = batch_ics[valid_idxs]
                            logger.info(
                                f"Found {len(valid_idxs)} valid initial conditions on attempt {attempt + 1}"
                            )
                            break
                        # if not found, increase raw_samples more aggressively and retry
                        new_raw_samples = int(
                            self.bo_config["raw_samples"] * (2.0**attempt)
                        )
                        logger.info(
                            f"Attempt {attempt + 1}: No valid conditions found. Retrying with raw_samples={new_raw_samples}"
                        )
                        self.bo_config["raw_samples"] = new_raw_samples

                    if valid_initials is None:
                        logger.warning(
                            "Could not find valid initial conditions with aggressive sampling. "
                            "Attempting unconstrained optimization with relaxed fallback..."
                        )
                        # As a last resort, attempt an unconstrained optimization
                        # (no nonlinear inequality) with larger sampling and
                        # check whether the returned candidate satisfies the
                        # constraint; if so, accept it. This provides a graceful
                        # fallback instead of immediate failure.
                        fallback_found = False
                        for retry in range(5):  # Increased from 3
                            try:
                                raw_mult = 4 + retry * 2  # More aggressive escalation
                                cand_uc, acq_uc = optimize_acqf(
                                    acq_func,
                                    bounds=bounds,
                                    q=batch_size,
                                    num_restarts=max(
                                        1, self.bo_config["num_restarts"] * 2
                                    ),
                                    raw_samples=int(
                                        self.bo_config["raw_samples"] * raw_mult
                                    ),
                                    sequential=sequential,
                                )
                            except Exception as exc:
                                logger.warning(
                                    f"Unconstrained optimization attempt {retry + 1} failed: {exc}"
                                )
                                cand_uc = None
                                acq_uc = None
                            if cand_uc is None:
                                continue
                            try:
                                res = constraint_fn(cand_uc)
                            except Exception:
                                res = None
                            if res is not None:
                                try:
                                    ok = torch.all(res >= 0)
                                except Exception:
                                    ok = False
                                if ok:
                                    candidate, acq_value = cand_uc, acq_uc
                                    fallback_found = True
                                    logger.info(
                                        f"Unconstrained optimization succeeded on attempt {retry + 1}"
                                    )
                                    break

                        if not fallback_found:
                            logger.error(
                                "All standard fallback strategies failed. "
                                "Attempting with constraint margin relaxation..."
                            )
                            # Last-ditch effort: try with significant constraint margin
                            try:
                                # Recursively call with relaxed constraint
                                return self._optimize_acquisition_function(
                                    acq_func,
                                    batch_size,
                                    sequential=sequential,
                                    constraint_model=constraint_model,
                                    constraint_output_index=constraint_output_index,
                                    constraint_threshold=constraint_threshold,
                                    constraint_greater_than=constraint_greater_than,
                                    constraint_margin=0.01,  # 1% margin relaxation
                                )
                            except Exception as margin_err:
                                raise ValueError(
                                    "Failed to generate feasible batch initial conditions that satisfy the non-linear inequality constraint. "
                                    "Tried: (1) standard generation with escalating raw_samples, "
                                    "(2) unconstrained optimization, "
                                    "(3) constraint margin relaxation. "
                                    "Check constraint threshold/scale, data normalization, or increase sampling parameters further."
                                ) from e
                    else:
                        # We found valid initials. Try to optimize with them.
                        # If that also fails, retry with margin relaxation.
                        logger.info(
                            f"Attempting optimization with {valid_initials.shape[0]} valid initial conditions..."
                        )
                        try:
                            candidate, acq_value = optimize_acqf(
                                acq_func,
                                bounds=bounds,
                                q=batch_size,
                                num_restarts=valid_initials.shape[0],
                                raw_samples=self.bo_config["raw_samples"],
                                sequential=sequential,
                                nonlinear_inequality_constraints=nonlinear_ineq_constraints,
                                batch_initial_conditions=valid_initials,
                            )
                            logger.info("Optimization with valid initials succeeded!")
                        except Exception as opt_err:
                            # If optimization still fails even with valid initials,
                            # try margin relaxation
                            logger.warning(
                                f"Optimization with valid initials failed: {type(opt_err).__name__}. "
                                "Retrying with constraint margin relaxation..."
                            )
                            # Recursively call with relaxed constraint
                            return self._optimize_acquisition_function(
                                acq_func,
                                batch_size,
                                sequential=sequential,
                                constraint_model=constraint_model,
                                constraint_output_index=constraint_output_index,
                                constraint_threshold=constraint_threshold,
                                constraint_greater_than=constraint_greater_than,
                                constraint_margin=0.01,  # 1% margin relaxation
                            )
                else:
                    raise
        else:
            candidate, acq_value = optimize_acqf(
                acq_func,
                bounds=bounds,
                q=batch_size,
                num_restarts=self.bo_config["num_restarts"],
                raw_samples=self.bo_config["raw_samples"],
                sequential=sequential,
                nonlinear_inequality_constraints=nonlinear_ineq_constraints,
            )
        return candidate, acq_value
