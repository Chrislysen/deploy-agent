"""Core optimisation engine — wraps the TBA hybrid optimizer.

Constructs a deployment-specific search space, runs the ask / tell loop, logs
each trial to a JSONL file, and invokes a callback after every evaluation so
callers (CLI, API, WebSocket) can react in real time.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .benchmarker import benchmark_config
from .models import OptimizationRequest, OptimizationStatus, TrialConfig, TrialResult

# TBA imports — from the user's Constrained-ML-Deployment repo
from tba.optimizer.search_space import SearchSpace
from tba.optimizer.tba_tpe_hybrid import TBATPEHybrid
from tba.types import EvalResult, VariableDef


def _build_search_space() -> SearchSpace:
    """Deployment search space (backend × quantization × batch × threads)."""
    return SearchSpace([
        VariableDef(
            name="backend",
            var_type="categorical",
            choices=["pytorch_eager", "torch_compile", "onnxruntime"],
        ),
        VariableDef(
            name="quantization",
            var_type="categorical",
            choices=["fp32", "fp16", "int8_dynamic"],
        ),
        VariableDef(
            name="batch_size",
            var_type="integer",
            low=1,
            high=64,
            log_scale=True,
        ),
        VariableDef(
            name="num_threads",
            var_type="integer",
            low=1,
            high=8,
            condition="backend in ['pytorch_eager', 'onnxruntime']",
        ),
    ])


def _build_constraints(req: OptimizationRequest) -> dict[str, float]:
    constraints: dict[str, float] = {}
    if req.max_latency_ms is not None:
        constraints["latency_p95_ms"] = req.max_latency_ms
    if req.max_memory_mb is not None:
        constraints["memory_peak_mb"] = req.max_memory_mb
    return constraints


def _objective_value(metrics: dict[str, Any], objective: str) -> float:
    """Extract the scalar objective from benchmark metrics."""
    if objective == "minimize_latency":
        v = metrics.get("latency_p95_ms")
        return -v if v is not None else -1e9
    # Default: maximize throughput
    v = metrics.get("throughput_qps")
    return v if v is not None else -1e9


def _check_feasibility(
    metrics: dict[str, Any],
    constraints: dict[str, float],
) -> bool:
    if metrics.get("crashed"):
        return False
    for key, bound in constraints.items():
        actual = metrics.get(key)
        if actual is None or actual > bound:
            return False
    return True


class DeploymentOptimizer:
    """Run a full deployment optimisation job."""

    def __init__(
        self,
        request: OptimizationRequest,
        log_dir: str | Path = "./logs",
        seed: int = 42,
    ) -> None:
        self.request = request
        self.job_id = uuid.uuid4().hex[:12]
        self.log_dir = Path(log_dir) / self.job_id
        self.seed = seed

        self.constraints = _build_constraints(request)
        self.search_space = _build_search_space()
        self.status = OptimizationStatus(
            job_id=self.job_id,
            state="pending",
            total_budget=request.budget,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        on_trial: Callable[[TrialResult], None] | None = None,
    ) -> OptimizationStatus:
        """Execute the optimisation loop. Blocks until done.

        Parameters
        ----------
        on_trial:
            Optional callback invoked after every trial with the result.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "trials.jsonl"

        optimizer = TBATPEHybrid(
            search_space=self.search_space,
            constraints=self.constraints,
            objective=self.request.objective,
            budget=self.request.budget,
            seed=self.seed,
        )

        self.status.state = "running"

        for trial_idx in range(self.request.budget):
            # --- ask ---
            raw_config: dict[str, Any] = optimizer.ask()
            config = TrialConfig(
                backend=raw_config["backend"],
                quantization=raw_config["quantization"],
                batch_size=int(raw_config["batch_size"]),
                num_threads=int(raw_config.get("num_threads", 4)),
            )

            # --- evaluate ---
            t0 = time.monotonic()
            metrics = benchmark_config(
                model_name=self.request.model_name,
                config=config,
                device=self.request.gpu,
            )
            eval_time = time.monotonic() - t0

            # --- build EvalResult for TBA ---
            constraint_vals = {
                k: metrics.get(k, 1e9) if metrics.get(k) is not None else 1e9
                for k in self.constraints
            }
            feasible = _check_feasibility(metrics, self.constraints)
            obj_val = _objective_value(metrics, self.request.objective)

            eval_result = EvalResult(
                objective_value=obj_val,
                constraints=constraint_vals,
                feasible=feasible,
                crashed=metrics.get("crashed", False),
                eval_time_s=eval_time,
                error_msg=metrics.get("error_msg", ""),
            )

            # --- tell ---
            optimizer.tell(raw_config, eval_result)

            # --- record trial ---
            trial = TrialResult(
                trial_id=trial_idx,
                config=config,
                latency_p95_ms=metrics.get("latency_p95_ms"),
                memory_peak_mb=metrics.get("memory_peak_mb"),
                throughput_qps=metrics.get("throughput_qps"),
                accuracy=metrics.get("accuracy"),
                feasible=feasible,
                crashed=metrics.get("crashed", False),
                error_msg=metrics.get("error_msg", ""),
                eval_time_s=eval_time,
            )
            self.status.trials.append(trial)
            self.status.trials_completed = trial_idx + 1

            # Update best
            if feasible and (
                self.status.best_objective is None or obj_val > self.status.best_objective
            ):
                self.status.best_config = config
                self.status.best_objective = obj_val

            # --- log as JSON ---
            with open(log_path, "a") as f:
                f.write(trial.model_dump_json() + "\n")

            # --- callback ---
            if on_trial is not None:
                on_trial(trial)

        self.status.state = "completed"
        return self.status
