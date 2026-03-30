from __future__ import annotations

from pydantic import BaseModel


class TrialConfig(BaseModel):
    backend: str
    quantization: str
    batch_size: int
    num_threads: int = 4


class TrialResult(BaseModel):
    trial_id: int
    config: TrialConfig
    latency_p95_ms: float | None = None
    memory_peak_mb: float | None = None
    throughput_qps: float | None = None
    accuracy: float | None = None
    feasible: bool = False
    crashed: bool = False
    error_msg: str = ""
    eval_time_s: float = 0.0


class OptimizationRequest(BaseModel):
    model_name: str
    max_latency_ms: float | None = None
    max_memory_mb: float | None = None
    gpu: str = "auto"
    budget: int = 25
    objective: str = "maximize_throughput"


class OptimizationStatus(BaseModel):
    job_id: str
    state: str = "pending"
    trials_completed: int = 0
    total_budget: int = 25
    best_config: TrialConfig | None = None
    best_objective: float | None = None
    trials: list[TrialResult] = []
    report: str | None = None
