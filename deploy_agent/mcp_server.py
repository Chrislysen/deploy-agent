"""MCP (Model Context Protocol) server — exposes deploy-agent as a tool for LLMs."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from .models import OptimizationRequest
from .report import to_json

mcp_app = FastMCP(
    "deploy-agent",
    instructions="Automated ML deployment optimiser. Finds the fastest feasible "
    "backend, quantization, and batch size for a given model under "
    "latency and memory constraints.",
)


@mcp_app.tool()
def optimize_deployment(
    model_name: str,
    max_latency_ms: float | None = None,
    max_memory_mb: float | None = None,
    gpu: str = "auto",
    budget: int = 25,
    objective: str = "maximize_throughput",
) -> str:
    """Optimise deployment of a model by searching over backends, quantization, and batch sizes.

    Tries pytorch_eager, torch_compile, and onnxruntime with fp32/fp16/int8
    quantization.  Returns the best configuration that meets the latency and
    memory constraints.

    Args:
        model_name: torchvision model name (e.g. resnet50, mobilenet_v2)
        max_latency_ms: Maximum p95 latency in milliseconds (constraint)
        max_memory_mb: Maximum peak GPU memory in megabytes (constraint)
        gpu: Device selection — "auto", "cuda", or "cpu"
        budget: Number of trial configurations to evaluate
        objective: "maximize_throughput" or "minimize_latency"

    Returns:
        JSON string with the optimisation report including best config and all trials.
    """
    from .optimizer import DeploymentOptimizer

    request = OptimizationRequest(
        model_name=model_name,
        max_latency_ms=max_latency_ms,
        max_memory_mb=max_memory_mb,
        gpu=gpu,
        budget=budget,
        objective=objective,
    )

    opt = DeploymentOptimizer(request)
    status = opt.run()
    return json.dumps(to_json(status), indent=2)


@mcp_app.tool()
def list_supported_models() -> str:
    """List the model names that deploy-agent can optimise."""
    models = [
        "resnet18",
        "resnet50",
        "mobilenet_v2",
        "efficientnet_b0",
        "vit_b_16",
    ]
    return json.dumps({"supported_models": models, "note": "Any torchvision model name is accepted."})
