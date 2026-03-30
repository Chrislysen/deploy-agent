"""Click CLI — ``deploy-agent optimize|serve|mcp``."""

from __future__ import annotations

import re
import sys

import click
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from .models import OptimizationRequest, TrialResult

console = Console()


# ---------------------------------------------------------------------------
# Unit parsers
# ---------------------------------------------------------------------------

def _parse_latency(raw: str) -> float:
    """``20ms`` / ``0.02s`` → float milliseconds."""
    m = re.fullmatch(r"([\d.]+)\s*(ms|s|us)", raw.strip().lower())
    if not m:
        return float(raw)
    val, unit = float(m.group(1)), m.group(2)
    return val * {"ms": 1, "s": 1_000, "us": 0.001}[unit]


def _parse_memory(raw: str) -> float:
    """``512MB`` / ``1GB`` → float megabytes."""
    m = re.fullmatch(r"([\d.]+)\s*(kb|mb|gb|tb)", raw.strip().lower())
    if not m:
        return float(raw)
    val, unit = float(m.group(1)), m.group(2)
    return val * {"kb": 1 / 1024, "mb": 1, "gb": 1024, "tb": 1024**2}[unit]


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="deploy-agent")
def main() -> None:
    """deploy-agent — automated ML deployment optimiser."""


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------

@main.command()
@click.option("--model", required=True, help="Model name (e.g. resnet50, mobilenet_v2).")
@click.option("--max-latency", default=None, help="Latency constraint, e.g. '20ms'.")
@click.option("--max-memory", default=None, help="Memory constraint, e.g. '512MB'.")
@click.option("--gpu", default="auto", help="Device: auto | cuda | cpu.")
@click.option("--budget", default=25, type=int, help="Number of trials.")
@click.option("--objective", default="maximize_throughput",
              type=click.Choice(["maximize_throughput", "minimize_latency"]),
              help="Optimisation objective.")
@click.option("--log-dir", default="./logs", help="Directory for trial JSONL logs.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def optimize(
    model: str,
    max_latency: str | None,
    max_memory: str | None,
    gpu: str,
    budget: int,
    objective: str,
    log_dir: str,
    seed: int,
) -> None:
    """Run deployment optimisation for MODEL."""
    from .optimizer import DeploymentOptimizer
    from .report import print_report

    request = OptimizationRequest(
        model_name=model,
        max_latency_ms=_parse_latency(max_latency) if max_latency else None,
        max_memory_mb=_parse_memory(max_memory) if max_memory else None,
        gpu=gpu,
        budget=budget,
        objective=objective,
    )

    console.rule(f"[bold]Optimising {model}[/bold]")
    console.print(f"  Constraints: latency <= {max_latency or 'none'}, memory <= {max_memory or 'none'}")
    console.print(f"  Device: {gpu}  |  Budget: {budget}  |  Objective: {objective}")
    console.print()

    def _on_trial(trial: TrialResult) -> None:
        status_str = (
            "[red]CRASH[/red]" if trial.crashed
            else "[green]OK[/green]" if trial.feasible
            else "[yellow]INFEAS[/yellow]"
        )
        lat = f"{trial.latency_p95_ms:.1f}ms" if trial.latency_p95_ms is not None else "-"
        mem = f"{trial.memory_peak_mb:.0f}MB" if trial.memory_peak_mb is not None else "-"
        qps = f"{trial.throughput_qps:.0f}qps" if trial.throughput_qps is not None else "-"
        console.print(
            f"  [{trial.trial_id + 1:>3}/{budget}] "
            f"{trial.config.backend:<16} {trial.config.quantization:<12} "
            f"bs={trial.config.batch_size:<3}  "
            f"{lat:>10}  {mem:>8}  {qps:>8}  {status_str}"
        )

    opt = DeploymentOptimizer(request, log_dir=log_dir, seed=seed)
    status = opt.run(on_trial=_on_trial)

    print_report(status)
    console.print(f"Logs saved to [bold]{opt.log_dir}[/bold]")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@main.command()
@click.option("--host", default="127.0.0.1", help="Bind address.")
@click.option("--port", default=8000, type=int, help="Port.")
def serve(host: str, port: int) -> None:
    """Start the FastAPI server with web dashboard."""
    import uvicorn

    from .server import app  # noqa: F811

    console.print(f"Starting server at http://{host}:{port}")
    console.print(f"Dashboard:  http://{host}:{port}/")
    console.print(f"API docs:   http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# mcp
# ---------------------------------------------------------------------------

@main.command()
def mcp() -> None:
    """Start the MCP (Model Context Protocol) server over stdio."""
    from .mcp_server import mcp_app

    mcp_app.run()
