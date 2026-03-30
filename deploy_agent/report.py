"""Report generation — terminal tables and JSON summaries."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.table import Table

from .models import OptimizationStatus


def print_report(status: OptimizationStatus) -> None:
    """Print a rich terminal report for *status*."""
    console = Console()

    # --- header ---
    console.print()
    console.rule("[bold]Deploy-Agent Optimisation Report[/bold]")
    console.print(f"  Job:    {status.job_id}")
    console.print(f"  Trials: {status.trials_completed} / {status.total_budget}")

    feasible = [t for t in status.trials if t.feasible]
    crashed = [t for t in status.trials if t.crashed]
    console.print(f"  Feasible: {len(feasible)}  |  Crashed: {len(crashed)}")

    # --- best configuration ---
    console.print()
    if status.best_config:
        console.rule("[green]Best Feasible Configuration[/green]")
        best_trial = next(
            (t for t in status.trials if t.config == status.best_config and t.feasible),
            None,
        )
        console.print(f"  Backend:      {status.best_config.backend}")
        console.print(f"  Quantization: {status.best_config.quantization}")
        console.print(f"  Batch size:   {status.best_config.batch_size}")
        console.print(f"  Num threads:  {status.best_config.num_threads}")
        if best_trial:
            console.print(f"  Latency p95:  {best_trial.latency_p95_ms:.2f} ms" if best_trial.latency_p95_ms else "")
            console.print(f"  Memory peak:  {best_trial.memory_peak_mb:.1f} MB" if best_trial.memory_peak_mb else "")
            console.print(f"  Throughput:   {best_trial.throughput_qps:.1f} qps" if best_trial.throughput_qps else "")
    else:
        console.print("[red]No feasible configuration found.[/red]")

    # --- trial table ---
    console.print()
    console.rule("All Trials")
    table = Table(show_lines=False)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Backend", width=16)
    table.add_column("Quant", width=12)
    table.add_column("BS", justify="right", width=4)
    table.add_column("Latency p95", justify="right", width=12)
    table.add_column("Memory MB", justify="right", width=10)
    table.add_column("QPS", justify="right", width=10)
    table.add_column("Status", width=10)

    for t in status.trials:
        style = ""
        if t.crashed:
            stat = "[red]CRASH[/red]"
        elif t.feasible:
            stat = "[green]OK[/green]"
            if status.best_config and t.config == status.best_config:
                style = "bold green"
        else:
            stat = "[yellow]INFEAS[/yellow]"

        table.add_row(
            str(t.trial_id),
            t.config.backend,
            t.config.quantization,
            str(t.config.batch_size),
            f"{t.latency_p95_ms:.2f}" if t.latency_p95_ms is not None else "-",
            f"{t.memory_peak_mb:.0f}" if t.memory_peak_mb is not None else "-",
            f"{t.throughput_qps:.1f}" if t.throughput_qps is not None else "-",
            stat,
            style=style,
        )

    console.print(table)
    console.print()


def to_json(status: OptimizationStatus) -> dict[str, Any]:
    """Return the full status as a JSON-serialisable dict."""
    return json.loads(status.model_dump_json())
