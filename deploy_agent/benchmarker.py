"""Hardware benchmarking for model deployment configurations.

Evaluates a single (backend, quantization, batch_size, num_threads) tuple on
real hardware and returns latency / memory / throughput metrics.  Every failure
is caught and surfaced as ``crashed=True`` so the TBA optimizer can learn from
infeasible regions.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import numpy as np

from .models import TrialConfig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def benchmark_config(
    model_name: str,
    config: TrialConfig,
    device: str = "cuda",
    warmup_iters: int = 10,
    bench_iters: int = 50,
) -> dict[str, Any]:
    """Benchmark *config* for *model_name* on *device*.

    Returns a dict with keys matching :class:`TrialResult` fields.
    On any exception the dict contains ``crashed=True`` and the error message.
    """
    result: dict[str, Any] = {
        "latency_p95_ms": None,
        "memory_peak_mb": None,
        "throughput_qps": None,
        "accuracy": None,
        "crashed": False,
        "error_msg": "",
    }

    try:
        if config.backend == "onnxruntime":
            _benchmark_onnx(model_name, config, device, warmup_iters, bench_iters, result)
        else:
            _benchmark_pytorch(model_name, config, device, warmup_iters, bench_iters, result)
    except Exception as exc:  # noqa: BLE001 – intentionally broad
        result["crashed"] = True
        result["error_msg"] = f"{type(exc).__name__}: {exc}"

    return result


# ---------------------------------------------------------------------------
# PyTorch (eager / torch.compile)
# ---------------------------------------------------------------------------

def _benchmark_pytorch(
    model_name: str,
    config: TrialConfig,
    device: str,
    warmup_iters: int,
    bench_iters: int,
    result: dict[str, Any],
) -> None:
    import torch
    import torchvision.models as models

    dev = _resolve_device(device)

    # --- load model ---
    model = getattr(models, model_name)(weights="DEFAULT").eval()

    if config.quantization == "fp16":
        model = model.half().to(dev)
    elif config.quantization == "int8_dynamic":
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8,
        )
        model = model.to(dev)
    else:
        model = model.to(dev)

    if config.backend == "torch_compile":
        model = torch.compile(model)

    # --- prepare input ---
    dtype = torch.float16 if (config.quantization == "fp16" and dev.type == "cuda") else torch.float32
    dummy = torch.randn(config.batch_size, 3, 224, 224, device=dev, dtype=dtype)

    if dev.type == "cpu":
        torch.set_num_threads(config.num_threads)

    # --- memory tracking ---
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)

    # --- warmup ---
    with torch.no_grad():
        for _ in range(warmup_iters):
            model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    # --- timed iterations ---
    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(bench_iters):
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            model(dummy)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

    arr = np.asarray(latencies)
    result["latency_p95_ms"] = float(np.percentile(arr, 95))
    result["throughput_qps"] = float(config.batch_size * 1000.0 / np.mean(arr))

    if dev.type == "cuda":
        result["memory_peak_mb"] = float(
            torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
        )


# ---------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------

def _benchmark_onnx(
    model_name: str,
    config: TrialConfig,
    device: str,
    warmup_iters: int,
    bench_iters: int,
    result: dict[str, Any],
) -> None:
    import torch
    import torchvision.models as models
    import onnxruntime as ort

    dev = _resolve_device(device)

    # --- export model to ONNX ---
    model = getattr(models, model_name)(weights="DEFAULT").eval()
    dummy_export = torch.randn(1, 3, 224, 224)

    fd, onnx_path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    try:
        torch.onnx.export(
            model, dummy_export, onnx_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

        providers: list[str] = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if dev.type == "cuda"
            else ["CPUExecutionProvider"]
        )
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = config.num_threads
        session = ort.InferenceSession(onnx_path, sess_opts, providers=providers)
    finally:
        os.unlink(onnx_path)

    # --- prepare input ---
    dummy_np = np.random.randn(config.batch_size, 3, 224, 224).astype(np.float32)

    # --- warmup ---
    for _ in range(warmup_iters):
        session.run(None, {"input": dummy_np})

    # --- timed iterations ---
    latencies: list[float] = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        session.run(None, {"input": dummy_np})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

    arr = np.asarray(latencies)
    result["latency_p95_ms"] = float(np.percentile(arr, 95))
    result["throughput_qps"] = float(config.batch_size * 1000.0 / np.mean(arr))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
