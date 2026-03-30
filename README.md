# deploy-agent

Automated ML deployment optimizer. Finds the fastest feasible backend, quantization, and batch size for your model under latency and memory constraints.

Uses [Thermal Budget Annealing (TBA)](https://github.com/Chrislysen/Constrained-ML-Deployment) as the search backend — a two-phase hybrid optimizer that discovers crash zones via simulated annealing, then optimizes within the feasible region using TPE.

## What it does

Given a model and constraints like `--max-latency 20ms --max-memory 512MB`, deploy-agent automatically searches over:

- **Backends**: PyTorch eager, `torch.compile`, ONNX Runtime
- **Quantization**: FP32, FP16, INT8 (dynamic)
- **Batch sizes**: 1–64 (log scale)
- **Thread counts**: 1–8 (conditional on backend)

Each configuration is benchmarked on real hardware. Crashes (OOM, unsupported combos) are handled gracefully and fed back to the optimizer so it learns to avoid infeasible regions.

## Install

```bash
pip install -e .
```

## Usage

### CLI

```bash
deploy-agent optimize --model resnet50 --max-latency 20ms --max-memory 512MB --gpu auto
```

### API server + web dashboard

```bash
deploy-agent serve --port 8000
# Dashboard:  http://localhost:8000
# API docs:   http://localhost:8000/docs
```

The dashboard shows live optimization progress with charts for latency, throughput, memory, and a trial log table — all updating in real time via WebSocket.

### MCP server (for LLM tool use)

```bash
deploy-agent mcp
```

Exposes `optimize_deployment` and `list_supported_models` as MCP tools that any LLM client can call.

## API

```
POST /optimize     — Start an optimization job (returns job_id)
GET  /jobs/{id}    — Get job status and all trials
GET  /jobs/{id}/report — Get JSON report
GET  /jobs         — List all jobs
WS   /ws/{id}      — WebSocket for live trial updates
```

## Architecture

| Module | Purpose |
|---|---|
| `cli.py` | Click CLI (`optimize`, `serve`, `mcp`) |
| `optimizer.py` | TBA hybrid optimizer ask/tell loop |
| `benchmarker.py` | Hardware benchmarking (PyTorch + ONNX Runtime) |
| `server.py` | FastAPI server with REST + WebSocket |
| `mcp_server.py` | MCP server for LLM tool use |
| `report.py` | Rich terminal + JSON reports |
| `models.py` | Shared Pydantic models |
| `static/index.html` | Web dashboard (Chart.js, live updates) |
