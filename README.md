# deploy-agent

**Automated ML deployment optimizer.** Give it a model and hardware constraints — it finds the fastest feasible deployment configuration automatically.

> *"Make resnet50 run under 20ms and 512MB on this GPU."*
> deploy-agent searches over backends, quantization modes, and batch sizes, handles crashes gracefully, and returns the best config with full evidence.

Built on [Thermal Budget Annealing (TBA)](https://github.com/Chrislysen/Constrained-ML-Deployment), a research optimizer for crash-heavy constrained deployment spaces.

---

## What it does

Deploying ML models in production means choosing the right combination of:

- **Backend**: PyTorch eager, `torch.compile`, ONNX Runtime
- **Quantization**: fp32, fp16, int8 dynamic
- **Batch size**: 1–64
- **Thread count**: 1–8 (backend-dependent)

Many of these combinations crash — out-of-memory errors, unsupported operator exceptions, compiler failures. Others run but violate latency or memory constraints. Manually searching through hundreds of configs is tedious and error-prone.

**deploy-agent automates this.** It uses TBA's two-phase hybrid optimizer:

1. **Phase 1 (Exploration):** Crash-aware simulated annealing maps out which regions of the config space are valid, which crash, and which violate constraints.
2. **Phase 2 (Exploitation):** Warm-starts Optuna's constrained TPE from the exploration data to find the optimal config within the feasible region.

Every trial is logged as structured JSON. Crashes are caught and recorded — they're data, not errors.

---

## Installation

### From GitHub

```bash
pip install git+https://github.com/Chrislysen/deploy-agent.git
```

### For development

```bash
git clone https://github.com/Chrislysen/deploy-agent.git
cd deploy-agent
pip install -e .
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 with CUDA (for GPU deployment)
- ONNX Runtime (optional, for ONNX backend search)

---

## Usage

deploy-agent has three interfaces: a **CLI** for direct use, a **FastAPI server** with a web dashboard for interactive use, and an **MCP server** for LLM tool integration.

### 1. CLI — Command Line

The simplest way to optimize a model deployment:

```bash
deploy-agent optimize --model resnet50 --max-latency 20ms --max-memory 512MB
```

**All CLI options:**

```
deploy-agent optimize [OPTIONS] MODEL

Arguments:
  MODEL                  Model name (e.g., resnet50, mobilenet_v2, efficientnet_b0, vit_tiny, resnet18)

Options:
  --max-latency TEXT     Latency constraint, e.g., "20ms" or "0.02s" (default: 20ms)
  --max-memory TEXT      Memory constraint, e.g., "512MB" or "1GB" (default: 512MB)
  --gpu TEXT             GPU selection: "auto" detects available GPU (default: auto)
  --budget INTEGER       Number of trials to run (default: 25)
  --output PATH          Path for JSON results log
```

**Example output:**

```
╭─────────────────── deploy-agent ───────────────────╮
│ Model:       resnet50                               │
│ Constraints: latency_p95 ≤ 20ms, memory ≤ 512MB    │
│ Budget:      25 trials                              │
│ GPU:         NVIDIA RTX 5080                        │
╰─────────────────────────────────────────────────────╯

  Trial  Backend          Quant    BS  Latency   Memory  Throughput  Status
  ─────  ───────          ─────    ──  ───────   ──────  ──────────  ──────
  0      onnxruntime      fp32     2   —         —       —           CRASH
  1      torch_compile    fp32     4   5.1ms     72MB    830 qps     OK
  2      torch_compile    int8     2   967ms     —       3 qps       INFEAS
  3      pytorch_eager    fp32     15  18.3ms    392MB   956 qps     OK ★
  ...

╭─────────────── Best Feasible Config ───────────────╮
│ Backend:     pytorch_eager                          │
│ Quantization: fp32                                  │
│ Batch size:  15                                     │
│ Latency p95: 18.3ms                                 │
│ Memory:      392 MB                                 │
│ Throughput:  956.2 qps                              │
╰─────────────────────────────────────────────────────╯
```

### 2. Web Dashboard — FastAPI Server

Start the server:

```bash
deploy-agent serve --port 8000
```

Then open **http://localhost:8000** in your browser.

The dashboard provides:

- **Live optimization controls** — enter model, latency, memory constraints, and hit Run
- **Real-time charts** — latency & throughput over trials, peak memory bars, latency-vs-throughput scatter plot with feasible/infeasible/crashed color coding
- **Best config card** — shows the current best deployment configuration with all metrics
- **Trial log table** — every trial with backend, quantization, batch size, latency, memory, throughput, and status (OK / CRASH / INFEAS)
- **WebSocket live updates** — charts and tables update in real time as trials complete

**API endpoints** (docs at http://localhost:8000/docs):

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/optimize` | Start an optimization job. Returns a `job_id` immediately. |
| `GET` | `/jobs/{job_id}` | Get full job status, all trial results, and best config. |
| `GET` | `/jobs` | List all jobs. |
| `WS` | `/ws/{job_id}` | WebSocket stream — receives each trial result as it completes. |

**Example API call:**

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"model": "resnet50", "max_latency_ms": 20, "max_memory_mb": 512, "budget": 25}'
```

Response:
```json
{"job_id": "bce31fd31169"}
```

Then poll for results:
```bash
curl http://localhost:8000/jobs/bce31fd31169
```

### 3. MCP Server — LLM Tool Integration

deploy-agent exposes itself as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server, allowing LLMs like Claude to call it as a tool:

```bash
deploy-agent mcp
```

This starts an MCP server with two tools:

| Tool | Description |
|------|-------------|
| `optimize_deployment` | Run a full deployment optimization search for a given model and constraints. Returns the best feasible config with evidence. |
| `list_supported_models` | List all models that deploy-agent can optimize. |

**Example MCP interaction** (from an LLM's perspective):

```
User: "Find the best way to deploy resnet50 under 20ms latency and 512MB memory."
LLM → calls optimize_deployment(model="resnet50", max_latency_ms=20, max_memory_mb=512)
LLM ← receives structured result with best config, all trials, crash analysis
LLM: "The best config is pytorch_eager with fp32 at batch size 15, achieving 18.3ms latency and 956 qps."
```

---

## How it works

### Search space

For each model, deploy-agent constructs a hierarchical search space:

| Variable | Type | Domain |
|----------|------|--------|
| backend | categorical | pytorch_eager, torch_compile, onnxruntime |
| quantization | categorical | fp32, fp16, int8_dynamic |
| batch_size | integer (log) | 1–64 |
| num_threads | integer | 1–8 (only active when backend ≠ torch_compile) |

This produces ~1,500+ possible configurations. Many will crash or violate constraints — that's expected and handled.

### Optimization strategy

deploy-agent uses TBA→TPE, a two-phase hybrid:

1. **Phase 1 — Thermal Budget Annealing**: Simulated annealing with crash-aware transitions, adaptive temperature control, and structural mutations. Explores broadly to discover which backends, quantization modes, and batch sizes are viable on the target hardware.

2. **Phase 2 — Constrained TPE**: Once enough feasible and infeasible observations exist, hands off to Optuna's Tree-structured Parzen Estimator with constraint awareness. All Phase 1 data (including crashes) is injected as prior knowledge.

This approach is based on the research paper: *"Feasible-First Exploration for Constrained ML Deployment Optimization in Crash-Prone Hierarchical Search Spaces"* ([GitHub](https://github.com/Chrislysen/Constrained-ML-Deployment)).

### Crash handling

Every benchmark trial is wrapped in exception handling. When a config crashes (OOM, unsupported ops, compiler error), deploy-agent:

- Catches the exception
- Logs it as a crashed trial with full metadata
- Reports it to the optimizer so it learns crash boundaries
- Continues to the next trial

Crashes are data, not failures. The optimizer uses crash information to avoid similar configs in later trials.

### Benchmarking

Each configuration is evaluated with:

- Model loading and optional quantization
- Backend compilation or ONNX export
- 10 warmup iterations (excluded from timing)
- 50 timed iterations
- p95 and p99 latency measurement
- Peak GPU memory measurement
- Throughput calculation (queries per second)

---

## Project structure

```
deploy-agent/
├── pyproject.toml              # Package config, dependencies, CLI entry point
├── README.md
└── deploy_agent/
    ├── __init__.py
    ├── models.py               # Pydantic data models (TrialConfig, TrialResult, etc.)
    ├── benchmarker.py          # Hardware benchmarking with crash handling
    ├── optimizer.py            # TBA hybrid optimizer wrapper (ask/tell loop)
    ├── report.py               # Terminal tables and JSON report generation
    ├── cli.py                  # Click CLI (optimize, serve, mcp commands)
    ├── server.py               # FastAPI server with REST + WebSocket
    ├── mcp_server.py           # MCP server for LLM tool integration
    └── static/
        └── index.html          # Web dashboard (Chart.js, WebSocket live updates)
```

---

## Supported models

Any torchvision model works. Tested with:

| Model | Parameters | Typical accuracy (ImageNette) |
|-------|-----------|-------------------------------|
| resnet18 | 11.7M | 72.0% |
| resnet50 | 25.6M | 74.6% |
| mobilenet_v2 | 3.4M | 71.6% |
| efficientnet_b0 | 5.3M | 74.2% |
| vit_tiny | 5.7M | 76.6% |

To add a custom model, modify the model loading in `benchmarker.py`.

---

## Example results

Running `deploy-agent optimize --model resnet50 --max-latency 20ms --max-memory 512MB --budget 25` on an NVIDIA RTX 5080 Laptop GPU:

| Metric | Value |
|--------|-------|
| Trials run | 25 |
| Feasible configs found | 16 (64%) |
| Crashed configs | 9 (36%) |
| Best latency p95 | 18.3ms |
| Best memory | 392 MB |
| Best throughput | 956.2 qps |
| Best config | pytorch_eager / fp32 / batch_size=15 |

The optimizer discovered that ONNX Runtime crashes for this model on this hardware, int8_dynamic quantization violates the latency constraint, and pytorch_eager with fp32 at batch size 15 is the sweet spot.

---

## Research

deploy-agent is built on published research:

- **Paper**: *Feasible-First Exploration for Constrained ML Deployment Optimization in Crash-Prone Hierarchical Search Spaces*
- **Research repo**: [Constrained-ML-Deployment](https://github.com/Chrislysen/Constrained-ML-Deployment)
- **Key finding**: In crash-heavy deployment spaces (30–80% invalidity), explicit feasible-first exploration improves model-family discovery and reduces wasted budget compared to cold-start model-guided search.

---

## License

MIT
