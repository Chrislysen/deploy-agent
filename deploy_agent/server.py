"""FastAPI server — REST endpoints, WebSocket live updates, and web dashboard."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .models import OptimizationRequest, OptimizationStatus, TrialResult

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="deploy-agent", version="0.1.0")

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

_jobs: dict[str, OptimizationStatus] = {}


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class _WSManager:
    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, job_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(job_id, []).append(ws)

    def disconnect(self, job_id: str, ws: WebSocket) -> None:
        conns = self._connections.get(job_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, job_id: str, data: dict[str, Any]) -> None:
        for ws in list(self._connections.get(job_id, [])):
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(job_id, ws)


ws_manager = _WSManager()


# ---------------------------------------------------------------------------
# Background optimisation runner
# ---------------------------------------------------------------------------

async def _run_optimization(request: OptimizationRequest, job_id_override: str | None = None) -> str:
    """Run optimisation in a thread, broadcasting results via WebSocket."""
    from .optimizer import DeploymentOptimizer
    from .report import to_json

    opt = DeploymentOptimizer(request)
    if job_id_override:
        opt.job_id = job_id_override
        opt.status.job_id = job_id_override

    _jobs[opt.job_id] = opt.status
    loop = asyncio.get_running_loop()

    def on_trial(trial: TrialResult) -> None:
        # Schedule WebSocket broadcast from the worker thread
        asyncio.run_coroutine_threadsafe(
            ws_manager.broadcast(opt.job_id, {
                "type": "trial",
                "trial": json.loads(trial.model_dump_json()),
                "trials_completed": opt.status.trials_completed,
                "total_budget": opt.status.total_budget,
                "best_config": json.loads(opt.status.best_config.model_dump_json()) if opt.status.best_config else None,
                "best_objective": opt.status.best_objective,
            }),
            loop,
        )

    await loop.run_in_executor(None, lambda: opt.run(on_trial=on_trial))

    # Final broadcast
    await ws_manager.broadcast(opt.job_id, {
        "type": "done",
        "status": to_json(opt.status),
    })

    return opt.job_id


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.post("/optimize")
async def start_optimization(request: OptimizationRequest) -> dict[str, str]:
    """Launch an optimisation job. Returns ``{job_id}`` immediately."""
    from .optimizer import DeploymentOptimizer

    # Create optimizer just to get a job_id, then run in background
    tmp = DeploymentOptimizer(request)
    job_id = tmp.job_id
    _jobs[job_id] = tmp.status
    asyncio.create_task(_run_optimization(request, job_id_override=job_id))
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> OptimizationStatus:
    """Get the current status of a job."""
    if job_id not in _jobs:
        from fastapi import HTTPException
        raise HTTPException(404, f"Job {job_id} not found")
    return _jobs[job_id]


@app.get("/jobs/{job_id}/report")
async def get_report(job_id: str) -> dict[str, Any]:
    """Get the JSON report for a completed job."""
    if job_id not in _jobs:
        from fastapi import HTTPException
        raise HTTPException(404, f"Job {job_id} not found")
    from .report import to_json
    return to_json(_jobs[job_id])


@app.get("/jobs")
async def list_jobs() -> dict[str, list[dict[str, Any]]]:
    """List all jobs with summary info."""
    return {
        "jobs": [
            {
                "job_id": s.job_id,
                "state": s.state,
                "trials_completed": s.trials_completed,
                "total_budget": s.total_budget,
            }
            for s in _jobs.values()
        ]
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(ws: WebSocket, job_id: str) -> None:
    await ws_manager.connect(job_id, ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive
    except WebSocketDisconnect:
        ws_manager.disconnect(job_id, ws)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
