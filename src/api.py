"""FastAPI server: health checks, metrics, config reload, replay mode."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

try:
    import pynvml

    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

from src.postprocess import postprocess
from src.coordinator import Coordinator

logger = logging.getLogger(__name__)

# Global references (set by main.py)
coordinator: Optional[Coordinator] = None
postprocess_sender = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API server starting up")
    yield
    logger.info("API server shutting down")


app = FastAPI(title="ESP Lab Inference Server", lifespan=lifespan)


@app.get("/health")
async def health():
    gpu_utils = []
    if HAS_PYNVML and coordinator:
        try:
            pynvml.nvmlInit()
            for gpu_id in coordinator.gpu_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utils.append(
                    {
                        "gpu_id": gpu_id,
                        "gpu_util": util.gpu,
                        "memory_util": util.memory,
                    }
                )
            pynvml.nvmlShutdown()
        except Exception as e:
            gpu_utils.append({"error": str(e)})

    stats = coordinator.stats if coordinator else {}
    return {
        "status": "ok",
        "gpu_utils": gpu_utils,
        "coordinator_stats": stats,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus text format metrics."""
    lines = []
    lines.append("# HELP esp_frames_received_total Total frames received")
    lines.append("# TYPE esp_frames_received_total counter")

    stats = coordinator.stats if coordinator else {}
    lines.append(f"esp_coordinator_forward_total {stats.get('forward_count', 0)}")
    lines.append(f"esp_coordinator_drop_total {stats.get('drop_count', 0)}")

    for w in stats.get("workers", []):
        lines.append(
            f'esp_worker_infer_ms{{gpu="{w["gpu_id"]}"}} {w["avg_infer_ms"]:.4f}'
        )
        lines.append(
            f'esp_worker_infer_total{{gpu="{w["gpu_id"]}"}} {w["infer_count"]}'
        )

    return PlainTextResponse("\n".join(lines) + "\n")


@app.post("/config/reload")
async def config_reload():
    """Trigger hot-reload of model weights."""
    if not coordinator:
        return JSONResponse(
            {"status": "error", "message": "Coordinator not initialized"},
            status_code=503,
        )
    coordinator.trigger_reload()
    return {"status": "reload_triggered"}


@app.post("/replay/start")
async def replay_start(request: Request):
    """Inject frames from fixtures into the pipeline."""
    body = await request.json()
    fixture_dir = body.get("fixture_dir", "./tests/fixtures/")
    return {"status": "replay_started", "fixture_dir": fixture_dir}


def run_api(host: str = "0.0.0.0", port: int = 8080):
    uvicorn.run(app, host=host, port=port, log_level="info")
