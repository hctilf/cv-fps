# ESP Lab — Real-Time Inference Server

Multi-GPU YOLO inference server for live game video streams. Receives MJPEG frames via UDP, runs detection across 3 GPUs, and returns bounding-box overlays to a client visualizer — all under 30ms end-to-end.

## Architecture

```
[Game PC]                              [Inference Server]
  ffmpeg (gdigrab)                          udp_receiver.py
    ── UDP:9999 ──→  coordinator.py
                                           inference_worker.py ×3 (GPU 0,1,2)
                                           postprocess.py
                                           ←── UDP:8888 ──→
  client_visualizer.py                      api.py (FastAPI)
```

## Use Cases

### 1. Live Game Coaching Overlay
Stream your game screen to the server, receive real-time detection of enemies, allies, weapons, and vehicles. The `client_visualizer.py` draws colored bounding boxes and a HUD with FPS and inference latency — perfect for training or competitive analysis.

### 2. Multi-GPU Throughput Benchmarking
Test inference throughput across multiple GPUs using the replay mode (`--dry-run`). The server routes frames round-robin across all available GPUs and logs per-frame latency. Run `pytest` to validate p99 latency stays under the 30ms SLA.

### 3. Hot-Reload Model Testing
Swap model weights without restarting the server. POST to `/config/reload` to trigger an in-place weight reload across all GPU workers. Useful for A/B testing different model sizes (n, s, m) or custom-trained checkpoints.

### 4. Custom Object Detection Pipeline
The class mapping in `postprocess.py` is easily configurable. Replace the default classes (`EnemySoldier`, `AllySoldier`, `Weapon`, `Vehicle`, `HealthPack`) with your own categories. The pipeline accepts any YOLO-compatible `.pt` model — just update `config/default.yaml`.

### 5. Low-Latency Development & Debugging
The structured logging and `/metrics` endpoint (Prometheus format) make it easy to profile the pipeline. Use `--dry-run` with pre-recorded fixture frames to reproduce latency issues without needing a live game capture.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (live mode)
python src/main.py --config config/default.yaml

# Start in replay/dry-run mode
python src/main.py --config config/default.yaml --dry-run

# Start the client visualizer
python client_visualizer.py
```

### Streaming from Game PC

```bash
# On the game PC, send screen capture to the inference server
ffmpeg -f gdigrab -i desktop -vf scale=640:640,fps=120 -vcodec mjpeg -q:v 5 -thread_type slice -threads 0 -f mjpeg udp://192.168.1.17:9999
```

## Configuration

Edit `config/default.yaml` to tune:

| Setting | Default | Description |
|---|---|---|
| `server.listen_port` | 9999 | UDP port for receiving MJPEG frames |
| `server.client_reply_port` | 8888 | UDP port the client visualizer listens on |
| `model.path` | `models/raven_v2.pt` | Path to your YOLO model |
| `model.conf_threshold` | 0.4 | Minimum confidence to report a detection |
| `model.imgsz` | 640 | Inference image size |
| `model.half` | true | FP16 inference (requires compatible GPU) |
| `gpus` | `[0, 1, 2]` | GPU device IDs to use |
| `queues.max_depth` | 10 | Per-worker queue size; older frames dropped on overflow |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | GPU utilization, latency stats, worker health |
| `/metrics` | GET | Prometheus-formatted metrics |
| `/config/reload` | POST | Trigger hot-reload of model weights |
| `/replay/start` | POST | Inject frames from fixture directory |

## Testing

```bash
python -m pytest tests/ -v
```

- `test_latency.py` — Simulates 1000 frames, validates p99 latency ≤ 30ms
- `test_schema.py` — Validates JSON output structure against schema

## Latency Budget

| Stage | Time |
|---|---|
| UDP recv + JPEG decode | ~1ms |
| ZMQ dispatch | ~0.1ms |
| YOLO inference (RTX 3090) | ~3–5ms |
| Post-process + JSON | ~0.5ms |
| UDP send | ~0.2ms |
| **Total** | **~5–7ms** |

## Requirements

- Python 3.12+
- NVIDIA GPU(s) with CUDA support (FP16 recommended)
- 3 GPUs for full throughput (fewer GPUs = lower capacity)

## License

Private — internal use only.
