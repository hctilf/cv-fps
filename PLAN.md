# ESP Lab — Implementation Plan (Option A: Two-Machine)

## Topology

```
[Game PC — any IP]                      [Inference Server — 192.168.1.17]
                                                                          
  ffmpeg (desktop capture)                                                
    -f gdigrab -i desktop                                                 
    -vf scale=640:640,fps=120                                             
    -vcodec mjpeg -q:v 5                                                  
    ──── UDP:9999 ──────────────────────────→  udp_receiver.py           
                                               coordinator.py             
                                               inference_worker.py ×3     
                                               postprocess.py             
                                               ←── UDP:CLIENT_IP:8888 ───
  client_visualizer.py                                                    
  (receives bbox JSON, draws                                              
   overlay via OpenCV)                                                    
```

The server discovers the client's return IP from the incoming UDP datagram's source address.  
No hardcoded client IP — works as long as both machines are on `192.168.1.x`.

---

## Architecture Decisions

| Decision | Choice | Reason |
|---|---|---|
| Worker model | **Threads** (not processes) | Required for inproc ZMQ; CUDA releases GIL → true GPU parallelism |
| IPC transport | **ZMQ inproc** | Zero-copy within process, nanosecond latency |
| Frame decode | **libjpeg-turbo** via `cv2` | Fastest JPEG decode on CPU |
| GPU transfer | **Pinned memory** tensors | Avoids extra host↔device copy |
| FP precision | **FP16 + torch.compile** | ~2× inference speed vs FP32 on RTX 3090 |
| Kalman filter | **Deferred to V2** | 120 FPS input is inherently smooth |
| Coordinator dispatch | **Round-robin + load check** | Simple, deterministic, O(1) |
| Queue overflow | **Drop oldest** (maxlen=10) | Prevents lag spiral under load |
| Result delivery | **UDP → client source IP:8888** | Client IP auto-detected from recv datagram |
| Latency target | **≤ 30ms end-to-end** | Hard SLA |

---

## Threading Model (single server process)

```
asyncio UDP recv (port 9999)
    │  raw JPEG bytes + source (ip, port)
    ▼
[Decode Thread]  cv2.imdecode → float32 pinned tensor
    │  (frame_id, ts_recv, tensor, client_addr)
    ▼  ZMQ inproc PUSH
[ZMQ inproc ROUTER — coordinator.py]
    │  round-robin dispatch
    ├──▶ [GPU Thread cuda:0 — inference_worker.py]──┐
    ├──▶ [GPU Thread cuda:1 — inference_worker.py]──┤
    └──▶ [GPU Thread cuda:2 — inference_worker.py]──┘
                                                    │  ZMQ inproc PULL
                                                    ▼
                                           [Output Thread — postprocess.py]
                                                    │  bbox JSON
                                                    ▼
                                           UDP → client_addr[0]:8888
```

---

## Latency Budget @ 120 FPS (8.3 ms/frame)

| Stage | Est. Time |
|---|---|
| UDP recv + JPEG decode | ~1 ms |
| Queue dispatch (inproc ZMQ) | ~0.1 ms |
| YOLO11n FP16 inference (RTX 3090) | ~3–5 ms |
| Post-process + JSON serialize | ~0.5 ms |
| UDP send to client | ~0.2 ms |
| **Total** | **~5–7 ms** ✅ |

3 GPUs at ~5ms each = ~200 FPS inference capacity for 120 FPS input — plenty of headroom.

---

## Project Structure

```
esp-lab/
├── config/
│   └── default.yaml          # all tunables — model path, thresholds, ports, GPU IDs
├── src/
│   ├── udp_receiver.py       # asyncio UDP listener, JPEG decode, ZMQ PUSH
│   ├── inference_worker.py   # per-GPU thread, YOLO init, hot-reload support
│   ├── coordinator.py        # spawns workers, routes frames, merges results
│   ├── postprocess.py        # class filter, JSON serialize, UDP send to client
│   └── api.py                # FastAPI: /health /metrics /config/reload /replay/start
├── tests/
│   ├── test_latency.py       # replay mode, p50/p95/p99, fail if p99 > 30ms
│   ├── test_schema.py        # JSON Schema validation of every output packet
│   └── fixtures/             # pre-recorded frames + expected detections
├── client_visualizer.py      # recv bbox JSON, draw overlay via OpenCV, FPS HUD
├── docker-compose.yml        # optional: 3 GPU containers
├── requirements.txt
├── PLAN.md                   # this file
└── OPTION_B_NOTES.md         # single-machine variant notes
```

---

## Module Responsibilities

### `udp_receiver.py`
- `asyncio` UDP protocol on `0.0.0.0:9999`
- Records `(client_ip, client_port)` from first datagram — passed downstream for reply routing
- Per-datagram: validate JPEG magic bytes → `cv2.imdecode` → `torch.from_numpy` → pin memory
- Silently drops corrupt/truncated frames
- Pushes `FramePacket(frame_id, ts_recv_ns, tensor, client_addr)` via ZMQ inproc PUSH

### `inference_worker.py`
- One thread per GPU, `device_id` injected at construction
- `YOLO(config.model.path, task='detect').to(f'cuda:{device_id}')` with `half=True`
- Hot-reload: watches a `threading.Event`; on trigger reloads weights in-place
- Emits `InferenceResult(frame_id, device_id, preprocess_ms, infer_ms, raw_results, client_addr)`

### `coordinator.py`
- Starts 3 `inference_worker` threads
- Round-robin dispatch; if a worker queue is full (>10), drop oldest frame and increment `drop_counter`
- Result collector thread re-orders by `frame_id` (sliding window ≤3 frames OOO) then forwards to postprocess

### `postprocess.py`
- Maps YOLO class indices → `{0: 'EnemySoldier', 1: 'AllySoldier', 2: 'Weapon', 3: 'Vehicle', 4: 'HealthPack'}`
- Confidence threshold from config (default 0.4)
- Serializes to JSON:
  ```json
  {
    "frame_id": 12345,
    "ts_infer_ms": 4.8,
    "detections": [
      {"class": "EnemySoldier", "bbox": [x1, y1, x2, y2], "conf": 0.94}
    ]
  }
  ```
- Sends via `socket.sendto` to `client_addr[0]:8888`

### `api.py` (FastAPI + Uvicorn)
| Endpoint | Detail |
|---|---|
| `GET /health` | GPU util via `pynvml`, rolling 1s FPS, latency p50/p99 |
| `POST /config/reload` | Sets hot-reload Event, waits for worker ACK |
| `GET /metrics` | Prometheus text format |
| `POST /replay/start` | Injects frames from `./tests/fixtures/` into the pipeline |

### `client_visualizer.py`
- Listens on `0.0.0.0:8888`
- Deserializes bbox JSON
- Draws colored boxes + class labels on a blank 640×640 canvas (or captured frame if available)
- Top-left HUD: FPS, last `ts_infer_ms`, server round-trip ms

---

## Config (`config/default.yaml`)

```yaml
server:
  listen_host: "0.0.0.0"
  listen_port: 9999
  client_reply_port: 8888

model:
  path: "yolo11n.pt"          # swap to yolo11s.pt / yolo11m.pt freely
  imgsz: 640
  conf_threshold: 0.4
  iou_threshold: 0.45
  half: true

gpus: [0, 1, 2]

queues:
  max_depth: 10               # per worker; older frames dropped on overflow

api:
  host: "0.0.0.0"
  port: 8080

logging:
  level: "INFO"
  structured: true            # JSON lines for log ingestion
```

---

## AQA / Test Hooks

- `--dry-run` CLI flag: routes `./tests/fixtures/` frames through the full pipeline, no live UDP
- Structured per-frame log: `frame_id, receive_ts, preprocess_ms, infer_ms, send_ts`
- `pytest` fixtures: `mock_udp_stream` yields `(frame_bytes, expected_detections)`
- `test_latency.py` fails if p99 latency > 30ms over a 1000-frame replay
- `test_schema.py` validates every output packet against JSON Schema

---

## SLAs

| Metric | Target |
|---|---|
| End-to-end latency p99 | ≤ 30 ms |
| Sustained throughput | ≥ 100 FPS inference |
| Frame drop rate | < 1% under normal load |
| Queue max depth | 10 frames per worker |
| Graceful shutdown | finish in-flight frames on SIGINT, save metrics snapshot |

---

## V2 Roadmap (out of scope now)

- TensorRT engine export for sub-3ms inference
- Kalman / SORT tracker for bbox smoothing
- NCCL-based model parallelism for larger models (YOLO11x, RT-DETR)
- WebSocket output for reliable delivery + sequencing
- Grafana dashboard wired to `/metrics`
