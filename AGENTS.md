# AGENTS.md — Guidelines for Agentic Coding Agents

## Project Overview

ESP Lab is a multi-GPU YOLO inference server for real-time game video streams.
Receives MJPEG frames via UDP, runs detection across 3 GPUs, returns bounding-box
overlays to a client visualizer — all under 30ms end-to-end.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_schema.py -v

# Run a single test method
python -m pytest tests/test_schema.py::TestSchema::test_invalid_conf_range -v

# Run the server (live mode)
python src/main.py --config config/default.yaml

# Run the server (dry-run replay mode)
python src/main.py --config config/default.yaml --dry-run

# Run the client visualizer
python client_visualizer.py

# No linter/formatter/type-checker is configured.
```

## Code Style

### Imports
Order: **stdlib → third-party → local (`src.`)**. No blank lines between groups.
```python
import asyncio
import json
import logging
from typing import Optional

import yaml
import zmq

from src.postprocess import postprocess
```

### String Quoting
**Single quotes** for all string literals. Double quotes only inside f-strings for
dict keys: `f"host={cfg['host']}"`.

### Naming
- **snake_case**: functions, methods, variables, modules
- **PascalCase**: classes
- **UPPER_SNAKE_CASE**: module-level constants and sentinels
- Private attributes: single underscore prefix (`_socket`, `_running`)

### Types
- Annotate **function parameters and return types** consistently.
- Use `Optional[T]` for nullable parameters.
- Use `@dataclass` for structured data (e.g. `FramePacket`, `InferenceResult`).
- Do **not** annotate local variables.

### Docstrings
Module-level and function docstrings: **brief, one-line description**.
No multi-line Google/NumPy format needed.

```python
"""Postprocess: maps YOLO class indices, filters by confidence, serializes to JSON."""
```

### Error Handling
Use **broad `except Exception`** with **silent failure + logging**. No custom
exceptions. Always log the error and continue.

```python
except Exception as e:
    logger.debug(f"Dropped frame (decode error): {e}")
```

ZMQ non-blocking reads: catch `zmq.Again` specifically.

### Async
- Use `asyncio.run()` at entry points.
- Inside async functions, use `await` directly — **never** `run_until_complete()`.
- Use `asyncio.get_running_loop()` (not `get_event_loop()`).

### Line Length
Keep lines under **88 characters** (Ruff default).

## Architecture Notes

- **`src/main.py`**: Entry point. Wires `UDPReceiver → Coordinator → PostprocessSender`.
- **`src/coordinator.py`**: Spawns per-GPU workers, round-robin dispatch, result collection.
- **`src/inference_worker.py`**: One thread per GPU, YOLO model, hot-reload via `threading.Event`.
- **`src/udp_receiver.py`**: Asyncio UDP listener, JPEG decode, pinned tensors, ZMQ PUSH.
- **`src/postprocess.py`**: Class mapping dict, confidence filter, JSON serializes, UDP send.
- **`src/api.py`**: FastAPI server on separate thread. `coordinator` is a global set by main.
- **`client_visualizer.py`**: Standalone script (not in `src/`). Listens on UDP:8888, draws OpenCV overlay.
- **`config/default.yaml`**: All tunables. Models reference `models/raven_v2.pt`.
- **`models/`**: Contains `.pt`, `.onnx`, `.engine` files — never modify these.

## Testing

- Tests use `unittest` with `pytest` as the runner.
- `test_latency.py`: Simulates 1000 frames, validates p99 latency ≤ 30ms.
- `test_schema.py`: Validates JSON output against `jsonschema` schema.
- Tests do **not** require GPUs or live UDP — they use synthetic fixtures.
