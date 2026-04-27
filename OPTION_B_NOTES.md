# Option B — Single-Machine Variant

Everything runs on one box. ffmpeg captures the local desktop and sends to `127.0.0.1:9999`.
The inference server outputs bbox JSON to `127.0.0.1:8888`. The visualizer reads from the same localhost port.

```
[Single Machine]
ffmpeg -f gdigrab ... udp://127.0.0.1:9999
        ↓
inference server (192.168.1.17 or localhost)
        ↓ UDP
127.0.0.1:8888 → client_visualizer.py (overlay on same screen)
```

**Differences from Option A:**
- No client IP discovery needed — reply address is always `127.0.0.1`
- No network latency between capture and inference
- Harder to isolate GPU load from game GPU (if game and inference share the same machine)
- `config/default.yaml` change: set `client_reply_port_host: "127.0.0.1"` instead of auto-detecting source IP

**When to use:** Development/testing without a second machine, or if the inference rig also runs the game.
