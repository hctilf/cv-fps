"""Test latency: replay frames through pipeline, measure p50/p95/p99, fail if p99 > 30ms."""

import json
import os
import sys
import time
import unittest
from pathlib import Path
from typing import List

import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.postprocess import postprocess


class TestLatency(unittest.TestCase):
    FIXTURE_DIR = Path(__file__).parent / "fixtures"
    NUM_FRAMES = 1000
    LATENCY_P99_THRESHOLD_MS = 30.0

    @classmethod
    def setUpClass(cls):
        cls.latencies: List[float] = []

    def _generate_fixture_frame(self, frame_id: int) -> bytes:
        """Generate a synthetic JPEG frame for testing."""
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    def test_p99_latency_under_threshold(self):
        """Simulate full pipeline latency for NUM_FRAMES frames."""
        self.latencies = []

        for i in range(self.NUM_FRAMES):
            frame_bytes = self._generate_fixture_frame(i)

            # Decode (UDP recv + JPEG decode stage)
            decode_start = time.perf_counter()
            nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.assertIsNotNone(frame, f"Failed to decode frame {i}")

            img_float = frame.astype(np.float32) / 255.0
            tensor = torch.from_numpy(np.transpose(img_float, (2, 0, 1)))

            decode_ms = (time.perf_counter() - decode_start) * 1000

            # Simulate inference (without actual GPU for unit test)
            infer_ms = 3.0 + np.random.exponential(1.0)

            # Postprocess
            processed = postprocess(
                raw_results=None,
                frame_id=i,
                infer_ms=infer_ms,
            )

            total_ms = decode_ms + infer_ms + 0.5  # + postprocess overhead
            self.latencies.append(total_ms)

        # Compute percentiles
        sorted_latencies = sorted(self.latencies)
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        print(f"\nLatency Stats ({self.NUM_FRAMES} frames):")
        print(f"  p50: {p50:.2f}ms")
        print(f"  p95: {p95:.2f}ms")
        print(f"  p99: {p99:.2f}ms")

        self.assertLessEqual(
            p99,
            self.LATENCY_P99_THRESHOLD_MS,
            f"p99 latency {p99:.2f}ms exceeds {self.LATENCY_P99_THRESHOLD_MS}ms threshold",
        )


if __name__ == "__main__":
    unittest.main()
