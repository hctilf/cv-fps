"""Client visualizer: receives bbox JSON from server, draws overlay via OpenCV, shows FPS HUD."""

import asyncio
import json
import logging
import sys
import time
from typing import List, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CLASS_COLORS = {
    "EnemySoldier": (0, 0, 255),
    "AllySoldier": (0, 255, 0),
    "Weapon": (255, 255, 0),
    "Vehicle": (0, 255, 255),
    "HealthPack": (255, 0, 255),
}


class ClientVisualizer:
    def __init__(self, listen_port: int = 8888, window_name: str = "ESP Lab"):
        self.listen_port = listen_port
        self.window_name = window_name
        self.running = False
        self.detections: List[Dict] = []
        self.last_infer_ms = 0.0
        self.frame_count = 0
        self.fps = 0.0
        self._start_time = time.time()
        self._socket = None

    def start(self):
        self.running = True
        self._socket = asyncio.get_event_loop().run_until_complete(self._setup_udp())
        logger.info(f"Client visualizer listening on port {self.listen_port}")

    async def _setup_udp(self):
        loop = asyncio.get_event_loop()

        class Protocol(asyncio.DatagramProtocol):
            def __init__(inner_self):
                inner_self.vis = self

            def datagram_received(inner_self, data, addr):
                try:
                    msg = json.loads(data.decode("utf-8"))
                    self.detections = msg.get("detections", [])
                    self.last_infer_ms = msg.get("ts_infer_ms", 0)
                except Exception as e:
                    logger.debug(f"JSON parse error: {e}")

        _, protocol = await loop.create_datagram_endpoint(
            lambda: Protocol(), local_addr=("0.0.0.0", self.listen_port)
        )
        return protocol

    def render(self, frame: np.ndarray = None) -> np.ndarray:
        if frame is None:
            frame = np.zeros((640, 640, 3), dtype=np.uint8)

        for det in self.detections:
            cls_name = det["class"]
            bbox = det["bbox"]
            conf = det["conf"]
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))

            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(
                frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        # HUD
        self.frame_count += 1
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed

        hud_lines = [
            f"FPS: {self.fps:.1f}",
            f"Infer: {self.last_infer_ms:.1f}ms",
            f"Detections: {len(self.detections)}",
        ]
        y_offset = 24
        for line in hud_lines:
            cv2.putText(
                frame,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            y_offset += 24

        return frame

    async def run(self):
        self.start()
        logger.info("Press 'q' to quit")
        try:
            while self.running:
                img = self.render()
                cv2.imshow(self.window_name, img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.running = False
                    break
        except KeyboardInterrupt:
            self.running = False
        finally:
            cv2.destroyAllWindows()

    def stop(self):
        self.running = False


async def main():
    visualizer = ClientVisualizer()
    await visualizer.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
