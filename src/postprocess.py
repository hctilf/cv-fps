"""Postprocess: maps YOLO class indices, filters by confidence, serializes to JSON, sends via UDP."""

import json
import logging
import socket
import time
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

CLASS_MAP: Dict[int, str] = {
    0: "EnemySoldier",
    1: "AllySoldier",
    2: "Weapon",
    3: "Vehicle",
    4: "HealthPack",
}


def postprocess(
    raw_results, frame_id: int, infer_ms: float, conf_threshold: float = 0.4
) -> dict:
    """Convert YOLO raw results to filtered detection JSON."""
    detections = []
    if not raw_results or len(raw_results) == 0:
        return {
            "frame_id": frame_id,
            "ts_infer_ms": round(infer_ms, 2),
            "detections": [],
        }

    result = raw_results[0]
    if result.boxes is None:
        return {
            "frame_id": frame_id,
            "ts_infer_ms": round(infer_ms, 2),
            "detections": [],
        }

    boxes = result.boxes
    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        if conf < conf_threshold:
            continue
        cls_id = int(boxes.cls[i])
        cls_name = CLASS_MAP.get(cls_id, f"class_{cls_id}")
        xyxy = boxes.xyxy[i].cpu().tolist()
        detections.append(
            {
                "class": cls_name,
                "bbox": [round(v, 1) for v in xyxy],
                "conf": round(conf, 3),
            }
        )

    return {
        "frame_id": frame_id,
        "ts_infer_ms": round(infer_ms, 2),
        "detections": detections,
    }


class PostprocessSender:
    def __init__(self, conf_threshold: float = 0.4):
        self.conf_threshold = conf_threshold
        self._socket = None
        self._send_count = 0
        self._drop_count = 0
        self._client_addr: Tuple[str, int] = None

    def set_client_addr(self, addr: Tuple[str, int]):
        self._client_addr = addr

    def start(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.IPPROTO_UDP, socket.UDP_CHECKSUM, 0)
        logger.info("PostprocessSender started")

    def send(self, result_dict: dict, client_addr: Tuple[str, int] = None):
        if self._socket is None:
            return

        target = client_addr or self._client_addr
        if target is None:
            logger.warning("No client address set, dropping result")
            self._drop_count += 1
            return

        try:
            payload = json.dumps(result_dict).encode("utf-8")
            self._socket.sendto(payload, target)
            self._send_count += 1
        except Exception as e:
            self._drop_count += 1
            logger.debug(f"UDP send failed: {e}")

    @property
    def stats(self):
        return {
            "send_count": self._send_count,
            "drop_count": self._drop_count,
            "client_addr": self._client_addr,
        }

    def stop(self):
        if self._socket:
            self._socket.close()
