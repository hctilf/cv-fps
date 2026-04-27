"""Per-GPU inference worker: loads YOLO model on a specific GPU, runs inference on incoming tensors."""

import logging
import time
import threading
import zmq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    frame_id: int
    device_id: int
    preprocess_ms: float
    infer_ms: float
    raw_results: List
    client_addr: Tuple[str, int] = field(default_factory=tuple)


class InferenceWorker:
    def __init__(
        self,
        gpu_id: int,
        model_path: str,
        imgsz: int = 640,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        half: bool = True,
        zmq_ctx: Optional[zmq.Context] = None,
        pull_url: str = "",
        push_url: str = "",
    ):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.half = half
        self.running = False
        self._reload_event = threading.Event()
        self._model = None
        self._zmq_ctx = zmq_ctx or zmq.Context()
        self._pull_url = pull_url
        self._push_url = push_url
        self._socket_pull = None
        self._socket_push = None
        self._infer_count = 0
        self._total_infer_ms = 0.0

    def init_model(self):
        logger.info(f"Loading model {self.model_path} on {self.device}")
        self._model = YOLO(self.model_path, task="detect")
        self._model.to(self.device)
        self._model.fuse()
        if self.half:
            self._model.half()
        logger.info(f"Model loaded on {self.device}")

    def start(self, zmq_ctx: Optional[zmq.Context] = None):
        self.running = True
        self.init_model()

        ctx = zmq_ctx or self._zmq_ctx
        self._socket_pull = ctx.socket(zmq.PULL)
        self._socket_pull.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket_pull.bind(self._pull_url)

        self._socket_push = ctx.socket(zmq.PUSH)
        self._socket_push.bind(self._push_url)

        logger.info(
            f"InferenceWorker gpu:{self.gpu_id} started, pulling on {self._pull_url}"
        )
        self._run_loop()

    def _run_loop(self):
        while self.running:
            try:
                msg = self._socket_pull.recv_pyobj(zmq.NOBLOCK)
            except zmq.Again:
                if not self.running:
                    break
                time.sleep(0.0001)
                continue

            if hasattr(msg, "reload"):
                self._reload_model()
                continue

            tensor = msg.tensor
            frame_id = msg.frame_id
            ts_recv_ns = msg.ts_recv_ns
            client_addr = msg.client_addr
            preprocess_ms = msg.preprocess_ms

            infer_start = time.perf_counter_ns()

            try:
                results = self._model(
                    tensor.unsqueeze(0),
                    imgsz=self.imgsz,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    half=self.half,
                    verbose=False,
                )
            except Exception as e:
                logger.error(f"Inference error on gpu:{self.gpu_id}: {e}")
                continue

            infer_ns = time.perf_counter_ns()
            infer_ms = (infer_ns - infer_start) / 1e6

            self._infer_count += 1
            self._total_infer_ms += infer_ms

            result = InferenceResult(
                frame_id=frame_id,
                device_id=self.gpu_id,
                preprocess_ms=preprocess_ms,
                infer_ms=infer_ms,
                raw_results=results,
                client_addr=client_addr,
            )
            try:
                self._socket_push.send_pyobj(result, zmq.NOBLOCK)
            except zmq.Again:
                logger.warning(
                    f"Worker gpu:{self.gpu_id} push queue full, dropping result for frame {frame_id}"
                )

    def _reload_model(self):
        logger.info(f"Hot-reload triggered for gpu:{self.gpu_id}")
        self._reload_event.wait(timeout=5.0)
        self._reload_event.clear()
        self.init_model()
        logger.info(f"Model reloaded on gpu:{self.gpu_id}")

    @property
    def avg_infer_ms(self):
        if self._infer_count == 0:
            return 0.0
        return self._total_infer_ms / self._infer_count

    def stop(self):
        self.running = False
        if self._socket_pull:
            self._socket_pull.close()
        if self._socket_push:
            self._socket_push.close()
