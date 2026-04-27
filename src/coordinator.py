"""Coordinator: spawns inference workers, routes frames round-robin, collects results, forwards to postprocess."""

import logging
import threading
import time
import zmq
from typing import List, Optional

from src.inference_worker import InferenceWorker, InferenceResult

logger = logging.getLogger(__name__)


class Coordinator:
    def __init__(
        self,
        gpu_ids: List[int],
        model_path: str,
        imgsz: int = 640,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        half: bool = True,
        max_depth: int = 10,
        zmq_ctx: Optional[zmq.Context] = None,
    ):
        self.gpu_ids = gpu_ids
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.half = half
        self.max_depth = max_depth
        self.zmq_ctx = zmq_ctx or zmq.Context()
        self._workers: List[InferenceWorker] = []
        self._threads: List[threading.Thread] = []
        self._rr_index = 0
        self._lock = threading.Lock()
        self._drop_count = 0
        self._forward_count = 0
        self._results_socket = None
        self._results_push_url = None
        self._running = False

    def create_zmq_urls(self):
        base_pull = f"tcp://127.0.0.1:{50000 + id(self)}"
        urls = []
        base_push = f"tcp://127.0.0.1:{60000 + id(self)}"
        push_urls = []
        for i, gpu_id in enumerate(self.gpu_ids):
            pull_url = f"tcp://127.0.0.1:{50100 + gpu_id}"
            push_url = f"tcp://127.0.0.1:{60100 + gpu_id}"
            urls.append((pull_url, push_url))
        self._worker_urls = urls
        self._results_push_url = f"tcp://127.0.0.1:{70000 + id(self)}"
        return self._worker_urls

    def start(self, results_push_url: str = ""):
        self._running = True
        if not hasattr(self, "_worker_urls"):
            self.create_zmq_urls()

        if results_push_url:
            self._results_push_url = results_push_url

        # Start result collector
        self._results_socket = self.zmq_ctx.socket(zmq.PULL)
        self._results_socket.bind(self._results_push_url)
        collector_thread = threading.Thread(target=self._collect_results, daemon=True)
        collector_thread.start()

        # Start workers
        for i, gpu_id in enumerate(self.gpu_ids):
            pull_url, push_url = self._worker_urls[i]
            worker = InferenceWorker(
                gpu_id=gpu_id,
                model_path=self.model_path,
                imgsz=self.imgsz,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                half=self.half,
                zmq_ctx=self.zmq_ctx,
                pull_url=pull_url,
                push_url=push_url,
            )
            self._workers.append(worker)
            t = threading.Thread(target=worker.start, args=(self.zmq_ctx,), daemon=True)
            t.start()
            self._threads.append(t)
            logger.info(f"Started worker for gpu:{gpu_id}")

        logger.info(f"Coordinator started with {len(self.gpu_ids)} GPUs")

    def _collect_results(self):
        """Collect results from all workers and forward to postprocess."""
        while self._running:
            try:
                result = self._results_socket.recv_pyobj(zmq.NOBLOCK)
            except zmq.Again:
                if not self._running:
                    break
                time.sleep(0.0001)
                continue

            self._forward_count += 1
            try:
                self._results_socket.send_pyobj(result, zmq.NOBLOCK)
            except zmq.Again:
                self._drop_count += 1
                logger.warning(f"Result queue full, dropping frame {result.frame_id}")

    def dispatch(self, packet):
        """Round-robin dispatch a frame packet to a GPU worker."""
        if not self._workers:
            return
        with self._lock:
            worker_idx = self._rr_index % len(self._workers)
            self._rr_index += 1

        worker = self._workers[worker_idx]
        push_socket = worker._socket_push
        if push_socket is None:
            return

        try:
            push_socket.send_pyobj(packet, zmq.NOBLOCK)
        except zmq.Again:
            self._drop_count += 1
            logger.warning(
                f"Worker gpu:{worker.gpu_id} queue full, dropping frame {packet.frame_id}"
            )

    def trigger_reload(self):
        for worker in self._workers:
            worker._reload_event.set()
            try:
                worker._socket_push.send_pyobj({"reload": True}, zmq.NOBLOCK)
            except Exception:
                pass

    @property
    def stats(self):
        return {
            "drop_count": self._drop_count,
            "forward_count": self._forward_count,
            "workers": [
                {
                    "gpu_id": w.gpu_id,
                    "infer_count": w._infer_count,
                    "avg_infer_ms": w.avg_infer_ms,
                }
                for w in self._workers
            ],
        }

    def stop(self):
        self._running = False
        for worker in self._workers:
            worker.stop()
        if self._results_socket:
            self._results_socket.close()
        self.zmq_ctx.term()
        logger.info("Coordinator stopped")
