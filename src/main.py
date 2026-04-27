"""Main entry point: starts the full inference pipeline."""

import argparse
import asyncio
import json
import logging
import signal
import sys
import threading
import time

import yaml
import zmq
import zmq.asyncio

from src.udp_receiver import UDPReceiver
from src.coordinator import Coordinator
from src.postprocess import PostprocessSender, postprocess
from src.api import run_api, coordinator as api_coordinator

logger = logging.getLogger(__name__)


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Pipeline:
    def __init__(self, config: dict):
        self.config = config
        self.zmq_ctx = zmq.Context()
        self.zmq_async_ctx = zmq.asyncio.Context()

        self.server = config["server"]
        self.model_cfg = config["model"]
        self.gpus = config["gpus"]
        self.queues = config["queues"]
        self.api_cfg = config["api"]

        # ZMQ URLs
        self.push_url = f"tcp://127.0.0.1:{55000}"
        self.results_url = f"tcp://127.0.0.1:{55100}"
        self.worker_urls = None

        self.receiver = UDPReceiver(
            listen_host=self.server["listen_host"],
            listen_port=self.server["listen_port"],
            zmq_ctx=self.zmq_async_ctx,
            push_url=self.push_url,
        )
        self.coordinator = Coordinator(
            gpu_ids=self.gpus,
            model_path=self.model_cfg["path"],
            imgsz=self.model_cfg["imgsz"],
            conf_threshold=self.model_cfg["conf_threshold"],
            iou_threshold=self.model_cfg["iou_threshold"],
            half=self.model_cfg["half"],
            max_depth=self.queues["max_depth"],
            zmq_ctx=self.zmq_ctx,
        )
        self.sender = PostprocessSender(conf_threshold=self.model_cfg["conf_threshold"])

        self._running = False
        self._api_thread = None
        self._receiver_task = None

    def start(self):
        self._running = True
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Start API server in background thread
        self._api_thread = threading.Thread(
            target=run_api,
            args=(self.api_cfg["host"], self.api_cfg["port"]),
            daemon=True,
        )
        self._api_thread.start()
        logger.info(f"API server on {self.api_cfg['host']}:{self.api_cfg['port']}")

        # Start coordinator
        self.coordinator.create_zmq_urls()
        self.coordinator.start(results_push_url=self.results_url)

        # Wire coordinator dispatch to receiver
        self.receiver._socket = None  # will be set by start()
        self._receiver_task = asyncio.ensure_future(self._run_pipeline())

        logger.info("Pipeline started")

    async def _run_pipeline(self):
        """Main async loop: receiver pushes to ZMQ, coordinator picks it up."""
        # Start receiver
        receiver_task = asyncio.ensure_future(self.receiver.start())

        # Start result collector from coordinator -> postprocess sender
        collector_task = asyncio.ensure_future(self._collect_and_send())

        try:
            await asyncio.gather(receiver_task, collector_task)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _collect_and_send(self):
        """Pull results from coordinator and send to client."""
        pull_socket = self.zmq_ctx.socket(zmq.PULL)
        pull_socket.connect(self.results_url)

        while self._running:
            try:
                result = pull_socket.recv_pyobj(zmq.NOBLOCK)
            except zmq.Again:
                await asyncio.sleep(0.001)
                continue

            self.sender.set_client_addr(result.client_addr)
            if not self.sender._socket:
                self.sender.start()

            processed = postprocess(
                result.raw_results,
                frame_id=result.frame_id,
                infer_ms=result.infer_ms,
                conf_threshold=self.model_cfg["conf_threshold"],
            )
            self.sender.send(processed, result.client_addr)

        pull_socket.close()

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    async def stop(self):
        self._running = False
        await self.receiver.stop()
        self.coordinator.stop()
        self.sender.stop()
        self.zmq_ctx.term()
        self.zmq_async_ctx.term()
        logger.info("Pipeline stopped")


def main():
    parser = argparse.ArgumentParser(description="ESP Lab Inference Server")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Config file path"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Replay mode from fixtures"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, config.get("logging", {}).get("level", "INFO")),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    pipeline = Pipeline(config)
    pipeline.start()

    if args.dry_run:
        logger.info("Dry-run mode: replaying from fixtures")
        # Fixture replay logic would go here
    else:
        logger.info(
            "Live mode: waiting for UDP frames on port %d",
            config["server"]["listen_port"],
        )

    try:
        while pipeline._running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        asyncio.run(pipeline.stop())


if __name__ == "__main__":
    main()
