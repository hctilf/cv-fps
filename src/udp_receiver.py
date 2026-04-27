"""Asyncio UDP receiver: receives MJPEG frames, decodes JPEG, converts to pinned tensor, pushes via ZMQ inproc."""

import asyncio
import logging
import time
import zmq
import zmq.asyncio
import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

JPEG_MAGIC = b"\xff\xd8"


@dataclass
class FramePacket:
    frame_id: int
    ts_recv_ns: int
    tensor: torch.Tensor
    client_addr: Tuple[str, int] = field(default_factory=tuple)
    preprocess_ms: float = 0.0


class UDPReceiver:
    def __init__(
        self,
        listen_host: str,
        listen_port: int,
        zmq_ctx: zmq.asyncio.Context,
        push_url: str,
        max_batch: int = 1024,
    ):
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.zmq_ctx = zmq_ctx
        self.push_url = push_url
        self.max_batch = max_batch
        self.frame_id = 0
        self.client_addr: Optional[Tuple[str, int]] = None
        self.running = False
        self._socket: Optional[zmq.asyncio.Socket] = None
        self._recv_count = 0
        self._drop_count = 0

    async def start(self):
        self.running = True
        self._socket = self.zmq_ctx.socket(zmq.asyncio.PUSH)
        self._socket.setsockopt_int(zmq.SNDHWM, self.max_batch)
        self._socket.connect(self.push_url)
        logger.info(f"UDP receiver listening on {self.listen_host}:{self.listen_port}")
        await self._receive_loop()

    async def _receive_loop(self):
        loop = asyncio.get_event_loop()
        reader = await loop.create_datagram_endpoint(
            lambda: _UDPReceiverProtocol(self),
            local_addr=(self.listen_host, self.listen_port),
        )
        _, protocol = reader
        try:
            while self.running:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            protocol.close()

    async def _handle_datagram(self, data: bytes, addr: Tuple[str, int]):
        if len(data) < 2:
            return
        if data[:2] != JPEG_MAGIC:
            self._drop_count += 1
            return

        if self.client_addr is None:
            self.client_addr = addr
            logger.info(f"Client discovered: {addr[0]}:{addr[1]}")

        start_ns = time.time_ns()

        try:
            nparr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                self._drop_count += 1
                return

            img_float = frame.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_float, (2, 0, 1))
            tensor = torch.from_numpy(img_transposed).pin_memory()

            preprocess_ms = (time.time_ns() - start_ns) / 1e6
            self.frame_id += 1
            self._recv_count += 1

            packet = FramePacket(
                frame_id=self.frame_id,
                ts_recv_ns=start_ns,
                tensor=tensor,
                client_addr=self.client_addr,
                preprocess_ms=preprocess_ms,
            )
            await self._socket.send_pyobj(packet)
        except Exception as e:
            self._drop_count += 1
            logger.debug(f"Dropped frame (decode error): {e}")

    @property
    def stats(self):
        return {
            "recv_count": self._recv_count,
            "drop_count": self._drop_count,
            "client_addr": self.client_addr,
        }

    async def stop(self):
        self.running = False
        if self._socket:
            await self._socket.close()


class _UDPReceiverProtocol(asyncio.DatagramProtocol):
    def __init__(self, receiver: UDPReceiver):
        self.receiver = receiver

    def datagram_received(self, data, addr):
        asyncio.ensure_future(self.receiver._handle_datagram(data, addr))

    def close(self):
        self.receiver.running = False
