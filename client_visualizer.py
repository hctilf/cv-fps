"""Client visualizer: receives bbox JSON from server, draws overlay on a target window."""

import asyncio
import json
import logging
import time
import ctypes
from typing import List, Dict

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Windows constants
# ---------------------------------------------------------------------------
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_POPUP = 0x800000
LWA_COLORKEY = 0x00000001
LWA_ALPHA = 0x00000002

# Color key: pixels matching this RGB are fully transparent
COLOR_KEY = (0x80, 0x80, 0x80)  # medium gray

CLASS_COLORS = {
    "EnemySoldier": (0, 0, 255),
    "AllySoldier": (0, 255, 0),
    "Weapon": (255, 255, 0),
    "Vehicle": (0, 255, 255),
    "HealthPack": (255, 0, 255),
}


def _find_window_by_process(process_name: str):
    """Find the first top-level window whose process name matches."""
    import win32gui
    import win32process

    results = []

    def enum_cb(hwnd, acc):
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            handle = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid)
            if handle:
                try:
                    buf = ctypes.create_unicode_buffer(260)
                    ctypes.windll.psapi.GetProcessImageFileNameW(handle, buf, 260)
                    ctypes.windll.kernel32.CloseHandle(handle)
                    if process_name.lower() in buf.value.lower():
                        acc.append((hwnd, win32gui.GetWindowRect(hwnd)))
                except Exception:
                    ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass
        return True

    win32gui.EnumWindows(enum_cb, results)
    return results


def _create_overlay_window(
    left: int,
    top: int,
    right: int,
    bottom: int,
):
    """Create a layered popup window positioned over the target."""
    import win32gui
    import win32con

    width = right - left
    height = bottom - top

    wclass = win32gui.WNDCLASS()
    wclass.lpfnWndProc = _wnd_proc
    wclass.lpszClassName = "ESPLayerOverlay"
    wclass.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
    wclass.hInstance = win32gui.GetModuleHandle(None)
    class_atom = win32gui.RegisterClass(wclass)

    hwnd = win32gui.CreateWindowEx(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST,
        "ESPLayerOverlay",
        "ESP Overlay",
        WS_POPUP,
        left,
        top,
        width,
        height,
        None,
        None,
        None,
        None,
    )

    ctypes.windll.user32.SetLayeredWindowAttributes(
        hwnd, ctypes.rgb(*COLOR_KEY), 0, LWA_COLORKEY
    )

    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
    return hwnd


# ---------------------------------------------------------------------------
# Window procedure for the overlay
# ---------------------------------------------------------------------------

_overlay_state = {
    "hwnd": None,
    "width": 0,
    "height": 0,
    "detections": [],
    "last_infer_ms": 0.0,
    "fps": 0.0,
    "frame_count": 0,
    "_start_time": None,
}


def _wnd_proc(hwnd, msg, wParam, lParam):
    import win32gui
    import win32con
    import win32api

    WM_PAINT = 0x000F
    WM_DESTROY = 0x0002
    WM_ERASEBKGND = 0x0014
    WM_SIZE = 0x0005

    if msg == WM_PAINT:
        _render_overlay(hwnd)
        return 0

    if msg == WM_SIZE:
        w = win32api.LOWORD(lParam)
        h = win32api.HIWORD(lParam)
        if w > 0 and h > 0:
            _overlay_state["width"] = w
            _overlay_state["height"] = h
        return 0

    if msg == WM_DESTROY:
        return 0

    if msg == WM_ERASEBKGND:
        return 1

    return win32gui.DefWindowProc(hwnd, msg, wParam, lParam)


def _render_overlay(hwnd):
    """Render detections onto the overlay window using GDI."""
    import win32gui
    import win32ui
    import win32con

    state = _overlay_state
    w = state["width"]
    h = state["height"]
    if w <= 0 or h <= 0:
        return

    hdc = win32gui.GetDC(hwnd)
    if not hdc:
        return

    try:
        mem_dc = win32ui.CreateCompatibleDC(hdc)
        bitmap = win32ui.CreateCompatibleBitmap(hdc, w, h)
        mem_dc.SelectObject(bitmap)

        mem_dc.FillSolidRect(0, 0, w, h, *COLOR_KEY)

        detections = state.get("detections", [])
        for det in detections:
            cls_name = det.get("class", "unknown")
            bbox = det.get("bbox", [0, 0, 0, 0])
            conf = det.get("conf", 0.0)
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))

            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            pen = win32ui.CreatePen(win32con.PS_SOLID, 2, color)
            old_pen = mem_dc.SelectObject(pen)
            mem_dc.Rectangle([x1, y1, x2, y2])
            mem_dc.SelectObject(old_pen)
            mem_dc.DeleteObject(pen)

            label = f"{cls_name} {conf:.2f}"
            text_w, text_h = mem_dc.GetTextExtent(label)
            label_x, label_y = x1, y1 - text_h - 6
            if label_y < 0:
                label_y = y1 + 4
            mem_dc.FillSolidRect(label_x, label_y, text_w + 6, text_h + 4, *color)
            mem_dc.SetTextColor((0, 0, 0))
            mem_dc.SetBkMode(win32con.TRANSPARENT)
            mem_dc.TextOut(label_x + 3, label_y + 2, label)

        # HUD
        state["frame_count"] += 1
        elapsed = time.time() - (state["_start_time"] or time.time())
        if elapsed > 0:
            state["fps"] = state["frame_count"] / elapsed

        hud_lines = [
            f"FPS: {state['fps']:.1f}",
            f"Infer: {state['last_infer_ms']:.1f}ms",
            f"Detections: {len(detections)}",
        ]
        y_offset = 20
        for line in hud_lines:
            text_w, _ = mem_dc.GetTextExtent(line)
            mem_dc.FillSolidRect(6, y_offset - 16, text_w + 10, 20, (0, 0, 0))
            mem_dc.SetTextColor((255, 255, 255))
            mem_dc.SetBkMode(win32con.TRANSPARENT)
            mem_dc.TextOut(10, y_offset, line)
            y_offset += 22

        win32gui.BitBlt(
            hdc,
            0,
            0,
            w,
            h,
            mem_dc.GetSafeHdc(),
            0,
            0,
            win32con.SRCCOPY,
        )

        mem_dc.DeleteDC()
        bitmap.DeleteObject()
        win32gui.ReleaseDC(hwnd, hdc)
    except Exception as e:
        logger.debug(f"Render error: {e}")
        try:
            win32gui.ReleaseDC(hwnd, hdc)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------


class ClientVisualizer:
    def __init__(
        self,
        listen_port: int = 8888,
        process_name: str = "",
    ):
        self.listen_port = listen_port
        self.process_name = process_name
        self.running = False
        self.detections: List[Dict] = []
        self.last_infer_ms = 0.0
        self._socket = None
        self._hwnd = None
        self._target_rect = None
        self._last_update = 0.0

    async def start(self):
        self.running = True
        self._socket = await self._setup_udp()
        if self.process_name:
            self._find_and_create_overlay()
        logger.info(
            f"Client visualizer listening on port {self.listen_port}"
            + (
                f"  target={self.process_name}"
                if self.process_name
                else "  (standalone mode)"
            )
        )

    def _find_and_create_overlay(self):
        matches = _find_window_by_process(self.process_name)
        if not matches:
            logger.warning(
                f"No window found for '{self.process_name}'. "
                f"Overlay will be created when window appears."
            )
            return
        hwnd, rect = matches[0]
        self._target_rect = rect
        self._create_overlay_for_rect(rect)
        logger.info(f"Overlay created on window: {hwnd}  rect={rect}")

    def _create_overlay_for_rect(self, rect):
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return

        import win32gui

        if self._hwnd is not None:
            try:
                win32gui.DestroyWindow(self._hwnd)
            except Exception:
                pass
            self._hwnd = None

        self._hwnd = _create_overlay_window(left, top, right, bottom)
        _overlay_state["hwnd"] = self._hwnd
        _overlay_state["width"] = width
        _overlay_state["height"] = height
        _overlay_state["_start_time"] = time.time()
        if self._hwnd:
            logger.info(
                f"Overlay window {self._hwnd} at ({left},{top}) {width}x{height}"
            )

    async def _setup_udp(self):
        loop = asyncio.get_running_loop()

        class Protocol(asyncio.DatagramProtocol):
            def __init__(inner_self):
                inner_self.vis = self

            def datagram_received(inner_self, data, addr):
                try:
                    msg = json.loads(data.decode("utf-8"))
                    self.detections = msg.get("detections", [])
                    self.last_infer_ms = msg.get("ts_infer_ms", 0)
                    _overlay_state["detections"] = self.detections
                    _overlay_state["last_infer_ms"] = self.last_infer_ms
                    if self._hwnd is not None:
                        try:
                            import win32con
                            import win32gui

                            win32gui.RedrawWindow(
                                self._hwnd,
                                None,
                                None,
                                win32con.RDW_INVALIDATE
                                | win32con.RDW_UPDATENOW
                                | win32con.RDW_NOCHILDREN,
                            )
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"JSON parse error: {e}")

        _, protocol = await loop.create_datagram_endpoint(
            lambda: Protocol(), local_addr=("0.0.0.0", self.listen_port)
        )
        return protocol

    def _update_overlay_position(self):
        if not self.process_name or self._hwnd is None:
            return
        matches = _find_window_by_process(self.process_name)
        if not matches:
            return
        _, rect = matches[0]
        if rect == self._target_rect:
            return
        self._target_rect = rect
        self._create_overlay_for_rect(rect)

    async def run(self):
        await self.start()
        logger.info("Press 'q' to quit")
        try:
            while self.running:
                now = time.time()
                if now - self._last_update > 1.0:
                    self._last_update = now
                    self._update_overlay_position()
                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            self.running = False
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self._hwnd is not None:
            try:
                import win32gui

                win32gui.DestroyWindow(self._hwnd)
            except Exception:
                pass
            self._hwnd = None
        logger.info("Visualizer stopped")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="ESP Lab Client Visualizer")
    parser.add_argument(
        "--port", type=int, default=8888, help="UDP listen port (default: 8888)"
    )
    parser.add_argument(
        "--process",
        type=str,
        default="",
        help="Process name to overlay on (e.g. 'Game.exe'). Leave empty for standalone window mode.",
    )
    args = parser.parse_args()

    visualizer = ClientVisualizer(listen_port=args.port, process_name=args.process)
    await visualizer.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
