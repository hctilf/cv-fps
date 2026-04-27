"""Client visualizer: receives bbox JSON from server, draws overlay on a target window."""

import asyncio
import json
import logging
import time
import ctypes
from typing import List, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Windows constants
# ---------------------------------------------------------------------------
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_POPUP = 0x800000
LWA_COLORKEY = 0x00000001
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01

# Color key: pixels matching this RGB are fully transparent
COLOR_KEY = (0x80, 0x80, 0x80)


def _win_rgb(r: int, g: int, b: int) -> int:
    """Windows RGB macro: R | (G << 8) | (B << 16)."""
    return r | (g << 8) | (b << 16)


# ---------------------------------------------------------------------------
# GDI helpers
# ---------------------------------------------------------------------------

# Register window class once at module load
_registered_class = False


def _ensure_window_class():
    """Register the overlay window class exactly once."""
    global _registered_class
    if _registered_class:
        return
    import win32gui
    import win32con

    wclass = win32gui.WNDCLASS()
    wclass.lpfnWndProc = _wnd_proc
    wclass.lpszClassName = "ESPLayerOverlay"
    wclass.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
    wclass.hInstance = win32gui.GetModuleHandle(None)
    win32gui.RegisterClass(wclass)
    _registered_class = True


class _GdiObject:
    """RAII wrapper for GDI objects."""

    def __init__(self, create_fn, delete_fn):
        self._hobj = create_fn()
        self._delete_fn = delete_fn

    def get(self):
        return self._hobj

    def delete(self):
        if self._hobj:
            self._delete_fn(self._hobj)
            self._hobj = 0

    def __del__(self):
        self.delete()


# ---------------------------------------------------------------------------
# Window finding
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Window creation
# ---------------------------------------------------------------------------


def _create_overlay_window(left, top, right, bottom):
    """Create a layered popup window positioned over the target."""
    import win32gui
    import win32con

    width = right - left
    height = bottom - top

    _ensure_window_class()

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
        hwnd, _win_rgb(*COLOR_KEY), 0, LWA_COLORKEY
    )

    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
    return hwnd


# ---------------------------------------------------------------------------
# Window procedure
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


# ---------------------------------------------------------------------------
# Render: uses UpdateLayeredWindow (correct API for layered windows)
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "EnemySoldier": (0, 0, 255),
    "AllySoldier": (0, 255, 0),
    "Weapon": (255, 255, 0),
    "Vehicle": (0, 255, 255),
    "HealthPack": (255, 0, 255),
}


def _render_overlay(hwnd):
    """Render detections using UpdateLayeredWindow."""
    import win32gui
    import win32con
    import win32ui

    state = _overlay_state
    w = state["width"]
    h = state["height"]
    if w <= 0 or h <= 0:
        return

    try:
        # Create memory DC and bitmap
        hdc_screen = win32gui.GetDC(0)  # desktop DC
        hdc_mem = win32ui.CreateCompatibleDC(hdc_screen)
        hbmp = win32ui.CreateCompatibleBitmap(hdc_screen, w, h)
        hdc_mem.SelectObject(hbmp)

        # Fill with color key (transparent)
        hdc_mem.FillSolidRect(0, 0, w, h, _win_rgb(*COLOR_KEY))

        detections = state.get("detections", [])
        for det in detections:
            cls_name = det.get("class", "unknown")
            bbox = det.get("bbox", [0, 0, 0, 0])
            conf = det.get("conf", 0.0)
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))

            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Draw rectangle
            hpen = win32gui.CreatePen(win32con.PS_SOLID, 2, _win_rgb(*color))
            old_pen = hdc_mem.SelectObject(hpen)
            hdc_mem.Rectangle([x1, y1, x2, y2])
            hdc_mem.SelectObject(old_pen)
            win32gui.DeleteObject(hpen)

            # Label
            label = f"{cls_name} {conf:.2f}"
            text_w, text_h = hdc_mem.GetTextExtent(label)
            lx, ly = x1, y1 - text_h - 6
            if ly < 0:
                ly = y1 + 4
            hdc_mem.FillSolidRect(lx, ly, text_w + 6, text_h + 4, _win_rgb(*color))
            hdc_mem.SetTextColor((0, 0, 0))
            hdc_mem.SetBkMode(win32con.TRANSPARENT)
            hdc_mem.TextOut(lx + 3, ly + 2, label)

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
        y_off = 20
        for line in hud_lines:
            tw, _ = hdc_mem.GetTextExtent(line)
            hdc_mem.FillSolidRect(6, y_off - 16, tw + 10, 20, _win_rgb(0, 0, 0))
            hdc_mem.SetTextColor((255, 255, 255))
            hdc_mem.SetBkMode(win32con.TRANSPARENT)
            hdc_mem.TextOut(10, y_off, line)
            y_off += 22

        # Get window position for UpdateLayeredWindow
        rect = win32gui.GetWindowRect(hwnd)
        pt = ctypes.wintypes.POINT(rect[0], rect[1])
        size = ctypes.wintypes.SIZE(w, h)

        # BLENDFUNCTION for color-key transparency
        blend = ctypes.wintypes.BLENDFUNCTION()
        blend.BlendOp = AC_SRC_OVER
        blend.BlendFlags = 0
        blend.SourceConstantAlpha = 255
        blend.AlphaFormat = AC_SRC_ALPHA

        # UpdateLayeredWindow replaces BitBlt for layered windows
        ctypes.windll.user32.UpdateLayeredWindow(
            hwnd,
            hdc_screen,
            ctypes.byref(pt),
            ctypes.byref(size),
            hdc_mem.GetSafeHdc(),
            ctypes.wintypes.POINT(0, 0),
            _win_rgb(*COLOR_KEY),
            ctypes.byref(blend),
            win32con.ULW_COLORKEY,
        )

        hdc_mem.DeleteDC()
        hbmp.DeleteObject()
        win32gui.ReleaseDC(0, hdc_screen)
    except Exception as e:
        logger.debug(f"Render error: {e}")


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------


class ClientVisualizer:
    def __init__(self, listen_port: int = 8888, process_name: str = ""):
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
