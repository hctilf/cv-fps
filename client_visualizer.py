"""Client visualizer: receives bbox JSON from server, draws overlay on a target window (Linux)."""

import asyncio
import json
import logging
import math
import time
from typing import List, Dict

import cv2
import glfw
import imgui
import mss
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
from OpenGL import GL as gl

logger = logging.getLogger(__name__)


def capture_window_linux(window_title: str):
    """Linux implementation using ewmh and Xlib."""
    try:
        from ewmh import EWMH
        from Xlib.display import Display

        ewmh = EWMH()
        display = Display()

        windows = [win for win in ewmh.getClientList()
                   if window_title in (ewmh.getWmName(win) or b'').decode('utf-8', errors='ignore')]

        if not windows:
            logger.debug(f"Window '{window_title}' not found")
            return None, None

        window = windows[0]
        geom = window.get_geometry()
        attrs = window.get_attributes()

        if attrs.map_state != 2:
            logger.debug(f"Window '{window_title}' is not visible")
            return None, None

        win_x, win_y = geom.x, geom.y
        win_width, win_height = geom.width, geom.height

        try:
            frame_extents = ewmh.getFrameExtents(window)
            left = win_x - frame_extents['left']
            top = win_y - frame_extents['top']
            width = win_width + frame_extents['left'] + frame_extents['right']
            height = win_height + frame_extents['top'] + frame_extents['bottom']
        except Exception:
            left, top = win_x, win_y
            width, height = win_width, win_height

        monitor = {
            "top": top,
            "left": left,
            "width": width,
            "height": height,
        }

        with mss.mss() as sct:
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img, monitor
    except Exception as e:
        logger.debug(f"Capture error: {e}")
        return None, None


# --- OVERLAY DRAWING LOGIC ---

detections: List[Dict] = []
_overlay_window = None
_model = None


def _esp(draw_list):
    """Draw ESP-style bounding boxes on overlay."""
    for det in detections:
        cls_name = det.get("class", "unknown")
        bbox = det.get("bbox", [0, 0, 0, 0])
        conf = det.get("conf", 0.0)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        label = f"{cls_name} {conf:.2f}"

        color = _get_class_color(cls_name)
        r, g, b = color
        draw_list.add_rect(x1, y1, x2, y2,
                           imgui.get_color_u32_rgba(r / 255.0, g / 255.0, b / 255.0, 1.0),
                           thickness=2)
        draw_list.add_text(x1, y1 - 15,
                           imgui.get_color_u32_rgba(1, 1, 1, 1), label)


def _get_class_color(cls_name: str):
    """Return (R, G, B) color for a class name."""
    colors = {
        "EnemySoldier": (0, 0, 255),
        "AllySoldier": (0, 255, 0),
        "Weapon": (255, 255, 0),
        "Vehicle": (0, 255, 255),
        "HealthPack": (255, 0, 255),
        "person": (0, 255, 0),
    }
    return colors.get(cls_name, (200, 200, 200))


# --- GLFW / IMGUI INITIALIZATION ---

def _init_overlay_window(width: int, height: int):
    global _overlay_window
    if not glfw.init():
        raise Exception("Failed to initialize GLFW")

    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)
    glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)

    _overlay_window = glfw.create_window(width, height, "ESP Overlay", None, None)
    if not _overlay_window:
        glfw.terminate()
        raise Exception("Could not create overlay window")

    glfw.set_window_pos(_overlay_window, 0, 0, width, height)
    glfw.make_context_current(_overlay_window)
    imgui.create_context()
    impl = GlfwRenderer(_overlay_window)
    glfw.set_input_mode(_overlay_window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

    return impl


# --- MAIN LOOP ---

def _run_detection_loop(window_title: str):
    """Generator that captures, runs inference, yields detections."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolo11n.pt").to("cpu")

    class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                   "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                   "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                   "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                   "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                   "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                   "teddy bear", "hair drier", "toothbrush"]

    while True:
        global detections
        detections.clear()

        img, monitor_info = capture_window_linux(window_title)
        if img is None or monitor_info is None:
            time.sleep(0.1)
            yield
            continue

        left, top = monitor_info['left'], monitor_info['top']
        width, height = monitor_info['width'], monitor_info['height']

        glfw.set_window_pos(_overlay_window, left, top)

        results = _model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = class_names[cls]
                detections.append({
                    "class": class_name,
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                })

        yield


# --- MAIN VISUALIZER ---

class ClientVisualizer:
    def __init__(self, listen_port: int = 8888, window_title: str = "", standalone: bool = True):
        self.listen_port = listen_port
        self.window_title = window_title
        self.standalone = standalone
        self.running = False
        self.detections: List[Dict] = []
        self._socket = None
        self._impl = None
        self._width = 0
        self._height = 0
        self._last_update = 0.0
        self._start_time = time.time()
        self._frame_count = 0
        self._fps = 0.0

    async def start(self):
        self.running = True
        self._socket = await self._setup_udp()

        if not self.standalone and self.window_title:
            self._find_and_create_overlay()
        else:
            self._create_standalone_overlay()

        logger.info(
            f"Client visualizer listening on port {self.listen_port}"
            + (f"  target={self.window_title}" if self.window_title and not self.standalone else "  (standalone mode)")
        )

    def _find_and_create_overlay(self):
        img, monitor_info = capture_window_linux(self.window_title)
        if img is None or monitor_info is None:
            logger.warning(f"No window found for '{self.window_title}'. Overlay will be created when window appears.")
            return
        left, top = monitor_info['left'], monitor_info['top']
        width, height = monitor_info['width'], monitor_info['height']
        self._create_overlay_at(left, top, width, height)

    def _create_standalone_overlay(self):
        self._create_overlay_at(100, 100, 1280, 720)

    def _create_overlay_at(self, x, y, width, height):
        global _overlay_window
        self._impl = _init_overlay_window(width, height)
        glfw.set_window_pos(_overlay_window, x, y)
        self._width = width
        self._height = height
        self._start_time = time.time()
        logger.info(f"Overlay at ({x},{y}) {width}x{height}")

    async def _setup_udp(self):
        loop = asyncio.get_running_loop()

        class Protocol(asyncio.DatagramProtocol):
            def __init__(inner_self):
                inner_self.vis = self

            def datagram_received(inner_self, data, addr):
                try:
                    msg = json.loads(data.decode("utf-8"))
                    self.detections = msg.get("detections", [])
                    if self._impl is not None:
                        try:
                            import imgui
                            imgui.reload()
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"JSON parse error: {e}")

        _, protocol = await loop.create_datagram_endpoint(
            lambda: Protocol(), local_addr=("0.0.0.0", self.listen_port)
        )
        return protocol

    async def run(self):
        await self.start()
        logger.info("Press 'q' to quit")

        if not self.standalone and self.window_title:
            detection_gen = _run_detection_loop(self.window_title)
        else:
            detection_gen = None

        try:
            while not glfw.window_should_close(_overlay_window if self._impl else None):
                glfw.poll_events()
                if self._impl:
                    self._impl.process_inputs()
                    imgui.new_frame()

                    imgui.set_next_window_size(self._width, self._height)
                    imgui.set_next_window_position(0, 0)
                    imgui.begin("overlay",
                                flags=imgui.WINDOW_NO_TITLE_BAR
                                      | imgui.WINDOW_NO_RESIZE
                                      | imgui.WINDOW_NO_SCROLLBAR
                                      | imgui.WINDOW_NO_COLLAPSE
                                      | imgui.WINDOW_NO_BACKGROUND)

                    draw_list = imgui.get_window_draw_list()

                    if detection_gen:
                        next(detection_gen)
                    else:
                        global detections
                        detections = self.detections

                    _esp(draw_list)

                    # HUD
                    self._frame_count += 1
                    elapsed = time.time() - self._start_time
                    if elapsed > 0:
                        self._fps = self._frame_count / elapsed

                    hud_lines = [
                        f"FPS: {self._fps:.1f}",
                        f"Detections: {len(detections)}",
                    ]
                    y_off = 20
                    for line in hud_lines:
                        # ImGui text drawing
                        draw_list.add_text(0, y_off, imgui.get_color_u32_rgba(1, 1, 1, 1), line)
                        y_off += 22

                    imgui.end()
                    imgui.end_frame()

                    gl.glClearColor(0, 0, 0, 0)
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                    imgui.render()
                    self._impl.render(imgui.get_draw_data())
                    glfw.swap_buffers(_overlay_window)

                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            self.running = False
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self._impl:
            self._impl.shutdown()
            glfw.terminate()
        logger.info("Visualizer stopped")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="ESP Lab Client Visualizer (Linux)")
    parser.add_argument("--port", type=int, default=8888, help="UDP listen port (default: 8888)")
    parser.add_argument("--window", type=str, default="", help="Window title to overlay on")
    parser.add_argument("--standalone", action="store_true", help="Run as standalone overlay window")
    args = parser.parse_args()

    visualizer = ClientVisualizer(
        listen_port=args.port,
        window_title=args.window,
        standalone=args.standalone or not args.window,
    )
    await visualizer.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
