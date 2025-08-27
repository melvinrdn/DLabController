from __future__ import annotations
import sys
import os
import time
import datetime
import numpy as np
from typing import Optional, Tuple
from PIL import Image, PngImagePlugin

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QMessageBox, QSplitter, QCheckBox, QGroupBox, QSpinBox
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import logging
import threading

from dlab.hardware.wrappers.daheng_controller import (
    DahengController, DahengControllerError,
    DEFAULT_EXPOSURE_US, MIN_EXPOSURE_US, MAX_EXPOSURE_US,
    DEFAULT_GAIN, MIN_GAIN, MAX_GAIN,
)
from dlab.diagnostics.utils import white_turbo
from dlab.boot import get_config
from dlab.core.device_registry import REGISTRY

# Pixel size (meters)
PIXEL_SIZE_M = 3.45e-6

# Min UI refresh interval (µs)
MIN_INTERVAL_US = 500_000

logging.getLogger("matplotlib").setLevel(logging.WARNING)
DATE_FORMAT = "%H:%M:%S"


def _data_root() -> str:
    cfg = get_config() or {}
    return str((cfg.get("paths", {}) or {}).get("data_dir", r"C:/data"))


# ------------------------ capture thread ------------------------

class LiveCaptureThread(QThread):
    image_signal = pyqtSignal(np.ndarray)

    def __init__(self, cam: DahengController, exposure_us: int, gain: int, interval_us: int, cap_lock=None):
        super().__init__()
        self.cam = cam
        self.exposure_us = exposure_us
        self.gain = gain
        self.interval_s = interval_us / 1e6
        self._running = True
        self._lock = threading.Lock()
        self._cap_lock = cap_lock  # shared with scan path if needed

    def update_parameters(self, exposure_us: int, gain: int, interval_us: int):
        with self._lock:
            self.exposure_us = exposure_us
            self.gain = gain
            self.interval_s = interval_us / 1e6

    def run(self) -> None:
        while self._running:
            try:
                with self._lock:
                    exp = self.exposure_us
                    g = self.gain
                    wait_s = self.interval_s

                # only one capture at a time (live OR scan)
                lock = self._cap_lock or threading.Lock()
                with lock:
                    frame = self.cam.capture_single(exp, g)

                self.image_signal.emit(frame)
                time.sleep(wait_s)
            except DahengControllerError as ce:
                print(f"Daheng capture error: {ce}")
                break
            except Exception as e:
                print(f"Unexpected capture error: {e}")
                break

    def stop(self):
        self._running = False


# ----------------------------- GUI ----------------------------------

class DahengLiveWindow(QWidget):
    """Live viewer for Daheng camera. Provides software ROI selection and crop."""
    closed = pyqtSignal()
    gui_update_image = pyqtSignal(object)   # thread-safe image updates
    gui_log = pyqtSignal(str)    

    def __init__(self, camera_name: str = "Daheng", fixed_index: int = 1):
        super().__init__()
        self.live_running = False
        self.capture_lock = threading.Lock()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.camera_name = camera_name
        self.fixed_index = fixed_index
        self.setWindowTitle(f"DahengLiveWindow - {camera_name}")

        self.cam: Optional[DahengController] = None
        self.thread: Optional[LiveCaptureThread] = None
        self.image_artist = None
        self.cbar = None
        self.last_frame: Optional[np.ndarray] = None
        self.fix_cbar = False
        self.fixed_vmax: Optional[float] = None

        self.cmap = white_turbo(512)

        # ROI state
        self.roi_px: Optional[Tuple[int, int, int, int]] = None  # (x0,y0,x1,y1) in pixels
        self.roi_artist = None
        self.rect_selector: Optional[RectangleSelector] = None

        # ROI controls
        self.use_roi_cb = QCheckBox("Use ROI")
        self.use_roi_cb.setChecked(False)
        self.preview_roi_cb = QCheckBox("Preview crop")
        self.preview_roi_cb.setChecked(False)

        # "Center on max" helpers
        self.center_w_spin = QSpinBox()
        self.center_h_spin = QSpinBox()
        for sp, dv in ((self.center_w_spin, 200), (self.center_h_spin, 200)):
            sp.setRange(4, 4096)
            sp.setSingleStep(10)
            sp.setValue(dv)

        self.initUI()
        self.gui_update_image.connect(self.update_image, Qt.QueuedConnection)
        self.gui_log.connect(self.log, Qt.QueuedConnection)

    # -------- UI --------

    def initUI(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter()

        # Left: params
        param_panel = QWidget()
        param_layout = QVBoxLayout(param_panel)

        # Camera index (read-only label)
        idx_layout = QHBoxLayout()
        idx_layout.addWidget(QLabel("Camera Index:"))
        idx_layout.addWidget(QLabel(str(self.fixed_index)))
        param_layout.addLayout(idx_layout)

        # Exposure (µs)
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure (µs):"))
        self.exposure_edit = QLineEdit(str(DEFAULT_EXPOSURE_US))
        self.exposure_edit.setValidator(QIntValidator(MIN_EXPOSURE_US, MAX_EXPOSURE_US, self))
        self.exposure_edit.textChanged.connect(self.update_params)
        exp_layout.addWidget(self.exposure_edit)
        param_layout.addLayout(exp_layout)

        # Update interval (µs) — derived, read-only
        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Update Interval (µs):"))
        init_interval = max(DEFAULT_EXPOSURE_US, MIN_INTERVAL_US)
        self.interval_edit = QLineEdit(str(init_interval))
        self.interval_edit.setEnabled(False)
        int_layout.addWidget(self.interval_edit)
        param_layout.addLayout(int_layout)

        # Gain
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.gain_edit = QLineEdit(str(DEFAULT_GAIN))
        self.gain_edit.setValidator(QIntValidator(MIN_GAIN, MAX_GAIN, self))
        self.gain_edit.textChanged.connect(self.update_params)
        gain_layout.addWidget(self.gain_edit)
        param_layout.addLayout(gain_layout)

        # Comment
        cmt_layout = QHBoxLayout()
        cmt_layout.addWidget(QLabel("Comment:"))
        self.comment_edit = QLineEdit()
        cmt_layout.addWidget(self.comment_edit)
        param_layout.addLayout(cmt_layout)

        # Frames to Save
        nsave_layout = QHBoxLayout()
        nsave_layout.addWidget(QLabel("Frames to Save:"))
        self.frames_to_save_edit = QLineEdit("1")
        self.frames_to_save_edit.setValidator(QIntValidator(1, 1000, self))
        nsave_layout.addWidget(self.frames_to_save_edit)
        param_layout.addLayout(nsave_layout)

        # Background flag
        self.background_cb = QCheckBox("Background")
        param_layout.addWidget(self.background_cb)

        # Fix Colorbar Max
        self.fix_cb = QCheckBox("Fix Colorbar Max")
        self.fix_cb.toggled.connect(self.on_fix_cbar)
        param_layout.addWidget(self.fix_cb)
        self.fix_value_edit = QLineEdit("10000")
        self.fix_value_edit.setValidator(QIntValidator(0, 1_000_000_000, self))
        self.fix_value_edit.setEnabled(False)
        self.fix_value_edit.textChanged.connect(self.on_fix_value_changed)
        self.fix_cb.toggled.connect(self.fix_value_edit.setEnabled)
        param_layout.addWidget(self.fix_value_edit)

        # --- ROI controls group ---
        roi_group = QGroupBox("ROI")
        roi_layout = QHBoxLayout(roi_group)
        set_btn = QPushButton("Set ROI")
        clear_btn = QPushButton("Clear ROI")
        center_btn = QPushButton("Center on max")
        set_btn.clicked.connect(self.start_roi_selection)
        clear_btn.clicked.connect(self.clear_roi)
        center_btn.clicked.connect(self.center_on_max)

        roi_layout.addWidget(self.use_roi_cb)
        roi_layout.addWidget(self.preview_roi_cb)
        roi_layout.addWidget(QLabel("W(px):")); roi_layout.addWidget(self.center_w_spin)
        roi_layout.addWidget(QLabel("H(px):")); roi_layout.addWidget(self.center_h_spin)
        roi_layout.addWidget(center_btn)
        roi_layout.addWidget(set_btn)
        roi_layout.addWidget(clear_btn)
        param_layout.addWidget(roi_group)

        # Buttons
        btn_layout = QVBoxLayout()
        def make_btn(text, slot, enabled=True):
            b = QPushButton(text)
            b.clicked.connect(slot)
            b.setEnabled(enabled)
            btn_layout.addWidget(b)
            return b

        self.activate_camera_btn = make_btn("Activate Camera", self.activate_camera)
        self.deactivate_camera_btn = make_btn("Deactivate Camera", self.deactivate_camera, False)
        self.start_capture_btn = make_btn("Start Live Capture", self.start_capture, False)
        self.stop_capture_btn = make_btn("Stop Live Capture", self.stop_capture, False)
        self.save_btn = make_btn("Save Frame(s)", self.save_frames)
        param_layout.addLayout(btn_layout)

        # Log box
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        param_layout.addWidget(self.log_text)

        splitter.addWidget(param_panel)

        # Right: plot
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax.set_title(f"DahengLiveWindow - {self.camera_name}")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.figure.subplots_adjust(right=0.85)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_panel)

        main_layout.addWidget(splitter)
        self.resize(1080, 720)

    # -------- helpers --------

    def log(self, message: str):
        now = datetime.datetime.now().strftime(DATE_FORMAT)
        self.log_text.append(f"[{now}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        logging.getLogger("dlab.ui.DahengLiveWindow").info(message)

    def _update_interval_field(self, exp_us: int):
        interval_us = exp_us if exp_us >= MIN_INTERVAL_US else MIN_INTERVAL_US
        self.interval_edit.setText(str(interval_us))
        return interval_us

    def _px_per_mm(self) -> float:
        # mm per pixel = PIXEL_SIZE_M * 1e3; so px per mm:
        return 1.0 / (PIXEL_SIZE_M * 1e3)

    # -------- ROI selection / overlay --------

    def start_roi_selection(self):
        """Enable interactive rectangle selection on the axes."""
        if self.last_frame is None:
            self.log("No image yet — start live or capture one to set ROI.")
            return

        # reset existing selector
        if self.rect_selector is not None:
            try: self.rect_selector.disconnect_events()
            except Exception: pass
            self.rect_selector = None

        def _on_select(eclick, erelease):
            if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
                return
            # Coordinates come in millimeters → convert to pixels
            px_per_mm = self._px_per_mm()
            x0_mm, y0_mm = eclick.xdata, eclick.ydata
            x1_mm, y1_mm = erelease.xdata, erelease.ydata
            x0 = int(np.floor(min(x0_mm, x1_mm) * px_per_mm))
            x1 = int(np.ceil (max(x0_mm, x1_mm) * px_per_mm))
            y0 = int(np.floor(min(y0_mm, y1_mm) * px_per_mm))
            y1 = int(np.ceil (max(y0_mm, y1_mm) * px_per_mm))

            # clamp to frame size
            h, w = self.last_frame.shape
            x0 = max(0, min(w-1, x0)); x1 = max(1, min(w, x1))
            y0 = max(0, min(h-1, y0)); y1 = max(1, min(h, y1))
            if x1 <= x0 or y1 <= y0:
                self.log("Invalid ROI")
                return
            self.roi_px = (x0, y0, x1, y1)
            self._draw_roi_overlay()
            self.log(f"ROI set: x={x0}:{x1}, y={y0}:{y1} (px)")
            
            if self.rect_selector:
                self.rect_selector.set_visible(False)
                self.canvas.draw_idle()

        self.rect_selector = RectangleSelector(
            self.ax, _on_select, useblit=True,
            button=[1],  # left mouse button
            minspanx=2, minspany=2, spancoords='pixels',
            interactive=True
        )

    def _draw_roi_overlay(self):
        """Draw or refresh the dashed rectangle overlay (in mm coordinates)."""
        try:
            if self.roi_artist is not None:
                self.roi_artist.remove()
                self.roi_artist = None
        except Exception:
            pass
        if self.roi_px is None:
            self.canvas.draw_idle()
            return
        x0, y0, x1, y1 = self.roi_px
        mm_per_px = PIXEL_SIZE_M * 1e3
        x0_mm, x1_mm = x0 * mm_per_px, x1 * mm_per_px
        y0_mm, y1_mm = y0 * mm_per_px, y1 * mm_per_px
        from matplotlib.patches import Rectangle
        self.roi_artist = self.ax.add_patch(
            Rectangle((x0_mm, y0_mm), (x1_mm - x0_mm), (y1_mm - y0_mm),
                      fill=False, linewidth=1.5, linestyle="--")
        )
        self.canvas.draw_idle()

    def clear_roi(self):
        """Clear ROI (no crop)."""
        self.roi_px = None
        self._draw_roi_overlay()
        self.log("ROI cleared")

    def center_on_max(self):
        """
        Center the ROI of chosen width/height on the brightest pixel of the current frame.
        If no frame is available, do nothing.
        """
        if self.last_frame is None or self.last_frame.size == 0:
            self.log("No image to center on.")
            return
        h, w = self.last_frame.shape
        # Find max location
        idx = int(np.argmax(self.last_frame))
        y_max, x_max = divmod(idx, w)

        rw = int(self.center_w_spin.value())
        rh = int(self.center_h_spin.value())
        rw = max(4, min(w, rw))
        rh = max(4, min(h, rh))
        half_w = rw // 2
        half_h = rh // 2

        x0 = max(0, x_max - half_w)
        y0 = max(0, y_max - half_h)
        x1 = min(w, x0 + rw)
        y1 = min(h, y0 + rh)

        # Adjust start if we hit the boundary
        x0 = max(0, x1 - rw)
        y0 = max(0, y1 - rh)

        if x1 <= x0 or y1 <= y0:
            self.log("Center ROI failed due to size.")
            return

        self.roi_px = (int(x0), int(y0), int(x1), int(y1))
        self._draw_roi_overlay()
        self.use_roi_cb.setChecked(True)
        self.preview_roi_cb.setChecked(True)
        self.log(f"ROI centered on max at ({x_max},{y_max}), size=({rw}x{rh})")

    # -------- params / lifecycle --------

    def update_params(self):
        """Keep update interval >= 500 ms; push exposure/gain to device or thread."""
        try:
            exp_us = int(self.exposure_edit.text())
            gain = int(self.gain_edit.text())
            interval_us = self._update_interval_field(exp_us)
            if self.thread:
                self.thread.update_parameters(exp_us, gain, interval_us)
            elif self.cam:
                self.cam.set_exposure(exp_us)
                self.cam.set_gain(gain)
        except ValueError:
            pass

    def activate_camera(self):
        if self.cam:
            try:
                self.cam.deactivate()
            except Exception:
                pass
            self.cam = None
        try:
            self.cam = DahengController(self.fixed_index)
            self.cam.activate()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate: {e}")
            self.log(f"Activation error: {e}")
            return
        self.log(f"Camera {self.fixed_index} activated")
        key_name  = f"camera:daheng:{self.camera_name.lower()}"     # e.g. camera:daheng:nomarski
        key_index = f"camera:daheng:index:{self.fixed_index}"       # e.g. camera:daheng:index:1
        
        for k in (key_name, key_index):
            prev = REGISTRY.get(k)
            if prev and prev is not self.cam:
                self.log(f"Registry key '{k}' already in use. Replacing.")
            REGISTRY.register(k, self)

        self.activate_camera_btn.setEnabled(False)
        self.deactivate_camera_btn.setEnabled(True)
        self.start_capture_btn.setEnabled(True)

    def deactivate_camera(self):
        if self.cam:
            try:
                self.cam.deactivate()
            except Exception:
                pass
            self.log(f"Camera {self.fixed_index} deactivated")
        for k in (f"camera:daheng:{self.camera_name.lower()}",
                f"camera:daheng:index:{self.fixed_index}"):
            REGISTRY.unregister(k)
        self.cam = None
        self.activate_camera_btn.setEnabled(True)
        self.deactivate_camera_btn.setEnabled(False)
        self.start_capture_btn.setEnabled(False)
        self.stop_capture_btn.setEnabled(False)

    # -------- live capture --------

    def start_capture(self):
        if not self.cam:
            QMessageBox.critical(self, "Error", "Camera not activated.")
            return
        try:
            exp_us = int(self.exposure_edit.text())
            gain = int(self.gain_edit.text())
            interval_us = self._update_interval_field(exp_us)
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameters.")
            return

        self.thread = LiveCaptureThread(self.cam, exp_us, gain, interval_us, cap_lock=self.capture_lock)
        self.thread.image_signal.connect(self.update_image)
        self.thread.start()
        self.live_running = True
        self.log("Live capture started")
        self.start_capture_btn.setEnabled(False)
        self.stop_capture_btn.setEnabled(True)

    def stop_capture(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None
            self.live_running = False
            self.log("Live capture stopped")
        self.start_capture_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(False)

    # -------- drawing --------

    def update_image(self, image: np.ndarray):
        """Update the displayed image; axes in millimeters. Optional preview ROI crop."""
        if not isinstance(image, np.ndarray):
            self.log("Invalid image received")
            return

        self.last_frame = image
        # Preview crop for display (does not affect M² unless force_roi=True in grab_frame_for_scan)
        disp = image
        extent_mm = None
        if self.preview_roi_cb.isChecked() and self.roi_px is not None and self.use_roi_cb.isChecked():
            x0, y0, x1, y1 = self.roi_px
            h0, w0 = disp.shape
            x0 = max(0, min(w0-1, x0)); x1 = max(1, min(w0, x1))
            y0 = max(0, min(h0-1, y0)); y1 = max(1, min(h0, y1))
            disp = disp[y0:y1, x0:x1]
            mm_per_px = PIXEL_SIZE_M * 1e3
            extent_mm = [x0*mm_per_px, x1*mm_per_px, y0*mm_per_px, y1*mm_per_px]

        vmin, vmax = float(disp.min()), float(disp.max())
        h, w = disp.shape
        if extent_mm is None:
            x_mm = w * (PIXEL_SIZE_M * 1e3)
            y_mm = h * (PIXEL_SIZE_M * 1e3)
            extent = [0, x_mm, 0, y_mm]
        else:
            extent = extent_mm

        if self.image_artist is None:
            self.ax.clear()
            self.image_artist = self.ax.imshow(
                disp,
                cmap=self.cmap,
                vmin=vmin, vmax=vmax,
                extent=extent,
                origin="lower",
                aspect="equal",
            )
            self.cbar = self.figure.colorbar(self.image_artist, ax=self.ax, fraction=0.046, pad=0.04)
        else:
            self.image_artist.set_data(disp)
            if self.fix_cbar and self.fixed_vmax is not None:
                self.image_artist.set_clim(vmin, self.fixed_vmax)
            else:
                self.image_artist.set_clim(vmin, vmax)
            self.image_artist.set_extent(extent)
            if self.cbar:
                self.cbar.update_normal(self.image_artist)

        # Draw overlay if not preview-cropping
        if not (self.preview_roi_cb.isChecked() and self.use_roi_cb.isChecked()):
            self._draw_roi_overlay()

        self.canvas.draw_idle()

    def on_fix_cbar(self, checked: bool):
        self.fix_cbar = checked
        if checked:
            try:
                self.fixed_vmax = float(self.fix_value_edit.text())
                self.log(f"Colorbar max set to {self.fixed_vmax:.1f}")
            except ValueError:
                self.fixed_vmax = None
                self.fix_cb.setChecked(False)
                self.log("Invalid colorbar max")
        else:
            self.fixed_vmax = None
            self.log("Colorbar auto scale")

    def on_fix_value_changed(self, text: str):
        if not self.fix_cb.isChecked() or not self.image_artist:
            return
        try:
            self.fixed_vmax = float(text)
            vmin, _ = self.image_artist.get_clim()
            self.image_artist.set_clim(vmin, self.fixed_vmax)
            if self.cbar:
                self.cbar.update_normal(self.image_artist)
            self.canvas.draw_idle()
            self.log(f"Colorbar max updated to {self.fixed_vmax:.1f}")
        except ValueError:
            pass

    # -------- saving N frames --------

    def _capture_one(self, exp_us: int, gain: int) -> np.ndarray:
        if not self.cam:
            raise DahengControllerError("Camera not activated.")
        return self.cam.capture_single(exp_us, gain)

    def save_frames(self):
        """
        Save N consecutive frames as 8-bit PNG (0..255) with metadata.
        Linear mapping from sensor values: uint16 >> 8.
        """
        # Parameters
        try:
            exp_us = int(self.exposure_edit.text())
            gain = int(self.gain_edit.text())
            n_frames = int(self.frames_to_save_edit.text())
            if n_frames <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid 'Frames to Save' / exposure / gain.")
            return

        now = datetime.datetime.now()
        base_dir = _data_root()
        folder = os.path.join(base_dir, f"{now:%Y-%m-%d}", self.camera_name)
        os.makedirs(folder, exist_ok=True)
        ts_prefix = now.strftime("%Y%m%d_%H%M%S")
        stem = f"{self.camera_name}_{'Background' if self.background_cb.isChecked() else 'Image'}"

        comment = self.comment_edit.text()

        saved: list[str] = []
        for i in range(1, n_frames + 1):
            try:
                frame = self._capture_one(exp_us, gain) if self.cam else self.last_frame
                if frame is None:
                    raise RuntimeError("No frame captured.")

                if self.use_roi_cb.isChecked() and self.roi_px is not None:
                    x0, y0, x1, y1 = self.roi_px
                    h0, w0 = frame.shape
                    x0 = max(0, min(w0-1, x0)); x1 = max(1, min(w0, x1))
                    y0 = max(0, min(h0-1, y0)); y1 = max(1, min(h0, y1))
                    frame = frame[y0:y1, x0:x1]

                f8 = np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))
                #self.log(f"f8 stats: max={int(f8.max())}, mean={float(f8.mean()):.2f}")

                fn = f"{stem}_{ts_prefix}_{i}.png"
                path = os.path.join(folder, fn)

                meta = PngImagePlugin.PngInfo()
                meta.add_text("Exposure_us", str(exp_us))
                meta.add_text("Gain", str(gain))
                meta.add_text("Comment", comment)

                Image.fromarray(f8, mode="L").save(path, format="PNG", pnginfo=meta)
                saved.append(fn)
                self.log(f"Saved {path}")
            except Exception as e:
                self.log(f"Error saving frame {i}: {e}")
                QMessageBox.critical(self, "Error", f"Error saving frame {i}: {e}")
                break

        if not saved:
            return

        # Append TSV log with .log extension
        log_name = f"{self.camera_name}_log_{now:%Y-%m-%d}.log"
        log_path = os.path.join(folder, log_name)
        header = "ImageFile\tExposure_us\tGain\tComment\n"
        try:
            if not os.path.exists(log_path):
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(header)
            with open(log_path, "a", encoding="utf-8") as f:
                for fn in saved:
                    f.write(f"{fn}\t{exp_us}\t{gain}\t{comment}\n")
            self.log(f"Logged {len(saved)} file(s) to {log_path}")
        except Exception as e:
            self.log(f"Error writing log: {e}")
            QMessageBox.critical(self, "Error", f"Error writing log: {e}")

    # -------- scan API (used by M² and grid scans) --------

    def grab_frame_for_scan(
        self,
        averages: int = 1,
        adaptive: dict | None = None,
        dead_pixel_cleanup: bool = False,
        background: bool = False,
        *,
        force_roi: bool = False,
    ):
        if not self.cam:
            raise DahengControllerError("Camera not activated.")

        was_live = bool(self.live_running)
        if was_live:
            try:
                self.stop_capture()
            except Exception:
                pass

        # Parse GUI
        try:
            exp_us = int(self.exposure_edit.text())
        except ValueError:
            exp_us = DEFAULT_EXPOSURE_US

        # Device needs a gain value; but scans must log Gain as blank.
        try:
            device_gain = int(self.gain_edit.text())
        except ValueError:
            device_gain = DEFAULT_GAIN

        # Adaptive config
        cfg = adaptive or {}
        use_adapt   = bool(cfg.get("enabled", False))
        target_frac = float(cfg.get("target_frac", 0.75))
        lo_frac     = float(cfg.get("low_frac", 0.60))
        hi_frac     = float(cfg.get("high_frac", 0.90))
        min_us      = int(cfg.get("min_us", 20))
        max_us      = int(cfg.get("max_us", 1_000_000))
        floor_cnt   = float(cfg.get("floor_counts", 200.0))
        max_iters   = int(cfg.get("max_iters", 6))

        def _cap_once(cur_exp_us: int):
            self.cam.set_exposure(cur_exp_us)
            self.cam.set_gain(device_gain)
            return self.cam.capture_single(cur_exp_us, device_gain)

        # --- Adaptive exposure ---
        if use_adapt:
            cur = int(exp_us)
            for _ in range(max_iters):
                # probe one frame at current exposure (camera gives 0..255 -> float array ici)
                f0 = np.asarray(_cap_once(cur), dtype=np.float64)

                # use ROI
                if (force_roi or self.use_roi_cb.isChecked()) and self.roi_px is not None:
                    x0, y0, x1, y1 = self.roi_px
                    h0, w0 = f0.shape
                    x0 = max(0, min(w0 - 1, x0)); x1 = max(1, min(w0, x1))
                    y0 = max(0, min(h0 - 1, y0)); y1 = max(1, min(h0, y1))
                    f0 = f0[y0:y1, x0:x1]

                # map to uint8 explicitly (no scaling, cam is 8-bit)
                f8 = np.clip(f0, 0, 255).astype(np.uint8)

                # ignore zeros (background) for the robust max
                nz = f8[f8 > 0]
                if nz.size == 0:
                    # fully dark: ramp up faster
                    cur = int(min(max_us, max(cur * 2, min_us)))
                    exp_us = cur
                    continue

                robust_max_8 = float(np.percentile(nz, 99.9))  # bright but robust
                frac = robust_max_8 / 255.0                    # target ~0.75 default

                if lo_frac <= frac <= hi_frac:
                    break

                # bounded multiplicative step (avoid oscillations)
                scale = target_frac / max(frac, 1e-6)
                scale = float(np.clip(scale, 0.5, 2.0))        # at most /2 or ×2 per iter
                next_exp = int(np.clip(cur * scale, min_us, max_us))

                # nudge if clamped/rounded to same value
                if next_exp == cur:
                    next_exp = min(max_us, cur + 1)

                cur = next_exp
                self.log(f"Adapt: robust_max={robust_max_8:.1f}/255 (frac={frac:.3f}) -> exp={cur} us")

            exp_us = int(cur)  # lock exposure for averaging


        # --- Averaging at LOCKED exposure + software ROI crop if requested ---
        n = max(1, int(averages))
        acc = None
        for _ in range(n):
            # capture returns uint8 when MONO8 is set; cast to float32 for accumulation
            f = np.asarray(_cap_once(exp_us), dtype=np.float32)
            if (force_roi or self.use_roi_cb.isChecked()) and self.roi_px is not None:
                x0, y0, x1, y1 = self.roi_px
                h0, w0 = f.shape
                x0 = max(0, min(w0 - 1, x0)); x1 = max(1, min(w0, x1))
                y0 = max(0, min(h0 - 1, y0)); y1 = max(1, min(h0, y1))
                f = f[y0:y1, x0:x1]
            acc = f if acc is None else (acc + f)
        avg = acc / n

        # Dead/hot pixel cleanup
        if dead_pixel_cleanup:
            avg[avg >= 65535.0] = 0.0
            avg[avg < 0.0] = 0.0
            p9999 = np.percentile(avg, 99.99)
            if p9999 > 65535.0:
                avg[avg > p9999] = 0.0

        frame_u8 = np.clip(avg, 0, 255).astype(np.uint8)

        #self.update_image(frame_u8)
        self.gui_update_image.emit(frame_u8)

        meta = {
            "CameraName": f"DahengCam_{self.fixed_index}",  # enforce naming
            "CameraIndex": self.fixed_index,
            "Exposure_us": exp_us,
            "Gain": "",                                    # blank in scans
            "Background": "1" if background else "0",
            "ROI_px": "" if self.roi_px is None else f"{self.roi_px[0]},{self.roi_px[1]},{self.roi_px[2]},{self.roi_px[3]}",
            "ROI_Used": "1" if (force_roi or (self.use_roi_cb.isChecked() and self.roi_px is not None)) else "0",
        }
        return frame_u8, meta

    # -------- close --------

    def closeEvent(self, event):
        if self.thread:
            self.stop_capture()
        if self.cam:
            try: self.cam.deactivate()
            except Exception: pass
        for k in (f"camera:daheng:{self.camera_name.lower()}",
                  f"camera:daheng:index:{self.fixed_index}"):
            REGISTRY.unregister(k)
        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DahengLiveWindow(camera_name="Nomarski", fixed_index=1)
    gui.show()
    sys.exit(app.exec_())
