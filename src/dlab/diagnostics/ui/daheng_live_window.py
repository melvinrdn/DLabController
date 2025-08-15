from __future__ import annotations
import sys
import os
import time
import datetime
import numpy as np
from typing import Optional
from PIL import Image, PngImagePlugin

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QMessageBox, QSplitter, QCheckBox
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)
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
        self._cap_lock = cap_lock          

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

                # ---- only one capture at a time (live OR scan) ----
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
    """Live viewer for Daheng camera."""
    closed = pyqtSignal()

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

        self.initUI()

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

        # Log box (no GUI logging handler; avoid duplicates)
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
        """Update the displayed image; axes in millimeters."""
        if not isinstance(image, np.ndarray):
            self.log("Invalid image received")
            return

        self.last_frame = image
        vmin, vmax = float(image.min()), float(image.max())
        h, w = image.shape
        x_mm = w * (PIXEL_SIZE_M * 1e3)
        y_mm = h * (PIXEL_SIZE_M * 1e3)

        if self.image_artist is None:
            self.ax.clear()
            self.image_artist = self.ax.imshow(
                image,
                cmap=self.cmap,
                vmin=vmin, vmax=vmax,
                extent=[0, x_mm, 0, y_mm],
                origin="lower",
                aspect="equal",
            )
            self.cbar = self.figure.colorbar(self.image_artist, ax=self.ax, fraction=0.046, pad=0.04)
        else:
            self.image_artist.set_data(image)
            if self.fix_cbar and self.fixed_vmax is not None:
                self.image_artist.set_clim(vmin, self.fixed_vmax)
            else:
                self.image_artist.set_clim(vmin, vmax)
            if self.cbar:
                self.cbar.update_normal(self.image_artist)

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
        Save N consecutive frames as 16-bit PNGs with metadata (Exposure_us, Gain, Comment).
        Filenames end with _1, _2, ..., and a .log TSV is appended.
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
                # Save as 16-bit PNG (I;16)
                frame_u16 = np.clip(frame, 0, 65535).astype(np.uint16, copy=False)
                img = Image.fromarray(frame_u16, mode="I;16")

                fn = f"{stem}_{ts_prefix}_{i}.png"
                path = os.path.join(folder, fn)

                meta = PngImagePlugin.PngInfo()
                meta.add_text("Exposure_us", str(exp_us))
                meta.add_text("Gain", str(gain))
                meta.add_text("Comment", comment)

                img.save(path, format="PNG", pnginfo=meta)
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
            
    def grab_frame_for_scan(
        self,
        averages: int = 1,
        adaptive: dict | None = None,
        dead_pixel_cleanup: bool = True,
        background: bool = False,
    ):
        """
        Synchronous capture for scans.
        - If live is running, stop it, capture, and update the view with the averaged frame.
        - Adaptive exposure finds a good exposure first; that exposure is then LOCKED
        for the averaging frames (no _1/_2 files; we return one averaged frame).
        - Dead pixels/hot pixels are zeroed.
        Returns (frame_u16, meta_dict) with Gain intentionally blank.
        """
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

        # --- Adaptive step (prepass) ---
        if use_adapt:
            cur = exp_us
            for _ in range(max_iters):
                f0 = np.asarray(_cap_once(cur), dtype=np.float64, copy=False)
                robust_max = float(np.percentile(f0, 99.9))
                if robust_max < floor_cnt:
                    robust_max = floor_cnt
                frac = robust_max / 65535.0
                if lo_frac <= frac <= hi_frac:
                    break
                scale = (target_frac / frac) if frac > 1e-9 else 2.0
                cur = int(max(min_us, min(max_us, cur * scale)))
            exp_us = cur  # lock exposure for averaging

        # --- Averaging at LOCKED exposure ---
        n = max(1, int(averages))
        acc = None
        for _ in range(n):
            f = np.asarray(_cap_once(exp_us), dtype=np.float64, copy=False)
            acc = f if acc is None else (acc + f)
        avg = acc / n

        # Dead/hot pixel cleanup
        if dead_pixel_cleanup:
            avg[avg >= 65535.0] = 0.0
            avg[avg < 0.0] = 0.0
            p9999 = np.percentile(avg, 99.99)
            if p9999 > 65535.0:
                avg[avg > p9999] = 0.0

        frame_u16 = np.clip(avg, 0, 65535).astype(np.uint16, copy=False)
        self.update_image(frame_u16)

        meta = {
            "CameraName": f"DahengCam_{self.fixed_index}",  # enforce naming
            "CameraIndex": self.fixed_index,
            "Exposure_us": exp_us,
            "Gain": "",                                    # blank in scans
            "Background": "1" if background else "0",
        }
        return frame_u16, meta

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
