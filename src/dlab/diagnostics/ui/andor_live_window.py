from __future__ import annotations
import sys
import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, PngImagePlugin

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QMessageBox, QSplitter, QCheckBox
)
    # QSpinBox, etc. not needed
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import logging
import threading

from dlab.hardware.wrappers.andor_controller import (
    AndorController, AndorControllerError,
    DEFAULT_EXPOSURE_US, MIN_EXPOSURE_US, MAX_EXPOSURE_US
)
from dlab.diagnostics.utils import white_turbo
from dlab.boot import get_config

# Quiet down noisy matplotlib logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)

LOG_FORMAT = "[%(asctime)s] %(message)s"
DATE_FORMAT = "%H:%M:%S"


def _data_root() -> str:
    """Base folder for saved frames, from YAML or default."""
    cfg = get_config() or {}
    return str((cfg.get("paths", {}) or {}).get("data_dir", r"C:/data"))


class QTextEditHandler(logging.Handler):
    """Pipe log records into the GUI text box."""
    def __init__(self, widget: QTextEdit):
        super().__init__()
        self.widget = widget
        self.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.widget.append(msg)
        self.widget.verticalScrollBar().setValue(self.widget.verticalScrollBar().maximum())


# ------------------------ Live capture thread (no averaging) ------------------------

class LiveCaptureThread(QThread):
    image_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_controller: AndorController, exposure: int, update_interval_ms: int):
        super().__init__()
        self.camera_controller = camera_controller
        self.exposure = exposure
        self.update_interval_ms = update_interval_ms
        self.interval_sec = update_interval_ms / 1000.0
        self._running = True
        self._param_lock = threading.Lock()

    def update_parameters(self, exposure: int, update_interval_ms: int):
        with self._param_lock:
            self.exposure = exposure
            self.update_interval_ms = update_interval_ms
            self.interval_sec = update_interval_ms / 1000.0

    def _capture_one(self, exp_us: int) -> np.ndarray:
        # Prefer controller.capture_single if available; fallback to take_image(exposure, 1)
        if hasattr(self.camera_controller, "capture_single"):
            return self.camera_controller.capture_single(exp_us)
        return self.camera_controller.take_image(exp_us, 1)  # legacy fallback

    def run(self):
        while self._running:
            try:
                with self._param_lock:
                    exp = self.exposure
                    sleep_interval = self.interval_sec
                image = self._capture_one(exp)
                self.image_signal.emit(image)
                time.sleep(sleep_interval)
            except AndorControllerError as ace:
                print(f"AndorControllerError in capture thread: {ace}")
                break
            except Exception as e:
                print("Error capturing image:", e)
                break

    def stop(self):
        self._running = False


# ----------------------------- GUI ----------------------------------

class AndorLiveWindow(QWidget):
    """PyQt5 GUI for live Andor capture without averaging."""
    closed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Andor Camera Feed")
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.cam: AndorController | None = None
        self.capture_thread: LiveCaptureThread | None = None
        self.image_artist = None
        self.cbar = None
        self.current_interval_ms = None
        self.fixed_cbar_max: float | None = None
        self.last_frame: np.ndarray | None = None

        self._logger = logging.getLogger("dlab.ui.AndorLiveWindow")

        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter()

        # ---- Left panel (params) ----
        param_panel = QWidget()
        param_layout = QVBoxLayout(param_panel)

        # Exposure
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure (µs):"))
        self.exposure_edit = QLineEdit(f"{DEFAULT_EXPOSURE_US}")
        self.exposure_edit.setValidator(QIntValidator(MIN_EXPOSURE_US, MAX_EXPOSURE_US, self))
        self.exposure_edit.textChanged.connect(self.update_capture_parameters)
        exp_layout.addWidget(self.exposure_edit)
        param_layout.addLayout(exp_layout)

        # Update interval
        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Update Interval (ms):"))
        self.interval_edit = QLineEdit("1000")
        self.interval_edit.setValidator(QIntValidator(100, 10000, self))
        self.interval_edit.textChanged.connect(self.update_capture_parameters)
        int_layout.addWidget(self.interval_edit)
        param_layout.addLayout(int_layout)

        # MCP voltage
        mcp_layout = QHBoxLayout()
        mcp_layout.addWidget(QLabel("MCP Voltage:"))
        self.mcp_voltage_edit = QLineEdit("Not specified")
        mcp_layout.addWidget(self.mcp_voltage_edit)
        param_layout.addLayout(mcp_layout)

        # Comment
        comment_layout = QHBoxLayout()
        comment_layout.addWidget(QLabel("Comment:"))
        self.comment_edit = QLineEdit()
        comment_layout.addWidget(self.comment_edit)
        param_layout.addLayout(comment_layout)

        # Frames-to-save (replaces "Averages" column for saving)
        nsave_layout = QHBoxLayout()
        nsave_layout.addWidget(QLabel("Frames to Save:"))
        self.frames_to_save_edit = QLineEdit("1")
        self.frames_to_save_edit.setValidator(QIntValidator(1, 1000, self))
        nsave_layout.addWidget(self.frames_to_save_edit)
        param_layout.addLayout(nsave_layout)

        # Colorbar controls
        self.fix_cbar_checkbox = QCheckBox("Fix Colorbar Max")
        param_layout.addWidget(self.fix_cbar_checkbox)
        self.fix_value_edit = QLineEdit("10000")
        self.fix_value_edit.setValidator(QIntValidator(0, 1_000_000_000, self))
        self.fix_value_edit.setEnabled(False)
        param_layout.addWidget(self.fix_value_edit)
        self.fix_cbar_checkbox.toggled.connect(self.fix_value_edit.setEnabled)
        self.fix_cbar_checkbox.toggled.connect(self.on_fix_cbar)
        self.fix_value_edit.textChanged.connect(self.on_fix_value_changed)

        # Background flag
        self.background_checkbox = QCheckBox("Background")
        self.background_checkbox.setChecked(False)
        param_layout.addWidget(self.background_checkbox)

        # Buttons
        btn_layout = QVBoxLayout()

        self.activate_button = QPushButton("Activate Camera")
        self.activate_button.clicked.connect(self.activate_camera)
        btn_layout.addWidget(self.activate_button)

        self.activate_dummy_button = QPushButton("Activate Dummy Camera")
        self.activate_dummy_button.clicked.connect(self.activate_dummy_camera)
        btn_layout.addWidget(self.activate_dummy_button)

        self.deactivate_button = QPushButton("Deactivate Camera")
        self.deactivate_button.clicked.connect(self.deactivate_camera)
        self.deactivate_button.setEnabled(False)
        btn_layout.addWidget(self.deactivate_button)

        self.start_button = QPushButton("Start Live Capture")
        self.start_button.clicked.connect(self.start_capture)
        self.start_button.setEnabled(False)
        btn_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Live Capture")
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False)
        btn_layout.addWidget(self.stop_button)

        self.save_button = QPushButton("Save Frame(s)")
        self.save_button.clicked.connect(self.save_frames)
        btn_layout.addWidget(self.save_button)

        param_layout.addLayout(btn_layout)

        # Log box + handler
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        param_layout.addWidget(self.log_text)

        ui_logger = logging.getLogger("dlab.ui.AndorLiveWindow")
        ui_logger.handlers = []
        ui_logger.setLevel(logging.INFO)
        ui_logger.propagate = True 

        splitter.addWidget(param_panel)

        # ---- Right panel (plots) ----
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.figure, (self.ax_img, self.ax_profile) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        self.ax_img.set_title("Live Andor Camera Image")
        self.ax_profile.set_title("Integrated Profile")
        self.figure.subplots_adjust(right=0.85)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_panel)

        main_layout.addWidget(splitter)
        self.resize(1080, 720)

    # ----------------------- Logging helper -----------------------

    def log(self, message: str):
        current_time = datetime.datetime.now().strftime(DATE_FORMAT)
        self.log_text.append(f"[{current_time}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        logging.getLogger("dlab.ui.AndorLiveWindow").info(message)

    # ----------------------- Param updates -----------------------

    def update_capture_parameters(self):
        try:
            exposure = int(self.exposure_edit.text())
            update_interval = int(self.interval_edit.text())
            if self.capture_thread:
                self.capture_thread.update_parameters(exposure, update_interval)
            elif self.cam:
                # Apply immediately if not running live
                if hasattr(self.cam, "set_exposure"):
                    self.cam.set_exposure(exposure)
        except ValueError:
            pass

    # ----------------------- Camera lifecycle -----------------------

    def activate_camera(self):
        try:
            self.cam = AndorController(device_index=0)
            self.cam.activate()
            self.log("Camera 0 activated.")
            self.activate_button.setEnabled(False)
            self.activate_dummy_button.setEnabled(False)
            self.deactivate_button.setEnabled(True)
            self.start_button.setEnabled(True)
        except AndorControllerError as ce:
            QMessageBox.critical(self, "Error", f"Failed to activate camera: {ce}")
            self.log(f"Error activating camera: {ce}")

    # Simple dummy controller for UI testing
    class _DummyAndorController:
        def __init__(self, device_index=0):
            self.device_index = device_index
            self.image_shape = (512, 512)
            self.current_exposure = DEFAULT_EXPOSURE_US
        def enable_debug(self, *_, **__): pass
        def activate(self): pass
        def set_exposure(self, exposure: int): self.current_exposure = exposure
        def capture_single(self, exposure_us: int) -> np.ndarray:
            amp = 10000 * (exposure_us / max(DEFAULT_EXPOSURE_US, 1))
            sigma = 50
            cy, cx = self.image_shape[0] // 2, self.image_shape[1] // 2
            y, x = np.indices(self.image_shape)
            gauss = amp * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)))
            noise = np.random.normal(0, amp * 0.05, self.image_shape)
            return np.clip(gauss + noise, 0, 65535).astype(np.float64)
        def deactivate(self): pass

    def activate_dummy_camera(self):
        try:
            self.cam = AndorLiveWindow._DummyAndorController(device_index=666)
            self.cam.activate()
            self.log("Dummy camera activated.")
            self.activate_button.setEnabled(False)
            self.activate_dummy_button.setEnabled(False)
            self.deactivate_button.setEnabled(True)
            self.start_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate dummy camera: {e}")
            self.log(f"Error activating dummy camera: {e}")

    def deactivate_camera(self):
        try:
            if self.cam:
                self.cam.deactivate()
                self.log("Camera 0 deactivated.")
            self.cam = None
            self.activate_button.setEnabled(True)
            self.activate_dummy_button.setEnabled(True)
            self.deactivate_button.setEnabled(False)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
        except AndorControllerError as ce:
            QMessageBox.critical(self, "Error", f"Failed to deactivate camera: {ce}")
            self.log(f"Error deactivating camera: {ce}")

    # ----------------------- Live capture -----------------------

    def start_capture(self):
        if self.cam is None:
            QMessageBox.critical(self, "Error", "Camera not activated.")
            return
        try:
            exposure = int(self.exposure_edit.text())
            update_interval = int(self.interval_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameter values.")
            return

        self.capture_thread = LiveCaptureThread(self.cam, exposure, update_interval)
        self.capture_thread.image_signal.connect(self.update_image)
        self.capture_thread.start()
        self.log("Live capture started.")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_capture(self):
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread.wait()
            self.log("Live capture stopped.")
            self.capture_thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    # ----------------------- Plot updates -----------------------

    def on_fix_cbar(self, checked: bool):
        if checked:
            try:
                self.fixed_cbar_max = float(self.fix_value_edit.text())
                self.log(f"Colorbar max set to {self.fixed_cbar_max:.1f}")
            except ValueError:
                self.log("Invalid colorbar max value")
                self.fix_cbar_checkbox.setChecked(False)
        else:
            self.fixed_cbar_max = None
            self.log("Colorbar auto scale")

    def on_fix_value_changed(self, text: str):
        if not self.fix_cbar_checkbox.isChecked() or not self.image_artist:
            return
        try:
            self.fixed_cbar_max = float(text)
            vmin, _ = self.image_artist.get_clim()
            self.image_artist.set_clim(vmin, self.fixed_cbar_max)
            if self.cbar:
                self.cbar.update_normal(self.image_artist)
            self.canvas.draw_idle()
            self.log(f"Colorbar max updated to {self.fixed_cbar_max:.1f}")
        except ValueError:
            pass

    def update_image(self, image: np.ndarray):
        """Update the displayed image and an integrated profile (preserve zoom)."""
        self.last_frame = image
        max_val = float(np.max(image))
        min_val = float(np.min(image))
        title_text = f"Live Andor Camera Image - Max: {max_val:.0f}"

        if self.image_artist is None:
            self.ax_img.clear()
            self.image_artist = self.ax_img.imshow(image, cmap=white_turbo(512))
            self.ax_img.set_title(title_text)
            self.cbar = self.figure.colorbar(self.image_artist, ax=self.ax_img, fraction=0.046, pad=0.04)
        else:
            xlim = self.ax_img.get_xlim()
            ylim = self.ax_img.get_ylim()
            self.image_artist.set_data(image)
            self.ax_img.set_xlim(xlim)
            self.ax_img.set_ylim(ylim)
            self.ax_img.set_title(title_text)
            if self.fix_cbar_checkbox.isChecked() and self.fixed_cbar_max is not None:
                self.image_artist.set_clim(min_val, self.fixed_cbar_max)
            else:
                self.fixed_cbar_max = None
                self.image_artist.set_clim(min_val, max_val)
            self.cbar.update_normal(self.image_artist)

        profile = np.sum(image, axis=1)
        self.ax_profile.clear()
        self.ax_profile.plot(np.arange(image.shape[0]), profile)
        self.ax_profile.set_xlabel("(px)")
        self.ax_profile.set_ylabel("Integrated Intensity")
        self.ax_profile.set_xlim(0, image.shape[0])
        self.canvas.draw_idle()

    # ----------------------- Save frames (N sequential captures) -----------------------

    def _capture_one(self, exp_us: int) -> np.ndarray:
        if self.cam is None:
            raise AndorControllerError("Camera not activated.")
        if hasattr(self.cam, "capture_single"):
            return self.cam.capture_single(exp_us)
        return self.cam.take_image(exp_us, 1)

    def save_frames(self):
        """
        Save N consecutive frames as 16-bit PNGs with metadata (Exposure, MCP Voltage, Comment).
        If 'Background' is checked, filenames include 'Background'.
        Log file is TSV without an 'Averages' column.
        """
        if self.cam is None and self.last_frame is None:
            QMessageBox.warning(self, "Warning", "No frame available to save.")
            return

        try:
            exposure_us = int(self.exposure_edit.text())
            n_frames = int(self.frames_to_save_edit.text())
            if n_frames <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid 'Frames to Save' or 'Exposure' value.")
            return

        now = datetime.datetime.now()
        base_dir = _data_root()
        dir_path = os.path.join(base_dir, now.strftime("%Y-%m-%d"), "AndorCamera")
        os.makedirs(dir_path, exist_ok=True)

        mcp_voltage_val = self.mcp_voltage_edit.text()
        comment_text = self.comment_edit.text()
        is_bg = self.background_checkbox.isChecked()

        ts_prefix = now.strftime("%Y%m%d_%H%M%S")
        stem = "AndorCamera_MCP_Background" if is_bg else "AndorCamera_MCP_Image"

        saved_files: list[str] = []

        # Capture N frames sequentially
        for i in range(1, n_frames + 1):
            try:
                frame = self._capture_one(exposure_us) if self.cam else self.last_frame
                if frame is None:
                    raise RuntimeError("No frame captured.")
                # ensure 16-bit
                frame_uint16 = np.clip(frame, 0, 65535).astype(np.uint16, copy=False)
                img = Image.fromarray(frame_uint16, mode="I;16")

                # filename with index suffix
                file_name = f"{stem}_{ts_prefix}_{i}.png"
                file_path = os.path.join(dir_path, file_name)

                # metadata (no Averages)
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text("Exposure", str(exposure_us))
                metadata.add_text("MCP Voltage", mcp_voltage_val)
                metadata.add_text("Comment", comment_text)

                img.save(file_path, format="PNG", pnginfo=metadata)
                saved_files.append(file_name)
                self.log(f"Saved {file_path}")
            except Exception as e:
                self.log(f"Error saving frame {i}: {e}")
                QMessageBox.critical(self, "Error", f"Error saving frame {i}: {e}")
                # continue to next or break? safer to break
                break

        if not saved_files:
            return

        # Append to TSV log (no Averages column)
        log_file_name = f"AndorCamera_log_{now.strftime('%Y-%m-%d')}.log"
        log_file_path = os.path.join(dir_path, log_file_name)
        header = "File Name\tExposure (µs)\tMCP Voltage\tComment\n"

        try:
            if not os.path.exists(log_file_path):
                with open(log_file_path, "w", encoding="utf-8") as log_file:
                    log_file.write(header)
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                for fn in saved_files:
                    log_file.write(f"{fn}\t{exposure_us}\t{mcp_voltage_val}\t{comment_text}\n")
            self.log(f"Logged {len(saved_files)} file(s) to {log_file_path}")
        except Exception as e:
            self.log(f"Error writing log file: {e}")
            QMessageBox.critical(self, "Error", f"Error writing log file: {e}")

    # ----------------------- Close -----------------------

    def closeEvent(self, event):
        if self.capture_thread:
            self.stop_capture()
        if self.cam:
            self.deactivate_camera()
        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AndorLiveWindow()
    gui.show()
    sys.exit(app.exec_())
