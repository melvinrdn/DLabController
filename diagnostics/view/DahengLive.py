import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, PngImagePlugin

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QMessageBox, QSplitter, QCheckBox
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)

from hardware.wrappers.DahengController import (
    DahengController, DahengControllerError,
    MAX_GAIN, MAX_EXPOSURE_US, MIN_GAIN, MIN_EXPOSURE_US,
    DEFAULT_AVERAGES, DEFAULT_GAIN, DEFAULT_EXPOSURE_US
)
import logging
import threading
from diagnostics.utils import white_turbo

# Suppress Matplotlib debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
LOG_FORMAT = "[%(asctime)s] %(message)s"
DATE_FORMAT = "%H:%M:%S"


class QTextEditHandler(logging.Handler):
    """Logging handler that writes to a QTextEdit widget."""
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)
        self.widget.verticalScrollBar().setValue(
            self.widget.verticalScrollBar().maximum()
        )


class LiveCaptureThread(QThread):
    """Thread that continuously captures images from the camera."""
    image_signal = pyqtSignal(np.ndarray)

    def __init__(self, cam_ctrl, exp, gain, avgs, interval_ms):
        super().__init__()
        self.cam = cam_ctrl
        self.exposure = exp
        self.gain = gain
        self.avgs = avgs
        self.interval = interval_ms / 1000.0
        self._running = True
        self._lock = threading.Lock()

    def update_parameters(self, exp, gain, avgs, interval_ms):
        with self._lock:
            self.exposure = exp
            self.gain = gain
            self.avgs = avgs
            self.interval = interval_ms / 1000.0

    def run(self):
        while self._running:
            try:
                with self._lock:
                    exp, g, av, wait = (self.exposure, self.gain, self.avgs, self.interval)
                img = self.cam.take_image(exp, g, av)
                self.image_signal.emit(img)
                time.sleep(wait)
            except DahengControllerError as ce:
                print(f"Camera error in capture thread: {ce}")
                break
            except Exception as e:
                print(f"Error capturing image: {e}")
                break

    def stop(self):
        self._running = False


class DummyCaptureThread(QThread):
    """Thread that simulates image capture with Gaussian + noise."""
    image_signal = pyqtSignal(np.ndarray)

    def __init__(self, exp, gain, avgs, interval_ms, shape=(512, 512)):
        super().__init__()
        self.exposure = exp
        self.gain = gain
        self.avgs = avgs
        self.interval = interval_ms / 1000.0
        self.shape = shape
        self._running = True
        self._lock = threading.Lock()

    def update_parameters(self, exp, gain, avgs, interval_ms):
        with self._lock:
            self.exposure = exp
            self.gain = gain
            self.avgs = avgs
            self.interval = interval_ms / 1000.0

    def run(self):
        x = np.linspace(-1, 1, self.shape[1])
        y = np.linspace(-1, 1, self.shape[0])
        xv, yv = np.meshgrid(x, y)
        sigma = 0.4
        base = np.exp(-(xv**2 + yv**2) / (2 * sigma**2)) * 255
        while self._running:
            try:
                noise = np.random.normal(0, 20, self.shape)
                img = np.clip(base + noise, 0, 255).astype(np.uint8)
                self.image_signal.emit(img)
                time.sleep(self.interval)
            except Exception as e:
                print(f"Error in dummy capture: {e}")
                break

    def stop(self):
        self._running = False


class DahengLive(QWidget):
    """Main GUI for live camera capture and display."""
    def __init__(self, camera_name="DahengLive", fixed_index=0):
        super().__init__()
        self.camera_name = camera_name
        self.fixed_index = fixed_index
        self.setWindowTitle(f"DahengLive - {camera_name}")

        self.cam = None
        self.thread = None
        self.dummy = False
        self.debug_mode = False

        self.image_artist = None
        self.cbar = None
        self.last_frame = None

        self.fix_cbar = False
        self.fixed_vmax = None

        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter()

        # Parameter panel
        param_panel = QWidget()
        param_layout = QVBoxLayout(param_panel)

        # Camera index display
        idx_layout = QHBoxLayout()
        idx_layout.addWidget(QLabel("Camera Index:"))
        idx_layout.addWidget(QLabel(str(self.fixed_index)))
        param_layout.addLayout(idx_layout)

        # Exposure input
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure (µs):"))
        self.exposure_edit = QLineEdit(str(DEFAULT_EXPOSURE_US))
        self.exposure_edit.setValidator(QIntValidator(MIN_EXPOSURE_US, MAX_EXPOSURE_US, self))
        self.exposure_edit.textChanged.connect(self.update_params)
        exp_layout.addWidget(self.exposure_edit)
        param_layout.addLayout(exp_layout)

        # Gain input
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.gain_edit = QLineEdit(str(DEFAULT_GAIN))
        self.gain_edit.setValidator(QIntValidator(MIN_GAIN, MAX_GAIN, self))
        self.gain_edit.textChanged.connect(self.update_params)
        gain_layout.addWidget(self.gain_edit)
        param_layout.addLayout(gain_layout)

        # Averages input
        avg_layout = QHBoxLayout()
        avg_layout.addWidget(QLabel("Averages:"))
        self.avgs_edit = QLineEdit(str(DEFAULT_AVERAGES))
        self.avgs_edit.setValidator(QIntValidator(1, 1000, self))
        self.avgs_edit.textChanged.connect(self.update_params)
        avg_layout.addWidget(self.avgs_edit)
        param_layout.addLayout(avg_layout)

        # Update interval input
        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Update Interval (ms):"))
        self.interval_edit = QLineEdit("1000")
        self.interval_edit.setValidator(QIntValidator(100, 10000, self))
        self.interval_edit.textChanged.connect(self.update_params)
        int_layout.addWidget(self.interval_edit)
        param_layout.addLayout(int_layout)

        # Comment field
        comment_layout = QHBoxLayout()
        comment_layout.addWidget(QLabel("Comment:"))
        self.comment_edit = QLineEdit()
        comment_layout.addWidget(self.comment_edit)
        param_layout.addLayout(comment_layout)

        # Auto-adjust exposure checkbox
        self.auto_adj = QCheckBox("Auto Adjust Exposure")
        param_layout.addWidget(self.auto_adj)

        # Background checkbox
        self.background_cb = QCheckBox("Background")
        param_layout.addWidget(self.background_cb)

        # Fix Colorbar Max checkbox and text field
        self.fix_cb = QCheckBox("Fix Colorbar Max")
        self.fix_cb.toggled.connect(self.on_fix_cbar)
        param_layout.addWidget(self.fix_cb)

        self.fix_value_edit = QLineEdit("255")
        self.fix_value_edit.setValidator(QIntValidator(0, 255, self))
        self.fix_value_edit.setEnabled(False)
        self.fix_value_edit.textChanged.connect(self.on_fix_value_changed)
        param_layout.addWidget(self.fix_value_edit)

        # Enable/disable text field when checkbox toggles
        self.fix_cb.toggled.connect(self.fix_value_edit.setEnabled)

        # Buttons
        btn_layout = QVBoxLayout()
        def make_btn(text, slot, enabled=True):
            b = QPushButton(text)
            b.clicked.connect(slot)
            b.setEnabled(enabled)
            btn_layout.addWidget(b)
            return b

        self.activate_camera_btn = make_btn("Activate Camera", self.activate_camera, True)
        self.activate_dummy_btn = make_btn("Activate Dummy", self.activate_dummy, True)
        self.deactivate_camera_btn = make_btn("Deactivate Camera", self.deactivate_camera, False)
        self.start_capture_btn = make_btn("Start Live Capture", self.start_capture, False)
        self.stop_capture_btn = make_btn("Stop Live Capture", self.stop_capture, False)
        self.debug_btn = make_btn("Enable Debug Mode", self.toggle_debug_mode, True)
        self.save_btn = make_btn("Save Frame", self.save_frame, True)
        param_layout.addLayout(btn_layout)

        # Log text box
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        param_layout.addWidget(self.log_text)

        splitter.addWidget(param_panel)

        # Plot panel
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax.set_title(f"DahengLive - {self.camera_name}")
        self.figure.subplots_adjust(right=0.85)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_panel)

        main_layout.addWidget(splitter)
        self.resize(1080, 720)

    def log(self, message):
        """Append a timestamped message to the log widget."""
        now = datetime.datetime.now().strftime(DATE_FORMAT)
        self.log_text.append(f"[{now}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def update_params(self):
        """Handle parameter edits."""
        try:
            exp = int(self.exposure_edit.text())
            gain = int(self.gain_edit.text())
            avgs = int(self.avgs_edit.text())
            interval = int(self.interval_edit.text())
            if self.thread:
                self.thread.update_parameters(exp, gain, avgs, interval)
            if self.cam:
                self.cam.set_exposure(exp)
                self.cam.set_gain(gain)
                self.cam.set_average(avgs)
        except ValueError:
            pass

    def on_fix_cbar(self, checked):
        """Handle Fix Colorbar Max checkbox toggling."""
        self.fix_cbar = checked
        if checked:
            try:
                val = int(self.fix_value_edit.text())
                self.fixed_vmax = float(val)
                self.log(f"Colorbar max set to {self.fixed_vmax:.1f}")
            except ValueError:
                self.log("Invalid value: please enter 0–255")
                self.fix_cb.setChecked(False)
        else:
            self.fixed_vmax = None
            self.log("Colorbar max refreshed automatically")

    def on_fix_value_changed(self, text):
        """Update colorbar immediately when the fix value is edited."""
        if not self.fix_cb.isChecked() or self.image_artist is None:
            return
        try:
            val = int(text)
        except ValueError:
            return
        self.fixed_vmax = float(val)
        vmin, _ = self.image_artist.get_clim()
        self.image_artist.set_clim(vmin, self.fixed_vmax)
        if self.cbar:
            self.cbar.update_normal(self.image_artist)
        self.canvas.draw_idle()
        self.log(f"Colorbar max updated to {self.fixed_vmax:.1f}")

    def toggle_debug_mode(self):
        """Enable or disable debug logging."""
        self.debug_mode = not self.debug_mode
        if self.cam:
            self.cam.enable_debug(self.debug_mode)
        state = "enabled" if self.debug_mode else "disabled"
        self.log(f"Debug mode {state}")
        self.debug_btn.setText("Disable Debug Mode" if self.debug_mode else "Enable Debug Mode")

    def activate_camera(self):
        """Activate the real camera."""
        try:
            self.cam = DahengController(self.fixed_index)
            self.cam.activate()
            self.cam.logger.handlers.clear()
            self.cam.logger.addHandler(QTextEditHandler(self.log_text))
            self.cam.enable_debug(False)
            self.log(f"Camera {self.fixed_index} activated")
            self.activate_camera_btn.setEnabled(False)
            self.activate_dummy_btn.setEnabled(False)
            self.deactivate_camera_btn.setEnabled(True)
            self.start_capture_btn.setEnabled(True)
            self.dummy = False
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.log(f"Error activating camera: {e}")

    def activate_dummy(self):
        """Activate dummy image capture."""
        self.dummy = True
        self.cam = None
        self.log("Dummy camera activated")
        self.activate_dummy_btn.setEnabled(False)
        self.activate_camera_btn.setEnabled(False)
        self.deactivate_camera_btn.setEnabled(True)
        self.start_capture_btn.setEnabled(True)

    def deactivate_camera(self):
        """Deactivate camera or dummy mode."""
        if self.cam:
            self.cam.deactivate()
            self.log(f"Camera {self.fixed_index} deactivated")
        else:
            self.log("Dummy camera deactivated")
        self.cam = None
        self.dummy = False
        self.activate_camera_btn.setEnabled(True)
        self.activate_dummy_btn.setEnabled(True)
        self.deactivate_camera_btn.setEnabled(False)
        self.start_capture_btn.setEnabled(False)
        self.stop_capture_btn.setEnabled(False)

    def start_capture(self):
        """Start live image capture."""
        try:
            exp = int(self.exposure_edit.text())
            gain = int(self.gain_edit.text())
            avgs = int(self.avgs_edit.text())
            interval = int(self.interval_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameters")
            return

        if self.dummy:
            self.thread = DummyCaptureThread(exp, gain, avgs, interval)
        else:
            self.thread = LiveCaptureThread(self.cam, exp, gain, avgs, interval)

        self.thread.image_signal.connect(self.update_image)
        self.thread.start()
        msg = "Dummy live capture started" if self.dummy else "Live capture started"
        self.log(msg)
        self.start_capture_btn.setEnabled(False)
        self.stop_capture_btn.setEnabled(True)

    def stop_capture(self):
        """Stop live image capture."""
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.log("Live capture stopped")
            self.thread = None
        self.start_capture_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(False)

    def update_image(self, image):
        """Update the displayed image and colorbar."""
        self.last_frame = image
        vmin, vmax = float(image.min()), float(image.max())

        if self.image_artist is None:
            self.ax.clear()
            self.image_artist = self.ax.imshow(image, cmap=white_turbo(512), vmin=vmin, vmax=vmax)
            self.cbar = self.figure.colorbar(self.image_artist, ax=self.ax, fraction=0.046, pad=0.04)
        else:
            xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
            self.image_artist.set_data(image)
            if self.fix_cbar and self.fixed_vmax is not None:
                self.image_artist.set_clim(vmin, self.fixed_vmax)
            else:
                self.image_artist.set_clim(vmin, vmax)
            if self.cbar:
                self.cbar.update_normal(self.image_artist)
            self.ax.set_xlim(*xlim)
            self.ax.set_ylim(*ylim)

        self.canvas.draw_idle()
        if self.auto_adj.isChecked():
            self.adjust_exposure(image)

    def adjust_exposure(self, image):
        """Auto-adjust exposure based on region-of-interest max value."""
        try:
            x0, x1 = map(int, self.ax.get_xlim())
            y0, y1 = map(int, self.ax.get_ylim())
            x0, x1 = sorted((np.clip(x0, 0, image.shape[1] - 1), np.clip(x1, 0, image.shape[1] - 1)))
            y0, y1 = sorted((np.clip(y0, 0, image.shape[0] - 1), np.clip(y1, 0, image.shape[0] - 1)))
            roi = image[y0:y1, x0:x1]
            m = roi.max()
            cur = int(self.exposure_edit.text())
            new_exp = cur // 2 if m > 254 else int((255 / m) * 0.8 * cur)
            new_exp = max(MIN_EXPOSURE_US, min(new_exp, MAX_EXPOSURE_US))
            if new_exp != cur:
                self.exposure_edit.setText(str(new_exp))
                if self.debug_mode:
                    self.log(f"Exposure optimized to {new_exp} µs")
                self.update_params()
        except Exception as e:
            self.log(f"Error adjusting exposure: {e}")

    def save_frame(self):
        """Save the current frame with metadata and log it."""
        if self.last_frame is None:
            QMessageBox.warning(self, "Warning", "No frame to save")
            return

        now = datetime.datetime.now()
        folder = f"C:/data/{now:%Y-%m-%d}/{self.camera_name}/"
        os.makedirs(folder, exist_ok=True)
        ts = now.strftime("%Y%m%d_%H%M%S%f")
        exp = self.exposure_edit.text()
        gain = self.gain_edit.text()
        comment = self.comment_edit.text()
        fn = f"{self.camera_name}_{'Background' if self.background_cb.isChecked() else 'Image'}_{ts}.png"
        path = os.path.join(folder, fn)

        try:
            arr = np.clip(self.last_frame, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            info = PngImagePlugin.PngInfo()
            info.add_text("Exposure", exp)
            info.add_text("Gain", gain)
            info.add_text("Comment", comment)
            img.save(path, pnginfo=info)
            self.log(f"Frame saved to {path}")
        except Exception as e:
            self.log(f"Error saving frame: {e}")
            QMessageBox.critical(self, "Error", str(e))
            return

        logfn = f"{self.camera_name}_log_{now:%Y_%m_%d}.txt"
        logp = os.path.join(folder, logfn)
        header = "File Name\tExposure\tGain\tComment\n"
        try:
            if not os.path.exists(logp):
                with open(logp, "w") as f:
                    f.write(header)
            with open(logp, "a") as f:
                f.write(f"{fn}\t{exp}\t{gain}\t{comment}\n")
            self.log(f"Parameters logged to {logp}")
        except Exception as e:
            self.log(f"Error writing log: {e}")
            QMessageBox.critical(self, "Error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DahengLive(camera_name="Nozzle", fixed_index=1)
    gui.show()
    sys.exit(app.exec_())
