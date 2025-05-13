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
from diagnostics.utils import black_red, white_turbo

# pixel size in meters
PIXEL_SIZE_M = 3.45e-6

# Minimum display interval = 500 ms = 500_000 µs
MIN_INTERVAL_US = 500_000

# Default camera exposure (µs)
DEFAULT_EXPOSURE_US_C = DEFAULT_EXPOSURE_US

# Suppress matplotlib debug messages
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
    """Continuously capture images from the camera."""
    image_signal = pyqtSignal(np.ndarray)

    def __init__(self, cam_ctrl, exposure_us, gain, avgs, interval_us):
        super().__init__()
        self.cam = cam_ctrl
        self.exposure = exposure_us
        self.gain = gain
        self.avgs = avgs
        self.interval = interval_us / 1e6
        self._running = True
        self._lock = threading.Lock()

    def update_parameters(self, exposure_us, gain, avgs, interval_us):
        with self._lock:
            self.exposure = exposure_us
            self.gain = gain
            self.avgs = avgs
            self.interval = interval_us / 1e6

    def run(self):
        while self._running:
            try:
                with self._lock:
                    exp, g, av, wait = (self.exposure, self.gain, self.avgs, self.interval)
                img = None
                for _ in range(3):
                    img = self.cam.take_image(exp, g, av)
                    if isinstance(img, np.ndarray) and img.dtype != np.object_:
                        break
                    time.sleep(0.1)
                if not (isinstance(img, np.ndarray) and img.dtype != np.object_):
                    continue
                self.image_signal.emit(img)
                time.sleep(wait)
            except DahengControllerError as ce:
                print(f"Camera error: {ce}")
                break
            except Exception as e:
                print(f"Capture error: {e}")

    def stop(self):
        self._running = False


class DummyCaptureThread(QThread):
    """Simulate image capture with Gaussian + noise."""
    image_signal = pyqtSignal(np.ndarray)

    def __init__(self, exposure_us, gain, avgs, interval_us, shape=(512, 512)):
        super().__init__()
        self.exposure = exposure_us
        self.gain = gain
        self.avgs = avgs
        self.interval = interval_us / 1e6
        self.shape = shape
        self._running = True
        self._lock = threading.Lock()

    def update_parameters(self, exposure_us, gain, avgs, interval_us):
        with self._lock:
            self.exposure = exposure_us
            self.gain = gain
            self.avgs = avgs
            self.interval = interval_us / 1e6

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
                print(f"Dummy capture error: {e}")
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

        self.cmap = black_red(512) if camera_name == 'Nomarski' else white_turbo(512)
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

        # Camera index
        idx_layout = QHBoxLayout()
        idx_layout.addWidget(QLabel("Camera Index:"))
        idx_layout.addWidget(QLabel(str(self.fixed_index)))
        param_layout.addLayout(idx_layout)

        # Exposure input
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure (µs):"))
        self.exposure_edit = QLineEdit(str(DEFAULT_EXPOSURE_US_C))
        self.exposure_edit.setValidator(QIntValidator(MIN_EXPOSURE_US, MAX_EXPOSURE_US, self))
        self.exposure_edit.textChanged.connect(self.update_params)
        exp_layout.addWidget(self.exposure_edit)
        param_layout.addLayout(exp_layout)

        # Update interval display
        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Update Interval (µs):"))
        init_interval = max(DEFAULT_EXPOSURE_US_C, MIN_INTERVAL_US)
        self.interval_edit = QLineEdit(str(init_interval))
        self.interval_edit.setEnabled(False)
        int_layout.addWidget(self.interval_edit)
        param_layout.addLayout(int_layout)

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

        # Comment field
        comment_layout = QHBoxLayout()
        comment_layout.addWidget(QLabel("Comment:"))
        self.comment_edit = QLineEdit()
        comment_layout.addWidget(self.comment_edit)
        param_layout.addLayout(comment_layout)

        # Auto-adjust exposure
        self.auto_adj = QCheckBox("Auto Adjust Exposure")
        param_layout.addWidget(self.auto_adj)

        # Background checkbox
        self.background_cb = QCheckBox("Background")
        param_layout.addWidget(self.background_cb)

        # Fix Colorbar Max
        self.fix_cb = QCheckBox("Fix Colorbar Max")
        self.fix_cb.toggled.connect(self.on_fix_cbar)
        param_layout.addWidget(self.fix_cb)
        self.fix_value_edit = QLineEdit("255")
        self.fix_value_edit.setValidator(QIntValidator(0, 255, self))
        self.fix_value_edit.setEnabled(False)
        self.fix_value_edit.textChanged.connect(self.on_fix_value_changed)
        param_layout.addWidget(self.fix_value_edit)
        self.fix_cb.toggled.connect(self.fix_value_edit.setEnabled)

        # Buttons
        btn_layout = QVBoxLayout()
        def make_btn(text, slot, enabled=True):
            b = QPushButton(text)
            b.clicked.connect(slot)
            b.setEnabled(enabled)
            btn_layout.addWidget(b)
            return b

        self.activate_camera_btn = make_btn("Activate Camera", self.activate_camera)
        self.activate_dummy_btn  = make_btn("Activate Dummy",  self.activate_dummy)
        self.deactivate_camera_btn = make_btn("Deactivate Camera", self.deactivate_camera, False)
        self.start_capture_btn   = make_btn("Start Live Capture", self.start_capture, False)
        self.stop_capture_btn    = make_btn("Stop Live Capture",  self.stop_capture, False)
        self.debug_btn           = make_btn("Enable Debug Mode",  self.toggle_debug_mode)
        self.save_btn            = make_btn("Save Frame",          self.save_frame)
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
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax.set_title(f"DahengLive - {self.camera_name}")
        self.figure.subplots_adjust(right=0.85)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_panel)

        main_layout.addWidget(splitter)
        self.resize(1080, 720)

    def closeEvent(self, event):
        """Ensure capture is stopped and camera deactivated on window close."""
        if self.thread:
            self.thread.stop()
            self.thread.wait()
        if self.cam:
            try:
                self.cam.deactivate()
            except Exception:
                pass
        event.accept()

    def log(self, message):
        now = datetime.datetime.now().strftime(DATE_FORMAT)
        self.log_text.append(f"[{now}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def update_params(self):
        """Sync update interval to exposure (with 500 ms minimum)."""
        try:
            exp_us = int(self.exposure_edit.text())
            gain   = int(self.gain_edit.text())
            avgs   = int(self.avgs_edit.text())
            interval_us = exp_us if exp_us >= MIN_INTERVAL_US else MIN_INTERVAL_US
            self.interval_edit.setText(str(interval_us))
            if self.thread:
                self.thread.update_parameters(exp_us, gain, avgs, interval_us)
            if self.cam:
                self.cam.set_exposure(exp_us)
                self.cam.set_gain(gain)
                self.cam.set_average(avgs)
        except ValueError:
            pass

    def on_fix_cbar(self, checked):
        self.fix_cbar = checked
        if checked:
            try:
                self.fixed_vmax = float(self.fix_value_edit.text())
                self.log(f"Colorbar max set to {self.fixed_vmax:.1f}")
            except ValueError:
                self.log("Invalid colorbar max value")
                self.fix_cb.setChecked(False)
        else:
            self.fixed_vmax = None
            self.log("Colorbar auto scale")

    def on_fix_value_changed(self, text):
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

    def toggle_debug_mode(self):
        self.debug_mode = not self.debug_mode
        if self.cam:
            self.cam.enable_debug(self.debug_mode)
        state = "enabled" if self.debug_mode else "disabled"
        self.log(f"Debug mode {state}")
        self.debug_btn.setText(
            "Disable Debug Mode" if self.debug_mode else "Enable Debug Mode"
        )

    def activate_camera(self):
        """Activate the real camera, cleaning up any leftover stream."""
        # first deactivate any existing handle
        if self.cam:
            try:
                self.cam.deactivate()
            except Exception:
                pass
            self.cam = None

        try:
            self.cam = DahengController(self.fixed_index)
            self.cam.activate()
        except DahengControllerError as e:
            errmsg = str(e)
            if "Device.stream_on" in errmsg:
                # retry once after forced close
                self.log("Stream already on—forcing camera close and retrying…")
                try:
                    self.cam.deactivate()
                except Exception:
                    pass
                try:
                    self.cam = DahengController(self.fixed_index)
                    self.cam.activate()
                except Exception as e2:
                    QMessageBox.critical(
                        self, "Error",
                        f"Failed first: {errmsg}\nRetry failed: {e2}"
                    )
                    self.log(f"Retry activation failed: {e2}")
                    return
            else:
                QMessageBox.critical(self, "Error", errmsg)
                self.log(f"Error activating camera: {errmsg}")
                return

        # on success
        self.cam.logger.handlers.clear()
        self.cam.logger.addHandler(QTextEditHandler(self.log_text))
        self.cam.enable_debug(False)
        self.log(f"Camera {self.fixed_index} activated")
        self.activate_camera_btn.setEnabled(False)
        self.activate_dummy_btn.setEnabled(False)
        self.deactivate_camera_btn.setEnabled(True)
        self.start_capture_btn.setEnabled(True)
        self.dummy = False

    def activate_dummy(self):
        self.dummy = True
        self.cam = None
        self.log("Dummy camera activated")
        self.activate_dummy_btn.setEnabled(False)
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
        try:
            exp_us = int(self.exposure_edit.text())
            gain = int(self.gain_edit.text())
            avgs = int(self.avgs_edit.text())
            interval_us = exp_us if exp_us >= MIN_INTERVAL_US else MIN_INTERVAL_US
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameters")
            return

        if self.dummy:
            self.thread = DummyCaptureThread(exp_us, gain, avgs, interval_us)
        else:
            self.thread = LiveCaptureThread(self.cam, exp_us, gain, avgs, interval_us)

        self.thread.image_signal.connect(self.update_image)
        self.thread.start()
        self.log("Dummy live capture started" if self.dummy else "Live capture started")
        self.start_capture_btn.setEnabled(False)
        self.stop_capture_btn.setEnabled(True)

    def stop_capture(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.log("Live capture stopped")
            self.thread = None
        self.start_capture_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(False)

    def update_image(self, image):
        """Update the displayed image, converting pixel axes to meters."""
        if not isinstance(image, np.ndarray) or image.dtype == np.object_:
            self.log("Invalid image received")
            return

        self.last_frame = image
        vmin, vmax = float(image.min()), float(image.max())
        height, width = image.shape

        # compute physical extent in mm:
        x_m = width * PIXEL_SIZE_M * 1e3
        y_m = height * PIXEL_SIZE_M * 1e3

        if self.image_artist is None:
            self.ax.clear()
            # use extent to map pixels → meters
            self.image_artist = self.ax.imshow(
                image,
                cmap=self.cmap,
                vmin=vmin, vmax=vmax,
                extent=[0, x_m, 0, y_m],
                origin='lower',
                aspect='equal'
            )
            self.cbar = self.figure.colorbar(
                self.image_artist, ax=self.ax, fraction=0.046, pad=0.04
            )
        else:
            # update data only; extent stays the same
            self.image_artist.set_data(image)
            if self.fix_cbar and self.fixed_vmax is not None:
                self.image_artist.set_clim(vmin, self.fixed_vmax)
            else:
                self.image_artist.set_clim(vmin, vmax)
            if self.cbar:
                self.cbar.update_normal(self.image_artist)

        # redraw
        self.canvas.draw_idle()

        # optionally auto-adjust exposure
        if self.auto_adj.isChecked():
            self.adjust_exposure(image)

    def adjust_exposure(self, image):
        """Auto-adjust exposure based on ROI max intensity."""
        try:
            x0, x1 = map(int, (self.ax.get_xlim()[0] / PIXEL_SIZE_M,
                               self.ax.get_xlim()[1] / PIXEL_SIZE_M))
            y0, y1 = map(int, (self.ax.get_ylim()[0] / PIXEL_SIZE_M,
                               self.ax.get_ylim()[1] / PIXEL_SIZE_M))
            x0, x1 = sorted((np.clip(x0, 0, image.shape[1] - 1),
                             np.clip(x1, 0, image.shape[1] - 1)))
            y0, y1 = sorted((np.clip(y0, 0, image.shape[0] - 1),
                             np.clip(y1, 0, image.shape[0] - 1)))
            roi = image[y0:y1, x0:x1]
            m = roi.max()
            cur_us = int(self.exposure_edit.text())
            new_us = cur_us // 2 if m > 254 else int((255 / m) * 0.8 * cur_us)
            new_us = max(MIN_EXPOSURE_US, min(new_us, MAX_EXPOSURE_US))
            if new_us != cur_us:
                self.exposure_edit.setText(str(new_us))
                if self.debug_mode:
                    self.log(f"Exposure optimized to {new_us} µs")
                self.update_params()
        except Exception as e:
            self.log(f"Error adjusting exposure: {e}")

    def save_frame(self):
        """Save the current frame with metadata and log parameters."""
        if self.last_frame is None:
            QMessageBox.warning(self, "Warning", "No frame to save")
            return

        now = datetime.datetime.now()
        folder = f"C:/data/{now:%Y-%m-%d}/{self.camera_name}/"
        os.makedirs(folder, exist_ok=True)
        ts = now.strftime("%Y%m%d_%H%M%S%f")
        exp_us = self.exposure_edit.text()
        gain = self.gain_edit.text()
        comment = self.comment_edit.text()
        fn = f"{self.camera_name}_{'Background' if self.background_cb.isChecked() else 'Image'}_{ts}.png"
        path = os.path.join(folder, fn)

        try:
            arr = np.clip(self.last_frame, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            info = PngImagePlugin.PngInfo()
            info.add_text("Exposure_us", exp_us)
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
        header = "File Name\tExposure_us\tGain\tComment\n"
        try:
            if not os.path.exists(logp):
                with open(logp, "w") as f:
                    f.write(header)
            with open(logp, "a") as f:
                f.write(f"{fn}\t{exp_us}\t{gain}\t{comment}\n")
            self.log(f"Parameters logged to {logp}")
        except Exception as e:
            self.log(f"Error writing log: {e}")
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DahengLive(camera_name="Nomarski", fixed_index=1)
    gui.show()
    sys.exit(app.exec_())