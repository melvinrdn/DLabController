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
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import logging
import threading

from hardware.wrappers.AndorController import (
    AndorController, AndorControllerError,
    DEFAULT_AVERAGES, DEFAULT_EXPOSURE_US, MIN_EXPOSURE_US, MAX_EXPOSURE_US
)
from diagnostics.utils import white_turbo

# Suppress Matplotlib debug messages.
logging.getLogger('matplotlib').setLevel(logging.WARNING)

LOG_FORMAT = "[%(asctime)s] %(message)s"
DATE_FORMAT = "%H:%M:%S"


###############################################################################
# Dummy Camera Controller
###############################################################################

class DummyAndorController:
    """
    A dummy controller that simulates an Andor camera.
    It always returns a 512x512 2D Gaussian image with some noise.
    """

    def __init__(self, device_index=0):
        self.device_index = device_index
        self.image_shape = (512, 512)
        self.current_exposure = DEFAULT_EXPOSURE_US
        self.logger = logging.getLogger(f"DummyAndorController_{self.device_index}")
        self.logger.propagate = False

    def enable_debug(self, debug_on=True):
        level = logging.DEBUG if debug_on else logging.INFO
        self.logger.setLevel(level)

    def activate(self):
        # Simulate activation by simply setting the image shape.
        self.logger.info(f"Dummy camera {self.device_index} activated with shape {self.image_shape}")

    def set_exposure(self, exposure):
        if not isinstance(exposure, int) or exposure <= 0:
            raise ValueError("Exposure must be a positive integer in microseconds.")
        self.current_exposure = exposure
        self.logger.debug(f"Dummy camera {self.device_index}: Exposure set to {exposure} µs")

    def take_image(self, exposure, avgs):
        """
        Simulate image capture by generating a 2D Gaussian with added noise.
        The amplitude is scaled with the exposure.
        """
        amplitude = 255 * (exposure / DEFAULT_EXPOSURE_US)
        sigma = 50
        center_x = self.image_shape[1] // 2
        center_y = self.image_shape[0] // 2
        x = np.arange(self.image_shape[1])
        y = np.arange(self.image_shape[0])
        xv, yv = np.meshgrid(x, y)
        gaussian = amplitude * np.exp(-(((xv - center_x) ** 2 + (yv - center_y) ** 2) / (2 * sigma ** 2)))
        noise = np.random.normal(0, amplitude * 0.05, self.image_shape)
        image = gaussian + noise
        image = np.clip(image, 0, 255)
        return image.astype(np.float64)

    def deactivate(self):
        self.logger.info(f"Dummy camera {self.device_index} deactivated.")


###############################################################################
# QTextEditHandler for logging to the GUI text box
###############################################################################

class QTextEditHandler(logging.Handler):
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


###############################################################################
# Live Capture Thread
###############################################################################


class LiveCaptureThread(QThread):
    image_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_controller, exposure, avgs, update_interval_ms):
        super().__init__()
        self.camera_controller = camera_controller
        self.exposure = exposure
        self.avgs = avgs
        self.update_interval_ms = update_interval_ms
        self.interval_sec = update_interval_ms / 1000.0
        self._running = True
        self._param_lock = threading.Lock()

    def update_parameters(self, exposure, avgs, update_interval_ms):
        with self._param_lock:
            self.exposure = exposure
            self.avgs = avgs
            self.update_interval_ms = update_interval_ms
            self.interval_sec = update_interval_ms / 1000.0

    def run(self):
        while self._running:
            try:
                with self._param_lock:
                    exp = self.exposure
                    av = self.avgs
                    sleep_interval = self.interval_sec
                image = self.camera_controller.take_image(exp, av)
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

###############################################################################
# AndorLive GUI Class
###############################################################################

class AndorLive(QWidget):
    """
    A PyQt5 GUI for live Andor camera image capture using AndorController.
    Includes options for real or dummy camera activation.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Andor Camera Feed")
        self.cam = None
        self.capture_thread = None
        self.debug_mode = False
        self.image_artist = None
        self.profile_line = None
        self.cbar = None
        self.current_interval_ms = None
        self.fixed_cbar_max = None
        self.last_frame = None  # To store the latest captured frame.
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter()

        # Parameter panel.
        param_panel = QWidget()
        param_layout = QVBoxLayout(param_panel)

        # Exposure parameter.
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure (µs):"))
        self.exposure_edit = QLineEdit(f"{DEFAULT_EXPOSURE_US}")
        self.exposure_edit.setValidator(QIntValidator(MIN_EXPOSURE_US, MAX_EXPOSURE_US, self))
        self.exposure_edit.textChanged.connect(self.update_capture_parameters)
        exp_layout.addWidget(self.exposure_edit)
        param_layout.addLayout(exp_layout)

        # Averaging parameter.
        avg_layout = QHBoxLayout()
        avg_layout.addWidget(QLabel("Averages:"))
        self.avgs_edit = QLineEdit(f"{DEFAULT_AVERAGES}")
        self.avgs_edit.setValidator(QIntValidator(1, 1000, self))
        self.avgs_edit.textChanged.connect(self.update_capture_parameters)
        avg_layout.addWidget(self.avgs_edit)
        param_layout.addLayout(avg_layout)

        # Update interval parameter.
        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Update Interval (ms):"))
        self.interval_edit = QLineEdit("1000")
        self.interval_edit.setValidator(QIntValidator(100, 10000, self))
        self.interval_edit.textChanged.connect(self.update_capture_parameters)
        int_layout.addWidget(self.interval_edit)
        param_layout.addLayout(int_layout)

        # MCP Voltage entry.
        mcp_layout = QHBoxLayout()
        mcp_layout.addWidget(QLabel("MCP Voltage:"))
        self.mcp_voltage_edit = QLineEdit("Not specified")
        mcp_layout.addWidget(self.mcp_voltage_edit)
        param_layout.addLayout(mcp_layout)

        # Comment entry.
        comment_layout = QHBoxLayout()
        comment_layout.addWidget(QLabel("Comment:"))
        self.comment_edit = QLineEdit()
        comment_layout.addWidget(self.comment_edit)
        param_layout.addLayout(comment_layout)

        # Checkbox to fix the colorbar.
        self.fix_cbar_checkbox = QCheckBox("Fix Colorbar Max")
        param_layout.addWidget(self.fix_cbar_checkbox)
        self.fix_value_edit = QLineEdit("10000")
        self.fix_value_edit.setValidator(QIntValidator(0, 1000000000, self))
        self.fix_value_edit.setEnabled(False)
        param_layout.addWidget(self.fix_value_edit)
        self.fix_cbar_checkbox.toggled.connect(self.fix_value_edit.setEnabled)
        self.fix_cbar_checkbox.toggled.connect(self.on_fix_cbar)
        self.fix_value_edit.textChanged.connect(self.on_fix_value_changed)

        # New checkbox for Background.
        self.background_checkbox = QCheckBox("Background")
        self.background_checkbox.setChecked(False)
        param_layout.addWidget(self.background_checkbox)

        # Buttons for camera control, live capture, and saving frame.
        btn_layout = QVBoxLayout()

        # Real camera activation.
        self.activate_button = QPushButton("Activate Camera")
        self.activate_button.clicked.connect(self.activate_camera)
        btn_layout.addWidget(self.activate_button)

        # Dummy camera activation.
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

        self.debug_button = QPushButton("Enable Debug Mode")
        self.debug_button.clicked.connect(self.toggle_debug_mode)
        btn_layout.addWidget(self.debug_button)

        # Save frame button.
        self.save_button = QPushButton("Save Frame")
        self.save_button.clicked.connect(self.save_frame)
        btn_layout.addWidget(self.save_button)

        param_layout.addLayout(btn_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        param_layout.addWidget(self.log_text)

        splitter.addWidget(param_panel)

        # Plot panel with Matplotlib canvas using gridspec.
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.figure, (self.ax_img, self.ax_profile) = plt.subplots(2, 1,
                                                                   gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        self.ax_img.set_title("Live Andor Camera Image")
        self.ax_profile.set_title("Integrated Profile")
        self.figure.subplots_adjust(right=0.85)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_panel)

        main_layout.addWidget(splitter)
        self.resize(1080, 720)

    def log(self, message):
        current_time = datetime.datetime.now().strftime(DATE_FORMAT)
        self.log_text.append(f"[{current_time}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def update_capture_parameters(self):
        try:
            exposure = int(self.exposure_edit.text())
            avgs = int(self.avgs_edit.text())
            update_interval = int(self.interval_edit.text())
            if (self.current_interval_ms is None or self.current_interval_ms != update_interval) and self.debug_mode:
                self.current_interval_ms = update_interval
                self.log(f"Update interval changed to {update_interval} ms.")
            if self.capture_thread:
                self.capture_thread.update_parameters(exposure, avgs, update_interval)
            elif self.cam:
                self.cam.set_exposure(exposure)
        except ValueError:
            pass

    def toggle_debug_mode(self):
        if self.cam:
            self.debug_mode = not self.debug_mode
            self.cam.enable_debug(debug_on=self.debug_mode)
            state = "enabled" if self.debug_mode else "disabled"
            self.log(f"Camera 0 Debug mode {state}.")
            self.debug_button.setText("Disable Debug Mode" if self.debug_mode else "Enable Debug Mode")
        else:
            QMessageBox.warning(self, "Warning", "Camera not activated. Activate the camera first.")

    def activate_camera(self):
        try:
            self.cam = AndorController(device_index=0)
            self.cam.activate()
            self.cam.logger.handlers = []
            qt_handler = QTextEditHandler(self.log_text)
            self.cam.logger.addHandler(qt_handler)
            self.cam.enable_debug(debug_on=False)
            self.log("Camera 0 activated.")
            self.activate_button.setEnabled(False)
            self.activate_dummy_button.setEnabled(False)
            self.deactivate_button.setEnabled(True)
            self.start_button.setEnabled(True)
        except AndorControllerError as ce:
            QMessageBox.critical(self, "Error", f"Failed to activate camera: {ce}")
            self.log(f"Error activating camera: {ce}")

    def activate_dummy_camera(self):
        try:
            self.cam = DummyAndorController(device_index=666)
            self.cam.activate()
            self.cam.enable_debug(debug_on=False)
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

    def start_capture(self):
        if self.cam is None:
            QMessageBox.critical(self, "Error", "Camera not activated.")
            return
        try:
            exposure = int(self.exposure_edit.text())
            avgs = int(self.avgs_edit.text())
            update_interval = int(self.interval_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameter values.")
            return
        self.capture_thread = LiveCaptureThread(self.cam, exposure, avgs, update_interval)
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

    def on_fix_cbar(self, checked):
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

    def on_fix_value_changed(self, text):
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

    def update_image(self, image):
        """
        Updates the displayed image on the Matplotlib canvas and the integrated profile,
        while preserving the current zoom settings.
        """
        self.last_frame = image  # Store the latest frame for saving.
        max_val = np.max(image)
        min_val = np.min(image)
        title_text = f"Live Andor Camera Image - Max: {max_val:.0f}"
        # Update top subplot.
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
        # Update bottom subplot (integrated intensity).
        profile = np.sum(image, axis=1)
        self.ax_profile.clear()
        self.ax_profile.plot(np.arange(image.shape[0]), profile, color='r')
        self.ax_profile.set_xlabel("(px)")
        self.ax_profile.set_ylabel("Integrated Intensity")
        self.ax_profile.set_xlim(0, image.shape[0])
        self.canvas.draw_idle()

    def save_frame(self):
        """
        Saves the most recent frame as a PNG file with embedded metadata (Exposure, Averages, MCP Voltage, and Comment).
        Also appends an entry to a log file recording the file name, Exposure, Averages, MCP Voltage, and Comment.
        The log file is named AndorCamera_log_YYYY_MM_DD.txt and uses tab ('\t') as the separator.
        If the Background checkbox is checked, the file name will include "Background" and the log comment will be "Background".
        """
        if self.last_frame is None:
            QMessageBox.warning(self, "Warning", "No frame available to save.")
            return

        now = datetime.datetime.now()
        # Build directory path based on current date.
        dir_path = f"C:/data/{now.strftime('%Y-%m-%d')}/AndorCamera/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        timestamp = now.strftime("%Y%m%d_%H%M%S%f")
        exposure_val = self.exposure_edit.text()
        avgs_val = self.avgs_edit.text()
        mcp_voltage_val = self.mcp_voltage_edit.text()
        comment_text = self.comment_edit.text()

        # If background checkbox is checked, override filename and comment.
        if self.background_checkbox.isChecked():
            file_name = f"AndorCamera_MCP_Background_{timestamp}.png"
            log_comment = comment_text
        else:
            file_name = f"AndorCamera_MCP_Image_{timestamp}.png"
            log_comment = comment_text

        file_path = os.path.join(dir_path, file_name)

        try:
            # Convert frame to uint8 and save with PNG metadata.
            frame_uint16 = self.last_frame.astype(np.uint16)

            # Create an image from the numpy array
            img = Image.fromarray(frame_uint16)

            # Add metadata
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Exposure", exposure_val)
            metadata.add_text("Averages", avgs_val)
            metadata.add_text("MCP Voltage", mcp_voltage_val)
            metadata.add_text("Comment", log_comment)

            # Save the image with metadata as a 16-bit PNG
            img.save(file_path, format='PNG', pnginfo=metadata)
            self.log(f"Frame saved to {file_path}")

            image = Image.fromarray(self.last_frame)
        except Exception as e:
            self.log(f"Error saving frame: {e}")
            QMessageBox.critical(self, "Error", f"Error saving frame: {e}")
            return

        # Log file: add header if file does not exist.
        log_file_name = f"AndorCamera_log_{now.strftime('%Y_%m_%d')}.txt"
        log_file_path = os.path.join(dir_path, log_file_name)
        header = "File Name\tExposure (µs)\tAverages\tMCP Voltage\tComment\n"

        try:
            if not os.path.exists(log_file_path):
                with open(log_file_path, "w") as log_file:
                    log_file.write(header)
            with open(log_file_path, "a") as log_file:
                log_file.write(f"{file_name}\t{exposure_val}\t{avgs_val}\t{mcp_voltage_val}\t{log_comment}\n")
            self.log(f"Camera parameters logged to {log_file_path}")
        except Exception as e:
            self.log(f"Error writing to log file: {e}")
            QMessageBox.critical(self, "Error", f"Error writing to log file: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AndorLive()
    gui.show()
    sys.exit(app.exec_())
