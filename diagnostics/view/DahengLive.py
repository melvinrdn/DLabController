import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QMessageBox, QSplitter, QComboBox, QCheckBox
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)

from hardware.wrappers.DahengController import (DahengController, DahengControllerError,
                                                MAX_GAIN, MAX_EXPOSURE_US, MIN_GAIN, MIN_EXPOSURE_US,
                                                DEFAULT_AVERAGES, DEFAULT_GAIN, DEFAULT_EXPOSURE_US)
import logging
import threading
from diagnostics.utils import white_turbo

# Suppress Matplotlib debug messages.
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Common log format and date format.
LOG_FORMAT = "[%(asctime)s] %(message)s"
DATE_FORMAT = "%H:%M:%S"


class QTextEditHandler(logging.Handler):
    """
    Custom logging handler to send log messages to a QTextEdit widget.
    """

    def __init__(self, widget):
        """
        Initializes the handler with the given QTextEdit widget.

        Parameters:
            widget (QTextEdit): The text widget to which log messages will be appended.
        """
        super().__init__()
        self.widget = widget
        self.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    def emit(self, record):
        """
        Emits a formatted log record to the text widget and auto-scrolls to the bottom.

        Parameters:
            record (logging.LogRecord): The record to be logged.
        """
        msg = self.format(record)
        self.widget.append(msg)
        self.widget.verticalScrollBar().setValue(self.widget.verticalScrollBar().maximum())


class LiveCaptureThread(QThread):
    """
    QThread subclass to continuously capture images from the camera.

    Attributes:
        image_signal (pyqtSignal): Signal emitted when an image is captured (np.ndarray).
    """
    image_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_controller, exposure, gain, avgs, update_interval_ms):
        """
        Initializes the live capture thread.

        Parameters:
            camera_controller (DahengController): The camera controller to capture images from.
            exposure (int): Initial exposure time in microseconds.
            gain (int): Initial gain value.
            avgs (int): Initial number of frames to average.
            update_interval_ms (int): Update interval in milliseconds.
        """
        super().__init__()
        self.camera_controller = camera_controller
        self.exposure = exposure
        self.gain = gain
        self.avgs = avgs
        self.update_interval_ms = update_interval_ms
        self.interval_sec = update_interval_ms / 1000.0  # Convert ms to seconds.
        self._running = True
        self._param_lock = threading.Lock()

    def update_parameters(self, exposure, gain, avgs, update_interval_ms):
        """
        Updates capture parameters dynamically.

        Parameters:
            exposure (int): New exposure time in microseconds.
            gain (int): New gain value.
            avgs (int): New number of frames to average.
            update_interval_ms (int): New update interval in milliseconds.
        """
        with self._param_lock:
            self.exposure = exposure
            self.gain = gain
            self.avgs = avgs
            self.update_interval_ms = update_interval_ms
            self.interval_sec = update_interval_ms / 1000.0

    def run(self):
        """
        Continuously captures images using updated parameters and emits them.
        """
        while self._running:
            try:
                with self._param_lock:
                    exp = self.exposure
                    g = self.gain
                    av = self.avgs
                    sleep_interval = self.interval_sec
                image = self.camera_controller.take_image(exp, g, av)
                self.image_signal.emit(image)
                time.sleep(sleep_interval)
            except DahengControllerError as ce:
                print(f"CameraError in capture thread: {ce}")
                break
            except Exception as e:
                print("Error capturing image:", e)
                break

    def stop(self):
        """
        Stops the live capture thread.
        """
        self._running = False


class DahengLive(QWidget):
    """
    A PyQt5 GUI for live camera image capture using DahengController.

    Provides controls for:
      - Selecting the camera index.
      - Setting exposure (in microseconds), gain, and averaging parameters (only integers allowed),
        and the update interval (in milliseconds).
      - Activating and deactivating the camera.
      - Starting and stopping live capture.
      - Toggling debug mode (which displays detailed log messages).
      - Enabling automatic exposure adjustment on the zoomed image.

    Captured images are displayed on a Matplotlib canvas. Log messages are shown in a text box
    with a consistent format and auto-scroll.
    """

    def __init__(self):
        """
        Initializes the DahengLive GUI.
        """
        super().__init__()
        self.setWindowTitle("Live Camera Feed")
        self.camera_controller = None
        self.capture_thread = None
        self.debug_mode = False  # Track debug mode state.
        self.image_artist = None  # To preserve zoomed axes.
        self.current_interval_ms = None  # Stores current update interval.
        self.initUI()

    def initUI(self):
        """
        Sets up the user interface including parameter controls, log box, and image display.
        """
        main_layout = QHBoxLayout(self)
        splitter = QSplitter()

        # Parameter panel.
        param_panel = QWidget()
        param_layout = QVBoxLayout(param_panel)

        # Camera index selection.
        idx_layout = QHBoxLayout()
        idx_layout.addWidget(QLabel("Camera Index:"))
        self.index_combo = QComboBox()
        self.refresh_indices()
        idx_layout.addWidget(self.index_combo)
        refresh_button = QPushButton("Refresh Indices")
        refresh_button.clicked.connect(self.refresh_indices)
        idx_layout.addWidget(refresh_button)
        param_layout.addLayout(idx_layout)

        # Exposure parameter with QIntValidator.
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure (µs):"))
        self.exposure_edit = QLineEdit(f"{DEFAULT_EXPOSURE_US}")
        self.exposure_edit.setValidator(QIntValidator(MIN_EXPOSURE_US, MAX_EXPOSURE_US, self))
        self.exposure_edit.textChanged.connect(self.update_capture_parameters)
        exp_layout.addWidget(self.exposure_edit)
        param_layout.addLayout(exp_layout)

        # Gain parameter with QIntValidator.
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.gain_edit = QLineEdit(f"{DEFAULT_GAIN}")
        self.gain_edit.setValidator(QIntValidator(MIN_GAIN, MAX_GAIN, self))
        self.gain_edit.textChanged.connect(self.update_capture_parameters)
        gain_layout.addWidget(self.gain_edit)
        param_layout.addLayout(gain_layout)

        # Averaging parameter with QIntValidator.
        avg_layout = QHBoxLayout()
        avg_layout.addWidget(QLabel("Averages:"))
        self.avgs_edit = QLineEdit(f"{DEFAULT_AVERAGES}")
        self.avgs_edit.setValidator(QIntValidator(1, 1000, self))
        self.avgs_edit.textChanged.connect(self.update_capture_parameters)
        avg_layout.addWidget(self.avgs_edit)
        param_layout.addLayout(avg_layout)

        # Update interval parameter with QIntValidator.
        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Update Interval (ms):"))
        self.interval_edit = QLineEdit("1000")
        self.interval_edit.setValidator(QIntValidator(100, 10000, self))
        self.interval_edit.textChanged.connect(self.update_capture_parameters)
        int_layout.addWidget(self.interval_edit)
        param_layout.addLayout(int_layout)

        # Checkbox for automatic exposure adjustment.
        self.auto_adjust_checkbox = QCheckBox("Auto Adjust Exposure")
        param_layout.addWidget(self.auto_adjust_checkbox)

        # Buttons for hardware control and live capture.
        btn_layout = QVBoxLayout()
        self.activate_button = QPushButton("Activate Camera")
        self.activate_button.clicked.connect(self.activate_camera)
        btn_layout.addWidget(self.activate_button)

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

        # Button to toggle debug mode.
        self.debug_button = QPushButton("Enable Debug Mode")
        self.debug_button.clicked.connect(self.toggle_debug_mode)
        btn_layout.addWidget(self.debug_button)

        param_layout.addLayout(btn_layout)

        # Log text box.
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        param_layout.addWidget(self.log_text)

        splitter.addWidget(param_panel)

        # Plot panel with Matplotlib canvas.
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.figure, self.ax = plt.subplots()
        self.ax.set_title("Live Camera Image")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_panel)

        main_layout.addWidget(splitter)
        self.resize(1080, 720)

    def refresh_indices(self):
        """
        Refreshes the list of available camera indices and populates the combo box.
        """
        self.index_combo.clear()
        indices = DahengController.get_available_indices()
        for idx in indices:
            self.index_combo.addItem(str(idx))

    def log(self, message):
        """
        Appends a log message to the log text box using the common format and auto-scrolls.

        Parameters:
            message (str): The message to log.
        """
        current_time = datetime.datetime.now().strftime(DATE_FORMAT)
        self.log_text.append(f"[{current_time}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def update_capture_parameters(self):
        """
        Reads updated parameters from the input fields and applies them immediately.
        Logs a message if the update interval has changed.
        """
        try:
            exposure = int(self.exposure_edit.text())
            gain = int(self.gain_edit.text())
            avgs = int(self.avgs_edit.text())
            update_interval = int(self.interval_edit.text())
            if self.debug_mode and (self.current_interval_ms is None or self.current_interval_ms != update_interval):
                self.current_interval_ms = update_interval
                self.log(f"Update interval changed to {update_interval} ms.")
            if self.capture_thread:
                self.capture_thread.update_parameters(exposure, gain, avgs, update_interval)
            if self.camera_controller:
                self.camera_controller.set_exposure(exposure)
                self.camera_controller.set_gain(gain)
                self.camera_controller.set_average(avgs)
        except ValueError:
            # Ignore invalid values.
            pass

    def adjust_exposure(self, image):
        """
        Automatically adjusts the exposure based on the brightness of the zoomed image region.

        The method extracts the sub-image corresponding to the current zoom (axes limits) and
        calculates its maximum pixel value. If the maximum is greater than 254, exposure is halved.
        Otherwise, new exposure is computed as (255 / max_pixel) * 0.8 * current_exposure.
        The new exposure is clamped to the range [MIN_EXPOSURE_US, MAX_EXPOSURE_US] and the GUI is updated.
        This adjustment and log message are only performed in debug mode.

        Parameters:
            image (np.ndarray): The latest captured full image.
        """
        try:
            # Get current axes limits and convert to integer indices.
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            x0 = int(np.clip(x0, 0, image.shape[1] - 1))
            x1 = int(np.clip(x1, 0, image.shape[1] - 1))
            y0 = int(np.clip(y0, 0, image.shape[0] - 1))
            y1 = int(np.clip(y1, 0, image.shape[0] - 1))
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            sub_image = image[y0:y1, x0:x1]
            max_val = np.max(sub_image)
            current_exposure = int(self.exposure_edit.text())
            if max_val <= 0:
                return
            if max_val > 254:
                new_exposure = int(current_exposure / 2)
            else:
                new_exposure = int((255 / max_val) * 0.8 * current_exposure)
            new_exposure = max(MIN_EXPOSURE_US, min(new_exposure, MAX_EXPOSURE_US))
            if new_exposure != current_exposure:
                self.exposure_edit.setText(str(new_exposure))
                if self.debug_mode:
                    self.log(f"Exposure optimized to {new_exposure} µs.")
                self.update_capture_parameters()
        except Exception as e:
            self.log(f"Error adjusting exposure: {e}")

    def toggle_debug_mode(self):
        """
        Toggles the debug mode of the camera controller and updates the GUI.
        """
        if self.camera_controller:
            self.debug_mode = not self.debug_mode
            self.camera_controller.enable_debug(debug_on=self.debug_mode)
            state = "enabled" if self.debug_mode else "disabled"
            self.log(f"Camera {self.camera_controller.index} Debug mode {state}.")
            self.debug_button.setText("Disable Debug Mode" if self.debug_mode else "Enable Debug Mode")
        else:
            QMessageBox.warning(self, "Warning", "Camera not activated. Activate the camera first.")

    def activate_camera(self):
        """
        Activates the camera based on the selected index and configures logging.
        """
        try:
            selected_index = int(self.index_combo.currentText())
            self.camera_controller = DahengController(selected_index)
            self.camera_controller.activate()
            self.camera_controller.logger.handlers = []
            qt_handler = QTextEditHandler(self.log_text)
            self.camera_controller.logger.addHandler(qt_handler)
            self.camera_controller.enable_debug(debug_on=False)
            self.log(f"Camera {selected_index} activated.")
            self.activate_button.setEnabled(False)
            self.deactivate_button.setEnabled(True)
            self.start_button.setEnabled(True)
        except DahengControllerError as ce:
            QMessageBox.critical(self, "Error", f"Failed to activate camera: {ce}")
            self.log(f"Error activating camera: {ce}")

    def deactivate_camera(self):
        """
        Deactivates the camera and resets the GUI controls.
        """
        try:
            if self.camera_controller:
                self.camera_controller.deactivate()
                self.log(f"Camera {self.camera_controller.index} deactivated.")
            self.camera_controller = None
            self.activate_button.setEnabled(True)
            self.deactivate_button.setEnabled(False)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
        except DahengControllerError as ce:
            QMessageBox.critical(self, "Error", f"Failed to deactivate camera: {ce}")
            self.log(f"Error deactivating camera: {ce}")

    def start_capture(self):
        """
        Starts the live capture thread using the current parameters.
        """
        if self.camera_controller is None:
            QMessageBox.critical(self, "Error", "Camera not activated.")
            return
        try:
            exposure = int(self.exposure_edit.text())
            gain = int(self.gain_edit.text())
            avgs = int(self.avgs_edit.text())
            update_interval = int(self.interval_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameter values.")
            return
        self.capture_thread = LiveCaptureThread(self.camera_controller, exposure, gain, avgs, update_interval)
        self.capture_thread.image_signal.connect(self.update_image)
        self.capture_thread.start()
        self.log("Live capture started.")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_capture(self):
        """
        Stops the live capture thread.
        """
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread.wait()
            self.log("Live capture stopped.")
            self.capture_thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_image(self, image):
        """
        Updates the displayed image on the Matplotlib canvas while preserving zoom.
        If auto adjust exposure is enabled, the adjustment is performed on the zoomed region.

        Parameters:
            image (np.ndarray): The new image data to display.
        """
        if self.image_artist is None:
            self.ax.clear()
            self.image_artist = self.ax.imshow(image, cmap=white_turbo(512))
            self.ax.set_title("Live Camera Image")
        else:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.image_artist.set_data(image)
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        self.canvas.draw_idle()
        if self.auto_adjust_checkbox.isChecked():
            self.adjust_exposure(image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DahengLive()
    gui.show()
    sys.exit(app.exec_())
