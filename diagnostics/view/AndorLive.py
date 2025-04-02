import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QMessageBox, QSplitter, QComboBox, QCheckBox
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)
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


class QTextEditHandler(logging.Handler):
    """
    Custom logging handler to send log messages to a QTextEdit widget.
    """

    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)
        self.widget.verticalScrollBar().setValue(self.widget.verticalScrollBar().maximum())


class LiveCaptureThread(QThread):
    """
    QThread subclass to continuously capture images from the Andor camera.

    Attributes:
        image_signal (pyqtSignal): Signal emitted when an image is captured (np.ndarray).
    """
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


class AndorLive(QWidget):
    """
    A PyQt5 GUI for live Andor camera image capture using AndorController.

    Provides controls for:
      - Setting exposure (in microseconds), averaging, and update interval (in milliseconds).
      - Activating/deactivating the camera.
      - Starting/stopping live capture.
      - Toggling debug mode.
      - Fixing the colorbar limits on the live image.

    The displayed figure contains two subplots: the top shows the live image (with a colorbar)
    and its title displays the maximum pixel value, while the bottom shows the integrated profile.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Andor Camera Feed")
        self.camera_controller = None
        self.capture_thread = None
        self.debug_mode = False
        self.image_artist = None
        self.profile_line = None
        self.current_interval_ms = None
        self.fixed_cbar_max = None
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

        # Checkbox to fix the colorbar.
        self.fix_cbar_checkbox = QCheckBox("Fix Colorbar")
        self.fix_cbar_checkbox.setChecked(False)
        param_layout.addWidget(self.fix_cbar_checkbox)

        # Buttons for camera control and live capture.
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

        self.debug_button = QPushButton("Enable Debug Mode")
        self.debug_button.clicked.connect(self.toggle_debug_mode)
        btn_layout.addWidget(self.debug_button)

        param_layout.addLayout(btn_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        param_layout.addWidget(self.log_text)

        splitter.addWidget(param_panel)

        # Plot panel with Matplotlib canvas using gridspec.
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        # Use gridspec to give the image subplot more space than the profile.
        self.figure, (self.ax_img, self.ax_profile) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},
                                                                   sharex=True)
        self.ax_img.set_title("Live Andor Camera Image")
        self.ax_profile.set_title("Integrated Profile")
        # Adjust right margin to make room for the colorbar.
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
            elif self.camera_controller:
                self.camera_controller.set_exposure(exposure)
                self.camera_controller.set_average(avgs)
        except ValueError:
            pass

    def toggle_debug_mode(self):
        if self.camera_controller:
            self.debug_mode = not self.debug_mode
            self.camera_controller.enable_debug(debug_on=self.debug_mode)
            state = "enabled" if self.debug_mode else "disabled"
            self.log(f"Camera 0 Debug mode {state}.")
            self.debug_button.setText("Disable Debug Mode" if self.debug_mode else "Enable Debug Mode")
        else:
            QMessageBox.warning(self, "Warning", "Camera not activated. Activate the camera first.")

    def activate_camera(self):
        try:
            self.camera_controller = AndorController(device_index=0)
            self.camera_controller.activate()
            self.camera_controller.logger.handlers = []
            qt_handler = QTextEditHandler(self.log_text)
            self.camera_controller.logger.addHandler(qt_handler)
            self.camera_controller.enable_debug(debug_on=False)
            self.log("Camera 0 activated.")
            self.activate_button.setEnabled(False)
            self.deactivate_button.setEnabled(True)
            self.start_button.setEnabled(True)
        except AndorControllerError as ce:
            QMessageBox.critical(self, "Error", f"Failed to activate camera: {ce}")
            self.log(f"Error activating camera: {ce}")

    def deactivate_camera(self):
        try:
            if self.camera_controller:
                self.camera_controller.deactivate()
                self.log("Camera 0 deactivated.")
            self.camera_controller = None
            self.activate_button.setEnabled(True)
            self.deactivate_button.setEnabled(False)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
        except AndorControllerError as ce:
            QMessageBox.critical(self, "Error", f"Failed to deactivate camera: {ce}")
            self.log(f"Error deactivating camera: {ce}")

    def start_capture(self):
        if self.camera_controller is None:
            QMessageBox.critical(self, "Error", "Camera not activated.")
            return
        try:
            exposure = int(self.exposure_edit.text())
            avgs = int(self.avgs_edit.text())
            update_interval = int(self.interval_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameter values.")
            return
        self.capture_thread = LiveCaptureThread(self.camera_controller, exposure, avgs, update_interval)
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

    def update_image(self, image):
        """
        Updates the displayed image on the Matplotlib canvas and the integrated profile,
        while preserving the current zoom settings.

        The top subplot displays the live image with a colorbar and the title shows the maximum pixel value.
        The bottom subplot displays the integrated intensity (sum along columns) plotted against the x–axis.

        If the "Fix Colorbar" checkbox is checked, the colorbar’s upper limit is fixed to the first
        maximum encountered; otherwise, it adjusts automatically.

        Parameters:
            image (np.ndarray): The new image data to display.
        """
        max_val = np.max(image)
        min_val = np.min(image)
        title_text = f"Live Andor Camera Image - Max: {max_val:.0f}"
        # Update top subplot.
        if self.image_artist is None:
            self.ax_img.clear()
            self.image_artist = self.ax_img.imshow(image, cmap=white_turbo(512))
            self.ax_img.set_title(title_text)
            # Create a colorbar in a separate axis.
            self.cbar = self.figure.colorbar(self.image_artist, ax=self.ax_img, fraction=0.046, pad=0.04)
        else:
            xlim = self.ax_img.get_xlim()
            ylim = self.ax_img.get_ylim()
            self.image_artist.set_data(image)
            self.ax_img.set_xlim(xlim)
            self.ax_img.set_ylim(ylim)
            self.ax_img.set_title(title_text)
            # Update colorbar limits using the min pixel value instead of 0.
            if self.fix_cbar_checkbox.isChecked():
                if self.fixed_cbar_max is None:
                    self.fixed_cbar_max = max_val
                self.image_artist.set_clim(min_val, self.fixed_cbar_max)
            else:
                self.fixed_cbar_max = None
                self.image_artist.set_clim(min_val, max_val)
            self.cbar.update_normal(self.image_artist)
        # Update bottom subplot (integrated intensity along columns).
        profile = np.sum(image, axis=0)
        self.ax_profile.clear()
        self.ax_profile.plot(np.arange(image.shape[1]), profile, color='r')
        self.ax_profile.set_xlabel("X (px)")
        self.ax_profile.set_ylabel("Integrated Intensity")
        self.ax_profile.set_xlim(0, image.shape[1])
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AndorLive()
    gui.show()
    sys.exit(app.exec_())
