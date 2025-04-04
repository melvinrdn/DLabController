import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, PngImagePlugin  # For saving images

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
        super().__init__()
        self.widget = widget
        self.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)
        self.widget.verticalScrollBar().setValue(self.widget.verticalScrollBar().maximum())


class LiveCaptureThread(QThread):
    """
    QThread subclass to continuously capture images from the camera.
    """
    image_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_controller, exposure, gain, avgs, update_interval_ms):
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
        with self._param_lock:
            self.exposure = exposure
            self.gain = gain
            self.avgs = avgs
            self.update_interval_ms = update_interval_ms
            self.interval_sec = update_interval_ms / 1000.0

    def run(self):
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
        self._running = False


class DahengLive(QWidget):
    """
    A PyQt5 GUI for live camera image capture using DahengController.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Camera Feed")
        self.camera_controller = None
        self.capture_thread = None
        self.debug_mode = False  # Track debug mode state.
        self.image_artist = None  # To preserve zoomed axes.
        self.last_frame = None    # To store the most recent frame for saving.
        self.current_interval_ms = None  # Stores current update interval.
        self.initUI()

    def initUI(self):
        """
        Sets up the user interface including parameter controls, log box, image display,
        the comment field, and the save button.
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

        # Exposure parameter.
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure (µs):"))
        self.exposure_edit = QLineEdit(f"{DEFAULT_EXPOSURE_US}")
        self.exposure_edit.setValidator(QIntValidator(MIN_EXPOSURE_US, MAX_EXPOSURE_US, self))
        self.exposure_edit.textChanged.connect(self.update_capture_parameters)
        exp_layout.addWidget(self.exposure_edit)
        param_layout.addLayout(exp_layout)

        # Gain parameter.
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.gain_edit = QLineEdit(f"{DEFAULT_GAIN}")
        self.gain_edit.setValidator(QIntValidator(MIN_GAIN, MAX_GAIN, self))
        self.gain_edit.textChanged.connect(self.update_capture_parameters)
        gain_layout.addWidget(self.gain_edit)
        param_layout.addLayout(gain_layout)

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

        # Comment field.
        comment_layout = QHBoxLayout()
        comment_layout.addWidget(QLabel("Comment:"))
        self.comment_edit = QLineEdit()
        comment_layout.addWidget(self.comment_edit)
        param_layout.addLayout(comment_layout)

        # Checkbox for automatic exposure adjustment.
        self.auto_adjust_checkbox = QCheckBox("Auto Adjust Exposure")
        param_layout.addWidget(self.auto_adjust_checkbox)

        # New Background checkbox.
        self.background_checkbox = QCheckBox("Background")
        self.background_checkbox.setChecked(False)
        param_layout.addWidget(self.background_checkbox)

        # Buttons for hardware control, live capture, and saving frame.
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

        # Save frame button.
        self.save_button = QPushButton("Save Frame")
        self.save_button.clicked.connect(self.save_frame)
        btn_layout.addWidget(self.save_button)

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
        """
        current_time = datetime.datetime.now().strftime(DATE_FORMAT)
        self.log_text.append(f"[{current_time}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def update_capture_parameters(self):
        """
        Reads updated parameters from the input fields and applies them immediately.
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
            pass

    def adjust_exposure(self, image):
        """
        Automatically adjusts the exposure based on the brightness of the zoomed image region.
        """
        try:
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
        """
        # Store the latest frame for saving.
        self.last_frame = image

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

    def save_frame(self):
        """
        Saves the most recent frame as a .bmp file in the designated folder.
        The file is saved with a timestamp in its name.
        If the Background checkbox is checked, the file name will include "Background" and the log comment will be "Background".
        Additionally, a log file entry is appended with tab-separated values.
        """
        if self.last_frame is None:
            QMessageBox.warning(self, "Warning", "No frame available to save.")
            return

        now = datetime.datetime.now()
        # Build the directory path using the current date.
        dir_path = f"C:/data/{now.strftime('%Y-%m-%d')}/DahengCamera/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        timestamp = now.strftime("%Y%m%d_%H%M%S%f")
        exposure_val = self.exposure_edit.text()
        gain_val = self.gain_edit.text()
        comment_text = self.comment_edit.text()

        # Override filename and comment if background checkbox is checked.
        if self.background_checkbox.isChecked():
            file_name = f"DahengCamera_Nozzle_Background_{timestamp}.png"
            log_comment = comment_text
        else:
            file_name = f"DahengCamera_Nozzle_Image_{timestamp}.png"
            log_comment = comment_text

        file_path = os.path.join(dir_path, file_name)

        try:
            frame_uint8 = np.uint8(np.clip(self.last_frame, 0, 255))
            img = Image.fromarray(frame_uint8)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Exposure", exposure_val)
            metadata.add_text("Gain", gain_val)
            metadata.add_text("Comment", log_comment)
            img.save(file_path, pnginfo=metadata)
            self.log(f"Frame saved to {file_path}")
        except Exception as e:
            self.log(f"Error saving frame: {e}")
            QMessageBox.critical(self, "Error", f"Error saving frame: {e}")
            return

        # Prepare the log file name and header.
        log_file_name = f"DahengCamera_log_{now.strftime('%Y_%m_%d')}.txt"
        log_file_path = os.path.join(dir_path, log_file_name)
        header = "File Name\tExposure (µs)\tGain\tComment\n"

        try:
            # Write header if the file does not exist.
            if not os.path.exists(log_file_path):
                with open(log_file_path, "w") as log_file:
                    log_file.write(header)
            # Append the log entry with tab-separated values.
            with open(log_file_path, "a") as log_file:
                log_file.write(f"{file_name}\t{exposure_val}\t{gain_val}\t{log_comment}\n")
            self.log(f"Camera parameters logged to {log_file_path}")
        except Exception as e:
            self.log(f"Error writing to log file: {e}")
            QMessageBox.critical(self, "Error", f"Error writing to log file: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DahengLive()
    gui.show()
    sys.exit(app.exec_())
