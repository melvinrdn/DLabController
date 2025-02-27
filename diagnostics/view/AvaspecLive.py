import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QTextEdit, QMessageBox, QSplitter, QCheckBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from hardware.wrappers.AvaspecController import AvaspecController

"""
# ---------------- Dummy AvaspecController ----------------
class DummyAvaspecController:
    def __init__(self, spec_handle):
        self.spec_handle = spec_handle
        self.wavelength = np.linspace(300, 400, 2048)
    def activate(self):
        print("Dummy spectrometer activated.")
    def measure_spectrum(self, int_time, no_avg):
        # Generate a Gaussian spectrum around 343 nm with amplitude A, center mu, sigma
        A = 1000  # peak counts
        mu = 343  # center wavelength in nm
        sigma = 1.2  # standard deviation
        data = A * np.exp(-0.5 * ((self.wavelength - mu)/sigma)**2)
        # Add some noise
        data += 50 * np.random.randn(len(self.wavelength))
        # Return a float timestamp and the spectrum data
        timestamp = time.time()
        return timestamp, data
    def deactivate(self):
        print("Dummy spectrometer deactivated.")
    @classmethod
    def list_spectrometers(cls):
        return ["DummySpec1"]

AvaspecController = DummyAvaspecController
"""

class LiveMeasurementThread(QThread):
    spectrum_signal = pyqtSignal(float, object)

    def __init__(self, avaspec_controller, int_time, no_avg):
        super().__init__()
        self.avaspec_controller = avaspec_controller
        self.int_time = int_time
        self.no_avg = no_avg
        self._running = True

    def run(self):
        while self._running:
            try:
                current_int_time = self.int_time
                current_update_interval = current_int_time / 1000.0
                timestamp, data = self.avaspec_controller.measure_spectrum(current_int_time, self.no_avg)
                self.spectrum_signal.emit(timestamp, data)
                time.sleep(current_update_interval)
            except Exception as e:
                print("Error in measurement thread:", e)
                break

    def stop(self):
        self._running = False

class AvaspecLive(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Spectrometer GUI")
        self.avaspec_controller = None
        self.measurement_thread = None
        self.spectrometer_handles = []
        self.line = None
        self.fit_line = None
        self.last_timestamp = None
        self.last_data = None
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter()

        param_panel = QWidget()
        param_layout = QVBoxLayout(param_panel)

        spec_select_layout = QVBoxLayout()
        self.spec_combo = QComboBox()
        self.search_button = QPushButton("Search Spectrometers")
        self.search_button.clicked.connect(self.search_spectrometers)
        spec_select_layout.addWidget(QLabel("Select Spectrometer:"))
        spec_select_layout.addWidget(self.spec_combo)
        spec_select_layout.addWidget(self.search_button)
        param_layout.addLayout(spec_select_layout)

        act_layout = QVBoxLayout()
        self.activate_button = QPushButton("Activate Hardware")
        self.activate_button.clicked.connect(self.activate_hardware)
        self.deactivate_button = QPushButton("Deactivate Hardware")
        self.deactivate_button.clicked.connect(self.deactivate_hardware)
        self.deactivate_button.setEnabled(False)
        act_layout.addWidget(self.activate_button)
        act_layout.addWidget(self.deactivate_button)
        param_layout.addLayout(act_layout)

        meas_layout = QVBoxLayout()
        meas_layout.addWidget(QLabel("Integration Time (ms):"))
        self.int_time_edit = QLineEdit("100")
        self.int_time_edit.textChanged.connect(self.integration_time_changed)
        meas_layout.addWidget(self.int_time_edit)
        meas_layout.addWidget(QLabel("Number of Averages:"))
        self.no_avg_edit = QLineEdit("1")
        meas_layout.addWidget(self.no_avg_edit)
        param_layout.addLayout(meas_layout)

        options_layout = QVBoxLayout()
        self.autoscale_checkbox = QCheckBox("Autoscale")
        self.autoscale_checkbox.setChecked(True)
        options_layout.addWidget(self.autoscale_checkbox)
        self.logscale_checkbox = QCheckBox("Log Scale")
        self.logscale_checkbox.setChecked(False)
        options_layout.addWidget(self.logscale_checkbox)
        self.gaussian_fit_checkbox = QCheckBox("Gaussian Fit")
        self.gaussian_fit_checkbox.setChecked(False)
        options_layout.addWidget(self.gaussian_fit_checkbox)
        param_layout.addLayout(options_layout)

        live_ctrl_layout = QVBoxLayout()
        self.start_live_button = QPushButton("Start Live Measurement")
        self.start_live_button.clicked.connect(self.start_live_measurement)
        self.stop_live_button = QPushButton("Stop Live Measurement")
        self.stop_live_button.clicked.connect(self.stop_live_measurement)
        self.stop_live_button.setEnabled(False)
        live_ctrl_layout.addWidget(self.start_live_button)
        live_ctrl_layout.addWidget(self.stop_live_button)
        param_layout.addLayout(live_ctrl_layout)

        self.save_button = QPushButton("Save Spectrum")
        self.save_button.clicked.connect(self.save_spectrum)
        param_layout.addWidget(self.save_button)

        param_layout.addWidget(QLabel("Comment:"))
        self.comment_edit = QLineEdit("")
        param_layout.addWidget(self.comment_edit)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        param_layout.addWidget(self.log_text)

        splitter.addWidget(param_panel)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.figure, self.ax = plt.subplots()
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Counts")
        self.ax.grid(True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_panel)

        main_layout.addWidget(splitter)
        self.resize(1000, 600)

    def search_spectrometers(self):
        self.spec_combo.clear()
        speclist = AvaspecController.list_spectrometers()
        if not speclist:
            QMessageBox.critical(self, "Error", "No spectrometer found.")
            self.update_log("No spectrometer found.")
            return
        self.spectrometer_handles = speclist
        self.spec_combo.addItems([f"Spectrometer {i + 1}" for i in range(len(speclist))])
        self.update_log(f"Found {len(speclist)} spectrometer(s).")

    def activate_hardware(self):
        selected_index = self.spec_combo.currentIndex()
        if selected_index < 0 or selected_index >= len(self.spectrometer_handles):
            QMessageBox.critical(self, "Error", "No spectrometer selected.")
            return
        spec_handle = self.spectrometer_handles[selected_index]
        try:
            self.avaspec_controller = AvaspecController(spec_handle)
            self.avaspec_controller.activate()
            self.update_log("Spectrometer activated.")
            self.activate_button.setEnabled(False)
            self.deactivate_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate spectrometer: {e}")
            self.update_log(f"Error activating spectrometer: {e}")

    def deactivate_hardware(self):
        try:
            if self.avaspec_controller:
                self.avaspec_controller.deactivate()
                self.update_log("Spectrometer deactivated.")
            self.avaspec_controller = None
            self.activate_button.setEnabled(True)
            self.deactivate_button.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to deactivate spectrometer: {e}")
            self.update_log(f"Error deactivating spectrometer: {e}")

    def start_live_measurement(self):
        if self.avaspec_controller is None:
            QMessageBox.critical(self, "Error", "Spectrometer not activated.")
            return
        try:
            int_time = float(self.int_time_edit.text())
            no_avg = int(self.no_avg_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid integration time or number of averages.")
            return
        self.measurement_thread = LiveMeasurementThread(self.avaspec_controller, int_time, no_avg)
        self.measurement_thread.spectrum_signal.connect(self.update_spectrum)
        self.measurement_thread.start()
        self.update_log("Live measurement started.")
        self.start_live_button.setEnabled(False)
        self.stop_live_button.setEnabled(True)

    def stop_live_measurement(self):
        if self.measurement_thread:
            self.measurement_thread.stop()
            self.measurement_thread.wait()
            self.update_log("Live measurement stopped.")
            self.measurement_thread = None
        self.start_live_button.setEnabled(True)
        self.stop_live_button.setEnabled(False)

    def update_spectrum(self, timestamp, data):
        self.last_timestamp = timestamp
        self.last_data = np.copy(data)
        if self.line is None:
            self.line, = self.ax.plot(self.avaspec_controller.wavelength, data, color='blue', label="Data")
        else:
            self.line.set_ydata(data)
        if self.gaussian_fit_checkbox.isChecked():
            def gaussian(x, A, mu, sigma):
                return A * np.exp(-0.5 * ((x - mu)/sigma)**2)
            try:
                A0 = np.max(data)
                mu0 = self.avaspec_controller.wavelength[np.argmax(data)]
                sigma0 = 2.0
                popt, _ = curve_fit(gaussian, self.avaspec_controller.wavelength, data, p0=[A0, mu0, sigma0])
                A_fit, mu_fit, sigma_fit = popt
                FWHM = 2.355 * sigma_fit
                mu_m = mu_fit * 1e-9
                FWHM_m = FWHM * 1e-9
                c = 3e8
                pulse_duration = 0.441 * mu_m**2 / (c * FWHM_m)
                pulse_duration_fs = pulse_duration * 1e15
                fitted_curve = gaussian(self.avaspec_controller.wavelength, *popt)
                if self.fit_line is None:
                    self.fit_line, = self.ax.plot(self.avaspec_controller.wavelength, fitted_curve, 'r--', label="Gaussian Fit")
                else:
                    self.fit_line.set_ydata(fitted_curve)
                self.ax.legend()
                self.ax.set_title(f"Peak: {A_fit:.2f}, FWHM: {FWHM:.2f} nm, Pulse: {pulse_duration_fs:.2f} fs")
            except Exception as e:
                self.ax.set_title("Gaussian fit failed")
                print("Gaussian fit failed:", e)
        else:
            if self.fit_line is not None:
                self.fit_line.remove()
                self.fit_line = None
            self.ax.set_title(f"Spectrum at {timestamp}")
        if self.autoscale_checkbox.isChecked():
            self.ax.relim()
            self.ax.autoscale_view()
        if self.logscale_checkbox.isChecked():
            self.ax.set_yscale('log')
        else:
            self.ax.set_yscale('linear')
        self.canvas.draw_idle()

    def integration_time_changed(self, text):
        try:
            new_int_time = float(text)
            if self.measurement_thread is not None:
                self.measurement_thread.int_time = new_int_time
                self.update_log(f"Integration time updated to {new_int_time} ms")
        except ValueError:
            pass

    def save_spectrum(self):
        if self.last_data is None:
            QMessageBox.warning(self, "Warning", "No spectrum available to save.")
            return
        # Use the current system time as the save time
        now = datetime.datetime.now()
        start_time = now.strftime("%Y-%m-%d %H:%M:%S")
        date_str = start_time.split(" ")[0]
        save_dir = os.path.join("C:\\data", date_str, "AvaspecLive")
        os.makedirs(save_dir, exist_ok=True)
        safe_time = start_time.replace(" ", "_").replace(":", "-")
        filename = os.path.join(save_dir, f"AvaspecLive_Spectrum_{safe_time}.txt")
        comment = self.comment_edit.text()
        try:
            with open(filename, "w") as f:
                if comment:
                    f.write(f"# Comment: {comment}\n")
                f.write(f"# Timestamp: {start_time}\n")
                f.write(f"# Integration Time (ms): {self.int_time_edit.text()}\n")
                f.write(f"# Number of Averages: {self.no_avg_edit.text()}\n")
                f.write("Wavelength (nm);Counts\n")
                for wl, count in zip(self.avaspec_controller.wavelength, self.last_data):
                    f.write(f"{wl};{count}\n")
            self.update_log(f"Spectrum saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save spectrum: {e}")
            self.update_log(f"Failed to save spectrum: {e}")

    def update_log(self, message):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{current_time}] {message}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AvaspecLive()
    gui.show()
    sys.exit(app.exec_())
