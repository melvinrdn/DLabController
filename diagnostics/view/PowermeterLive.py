import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTextEdit, QMessageBox, QSplitter, QLineEdit
)


from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from hardware.wrappers.PowermeterController import PowermeterController

class PowerMeasurementThread(QThread):
    measurement_signal = pyqtSignal(float, float)  # timestamp, power reading

    def __init__(self, powermeter_controller, update_interval=0.1):
        super().__init__()
        self.powermeter_controller = powermeter_controller
        self.update_interval = update_interval
        self._running = True

    def run(self):
        while self._running:
            try:
                power = self.powermeter_controller.read_power()
                timestamp = time.time()
                self.measurement_signal.emit(timestamp, power)
                time.sleep(self.update_interval)
            except Exception as e:
                print("Error in power measurement thread:", e)
                break

    def stop(self):
        self._running = False

class PowerMeterLive(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Powermeter")
        self.powermeter_controller = None
        self.measurement_thread = None
        self.times = []
        self.powers = []
        self.line = None
        self.time_window = 10
        self.wavelength = 343  # Default wavelength
        self.wavelength_colors = {343: 'blue', 515: 'green', 1030: 'red'}
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter()

        # Left panel: Controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

        self.visa_label = QLabel("Powermeter VISA ID:")
        self.visa_edit = QLineEdit("USB0::0x1313::0x8078::P0045634::INSTR")
        control_layout.addWidget(self.visa_label)
        control_layout.addWidget(self.visa_edit)

        self.activate_button = QPushButton("Activate Powermeter")
        self.activate_button.clicked.connect(self.activate_hardware)
        control_layout.addWidget(self.activate_button)

        self.deactivate_button = QPushButton("Deactivate Powermeter")
        self.deactivate_button.clicked.connect(self.deactivate_hardware)
        self.deactivate_button.setEnabled(False)
        control_layout.addWidget(self.deactivate_button)

        # Wavelength selection
        self.wavelength_label = QLabel("Select Wavelength (nm):")
        self.wavelength_combo = QComboBox()
        self.wavelength_combo.addItems(["343", "515", "1030"])
        self.wavelength_combo.currentIndexChanged.connect(self.update_wavelength)

        control_layout.addWidget(self.wavelength_label)
        control_layout.addWidget(self.wavelength_combo)

        self.avg_label = QLabel("Number of Averages:")
        self.avg_edit = QLineEdit("1")
        control_layout.addWidget(self.avg_label)
        control_layout.addWidget(self.avg_edit)

        self.update_interval_label = QLabel("Update Interval (s):")
        self.update_interval_edit = QLineEdit("0.5")
        self.set_update_interval_button = QPushButton("Set Update Interval")
        self.set_update_interval_button.clicked.connect(self.set_update_interval)

        control_layout.addWidget(self.update_interval_label)
        control_layout.addWidget(self.update_interval_edit)
        control_layout.addWidget(self.set_update_interval_button)

        self.start_button = QPushButton("Start Live Measurement")
        self.start_button.clicked.connect(self.start_live_measurement)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Live Measurement")
        self.stop_button.clicked.connect(self.stop_live_measurement)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        self.reset_plot_button = QPushButton("Reset Plot")
        self.reset_plot_button.clicked.connect(self.reset_plot)
        control_layout.addWidget(self.reset_plot_button)

        self.time_window_label = QLabel("Time Window (s):")
        self.time_window_edit = QLineEdit(str(self.time_window))  # Default to 10 seconds
        self.set_time_window_button = QPushButton("Set Time Window")
        self.set_time_window_button.clicked.connect(self.set_time_window)

        control_layout.addWidget(self.time_window_label)
        control_layout.addWidget(self.time_window_edit)
        control_layout.addWidget(self.set_time_window_button)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        control_layout.addWidget(self.log_text)

        splitter.addWidget(control_panel)

        # Right panel: Plot
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.figure, self.ax = plt.subplots()
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Power (W)")
        self.ax.grid(True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_panel)

        main_layout.addWidget(splitter)
        self.resize(1200, 600)

    def update_log(self, message):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{now}] {message}")

    def update_wavelength(self):
        self.wavelength = int(self.wavelength_combo.currentText())
        self.update_log(f"Wavelength set to {self.wavelength} nm")

    def activate_hardware(self):
        visa_id = self.visa_edit.text().strip()
        if not visa_id:
            QMessageBox.critical(self, "Error", "No VISA ID provided.")
            return
        try:
            self.powermeter_controller = PowermeterController(visa_id)
            self.powermeter_controller.activate()
            self.powermeter_controller.set_wavelength(self.wavelength)
            self.update_log(f"Powermeter activated with wavelength {self.wavelength} nm")
            self.activate_button.setEnabled(False)
            self.deactivate_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate powermeter: {e}")
            self.update_log(f"Error: {e}")

    def deactivate_hardware(self):
        try:
            if self.powermeter_controller:
                self.powermeter_controller.deactivate()
                self.update_log("Powermeter deactivated.")
            self.powermeter_controller = None
            self.activate_button.setEnabled(True)
            self.deactivate_button.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to deactivate powermeter: {e}")
            self.update_log(f"Error: {e}")

    def start_live_measurement(self):
        if self.powermeter_controller is None:
            QMessageBox.critical(self, "Error", "Powermeter not activated.")
            return

        try:
            no_avg = int(self.avg_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid averaging count.")
            return

        try:
            update_interval = float(self.update_interval_edit.text())
            if update_interval <= 0:
                raise ValueError("Update interval must be positive.")
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid update interval. Please enter a positive number.")
            return

        self.powermeter_controller.set_avg(no_avg)
        self.times = []
        self.powers = []
        self.line = None
        self.measurement_thread = PowerMeasurementThread(self.powermeter_controller, update_interval)
        self.measurement_thread.measurement_signal.connect(self.update_plot)
        self.measurement_thread.start()
        self.update_log(f"Live measurement started with update interval {update_interval} s.")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_live_measurement(self):
        if self.measurement_thread:
            self.measurement_thread.stop()
            self.measurement_thread.wait()
            self.update_log("Live measurement stopped.")
            self.measurement_thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_plot(self, timestamp, power):
        time_limit = timestamp - self.time_window
        self.times.append(timestamp)
        self.powers.append(power)

        filtered_data = [(t, p) for t, p in zip(self.times, self.powers) if t >= time_limit]

        if not filtered_data:
            self.times = []
            self.powers = []
            return

        self.times, self.powers = zip(*filtered_data)
        self.times, self.powers = list(self.times), list(self.powers)

        times_rel = np.array(self.times) - self.times[0]

        color = self.wavelength_colors[self.wavelength]

        if self.line is None:
            self.line, = self.ax.plot(times_rel, self.powers, color=color, marker='o')
        else:
            self.line.set_data(times_rel, self.powers)
            self.line.set_color(color)

        self.ax.relim()
        self.ax.autoscale_view()

        self.ax.set_title(f"Current Power: {power} W")
        self.canvas.draw_idle()


    def set_time_window(self):
        try:
            new_window = float(self.time_window_edit.text())
            if new_window <= 0:
                raise ValueError("Time window must be positive.")
            self.time_window = new_window
            self.update_log(f"Time window set to {self.time_window} seconds.")
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid time window. Please enter a positive number.")

    def set_update_interval(self):
        try:
            new_interval = float(self.update_interval_edit.text())
            if new_interval <= 0:
                raise ValueError("Update interval must be positive.")

            self.update_log(f"Update interval set to {new_interval} seconds.")
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid update interval. Please enter a positive number.")

    def reset_plot(self):
        self.times = []
        self.powers = []
        self.ax.clear()
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Power (W)")
        self.ax.grid(True)
        self.line = None
        self.canvas.draw_idle()
        self.update_log("Plot reset.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PowerMeterLive()
    gui.show()
    sys.exit(app.exec_())
