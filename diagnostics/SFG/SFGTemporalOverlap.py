import sys
import os
import time
import datetime

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QMessageBox, QTextEdit, QLineEdit, QLabel, QHBoxLayout,
                             QSplitter,QComboBox, QCheckBox, QTabWidget)
from PyQt5.QtCore import QThread, pyqtSignal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from hardware.wrappers.ThorlabsController import ThorlabsController
from hardware.wrappers.AvaspecController import AvaspecController
from hardware.wrappers.PowermeterController import PowermeterController

# ------------------ Spectrometer Measurement Thread ------------------

class MeasurementThread(QThread):
    log_signal = pyqtSignal(str)
    plot_signal = pyqtSignal(object, object)  # wavelength, spectrum data

    def __init__(self, motor_controller, avaspec_controller, positions, int_time, no_avg, save_data, header_text=""):
        super().__init__()
        self.thorlabs_controller = motor_controller
        self.avaspec_controller = avaspec_controller
        self.positions = positions
        self.int_time = int_time
        self.no_avg = no_avg
        self.save_data = save_data
        self.running = True
        self.header_text = header_text

        wavelength = self.avaspec_controller.wavelength
        self.spectrum_data = np.zeros((len(self.positions), len(wavelength)))

    def run(self):
        try:
            wavelength = self.avaspec_controller.wavelength
            start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            max_intensity = -np.inf
            max_position = None
            for i, position in enumerate(self.positions):
                if not self.running:
                    self.log_signal.emit("Measurement aborted.")
                    break

                self.log_signal.emit(f"Moving to position {position:.2f}...")
                self.thorlabs_controller.move_to(position, blocking=True)
                time.sleep(0.5)
                current_pos = self.thorlabs_controller.get_position()
                self.log_signal.emit(f"Reached position {current_pos:.2f}")

                timestamp, data = self.avaspec_controller.measure_spectrum(self.int_time, self.no_avg)
                self.log_signal.emit("Spectrum successfully taken.")
                self.spectrum_data[i, :] = data

                total_intensity = np.sum(data)
                if total_intensity > max_intensity:
                    max_intensity = total_intensity
                    max_position = current_pos

                self.plot_signal.emit(wavelength, self.spectrum_data.copy())

            if self.save_data:
                self.save_spectrum_data(np.array(wavelength), self.spectrum_data, start_time)

            if max_position is not None:
                self.log_signal.emit(f"Maximum intensity found at position: {max_position:.2f} mm")

        except Exception as e:
            self.log_signal.emit(f"Error: {e}")

    def save_spectrum_data(self, wavelength, spectrum_data, start_time):
        date_str = start_time.split(" ")[0]
        save_dir = os.path.join("C:\\data", date_str, "SFGTemporalOverlap")
        os.makedirs(save_dir, exist_ok=True)

        safe_time = start_time.replace(" ", "_").replace(":", "-")
        file_name = os.path.join(save_dir, f"SFGTemporalOverlap_Spectrum_{safe_time}.txt")
        try:
            with open(file_name, "w") as f:
                f.write("# SFG Temporal Overlap Scan\n")
                if self.header_text:
                    f.write(f"# {self.header_text}\n")
                f.write(f"# Date: {start_time}\n")
                f.write(f"# Integration Time (ms): {self.int_time}\n")
                f.write(f"# Number of Averages: {self.no_avg}\n")
                if len(self.positions) > 1:
                    step = self.positions[1] - self.positions[0]
                else:
                    step = 0
                f.write(f"# Positions: len = {len(self.positions)}, start = {self.positions[0]} mm, "
                        f"end = {self.positions[-1]} mm, step = {step} mm\n")
                f.write("\n")

                header = "Wavelength (nm)"
                for pos in self.positions:
                    header += f", {pos}"
                header += "\n"
                f.write(header)

                spectrum_data_T = spectrum_data.T
                for i, wl in enumerate(wavelength):
                    row = f"{wl}"
                    for intensity in spectrum_data_T[i]:
                        row += f"; {intensity}"
                    row += "\n"
                    f.write(row)
            self.log_signal.emit(f"Saved spectrum data to {file_name}")
        except Exception as e:
            self.log_signal.emit(f"Error saving data: {e}")

    def stop(self):
        self.running = False


# ------------------ Powermeter Measurement Thread ------------------

class PowerMeasurementThread(QThread):
    log_signal = pyqtSignal(str)
    plot_signal = pyqtSignal(object, object)  # positions, power data

    def __init__(self, motor_controller, powermeter_controller, positions, no_avg, save_data, header_text=""):
        super().__init__()
        self.motor_controller = motor_controller
        self.powermeter_controller = powermeter_controller
        self.positions = positions
        self.no_avg = no_avg
        self.save_data = save_data
        self.running = True
        self.header_text = header_text

        self.power_data = np.zeros(len(positions))

    def run(self):
        try:
            start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for i, pos in enumerate(self.positions):
                if not self.running:
                    self.log_signal.emit("Measurement aborted.")
                    break

                self.log_signal.emit(f"Moving to position {pos:.2f}...")
                self.motor_controller.move_to(pos, blocking=True)
                time.sleep(0.5)
                current_pos = self.motor_controller.get_position()
                self.log_signal.emit(f"Reached position {current_pos:.2f}")

                self.powermeter_controller.set_avg(self.no_avg)
                power = self.powermeter_controller.read_power()
                self.log_signal.emit(f"Power measured: {power}")
                self.power_data[i] = power

                self.plot_signal.emit(self.positions[:i + 1], self.power_data[:i + 1])

            if self.save_data:
                self.save_power_data(self.positions, self.power_data, start_time)

        except Exception as e:
            self.log_signal.emit(f"Error: {e}")

    def save_power_data(self, positions, power_data, start_time):
        date_str = start_time.split(" ")[0]
        save_dir = os.path.join("C:\\data", date_str, "SFGTemporalOverlap")
        os.makedirs(save_dir, exist_ok=True)

        safe_time = start_time.replace(" ", "_").replace(":", "-")
        file_name = os.path.join(save_dir, f"SFGTemporalOverlap_Power_{safe_time}.txt")
        try:
            with open(file_name, "w") as f:
                f.write("# Powermeter Measurement\n")
                if self.header_text:
                    f.write(f"# {self.header_text}\n")
                f.write(f"# Date: {start_time}\n")
                if len(positions) > 1:
                    step = positions[1] - positions[0]
                else:
                    step = 0
                f.write(f"# Positions: len = {len(positions)}, start = {positions[0]} mm, "
                        f"end = {positions[-1]} mm, step = {step} mm\n")
                f.write("# Position (mm), Power (W)\n")
                for pos, power in zip(positions, power_data):
                    f.write(f"{pos}; {power}\n")
            self.log_signal.emit(f"Saved power data to {file_name}")
        except Exception as e:
            self.log_signal.emit(f"Error saving data: {e}")

    def stop(self):
        self.running = False


# ------------------ Spectrometer Measurement GUI (Tab 1) ------------------

class SFGTemporalOverlapGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.spectrometer_handles = []
        self.motor_controller = None
        self.avaspec_controller = None
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()
        controls_layout = QVBoxLayout()

        # Thorlabs (Motor) settings
        self.motorIDLabel = QLabel("Motor ID:")
        self.motorIDInput = QLineEdit("83837725")
        controls_layout.addWidget(self.motorIDLabel)
        controls_layout.addWidget(self.motorIDInput)

        self.positionLabel = QLabel("Start Position:")
        self.positionInput = QLineEdit("13")
        self.endPositionLabel = QLabel("End Position:")
        self.endPositionInput = QLineEdit("14")
        self.stepSizeLabel = QLabel("Step Size:")
        self.stepSizeInput = QLineEdit("0.01")
        controls_layout.addWidget(self.positionLabel)
        controls_layout.addWidget(self.positionInput)
        controls_layout.addWidget(self.endPositionLabel)
        controls_layout.addWidget(self.endPositionInput)
        controls_layout.addWidget(self.stepSizeLabel)
        controls_layout.addWidget(self.stepSizeInput)

        self.intTimeLabel = QLabel("Integration Time (ms):")
        self.intTimeInput = QLineEdit("10")
        self.noAvgLabel = QLabel("Number of Averages:")
        self.noAvgInput = QLineEdit("1")
        controls_layout.addWidget(self.intTimeLabel)
        controls_layout.addWidget(self.intTimeInput)
        controls_layout.addWidget(self.noAvgLabel)
        controls_layout.addWidget(self.noAvgInput)

        self.spectrometerLabel = QLabel("Select Spectrometer:")
        self.spectrometerSelect = QComboBox()
        controls_layout.addWidget(self.spectrometerLabel)
        controls_layout.addWidget(self.spectrometerSelect)

        self.searchSpectrometerButton = QPushButton("Search Spectrometers")
        self.searchSpectrometerButton.clicked.connect(self.populate_spectrometers)
        controls_layout.addWidget(self.searchSpectrometerButton)

        self.activateButton = QPushButton("Activate Hardware")
        self.activateButton.clicked.connect(self.activateHardware)
        controls_layout.addWidget(self.activateButton)

        self.deactivateButton = QPushButton("Deactivate Hardware")
        self.deactivateButton.clicked.connect(self.deactivateHardware)
        controls_layout.addWidget(self.deactivateButton)

        self.activateButton.setEnabled(True)
        self.deactivateButton.setEnabled(False)

        self.saveDataCheckbox = QCheckBox("Save Data to File")
        controls_layout.addWidget(self.saveDataCheckbox)

        self.headerLabel = QLabel("File Header Comment:")
        self.headerInput = QLineEdit("")
        controls_layout.addWidget(self.headerLabel)
        controls_layout.addWidget(self.headerInput)

        button_layout = QHBoxLayout()
        self.startButton = QPushButton("Start Measurement")
        self.startButton.clicked.connect(self.start_measurement)
        button_layout.addWidget(self.startButton)

        self.abortButton = QPushButton("Abort Measurement")
        self.abortButton.clicked.connect(self.abort_measurement)
        button_layout.addWidget(self.abortButton)
        controls_layout.addLayout(button_layout)

        self.logText = QTextEdit()
        self.logText.setReadOnly(True)
        controls_layout.addWidget(self.logText)

        controls_container = QWidget()
        controls_container.setLayout(controls_layout)

        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)

        self.image = self.ax.imshow(np.zeros((1, 1)), aspect='auto', cmap='turbo',
                                     extent=[0, 1, 0, 1], origin='lower')
        self.cbar = self.figure.colorbar(self.image, ax=self.ax, orientation='vertical')
        self.cbar.set_label("Counts")
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Position (mm)")
        self.canvas.draw()

        splitter = QSplitter()
        splitter.addWidget(controls_container)
        splitter.addWidget(self.canvas)
        splitter.setSizes([300, 500])
        layout.addWidget(splitter)
        self.setLayout(layout)
        self.setWindowTitle("D-lab Controller - SFG Temporal Overlap")
        self.resize(1000, 500)

    def populate_spectrometers(self):
        self.spectrometerSelect.clear()
        speclist = AvaspecController.list_spectrometers()
        if not speclist:
            QMessageBox.critical(self, "Error", "No spectrometer found.")
            self.update_log("No spectrometer found.")
            return
        self.spectrometer_handles = speclist
        self.spectrometerSelect.addItems([f"Spectrometer {i + 1}" for i in range(len(speclist))])
        self.update_log(f"Found {len(speclist)} spectrometer(s).")

    def activateHardware(self):
        try:
            motor_id = int(self.motorIDInput.text())
            selected_index = self.spectrometerSelect.currentIndex()
            if selected_index < 0 or selected_index >= len(self.spectrometer_handles):
                QMessageBox.critical(self, "Error", "No spectrometer selected.")
                return
            spec_handle = self.spectrometer_handles[selected_index]

            self.update_log("Starting Thorlabs stage...")
            self.motor_controller = ThorlabsController(motor_id)
            self.motor_controller.activate()

            self.update_log("Starting Spectrometer...")
            self.avaspec_controller = AvaspecController(spec_handle)
            self.avaspec_controller.activate()
            self.update_log("Hardware activated.")

            self.activateButton.setEnabled(False)
            self.deactivateButton.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate hardware: {e}")
            self.update_log(f"Error activating hardware: {e}")

    def deactivateHardware(self):
        try:
            if self.avaspec_controller:
                self.avaspec_controller.deactivate()
                self.update_log("Spectrometer deactivated.")
            if self.motor_controller:
                self.motor_controller.disable()
                self.update_log("Motor disabled.")
            self.motor_controller = None
            self.avaspec_controller = None

            self.deactivateButton.setEnabled(False)
            self.activateButton.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to deactivate hardware: {e}")
            self.update_log(f"Error deactivating hardware: {e}")

    def start_measurement(self):
        try:
            if self.motor_controller is None or self.avaspec_controller is None:
                self.activateHardware()

            self.update_log("Starting measurement...")
            self.startButton.setEnabled(False)
            self.abortButton.setEnabled(True)

            start_pos = float(self.positionInput.text())
            end_pos = float(self.endPositionInput.text())
            step_size = float(self.stepSizeInput.text())
            int_time = float(self.intTimeInput.text())
            no_avg = int(self.noAvgInput.text())
            num_points = int(np.floor((end_pos - start_pos) / step_size)) + 1
            positions = np.linspace(start_pos, end_pos, num_points)
            save_data = self.saveDataCheckbox.isChecked()

            header_text = self.headerInput.text()
            self.thread = MeasurementThread(self.motor_controller, self.avaspec_controller,
                                            positions, int_time, no_avg, save_data, header_text)

            self.thread.log_signal.connect(self.update_log)
            self.thread.plot_signal.connect(self.update_plot)
            self.thread.finished.connect(self.measurement_finished)
            self.thread.start()
            self.update_log(f"Number of steps: {num_points}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start measurement: {e}")
            self.startButton.setEnabled(True)
            self.abortButton.setEnabled(False)

    def update_log(self, message):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.logText.append(f"[{current_time}] {message}")

    def update_plot(self, wavelength, spectrum_data):
        self.image.set_extent([wavelength[0], wavelength[-1],
                               self.thread.positions[0], self.thread.positions[-1]])
        self.image.set_data(spectrum_data)
        self.image.set_clim(vmin=np.min(spectrum_data), vmax=np.max(spectrum_data))
        self.cbar.update_normal(self.image)
        self.canvas.draw_idle()

    def abort_measurement(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            self.update_log("Measurement aborted.")

    def measurement_finished(self):
        self.update_log("Measurement thread has finished.")
        self.startButton.setEnabled(True)
        self.abortButton.setEnabled(False)

# ------------------ Powermeter Measurement GUI (Tab 2) ------------------

class PowermeterMeasurementGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.powermeter_controller = None
        self.motor_controller = None
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()
        controls_layout = QVBoxLayout()

        # Motor settings (same as before)
        self.motorIDLabel = QLabel("Motor ID:")
        self.motorIDInput = QLineEdit("83837725")
        controls_layout.addWidget(self.motorIDLabel)
        controls_layout.addWidget(self.motorIDInput)

        self.positionLabel = QLabel("Start Position:")
        self.positionInput = QLineEdit("13")
        self.endPositionLabel = QLabel("End Position:")
        self.endPositionInput = QLineEdit("14")
        self.stepSizeLabel = QLabel("Step Size:")
        self.stepSizeInput = QLineEdit("0.01")
        controls_layout.addWidget(self.positionLabel)
        controls_layout.addWidget(self.positionInput)
        controls_layout.addWidget(self.endPositionLabel)
        controls_layout.addWidget(self.endPositionInput)
        controls_layout.addWidget(self.stepSizeLabel)
        controls_layout.addWidget(self.stepSizeInput)

        self.noAvgLabel = QLabel("Number of Averages:")
        self.noAvgInput = QLineEdit("1")
        controls_layout.addWidget(self.noAvgLabel)
        controls_layout.addWidget(self.noAvgInput)

        # Instead of spectrometer selection, use Powermeter VISA ID
        self.powermeterLabel = QLabel("Powermeter VISA ID:")
        self.powermeterInput = QLineEdit("USB0::0x1313::0x8078::P0045634::INSTR")
        controls_layout.addWidget(self.powermeterLabel)
        controls_layout.addWidget(self.powermeterInput)

        self.activateButton = QPushButton("Activate Hardware")
        self.activateButton.clicked.connect(self.activateHardware)
        controls_layout.addWidget(self.activateButton)

        self.deactivateButton = QPushButton("Deactivate Hardware")
        self.deactivateButton.clicked.connect(self.deactivateHardware)
        controls_layout.addWidget(self.deactivateButton)
        self.activateButton.setEnabled(True)
        self.deactivateButton.setEnabled(False)

        self.saveDataCheckbox = QCheckBox("Save Data to File")
        controls_layout.addWidget(self.saveDataCheckbox)

        self.headerLabel = QLabel("File Header Comment:")
        self.headerInput = QLineEdit("")
        controls_layout.addWidget(self.headerLabel)
        controls_layout.addWidget(self.headerInput)

        button_layout = QHBoxLayout()
        self.startButton = QPushButton("Start Measurement")
        self.startButton.clicked.connect(self.start_measurement)
        button_layout.addWidget(self.startButton)

        self.abortButton = QPushButton("Abort Measurement")
        self.abortButton.clicked.connect(self.abort_measurement)
        button_layout.addWidget(self.abortButton)
        controls_layout.addLayout(button_layout)

        self.logText = QTextEdit()
        self.logText.setReadOnly(True)
        controls_layout.addWidget(self.logText)

        controls_container = QWidget()
        controls_container.setLayout(controls_layout)

        # Set up the matplotlib figure for a line plot (Power vs. Position)
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Position (mm)")
        self.ax.set_ylabel("Power")
        self.line, = self.ax.plot([], [], marker='o')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.draw()

        splitter = QSplitter()
        splitter.addWidget(controls_container)
        splitter.addWidget(self.canvas)
        splitter.setSizes([300, 500])
        layout.addWidget(splitter)
        self.setLayout(layout)
        self.setWindowTitle("D-lab Controller - Powermeter Measurement")
        self.resize(1000, 500)

    def activateHardware(self):
        try:
            motor_id = int(self.motorIDInput.text())
            powermeter_id = self.powermeterInput.text()
            if not powermeter_id:
                QMessageBox.critical(self, "Error", "No Powermeter VISA ID provided.")
                return

            self.logText.append("Starting Thorlabs stage...")
            self.motor_controller = ThorlabsController(motor_id)
            self.motor_controller.activate()

            self.logText.append("Starting Powermeter...")
            self.powermeter_controller = PowermeterController(powermeter_id)
            self.powermeter_controller.activate()
            self.logText.append("Hardware activated.")

            self.activateButton.setEnabled(False)
            self.deactivateButton.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate hardware: {e}")
            self.logText.append(f"Error activating hardware: {e}")

    def deactivateHardware(self):
        try:
            if self.powermeter_controller:
                self.powermeter_controller.deactivate()
                self.logText.append("Powermeter deactivated.")
            if self.motor_controller:
                self.motor_controller.disable()
                self.logText.append("Motor disabled.")
            self.powermeter_controller = None
            self.motor_controller = None

            self.deactivateButton.setEnabled(False)
            self.activateButton.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to deactivate hardware: {e}")
            self.logText.append(f"Error deactivating hardware: {e}")

    def start_measurement(self):
        try:
            if self.motor_controller is None or self.powermeter_controller is None:
                self.activateHardware()

            self.logText.append("Starting measurement...")
            self.startButton.setEnabled(False)
            self.abortButton.setEnabled(True)

            start_pos = float(self.positionInput.text())
            end_pos = float(self.endPositionInput.text())
            step_size = float(self.stepSizeInput.text())
            no_avg = int(self.noAvgInput.text())
            num_points = int(np.floor((end_pos - start_pos) / step_size)) + 1
            positions = np.linspace(start_pos, end_pos, num_points)
            save_data = self.saveDataCheckbox.isChecked()

            header_text = self.headerInput.text()
            self.thread = PowerMeasurementThread(self.motor_controller, self.powermeter_controller,
                                            positions, no_avg, save_data, header_text)

            self.thread.log_signal.connect(self.update_log)
            self.thread.plot_signal.connect(self.update_plot)
            self.thread.finished.connect(self.measurement_finished)
            self.thread.start()
            self.logText.append(f"Number of steps: {num_points}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start measurement: {e}")
            self.startButton.setEnabled(True)
            self.abortButton.setEnabled(False)

    def update_log(self, message):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.logText.append(f"[{current_time}] {message}")

    def update_plot(self, positions, power_data):
        self.ax.clear()
        self.ax.plot(positions, power_data, marker='o')
        self.ax.set_xlabel("Position (mm)")
        self.ax.set_ylabel("Power")
        self.canvas.draw_idle()

    def abort_measurement(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            self.logText.append("Measurement aborted.")

    def measurement_finished(self):
        self.logText.append("Measurement thread has finished.")
        self.startButton.setEnabled(True)
        self.abortButton.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    tabs = QTabWidget()

    spectrometer_tab = SFGTemporalOverlapGUI()
    tabs.addTab(spectrometer_tab, "Spectrometer Measurement")

    powermeter_tab = PowermeterMeasurementGUI()
    tabs.addTab(powermeter_tab, "Powermeter Measurement")

    tabs.resize(1000, 800)
    tabs.setWindowTitle("D-lab Controller - SFG Temporal Overlap")
    tabs.show()

    sys.exit(app.exec_())
