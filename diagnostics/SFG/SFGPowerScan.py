import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QMessageBox, QTextEdit, QLineEdit, QLabel,
                             QSplitter, QComboBox, QCheckBox, QTabWidget)

from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from hardware.wrappers.ThorlabsController import ThorlabsController
from hardware.wrappers.PowermeterController import PowermeterController
from hardware.wrappers.AvaspecController import AvaspecController


from diagnostics.utils import custom_cmap

# ------------------ Two-Motor Powermeter Measurement Thread ------------------

class TwoMotorPowermeterMeasurementThread(QThread):
    log_signal = pyqtSignal(str)
    plot_signal = pyqtSignal(object, object, object)  # motor2 positions, motor1 positions, 2D power map

    def __init__(self, motor1_controller, motor2_controller, powermeter_controller,
                 m1_positions, m2_positions, no_avg, save_data, header_text=""):
        super().__init__()
        self.motor1 = motor1_controller
        self.motor2 = motor2_controller
        self.powermeter = powermeter_controller
        self.m1_positions = m1_positions
        self.m2_positions = m2_positions
        self.no_avg = no_avg
        self.save_data = save_data
        self.header_text = header_text
        self.running = True

        self.power_map = np.zeros((len(m1_positions), len(m2_positions)))

    def run(self):
        try:
            start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for i, pos1 in enumerate(self.m1_positions):
                if not self.running:
                    self.log_signal.emit("Measurement aborted.")
                    break

                self.log_signal.emit(f"Moving Motor1 to {pos1:.3f}...")
                self.motor1.move_to(pos1, blocking=True)
                time.sleep(0.5)
                current_m1 = self.motor1.get_position()
                self.log_signal.emit(f"Motor1 reached {current_m1:.3f}")

                for j, pos2 in enumerate(self.m2_positions):
                    if not self.running:
                        self.log_signal.emit("Measurement aborted.")
                        break

                    self.log_signal.emit(f"Moving Motor2 to {pos2:.3f}...")
                    self.motor2.move_to(pos2, blocking=True)
                    time.sleep(0.5)
                    current_m2 = self.motor2.get_position()
                    self.log_signal.emit(f"Motor2 reached {current_m2:.3f}")

                    self.powermeter.set_avg(self.no_avg)
                    power = self.powermeter.read_power()
                    self.log_signal.emit(f"Measured power: {power}")
                    self.power_map[i, j] = power

                    self.plot_signal.emit(self.m2_positions, self.m1_positions, self.power_map.copy())

            if self.save_data:
                self.save_power_map(self.power_map, start_time)

            self.log_signal.emit("Measurement complete.")
        except Exception as e:
            self.log_signal.emit(f"Error: {e}")

    def save_power_map(self, power_map, start_time):
        date_str = start_time.split(" ")[0]
        save_dir = os.path.join("C:\\data", date_str, "SFGPowerScan")
        os.makedirs(save_dir, exist_ok=True)

        safe_time = start_time.replace(" ", "_").replace(":", "-")
        file_name = os.path.join(save_dir, f"SFGPowerScan_Power_{safe_time}.txt")
        try:
            with open(file_name, "w") as f:
                f.write("# Power scan with powermeter\n")
                if self.header_text:
                    f.write(f"# {self.header_text}\n")
                f.write(f"# Date: {start_time}\n")
                f.write(f"# Number of Averages: {self.no_avg}\n")
                f.write(f"# Motor 1 positions (rows): start = {self.m1_positions[0]}, end = {self.m1_positions[-1]}, count = {len(self.m1_positions)}\n")
                f.write(f"# Motor 2 positions (columns): start = {self.m2_positions[0]}, end = {self.m2_positions[-1]}, count = {len(self.m2_positions)}\n")
                f.write("\n")
                header = "Motor1 \\ Motor2"
                for pos2 in self.m2_positions:
                    header += f"\t{pos2}"
                f.write(header + "\n")
                for i, pos1 in enumerate(self.m1_positions):
                    row = f"{pos1}"
                    for power in power_map[i]:
                        row += f"\t{power}"
                    f.write(row + "\n")
            self.log_signal.emit(f"Saved power map to {file_name}")
        except Exception as e:
            self.log_signal.emit(f"Error saving data: {e}")

    def stop(self):
        self.running = False

# ------------------ Two-Motor Powermeter Measurement GUI ------------------

class TwoMotorPowermeterMeasurementGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.motor1_controller = None
        self.motor2_controller = None
        self.powermeter_controller = None
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        controls_layout = QVBoxLayout()

        self.motor1IDLabel = QLabel("Motor1 ID:")
        self.motor1IDInput = QLineEdit("83838295")
        controls_layout.addWidget(self.motor1IDLabel)
        controls_layout.addWidget(self.motor1IDInput)

        self.homing_motor1_Checkbox = QCheckBox("Home on Activate")
        self.homing_motor1_Checkbox.setChecked(False)
        controls_layout.addWidget(self.homing_motor1_Checkbox)

        self.motor1StartLabel = QLabel("Motor1 Start Position:")
        self.motor1StartInput = QLineEdit("0")
        self.motor1EndLabel = QLabel("Motor1 End Position:")
        self.motor1EndInput = QLineEdit("90")
        self.motor1StepLabel = QLabel("Motor1 Step Size:")
        self.motor1StepInput = QLineEdit("5")
        controls_layout.addWidget(self.motor1StartLabel)
        controls_layout.addWidget(self.motor1StartInput)
        controls_layout.addWidget(self.motor1EndLabel)
        controls_layout.addWidget(self.motor1EndInput)
        controls_layout.addWidget(self.motor1StepLabel)
        controls_layout.addWidget(self.motor1StepInput)

        self.motor2IDLabel = QLabel("Motor2 ID:")
        self.motor2IDInput = QLineEdit("83837725")
        controls_layout.addWidget(self.motor2IDLabel)
        controls_layout.addWidget(self.motor2IDInput)

        self.homing_motor2_Checkbox = QCheckBox("Home on Activate")
        self.homing_motor2_Checkbox.setChecked(False)
        controls_layout.addWidget(self.homing_motor2_Checkbox)

        self.motor2StartLabel = QLabel("Motor2 Start Position:")
        self.motor2StartInput = QLineEdit("0")
        self.motor2EndLabel = QLabel("Motor2 End Position:")
        self.motor2EndInput = QLineEdit("90")
        self.motor2StepLabel = QLabel("Motor2 Step Size:")
        self.motor2StepInput = QLineEdit("5")
        controls_layout.addWidget(self.motor2StartLabel)
        controls_layout.addWidget(self.motor2StartInput)
        controls_layout.addWidget(self.motor2EndLabel)
        controls_layout.addWidget(self.motor2EndInput)
        controls_layout.addWidget(self.motor2StepLabel)
        controls_layout.addWidget(self.motor2StepInput)

        self.powermeterLabel = QLabel("Powermeter VISA ID:")
        self.powermeterInput = QLineEdit("USB0::0x1313::0x8078::P0045634::INSTR")
        controls_layout.addWidget(self.powermeterLabel)
        controls_layout.addWidget(self.powermeterInput)

        self.wavelengthLabel = QLabel("Wavelength (nm):")
        self.wavelengthInput = QLineEdit("343")
        controls_layout.addWidget(self.wavelengthLabel)
        controls_layout.addWidget(self.wavelengthInput)

        self.noAvgLabel = QLabel("Number of Averages:")
        self.noAvgInput = QLineEdit("1")
        controls_layout.addWidget(self.noAvgLabel)
        controls_layout.addWidget(self.noAvgInput)

        self.saveDataCheckbox = QCheckBox("Save Data to File")
        controls_layout.addWidget(self.saveDataCheckbox)
        self.headerLabel = QLabel("Comment:")
        self.headerInput = QLineEdit("")
        controls_layout.addWidget(self.headerLabel)
        controls_layout.addWidget(self.headerInput)

        self.activateButton = QPushButton("Activate Hardware")
        self.activateButton.clicked.connect(self.activateHardware)
        controls_layout.addWidget(self.activateButton)

        self.deactivateButton = QPushButton("Deactivate Hardware")
        self.deactivateButton.clicked.connect(self.deactivateHardware)
        controls_layout.addWidget(self.deactivateButton)
        self.deactivateButton.setEnabled(False)

        buttons_layout = QHBoxLayout()
        self.startButton = QPushButton("Start Measurement")
        self.startButton.clicked.connect(self.start_measurement)
        buttons_layout.addWidget(self.startButton)
        self.abortButton = QPushButton("Abort Measurement")
        self.abortButton.clicked.connect(self.abort_measurement)
        buttons_layout.addWidget(self.abortButton)
        controls_layout.addLayout(buttons_layout)

        self.logText = QTextEdit()
        self.logText.setReadOnly(True)
        controls_layout.addWidget(self.logText)

        controls_container = QWidget()
        controls_container.setLayout(controls_layout)

        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.image = self.ax.imshow(np.zeros((10,10)), aspect='auto', cmap=custom_cmap(512), origin='lower')
        self.cbar = self.figure.colorbar(self.image, ax=self.ax, orientation='horizontal', location='top')
        self.cbar.set_label("Power (W)")
        self.ax.set_xlabel("Motor 2 Position (deg)")
        self.ax.set_ylabel("Motor 1 Position (deg)")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.draw()

        splitter = QSplitter()
        splitter.addWidget(controls_container)
        splitter.addWidget(self.canvas)
        splitter.setSizes([300, 500])
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        self.setWindowTitle("D-lab Controller - Power Scan - Spectrometer")
        self.resize(1000, 800)

    def activateHardware(self):
        try:
            motor1_id = int(self.motor1IDInput.text())
            self.update_log("Activating Motor1...")
            self.motor1_controller = ThorlabsController(motor1_id)
            self.motor1_controller.activate(homing=self.homing_motor1_Checkbox.isChecked())
            self.update_log(f"Current position of Motor1: {self.motor1_controller.get_position()}")

            motor2_id = int(self.motor2IDInput.text())
            self.update_log("Activating Motor2...")
            self.motor2_controller = ThorlabsController(motor2_id)
            self.motor2_controller.activate(homing=self.homing_motor2_Checkbox.isChecked())
            self.update_log(f"Current position of Motor2: {self.motor2_controller.get_position()}")

            powermeter_id = self.powermeterInput.text()
            if not powermeter_id:
                QMessageBox.critical(self, "Error", "No Powermeter VISA ID provided.")
                return
            self.update_log("Activating Powermeter...")
            self.powermeter_controller = PowermeterController(powermeter_id)
            self.powermeter_controller.activate()
            wavelength = float(self.wavelengthInput.text())
            self.powermeter_controller.set_wavelength(wavelength)
            self.update_log(f"Powermeter wavelength set to {wavelength} nm")
            self.update_log("Hardware activated.")

            self.activateButton.setEnabled(False)
            self.deactivateButton.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate hardware: {e}")
            self.update_log(f"Error activating hardware: {e}")

    def deactivateHardware(self):
        try:
            if self.powermeter_controller:
                self.powermeter_controller.deactivate()
                self.update_log("Powermeter deactivated.")
            if self.motor1_controller:
                self.motor1_controller.disable()
                self.update_log("Motor1 disabled.")
            if self.motor2_controller:
                self.motor2_controller.disable()
                self.update_log("Motor2 disabled.")
            self.powermeter_controller = None
            self.motor1_controller = None
            self.motor2_controller = None

            self.deactivateButton.setEnabled(False)
            self.activateButton.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to deactivate hardware: {e}")
            self.update_log(f"Error deactivating hardware: {e}")

    def start_measurement(self):
        try:
            if (self.motor1_controller is None or
                self.motor2_controller is None or
                self.powermeter_controller is None):
                self.activateHardware()

            self.update_log("Starting measurement...")
            self.startButton.setEnabled(False)
            self.abortButton.setEnabled(True)

            m1_start = float(self.motor1StartInput.text())
            m1_end = float(self.motor1EndInput.text())
            m1_step = float(self.motor1StepInput.text())
            m1_points = int(np.floor((m1_end - m1_start) / m1_step)) + 1
            m1_positions = np.linspace(m1_start, m1_end, m1_points)

            m2_start = float(self.motor2StartInput.text())
            m2_end = float(self.motor2EndInput.text())
            m2_step = float(self.motor2StepInput.text())
            m2_points = int(np.floor((m2_end - m2_start) / m2_step)) + 1
            m2_positions = np.linspace(m2_start, m2_end, m2_points)

            no_avg = int(self.noAvgInput.text())
            save_data = self.saveDataCheckbox.isChecked()
            header_text = self.headerInput.text()

            self.thread = TwoMotorPowermeterMeasurementThread(self.motor1_controller,
                                                               self.motor2_controller,
                                                               self.powermeter_controller,
                                                               m1_positions, m2_positions,
                                                               no_avg, save_data, header_text)
            self.thread.log_signal.connect(self.update_log)
            self.thread.plot_signal.connect(self.update_plot)
            self.thread.finished.connect(self.measurement_finished)
            self.thread.start()

            self.update_log(f"Grid: {m1_points} x {m2_points} points")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start measurement: {e}")
            self.startButton.setEnabled(True)
            self.abortButton.setEnabled(False)

    def update_log(self, message):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.logText.append(f"[{current_time}] {message}")

    def update_plot(self, m2_positions, m1_positions, power_map):
        self.image.set_extent([m2_positions[0], m2_positions[-1], m1_positions[0], m1_positions[-1]])
        self.image.set_data(power_map)
        self.image.set_clim(vmin=np.min(power_map), vmax=np.max(power_map))
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


# ------------------ Two-Motor Spectrometer Measurement Thread ------------------
class TwoMotorSpectrometerMeasurementThread(QThread):
    log_signal = pyqtSignal(str)
    plot_signal = pyqtSignal(object, object, object)

    def __init__(self, motor1_controller, motor2_controller, spectrometer_controller,
                 m1_positions, m2_positions, int_time, no_avg, save_data, header_text=""):
        super().__init__()
        self.motor1 = motor1_controller
        self.motor2 = motor2_controller
        self.spec_controller = spectrometer_controller
        self.m1_positions = m1_positions
        self.m2_positions = m2_positions
        self.int_time = int_time
        self.no_avg = no_avg
        self.save_data = save_data
        self.header_text = header_text
        self.running = True

        self.intensity_map = np.zeros((len(m1_positions), len(m2_positions)))

    def run(self):
        try:
            start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for i, pos1 in enumerate(self.m1_positions):
                if not self.running:
                    self.log_signal.emit("Measurement aborted.")
                    break

                self.log_signal.emit(f"Moving Motor1 to {pos1:.3f}...")
                self.motor1.move_to(pos1, blocking=True)
                time.sleep(0.5)
                current_m1 = self.motor1.get_position()
                self.log_signal.emit(f"Motor1 reached {current_m1:.3f}")

                for j, pos2 in enumerate(self.m2_positions):
                    if not self.running:
                        self.log_signal.emit("Measurement aborted.")
                        break

                    self.log_signal.emit(f"Moving Motor2 to {pos2:.3f}...")
                    self.motor2.move_to(pos2, blocking=True)
                    time.sleep(0.5)
                    current_m2 = self.motor2.get_position()
                    self.log_signal.emit(f"Motor2 reached {current_m2:.3f}")

                    timestamp, spectrum = self.spec_controller.measure_spectrum(self.int_time, self.no_avg)
                    self.log_signal.emit("Spectrum acquired.")
                    integrated_intensity = np.sum(spectrum)
                    self.intensity_map[i, j] = integrated_intensity

                    self.plot_signal.emit(self.m2_positions, self.m1_positions, self.intensity_map.copy())

            if self.save_data:
                self.save_spectrometer_data(self.intensity_map, start_time)

            self.log_signal.emit("Measurement complete.")
        except Exception as e:
            self.log_signal.emit(f"Error: {e}")

    def save_spectrometer_data(self, intensity_map, start_time):
        date_str = start_time.split(" ")[0]
        save_dir = os.path.join("C:\\data", date_str, "SFGPowerScan")
        os.makedirs(save_dir, exist_ok=True)

        safe_time = start_time.replace(" ", "_").replace(":", "-")
        file_name = os.path.join(save_dir, f"SFGPowerScan_Spectrum_{safe_time}.txt")
        try:
            with open(file_name, "w") as f:
                f.write("# Power scan with spectrometer\n")
                if self.header_text:
                    f.write(f"# {self.header_text}\n")
                f.write(f"# Date: {start_time}\n")
                f.write(f"# Integration Time (ms): {self.int_time}\n")
                f.write(f"# Number of Averages: {self.no_avg}\n")
                f.write(
                    f"# Motor 1 positions (rows): start = {self.m1_positions[0]}, end = {self.m1_positions[-1]}, count = {len(self.m1_positions)}\n")
                f.write(
                    f"# Motor 2 positions (columns): start = {self.m2_positions[0]}, end = {self.m2_positions[-1]}, count = {len(self.m2_positions)}\n")
                f.write("\n")
                header = "Motor1 \\ Motor2"
                for pos2 in self.m2_positions:
                    header += f"\t{pos2}"
                f.write(header + "\n")
                for i, pos1 in enumerate(self.m1_positions):
                    row = f"{pos1}"
                    for intensity in intensity_map[i]:
                        row += f"\t{intensity}"
                    f.write(row + "\n")
            self.log_signal.emit(f"Saved spectrometer data to {file_name}")
        except Exception as e:
            self.log_signal.emit(f"Error saving data: {e}")

    def stop(self):
        self.running = False


# ------------------ Two-Motor Spectrometer Measurement GUI ------------------
class TwoMotorSpectrometerMeasurementGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.motor1_controller = None
        self.motor2_controller = None
        self.spec_controller = None
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        controls_layout = QVBoxLayout()

        self.motor1IDLabel = QLabel("Motor1 ID:")
        self.motor1IDInput = QLineEdit("83838295")
        controls_layout.addWidget(self.motor1IDLabel)
        controls_layout.addWidget(self.motor1IDInput)

        self.homing_motor1_Checkbox = QCheckBox("Home on Activate")
        self.homing_motor1_Checkbox.setChecked(False)
        controls_layout.addWidget(self.homing_motor1_Checkbox)


        self.motor1StartLabel = QLabel("Motor1 Start Position:")
        self.motor1StartInput = QLineEdit("0")
        self.motor1EndLabel = QLabel("Motor1 End Position:")
        self.motor1EndInput = QLineEdit("90")
        self.motor1StepLabel = QLabel("Motor1 Step Size:")
        self.motor1StepInput = QLineEdit("5")
        controls_layout.addWidget(self.motor1StartLabel)
        controls_layout.addWidget(self.motor1StartInput)
        controls_layout.addWidget(self.motor1EndLabel)
        controls_layout.addWidget(self.motor1EndInput)
        controls_layout.addWidget(self.motor1StepLabel)
        controls_layout.addWidget(self.motor1StepInput)

        self.motor2IDLabel = QLabel("Motor2 ID:")
        self.motor2IDInput = QLineEdit("83837725")
        controls_layout.addWidget(self.motor2IDLabel)
        controls_layout.addWidget(self.motor2IDInput)

        self.homing_motor2_Checkbox = QCheckBox("Home on Activate")
        self.homing_motor2_Checkbox.setChecked(False)
        controls_layout.addWidget(self.homing_motor2_Checkbox)

        self.motor2StartLabel = QLabel("Motor2 Start Position:")
        self.motor2StartInput = QLineEdit("0")
        self.motor2EndLabel = QLabel("Motor2 End Position:")
        self.motor2EndInput = QLineEdit("90")
        self.motor2StepLabel = QLabel("Motor2 Step Size:")
        self.motor2StepInput = QLineEdit("5")
        controls_layout.addWidget(self.motor2StartLabel)
        controls_layout.addWidget(self.motor2StartInput)
        controls_layout.addWidget(self.motor2EndLabel)
        controls_layout.addWidget(self.motor2EndInput)
        controls_layout.addWidget(self.motor2StepLabel)
        controls_layout.addWidget(self.motor2StepInput)

        self.specLabel = QLabel("Select Spectrometer:")
        self.specSelect = QComboBox()
        controls_layout.addWidget(self.specLabel)
        controls_layout.addWidget(self.specSelect)
        self.searchSpecButton = QPushButton("Search Spectrometers")
        self.searchSpecButton.clicked.connect(self.populate_spectrometers)
        controls_layout.addWidget(self.searchSpecButton)

        self.intTimeLabel = QLabel("Integration Time (ms):")
        self.intTimeInput = QLineEdit("10")
        self.noAvgLabel = QLabel("Number of Averages:")
        self.noAvgInput = QLineEdit("1")
        controls_layout.addWidget(self.intTimeLabel)
        controls_layout.addWidget(self.intTimeInput)
        controls_layout.addWidget(self.noAvgLabel)
        controls_layout.addWidget(self.noAvgInput)

        self.saveDataCheckbox = QCheckBox("Save Data to File")
        controls_layout.addWidget(self.saveDataCheckbox)
        self.headerLabel = QLabel("Comment:")
        self.headerInput = QLineEdit("")
        controls_layout.addWidget(self.headerLabel)
        controls_layout.addWidget(self.headerInput)

        self.activateButton = QPushButton("Activate Hardware")
        self.activateButton.clicked.connect(self.activateHardware)
        controls_layout.addWidget(self.activateButton)

        self.deactivateButton = QPushButton("Deactivate Hardware")
        self.deactivateButton.clicked.connect(self.deactivateHardware)
        controls_layout.addWidget(self.deactivateButton)
        self.deactivateButton.setEnabled(False)

        buttons_layout = QHBoxLayout()
        self.startButton = QPushButton("Start Measurement")
        self.startButton.clicked.connect(self.start_measurement)
        buttons_layout.addWidget(self.startButton)
        self.abortButton = QPushButton("Abort Measurement")
        self.abortButton.clicked.connect(self.abort_measurement)
        buttons_layout.addWidget(self.abortButton)
        controls_layout.addLayout(buttons_layout)

        self.logText = QTextEdit()
        self.logText.setReadOnly(True)
        controls_layout.addWidget(self.logText)

        controls_container = QWidget()
        controls_container.setLayout(controls_layout)

        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.image = self.ax.imshow(np.zeros((10, 10)), aspect='auto', cmap=custom_cmap(512), origin='lower')
        self.cbar = self.figure.colorbar(self.image, ax=self.ax, orientation='horizontal', location='top')
        self.cbar.set_label("Counts")
        self.ax.set_xlabel("Motor 2 Position (deg)")
        self.ax.set_ylabel("Motor 1 Position (deg)")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.draw()

        splitter = QSplitter()
        splitter.addWidget(controls_container)
        splitter.addWidget(self.canvas)
        splitter.setSizes([300, 500])
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        self.setWindowTitle("D-lab Controller - Power Scan - Powermeter")
        self.resize(1000, 800)

    def populate_spectrometers(self):
        self.specSelect.clear()
        speclist = AvaspecController.list_spectrometers()
        if not speclist:
            QMessageBox.critical(self, "Error", "No spectrometer found.")
            self.update_log("No spectrometer found.")
            return
        self.spectrometers = speclist
        self.specSelect.addItems([f"Spectrometer {i + 1}" for i in range(len(speclist))])
        self.update_log(f"Found {len(speclist)} spectrometer(s).")

    def activateHardware(self):
        try:
            motor1_id = int(self.motor1IDInput.text())
            self.update_log("Activating Motor1...")
            self.motor1_controller = ThorlabsController(motor1_id)
            self.motor1_controller.activate(homing=self.homing_motor1_Checkbox.isChecked())
            self.update_log(f"Current position of Motor1: {self.motor1_controller.get_position()}")

            motor2_id = int(self.motor2IDInput.text())
            self.update_log("Activating Motor2...")
            self.motor2_controller = ThorlabsController(motor2_id)
            self.motor2_controller.activate(homing=self.homing_motor2_Checkbox.isChecked())
            self.update_log(f"Current position of Motor2: {self.motor2_controller.get_position()}")


            selected_index = self.specSelect.currentIndex()
            if selected_index < 0 or selected_index >= len(self.spectrometers):
                QMessageBox.critical(self, "Error", "No spectrometer selected.")
                return
            spec_handle = self.spectrometers[selected_index]
            self.update_log("Activating Spectrometer...")
            self.spec_controller = AvaspecController(spec_handle)
            self.spec_controller.activate()

            self.update_log("Hardware activated.")
            self.activateButton.setEnabled(False)
            self.deactivateButton.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate hardware: {e}")
            self.update_log(f"Error activating hardware: {e}")

    def deactivateHardware(self):
        try:
            if self.spec_controller:
                self.spec_controller.deactivate()
                self.update_log("Spectrometer deactivated.")
            if self.motor1_controller:
                self.motor1_controller.disable()
                self.update_log("Motor1 disabled.")
            if self.motor2_controller:
                self.motor2_controller.disable()
                self.update_log("Motor2 disabled.")
            self.spec_controller = None
            self.motor1_controller = None
            self.motor2_controller = None

            self.deactivateButton.setEnabled(False)
            self.activateButton.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to deactivate hardware: {e}")
            self.update_log(f"Error deactivating hardware: {e}")

    def start_measurement(self):
        try:
            if (self.motor1_controller is None or
                    self.motor2_controller is None or
                    self.spec_controller is None):
                self.activateHardware()

            self.update_log("Starting measurement...")
            self.startButton.setEnabled(False)
            self.abortButton.setEnabled(True)

            m1_start = float(self.motor1StartInput.text())
            m1_end = float(self.motor1EndInput.text())
            m1_step = float(self.motor1StepInput.text())
            m1_points = int(np.floor((m1_end - m1_start) / m1_step)) + 1
            m1_positions = np.linspace(m1_start, m1_end, m1_points)

            m2_start = float(self.motor2StartInput.text())
            m2_end = float(self.motor2EndInput.text())
            m2_step = float(self.motor2StepInput.text())
            m2_points = int(np.floor((m2_end - m2_start) / m2_step)) + 1
            m2_positions = np.linspace(m2_start, m2_end, m2_points)

            int_time = float(self.intTimeInput.text())
            no_avg = int(self.noAvgInput.text())
            save_data = self.saveDataCheckbox.isChecked()
            header_text = self.headerInput.text()

            self.thread = TwoMotorSpectrometerMeasurementThread(self.motor1_controller,
                                                                self.motor2_controller,
                                                                self.spec_controller,
                                                                m1_positions, m2_positions,
                                                                int_time, no_avg, save_data, header_text)
            self.thread.log_signal.connect(self.update_log)
            self.thread.plot_signal.connect(self.update_plot)
            self.thread.finished.connect(self.measurement_finished)
            self.thread.start()

            self.update_log(f"Grid: {m1_points} x {m2_points} points")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start measurement: {e}")
            self.startButton.setEnabled(True)
            self.abortButton.setEnabled(False)

    def update_log(self, message):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.logText.append(f"[{current_time}] {message}")

    def update_plot(self, m2_positions, m1_positions, intensity_map):
        self.image.set_extent([m2_positions[0], m2_positions[-1], m1_positions[0], m1_positions[-1]])
        self.image.set_data(intensity_map)
        self.image.set_clim(vmin=np.min(intensity_map), vmax=np.max(intensity_map))
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



if __name__ == "__main__":
    app = QApplication(sys.argv)
    tabs = QTabWidget()

    spectrometer_tab = TwoMotorSpectrometerMeasurementGUI()
    tabs.addTab(spectrometer_tab, "Spectrometer Measurement")

    powermeter_tab = TwoMotorPowermeterMeasurementGUI()
    tabs.addTab(powermeter_tab, "Powermeter Measurement")

    tabs.resize(1200, 900)
    tabs.setWindowTitle("D-lab Controller - Power Scans")
    tabs.show()

    sys.exit(app.exec_())
