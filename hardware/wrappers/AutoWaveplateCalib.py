import os
import sys
import time
from datetime import datetime
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit
)
from PyQt5.QtCore import QObject, pyqtSignal, QCoreApplication

# Import your hardware controller classes.
from hardware.wrappers.ThorlabsController import ThorlabsController
from hardware.wrappers.PowermeterController import PowermeterController


class AutoAttCalibWorker(QObject):
    """
    Worker for automatic attenuation calibration.

    This worker uses a Thorlabs motor controller to move a stage through a list of angles
    and a PowermeterController to read power values. It emits signals with each measurement
    so that a live plot and log area in the GUI can be updated.

    Once the measurement sequence is finished, the data is written to a file,
    and a finished signal is emitted.
    """
    measurement_updated = pyqtSignal(float, float)
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, motor_id: int, powermeter_id: str, wavelength: float,
                 angle_list: list, waveplate_name: str, waveplate_path: str,
                 stabilization_time: float = 1.0, parent: QObject = None):
        super().__init__(parent)
        self.motor_id = motor_id
        self.powermeter_id = powermeter_id
        self.wavelength = wavelength
        self.angle_list = angle_list
        self.waveplate_name = waveplate_name
        self.waveplate_path = waveplate_path
        self.stabilization_time = stabilization_time
        self.abort = False  # Flag to allow aborting the measurement

    def _get_unique_filename(self, base_file: str) -> str:
        """Return a unique file name by appending -2, -3, etc., if necessary."""
        if not os.path.exists(base_file):
            return base_file
        base, ext = os.path.splitext(base_file)
        counter = 2
        unique_file = f"{base}-{counter}{ext}"
        while os.path.exists(unique_file):
            counter += 1
            unique_file = f"{base}-{counter}{ext}"
        return unique_file

    def run(self):
        try:
            motor_controller = ThorlabsController(self.motor_id)
            motor_controller.activate(homing=True)
            self.log_signal.emit(f"Motor with ID {self.motor_id} activated and homed.")
        except Exception as e:
            self.log_signal.emit(f"Motor initialization error: {e}")
            return

        try:
            powermeter_controller = PowermeterController(self.powermeter_id)
            powermeter_controller.activate()
            powermeter_controller.set_wavelength(self.wavelength)
            self.log_signal.emit(f"Power meter activated successfully at {self.wavelength} nm.")
        except Exception as e:
            self.log_signal.emit(f"Power meter initialization error: {e}")
            return

        os.makedirs(self.waveplate_path, exist_ok=True)
        base_output_file = os.path.join(
            self.waveplate_path,
            f"calib_{self.waveplate_name}_{datetime.now().strftime('%Y_%m_%d')}.txt"
        )
        output_file = self._get_unique_filename(base_output_file)

        try:
            with open(output_file, "w") as f:
                for angle in self.angle_list:
                    if self.abort:
                        self.log_signal.emit("Calibration aborted by user.")
                        break

                    self.log_signal.emit(f"Moving motor to {angle}°.")
                    motor_controller.move_to(angle, blocking=True)
                    time.sleep(self.stabilization_time)

                    try:
                        power_value = powermeter_controller.read_power()
                    except Exception as e:
                        self.log_signal.emit(f"Failed to read power: {e}")
                        power_value = 0.0

                    self.log_signal.emit(f"Angle: {angle}°, Power: {power_value:.6f} W")
                    f.write(f"{angle}\t{power_value:.6f}\n")
                    self.measurement_updated.emit(angle, power_value)
                    QCoreApplication.processEvents()

            self.log_signal.emit("Power measurement completed.")
        except Exception as e:
            self.log_signal.emit(f"Error during measurement: {e}")
        finally:
            try:
                powermeter_controller.deactivate()
                self.log_signal.emit("Hardware connections closed.")
            except Exception as e:
                self.log_signal.emit(f"Error closing hardware: {e}")
            self.finished.emit(output_file)


class AutoWaveplateCalib(QMainWindow):
    """
    A GUI for automatic attenuation calibration.

    Allows the user to input motor ID, power meter VISA ID, wavelength,
    stabilization time, waveplate name, and parameters to build the angle array.
    A live plot and log area show progress, and an Abort button lets you cancel the calibration.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Attenuation Calibration")
        self.setup_ui()
        self.angle_data = []
        self.power_data = []
        self.worker = None

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel: Input controls
        input_layout = QVBoxLayout()

        self.motor_id_edit = QLineEdit("83837725")
        self.powermeter_id_edit = QLineEdit("USB0::0x1313::0x8078::P0045634::INSTR")
        self.wavelength_edit = QLineEdit("1030.0")
        self.stabilization_time_edit = QLineEdit("1")
        self.waveplate_name_edit = QLineEdit("wp_1")
        self.angle_start_edit = QLineEdit("0")
        self.angle_end_edit = QLineEdit("180")
        self.num_points_edit = QLineEdit("181")

        input_layout.addWidget(QLabel("Motor ID:"))
        input_layout.addWidget(self.motor_id_edit)
        input_layout.addWidget(QLabel("Powermeter ID:"))
        input_layout.addWidget(self.powermeter_id_edit)
        input_layout.addWidget(QLabel("Wavelength (nm):"))
        input_layout.addWidget(self.wavelength_edit)
        input_layout.addWidget(QLabel("Stabilization Time (s):"))
        input_layout.addWidget(self.stabilization_time_edit)
        input_layout.addWidget(QLabel("Waveplate Name:"))
        input_layout.addWidget(self.waveplate_name_edit)
        input_layout.addWidget(QLabel("Angle Start (°):"))
        input_layout.addWidget(self.angle_start_edit)
        input_layout.addWidget(QLabel("Angle End (°):"))
        input_layout.addWidget(self.angle_end_edit)
        input_layout.addWidget(QLabel("Number of Points:"))
        input_layout.addWidget(self.num_points_edit)

        self.calibration_path_label = QLabel()
        self.update_calibration_path()
        self.waveplate_name_edit.textChanged.connect(self.update_calibration_path)
        input_layout.addWidget(QLabel("Calibration Path:"))
        input_layout.addWidget(self.calibration_path_label)

        self.start_button = QPushButton("Start Calibration")
        self.start_button.clicked.connect(self.start_calibration)
        input_layout.addWidget(self.start_button)

        self.abort_button = QPushButton("Abort")
        self.abort_button.clicked.connect(self.abort_calibration)
        self.abort_button.setEnabled(False)
        input_layout.addWidget(self.abort_button)

        input_layout.addStretch()
        main_layout.addLayout(input_layout, 1)

        # Right panel: Plot and log area
        right_layout = QVBoxLayout()
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Angle (°)")
        self.ax.set_ylabel("Power (W)")
        self.line, = self.ax.plot([], [], 'bo-')
        right_layout.addWidget(self.canvas)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(QLabel("Log:"))
        right_layout.addWidget(self.log_text)

        main_layout.addLayout(right_layout, 2)

    def update_calibration_path(self):
        waveplate_name = self.waveplate_name_edit.text()
        path = os.path.join('..', '..', 'ressources', 'calibration', waveplate_name)
        self.calibration_path_label.setText(path)

    def start_calibration(self):
        # Clear previous data
        self.angle_data = []
        self.power_data = []
        self.line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.tight_layout()
        self.canvas.draw()

        try:
            motor_id = int(self.motor_id_edit.text())
            powermeter_id = self.powermeter_id_edit.text()
            wavelength = float(self.wavelength_edit.text())
            stabilization_time = float(self.stabilization_time_edit.text())
            waveplate_name = self.waveplate_name_edit.text()
            waveplate_path = os.path.join('..', '..', 'ressources', 'calibration', waveplate_name)
            angle_start = float(self.angle_start_edit.text())
            angle_end = float(self.angle_end_edit.text())
            num_points = int(self.num_points_edit.text())
        except Exception as e:
            self.append_log(f"Invalid input: {e}")
            return

        angles_to_measure = np.linspace(angle_start, angle_end, num_points)

        self.append_log("Calibration started...")
        self.abort_button.setEnabled(True)

        self.worker = AutoAttCalibWorker(
            motor_id, powermeter_id, wavelength,
            list(angles_to_measure), waveplate_name, waveplate_path, stabilization_time
        )
        self.worker.log_signal.connect(self.append_log)
        self.worker.measurement_updated.connect(self.update_plot)
        self.worker.finished.connect(self.calibration_finished)
        self.worker.run()

    def abort_calibration(self):
        if self.worker:
            self.worker.abort = True
            self.append_log("Abort requested by user.")
            self.abort_button.setEnabled(False)

    def update_plot(self, angle, power):
        self.angle_data.append(angle)
        self.power_data.append(power)
        self.line.set_data(self.angle_data, self.power_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.tight_layout()
        self.canvas.draw()

    def append_log(self, message):
        current_time = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{current_time}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def calibration_finished(self, output_file):
        self.append_log(f"Calibration finished. Data saved to {output_file}")
        self.abort_button.setEnabled(False)
        self.worker = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoWaveplateCalib()
    window.show()
    sys.exit(app.exec_())
