import sys
import os
import time
import numpy as np
import thorlabs_apt as apt
import hardware.avaspec_driver._avs_py as avs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QTextEdit, QLineEdit, QLabel, \
    QHBoxLayout, QComboBox, QSplitter, QCheckBox
from PyQt5.QtCore import QThread, pyqtSignal
import datetime


class MeasurementThread(QThread):
    log_signal = pyqtSignal(str)
    plot_signal = pyqtSignal(object, object)

    def __init__(self, motor, active_spec_handle, positions, int_time, no_avg, save_data):
        super().__init__()
        self.motor = motor
        self.active_spec_handle = active_spec_handle
        self.positions = positions
        self.int_time = int_time
        self.no_avg = no_avg
        self.running = True
        self.spectrum_data = []
        self.save_data = save_data

    def run(self):
        try:
            wavelength = avs.AVS_GetLambda(self.active_spec_handle)
            num_pixels = len(wavelength)
            start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for position in self.positions:
                if not self.running:
                    self.log_signal.emit("Measurement aborted.")
                    return

                log_message = f"Moving to position {position:.3f}"
                self.log_signal.emit(log_message)
                self.motor.move_to(position, blocking=True)
                time.sleep(0.5)
                self.log_signal.emit(f"Reached position {position:.3f}")

                avs.AVS_StopMeasure(self.active_spec_handle)
                time.sleep(0.5)
                avs.set_measure_params(self.active_spec_handle, self.int_time, self.no_avg)
                avs.AVS_Measure(self.active_spec_handle)
                timestamp, data = avs.get_spectrum(self.active_spec_handle)

                if len(data) == num_pixels:
                    self.spectrum_data.append(data)
                    self.log_signal.emit("Spectrum successfully taken.")
                    self.plot_signal.emit(np.array(wavelength), np.array(self.spectrum_data))
                else:
                    self.log_signal.emit(f"Warning: Skipped measurement at {position:.3f} due to size mismatch.")

            if self.save_data and len(self.spectrum_data) > 0:
                self.save_spectrum_data(np.array(wavelength), np.array(self.spectrum_data), start_time)
                self.log_signal.emit("Data saved successfully.")

            avs.AVS_Deactivate(self.active_spec_handle)
            self.log_signal.emit("Measurement complete.")
        except Exception as e:
            self.log_signal.emit(f"Error: {e}")

    def save_spectrum_data(self, wavelength, spectrum_data, start_time):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(os.getcwd(), f"SFGTemporalOverlap_{date_str}.txt")

        header = (f"Measurement Date: {start_time}\n"
                  f"Integration Time: {self.int_time} ms\n"
                  f"Number of Averages: {self.no_avg}\n"
                  f"Stage Positions: {self.positions}\n"
                  f"Data Columns: Wavelength (nm) + Spectrum Intensity at each position")

        np.savetxt(save_path, np.column_stack([wavelength] + spectrum_data.tolist()), header=header)

    def stop(self):
        self.running = False


class SFGTemporalOverlapGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.spectrometer_handles = []
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()
        controls_layout = QVBoxLayout()

        self.motorIDLabel = QLabel("Motor ID:")
        self.motorIDInput = QLineEdit("83838295")
        controls_layout.addWidget(self.motorIDLabel)
        controls_layout.addWidget(self.motorIDInput)

        self.positionLabel = QLabel("Start Position:")
        self.positionInput = QLineEdit("0")
        self.endPositionLabel = QLabel("End Position:")
        self.endPositionInput = QLineEdit("10")
        self.numPointsLabel = QLabel("Number of Points:")
        self.numPointsInput = QLineEdit("5")

        controls_layout.addWidget(self.positionLabel)
        controls_layout.addWidget(self.positionInput)
        controls_layout.addWidget(self.endPositionLabel)
        controls_layout.addWidget(self.endPositionInput)
        controls_layout.addWidget(self.numPointsLabel)
        controls_layout.addWidget(self.numPointsInput)

        self.intTimeLabel = QLabel("Integration Time (ms):")
        self.intTimeInput = QLineEdit("100")
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

        self.saveDataCheckbox = QCheckBox("Save Data to File")
        controls_layout.addWidget(self.saveDataCheckbox)

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

        self.canvas = FigureCanvas(plt.figure())

        splitter = QSplitter()
        splitter.addWidget(controls_container)
        splitter.addWidget(self.canvas)
        splitter.setSizes([300, 500])
        layout.addWidget(splitter)

        self.setLayout(layout)
        self.setWindowTitle("D-lab Controller - SFG Temporal Overlap")
        self.resize(1000, 500)

        self.populate_spectrometers()

    def populate_spectrometers(self):
        avs.AVS_Init()
        speclist = avs.AVS_GetList()
        if not speclist:
            QMessageBox.critical(self, "Error", "No spectrometer found!")
            return

        self.spectrometer_handles = speclist
        self.spectrometerSelect.addItems([f"Spectrometer {i + 1}" for i in range(len(speclist))])

    def start_measurement(self):
        try:
            motor_id = int(self.motorIDInput.text())
            start_pos = float(self.positionInput.text())
            end_pos = float(self.endPositionInput.text())
            num_points = int(self.numPointsInput.text())
            int_time = int(self.intTimeInput.text())
            no_avg = int(self.noAvgInput.text())
            positions = np.linspace(start_pos, end_pos, num_points)
            save_data = self.saveDataCheckbox.isChecked()

            motor = apt.Motor(motor_id)
            selected_index = self.spectrometerSelect.currentIndex()
            active_spec_handle = avs.AVS_Activate(self.spectrometer_handles[selected_index])

            self.thread = MeasurementThread(motor, active_spec_handle, positions, int_time, no_avg, save_data)
            self.thread.log_signal.connect(self.update_log)
            self.thread.plot_signal.connect(self.update_plot)
            self.thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start measurement: {e}")

    def update_log(self, message):
        self.logText.append(message)

    def update_plot(self, wavelength, spectrum_data):
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.imshow(spectrum_data, aspect='auto', extent=[wavelength[0], wavelength[-1], 0, len(spectrum_data)],
                  cmap='turbo')
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Position")
        self.canvas.draw()

    def abort_measurement(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            self.update_log("Measurement aborted.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SFGTemporalOverlapGUI()
    window.show()
    sys.exit(app.exec_())
