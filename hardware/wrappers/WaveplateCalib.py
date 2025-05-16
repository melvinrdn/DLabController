import os
import sys
import json
import datetime
import numpy as np
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QGroupBox, QFileDialog, QTextEdit, QTabWidget, QMainWindow, QCheckBox, QMessageBox
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D

# ----------------------------
# Waveplate Calibration Widget
# ----------------------------

# Constants for calibration
NUM_WAVEPLATES = 6
# Using six distinct colors.
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
CONFIG_FILE = os.path.join('ressources', 'calibration', 'calib_path.json')


class WaveplateCalibWidget(QWidget):
    """
    A PyQt5 widget for waveplate calibration.

    The calibration files are loaded from a JSON configuration file.
    The widget creates a grid of subplots (one per waveplate) without individual titles
    and uses a global legend (WP1, WP2, ...).

    Instead of having its own log, it calls a provided log_callback to log messages.
    Additionally, after a calibration file is loaded/updated for a waveplate,
    it calls a calibration_changed_callback passing the specific waveplate index and its parameters.
    """

    def __init__(self, log_callback=None, calibration_changed_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Waveplate Calibration")
        self.config_file = CONFIG_FILE
        self.log_callback = log_callback
        self.calibration_changed_callback = calibration_changed_callback
        # Dictionary to store calibration parameters for each waveplate:
        # key: waveplate index (int), value: (max_value, offset)
        self.calibration_params = {}

        self.default_calib = {}  # Will be loaded from JSON

        # For each waveplate, we will store max and offset entries.
        self.wp_entries = {}

        self.initUI()
        self.default_calib = self.load_config()
        self.load_default_calibrations()

    def initUI(self):
        main_layout = QHBoxLayout(self)

        # Left panel: Calibration options
        options_group = QGroupBox("Calibration Options")
        options_layout = QVBoxLayout(options_group)

        # Create entries for each waveplate.
        for i in range(1, NUM_WAVEPLATES + 1):
            h_layout = QHBoxLayout()
            label = QLabel(f"WP{i}:")
            label.setFixedWidth(50)
            h_layout.addWidget(label)

            max_label = QLabel("Max:")
            max_label.setFixedWidth(30)
            h_layout.addWidget(max_label)
            max_edit = QLineEdit("0")
            max_edit.setFixedWidth(60)
            h_layout.addWidget(max_edit)

            offset_label = QLabel("Offset:")
            offset_label.setFixedWidth(50)
            h_layout.addWidget(offset_label)
            offset_edit = QLineEdit("0")
            offset_edit.setFixedWidth(60)
            h_layout.addWidget(offset_edit)

            self.wp_entries[str(i)] = {"max": max_edit, "offset": offset_edit}
            options_layout.addLayout(h_layout)

        # Dropdown to select a waveplate.
        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(QLabel("Select WP:"))
        self.wp_dropdown = QComboBox()
        self.wp_dropdown.addItems([str(i) for i in range(1, NUM_WAVEPLATES + 1)])
        h_layout2.addWidget(self.wp_dropdown)
        options_layout.addLayout(h_layout2)

        # Button to update calibration file.
        self.update_calib_btn = QPushButton("Update Calibration File")
        self.update_calib_btn.clicked.connect(self.update_selected_calibration_file)
        options_layout.addWidget(self.update_calib_btn)

        main_layout.addWidget(options_group, 1)

        # Right panel: Plot area.
        self.fig = Figure(figsize=(5, 6), dpi=100)
        # Create a grid of subplots.
        rows = int(np.ceil(np.sqrt(NUM_WAVEPLATES)))
        cols = int(np.ceil(NUM_WAVEPLATES / rows))
        self.axes = [self.fig.add_subplot(rows, cols, i + 1) for i in range(NUM_WAVEPLATES)]
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, 2)

    def load_config(self):
        """Loads calibration configuration from a JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            valid_config = {k: v for k, v in config.items() if os.path.isfile(v)}
            self.log("Configuration loaded from JSON.")
            return valid_config
        except Exception as e:
            self.log(f"Calibration configuration not found or invalid: {e}")
            return {str(i): f"default_wp_{i}_calib.txt" for i in range(1, NUM_WAVEPLATES + 1)}

    def save_config(self):
        """Saves the current calibration configuration to the JSON file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.default_calib, f)
            self.log("Calibration configuration saved.")
        except Exception as e:
            self.log(f"Error saving calibration configuration: {e}")

    def load_default_calibrations(self):
        """Loads calibration data for each waveplate from the configuration."""
        for wp in self.default_calib.keys():
            self.open_calibration_file(int(wp))
        self.update_global_legend()

    def open_calibration_file(self, wp_index: int):
        filename = self.default_calib.get(str(wp_index), "")
        if not filename or not os.path.isfile(filename):
            self.log(f"Calibration file for WP{wp_index} not found.")
            return
        try:
            angles, powers = np.loadtxt(filename, delimiter='\t', unpack=True)
            amplitude, phase = self.cos_fit(angles, powers)
            # Save calibration parameters: use 2*amplitude as the max value.
            self.calibration_params[wp_index] = (2 * amplitude, phase)
            # Update the corresponding entries.
            self.wp_entries[str(wp_index)]["max"].setText(f"{2 * amplitude:.2f}")
            self.wp_entries[str(wp_index)]["offset"].setText(f"{phase:.2f}")
            # Plot on the corresponding axis.
            self.plot_waveplate_data(self.axes[wp_index - 1], angles, powers, COLORS[wp_index - 1], amplitude, phase)

            # Convert the filename to a relative path based on the config folder.
            relative_filename = os.path.relpath(filename, os.path.dirname(self.config_file))
            if not relative_filename.startswith("."):
                relative_filename = "./" + relative_filename

            self.log(f"Calibration data for WP{wp_index} loaded from {relative_filename}.")
            self.canvas.draw()
            self.update_global_legend()
            # Notify that calibration has changed for only this waveplate.
            if self.calibration_changed_callback:
                self.calibration_changed_callback(wp_index, self.calibration_params[wp_index])
        except Exception as e:
            self.log(f"Failed to load calibration data for WP{wp_index}: {e}")

    def update_selected_calibration_file(self):
        """Prompts the user to select a new calibration file for the chosen waveplate."""
        wp_index = self.wp_dropdown.currentText()
        default_folder = os.path.dirname(self.config_file)
        filename, _ = QFileDialog.getOpenFileName(self, f"Select calibration file for WP{wp_index}",
                                                  default_folder,
                                                  "Text Files (*.txt);;All Files (*)")
        if filename:
            self.default_calib[wp_index] = filename
            # Convert to a relative path for a nicer log message.
            relative_filename = os.path.relpath(filename, os.path.dirname(self.config_file))
            self.log(f"Calibration file for WP{wp_index} updated to {relative_filename}.")
            self.save_config()
            self.open_calibration_file(int(wp_index))

    def plot_waveplate_data(self, ax, angles, powers, color, amplitude, phase):
        ax.clear()
        ax.plot(angles, powers, marker='o', linestyle='None', color=color)
        x = np.linspace(0, 360, 361)
        ax.plot(x, self.cos_func(x, amplitude, phase), color=color)
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Power (W)")

    def update_global_legend(self):
        handles = [Line2D([], [], color=COLORS[i], marker='o', linestyle='None') for i in range(NUM_WAVEPLATES)]
        labels = [f"WP{i + 1}" for i in range(NUM_WAVEPLATES)]
        self.fig.legend(handles, labels, loc='upper right')
        self.fig.tight_layout(rect=[0, 0, 0.85, 1])
        self.canvas.draw()

    def cos_func(self, x, amplitude, phase):
        """Cosine function for calibration curve."""
        return amplitude * np.cos(2 * np.pi / 90 * x - 2 * np.pi / 90 * phase) + amplitude

    def cos_fit(self, x, y):
        """Fits a cosine function to the calibration data. Returns (amplitude, phase)."""
        initial_guess = (np.max(y) / 2, 0)
        popt, _ = curve_fit(self.cos_func, x, y, p0=initial_guess)
        return popt

    def log(self, message: str):
        """Uses the provided log_callback if available, otherwise prints to console."""
        full_msg = f"[WaveplateCalib] {message}"
        if self.log_callback:
            self.log_callback(full_msg)
        else:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] {full_msg}")
