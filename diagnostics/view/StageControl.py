
import sys
import datetime
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QTextEdit, QTabWidget, QMainWindow, QCheckBox, QMessageBox
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from hardware.wrappers.ThorlabsController import ThorlabsController
from hardware.wrappers.WaveplateCalib import WaveplateCalibWidget, NUM_WAVEPLATES
from hardware.wrappers.AutoWaveplateCalib import AutoWaveplateCalib


def power_to_angle(power: float, amplitude: float, offset: float) -> float:
    """
    Converts a power value to the corresponding angle using a cosine fit.
    This function assumes a fit of the form:
         y = amplitude * cos(2*pi/90 * x - 2*pi/90 * offset) + amplitude
    and returns the corresponding angle for a given power.
    """
    A = amplitude / 2.0
    try:
        angle = -(45 * np.arccos(power / A - 1)) / np.pi + offset
    except Exception:
        angle = offset
    return angle

class StageRow(QWidget):
    """
    A widget representing a single Thorlabs stage row with controls.

    Includes motor ID, activation, homing, target entry (angle or power),
    and current position. Also stores calibration parameters for power conversion.
    """
    def __init__(self, stage_number: int, description: str = "",
                 log_callback=None, parent: QWidget = None):
        super().__init__(parent)
        self.stage_number = stage_number
        self.controller = None
        self.log_callback = log_callback

        # Default calibration parameters for power conversion.
        # 'amplitude' is updated by calibration; it is also considered the maximum power.
        self.amplitude = 100.0
        self.offset = 0.0

        layout = QHBoxLayout(self)
        layout.setSpacing(5)

        # Stage label.
        if stage_number < 6:
            self.stage_label = QLabel(f"Waveplate {stage_number+1}:")
        else:
            self.stage_label = QLabel(f"Stage {stage_number+1}:")
        self.stage_label.setFixedWidth(120)
        layout.addWidget(self.stage_label)

        # Description label.
        self.desc_label = QLabel(f"{description}")
        self.desc_label.setFixedWidth(120)
        layout.addWidget(self.desc_label)

        # Motor ID entry.
        self.motor_id_edit = QLineEdit()
        default_id = "83837725" if stage_number == 0 else "000000"
        self.motor_id_edit.setText(default_id)
        self.motor_id_edit.setPlaceholderText("Motor ID")
        self.motor_id_edit.setFixedWidth(80)
        self.motor_id_edit.setValidator(QIntValidator(1, 99999999, self))
        layout.addWidget(self.motor_id_edit)

        # Activate button.
        self.activate_btn = QPushButton("Activate")
        self.activate_btn.clicked.connect(self.activate_stage)
        layout.addWidget(self.activate_btn)

        # Home on Activate checkbox.
        self.home_on_activate_checkbox = QCheckBox("Home on Activate")
        self.home_on_activate_checkbox.setChecked(False)
        layout.addWidget(self.home_on_activate_checkbox)

        # Home button.
        self.home_btn = QPushButton("Home")
        self.home_btn.clicked.connect(self.home_stage)
        self.home_btn.setEnabled(False)
        layout.addWidget(self.home_btn)

        # Target entry.
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("Target")
        self.target_edit.setFixedWidth(80)
        self.target_edit.setValidator(QDoubleValidator(-10000, 10000, 2, self))
        layout.addWidget(self.target_edit)

        # Power Mode checkbox.
        self.power_mode_checkbox = QCheckBox("Power Mode")
        layout.addWidget(self.power_mode_checkbox)

        # Move To button.
        self.move_btn = QPushButton("Move To")
        self.move_btn.clicked.connect(self.move_stage)
        self.move_btn.setEnabled(False)
        layout.addWidget(self.move_btn)

        # Current position display.
        self.current_edit = QLineEdit()
        self.current_edit.setPlaceholderText("Current")
        self.current_edit.setFixedWidth(80)
        self.current_edit.setReadOnly(True)
        layout.addWidget(self.current_edit)

        layout.addStretch(1)

    def log(self, message: str) -> None:
        full_msg = f"Stage {self.stage_number+1}: {message}"
        if self.log_callback:
            self.log_callback(full_msg)
        else:
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] {full_msg}")

    def activate_stage(self) -> None:
        motor_id_str = self.motor_id_edit.text().strip()
        if not motor_id_str:
            QMessageBox.warning(self, "Error", "Please enter a motor ID.")
            return
        try:
            motor_id = int(motor_id_str)
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid motor ID.")
            return
        try:
            self.controller = ThorlabsController(motor_id)
            home_flag = self.home_on_activate_checkbox.isChecked()
            self.controller.activate(homing=home_flag)
            pos = self.controller.get_position()
            if pos is None:
                raise Exception("Unknown serial number.")
            self.current_edit.setText(f"{pos:.3f}")
            self.activate_btn.setEnabled(False)
            self.motor_id_edit.setEnabled(False)
            self.home_btn.setEnabled(True)
            self.move_btn.setEnabled(True)
            self.log("Activated" + (" with homing." if home_flag else "."))
            self.stage_label.setStyleSheet("background-color: lightgreen;")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate stage: {e}")
            self.log(f"Activation failed: {e}")
            self.controller = None
            self.stage_label.setStyleSheet("")

    def home_stage(self) -> None:
        if self.controller:
            try:
                self.controller.home()
                pos = self.controller.get_position()
                self.current_edit.setText(f"{pos:.3f}" if pos is not None else "N/A")
                self.log("Homed.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to home stage: {e}")
                self.log(f"Home failed: {e}")

    def move_stage(self) -> None:
        if self.controller is None:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return
        target_str = self.target_edit.text().strip()
        if not target_str:
            QMessageBox.warning(self, "Error", "Please enter a target value.")
            return
        try:
            if self.power_mode_checkbox.isChecked():
                desired_power = float(target_str)
                # Cap the desired power to the maximum calibrated power (self.amplitude)
                if desired_power > self.amplitude:
                    self.log(f"Desired power {desired_power} exceeds maximum {self.amplitude:.2f}. Capping to maximum.")
                    desired_power = self.amplitude
                target_angle = power_to_angle(desired_power, self.amplitude, self.offset)
                target_angle = target_angle
                self.log(f"Converting power {desired_power} to angle {target_angle:.2f}.")
            else:
                target_angle = float(target_str)

            target_angle = target_angle % 360

            self.controller.move_to(target_angle, blocking=True)
            pos = self.controller.get_position()
            self.current_edit.setText(f"{pos:.3f}" if pos is not None else "N/A")
            self.log(f"Moved to {target_angle:.2f} (angle).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to move stage: {e}")
            self.log(f"Move failed: {e}")


class ThorlabsView(QWidget):
    """
    A GUI to control up to 10 Thorlabs stages.
    Contains several groups of StageRows and a common log area.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thorlabs Stage Control")
        self.stage_rows = []
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)

        # Left panel: Stage groups.
        left_layout = QVBoxLayout()

        # Group 1: "w/(2w,3w)" – Stage 1.
        group1 = QGroupBox("w/(2w,3w)")
        g1_layout = QVBoxLayout()
        activate_all_g1 = QPushButton("Activate All in Group")
        activate_all_g1.clicked.connect(lambda: self.activate_group([0]))
        g1_layout.addWidget(activate_all_g1)
        row1 = StageRow(0, description="w/(2w,3w)", log_callback=self.log_message)
        self.stage_rows.append(row1)
        g1_layout.addWidget(row1)
        group1.setLayout(g1_layout)
        left_layout.addWidget(group1)

        # Group 2: "Individual Attenuation" – Stages 2,3,4.
        group2 = QGroupBox("Individual Attenuation")
        g2_layout = QVBoxLayout()
        activate_all_g2 = QPushButton("Activate All in Group")
        activate_all_g2.clicked.connect(lambda: self.activate_group([1, 2, 3]))
        g2_layout.addWidget(activate_all_g2)
        row2 = StageRow(1, description="w", log_callback=self.log_message)
        row3 = StageRow(2, description="2w", log_callback=self.log_message)
        row4 = StageRow(3, description="3w", log_callback=self.log_message)
        self.stage_rows.extend([row2, row3, row4])
        g2_layout.addWidget(row2)
        g2_layout.addWidget(row3)
        g2_layout.addWidget(row4)
        group2.setLayout(g2_layout)
        left_layout.addWidget(group2)

        # Group 3: "SFG" – Stages 5,6,7.
        group3 = QGroupBox("SFG")
        g3_layout = QVBoxLayout()
        activate_all_g3 = QPushButton("Activate All in Group")
        activate_all_g3.clicked.connect(lambda: self.activate_group([4, 5, 6]))
        g3_layout.addWidget(activate_all_g3)
        row5 = StageRow(4, description="w", log_callback=self.log_message)
        row6 = StageRow(5, description="2w", log_callback=self.log_message)
        row7 = StageRow(6, description="Overlap", log_callback=self.log_message)
        self.stage_rows.extend([row5, row6, row7])
        g3_layout.addWidget(row5)
        g3_layout.addWidget(row6)
        g3_layout.addWidget(row7)
        group3.setLayout(g3_layout)
        left_layout.addWidget(group3)

        # Group 4: "Focus" – Stage 8.
        group4 = QGroupBox("Focus")
        g4_layout = QVBoxLayout()
        activate_all_g4 = QPushButton("Activate All in Group")
        activate_all_g4.clicked.connect(lambda: self.activate_group([7]))
        g4_layout.addWidget(activate_all_g4)
        row8 = StageRow(7, description="Focus", log_callback=self.log_message)
        self.stage_rows.append(row8)
        g4_layout.addWidget(row8)
        group4.setLayout(g4_layout)
        left_layout.addWidget(group4)

        left_layout.addStretch(1)

        # Right panel: Unique log area.
        right_layout = QVBoxLayout()
        log_label = QLabel("Log:")
        right_layout.addWidget(log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text)

        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)

    def activate_group(self, indices):
        for idx in indices:
            row = self.stage_rows[idx]
            try:
                row.activate_stage()
                if row.controller is not None:
                    row.stage_label.setStyleSheet("background-color: lightgreen;")
                    self.log_message(f"Stage {row.stage_number + 1} activated successfully.")
                else:
                    row.stage_label.setStyleSheet("")
            except Exception as e:
                self.log_message(f"Stage {row.stage_number + 1} activation error: {e}")
                row.stage_label.setStyleSheet("background-color: lightcoral;")

    def log_message(self, message: str):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{current_time}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def update_stage_calibration(self, wp_index, calibration):
        """
        Update calibration parameters for a single StageRow corresponding to a waveplate.
        wp_index: waveplate index (1-indexed)
        calibration: tuple (amplitude, offset)
        """
        for row in self.stage_rows:
            # Only update rows representing waveplates (stage_number 0 to 5)
            if row.stage_number < NUM_WAVEPLATES and (row.stage_number + 1) == wp_index:
                row.amplitude, row.offset = calibration
                self.log_message(
                    f"Updated calibration for Stage {row.stage_number + 1}: amplitude={calibration[0]:.2f}, offset={calibration[1]:.2f}")


# ----------------------------
# Main Window with Tabbed Interface
# ----------------------------

class StageControl(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stage Control")
        # Create a tab widget as central widget.
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create the ThorlabsView (first tab).
        self.thorlabs_view = ThorlabsView()
        self.tabs.addTab(self.thorlabs_view, "Stage Control")

        # Create the WaveplateCalibWidget (second tab).
        # Pass the ThorlabsView.log_message as the log callback.
        # Also, supply a calibration_changed_callback that updates only the relevant stage.
        self.calib_widget = WaveplateCalibWidget(
            log_callback=self.thorlabs_view.log_message,
            calibration_changed_callback=self.update_stage_calibrations
        )
        self.tabs.addTab(self.calib_widget, "Waveplate Calibration")

        # Create the AutoWavepalteCalib
        self.autocalib_view = AutoWaveplateCalib()
        self.tabs.addTab(self.autocalib_view, "Automatic Waveplate Calibration")

    def update_stage_calibrations(self, wp_index, calibration):
        """
        Called when calibration changes for a specific waveplate.
        Propagates calibration parameters from the calibration widget to the matching stage row.
        """
        self.thorlabs_view.update_stage_calibration(wp_index, calibration)


# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StageControl()
    window.show()
    sys.exit(app.exec_())
