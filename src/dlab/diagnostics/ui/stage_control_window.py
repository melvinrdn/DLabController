from PyQt5.QtCore import QTimer
from dlab.boot import get_config

import sys
import datetime
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QTextEdit, QTabWidget, QMainWindow, QCheckBox, QMessageBox
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from dlab.hardware.wrappers.thorlabs_controller import ThorlabsController
from dlab.hardware.wrappers.waveplate_calib import WaveplateCalibWidget, NUM_WAVEPLATES
from dlab.diagnostics.ui.auto_waveplate_calib_window import AutoWaveplateCalibWindow
from dlab.diagnostics.ui.grating_compressor import GratingCompressorWindow
from dlab.core.device_registry import REGISTRY

import logging
logger = logging.getLogger("dlab.ui.StageControlWindow")


def _wp_index_from_stage_number(stage_number: int) -> int:
    # StageRow.stage_number is 0-based for waveplates. Convert to 1-based index.
    return stage_number + 1

def _reg_key_powermode(wp_index: int) -> str:
    return f"waveplate:powermode:{wp_index}"  # bool

def _reg_key_calib(wp_index: int) -> str:
    return f"waveplate:calib:{wp_index}"      # (amplitude, offset)

def load_default_ids() -> dict[int, str]:
    """
    Returns a mapping {row_index -> motor_id_string} from YAML.
    Fallbacks to your current defaults if missing.
    Expected YAML:
      stages:
        default_ids:
          0: "83837714"
          1: "83837719"
          6: "83838295"
          7: "83837725"
    """
    cfg = get_config() or {}
    defaults = {
        0: "00000000", 
        1: "00000000",  
        2: "00000000",  
        3: "00000000",  
        4: "00000000",  
        5: "00000000",  
        6: "00000000",  
        7: "00000000", 
        8: "00000000", 
    }
    try:
        y = (cfg.get("stages", {}) or {}).get("default_ids", {}) or {}
        for k, v in y.items():
            try:
                ki = int(k)
                defaults[ki] = str(v)
            except Exception:
                pass
    except Exception:
        pass
    return defaults


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
    One stage row with: ID, Activate, Home, target (angle/power or position), Move, Current position.
    Waveplates (index < NUM_WAVEPLATES) keep 'Power Mode'; translation stages have it disabled.
    """
    def __init__(self, stage_number: int, description: str = "",
                 log_callback=None, parent: QWidget = None):
        super().__init__(parent)
        self.stage_number = stage_number
        self.is_waveplate = (stage_number < NUM_WAVEPLATES)
        self.controller: ThorlabsController | None = None
        self.log_callback = log_callback

        self._poll = QTimer(self)
        self._poll.setInterval(200)  # ms
        self._poll.timeout.connect(self._update_position)

        # power→angle fit params (only used for waveplates)
        self.amplitude = 100.0
        self.offset = 0.0

        layout = QHBoxLayout(self)
        layout.setSpacing(5)

        # Label
        title = f"Waveplate {stage_number+1}:" if self.is_waveplate else f"Stage {stage_number+1}:"
        self.stage_label = QLabel(title); self.stage_label.setFixedWidth(120)
        layout.addWidget(self.stage_label)

        self.desc_label = QLabel(f"{description}"); self.desc_label.setFixedWidth(120)
        layout.addWidget(self.desc_label)

        # Motor ID
        self.motor_id_edit = QLineEdit()
        defaults = load_default_ids()
        motor_id = defaults.get(stage_number, "00000000")
        self.motor_id_edit.setText(motor_id)
        self.motor_id_edit.setPlaceholderText("Motor ID")
        self.motor_id_edit.setFixedWidth(90)
        self.motor_id_edit.setValidator(QIntValidator(0, 99999999, self))
        layout.addWidget(self.motor_id_edit)

        # Activate / Home-on-activate
        self.activate_btn = QPushButton("Activate")
        self.activate_btn.clicked.connect(self.activate_stage)
        layout.addWidget(self.activate_btn)

        self.home_on_activate_checkbox = QCheckBox("Home on Activate")
        self.home_on_activate_checkbox.setChecked(False)
        layout.addWidget(self.home_on_activate_checkbox)

        self.home_btn = QPushButton("Home")
        self.home_btn.clicked.connect(self.home_stage)
        self.home_btn.setEnabled(False)
        layout.addWidget(self.home_btn)

        # Identify
        self.ident_btn = QPushButton("Identify")
        self.ident_btn.clicked.connect(self.identify_stage)
        self.ident_btn.setEnabled(False)
        layout.addWidget(self.ident_btn)

        # Target input
        self.target_edit = QLineEdit()
        ph = "Angle (deg)" if self.is_waveplate else "Position"
        self.target_edit.setPlaceholderText(ph)
        self.target_edit.setFixedWidth(100)
        self.target_edit.setValidator(QDoubleValidator(-1e6, 1e6, 3, self))
        layout.addWidget(self.target_edit)

        # Power mode (waveplates only)
        self.power_mode_checkbox = QCheckBox("Power Mode")
        if self.is_waveplate:
            # Load previous state if any
            wp_idx = _wp_index_from_stage_number(self.stage_number)
            pm = REGISTRY.get(_reg_key_powermode(wp_idx))
            if isinstance(pm, bool):
                self.power_mode_checkbox.setChecked(pm)

            # When toggled here, publish to REGISTRY
            def _on_pm_toggled(checked: bool):
                REGISTRY.register(_reg_key_powermode(wp_idx), bool(checked))
                # nothing else to do; GridScan will see it

            self.power_mode_checkbox.toggled.connect(_on_pm_toggled)
        if not self.is_waveplate:
            # Translation stages: disable & hide power mode
            self.power_mode_checkbox.setChecked(False)
            self.power_mode_checkbox.setEnabled(False)
            self.power_mode_checkbox.setVisible(False)
            self.power_mode_checkbox.setToolTip("Disabled for translation stages")
        layout.addWidget(self.power_mode_checkbox)

        # Move
        self.move_btn = QPushButton("Move To")
        self.move_btn.clicked.connect(self.move_stage)
        self.move_btn.setEnabled(False)
        layout.addWidget(self.move_btn)

        # Current pos (read-only)
        self.current_edit = QLineEdit()
        self.current_edit.setPlaceholderText("Current")
        self.current_edit.setFixedWidth(100)
        self.current_edit.setReadOnly(True)
        layout.addWidget(self.current_edit)

        layout.addStretch(1)

    def log(self, message: str) -> None:
        full_msg = f"Stage {self.stage_number+1}: {message}"
        if self.log_callback:
            self.log_callback(full_msg)
        logger.info(full_msg)

    def _update_position(self) -> None:
        if not self.controller:
            self._poll.stop()
            return
        try:
            pos = self.controller.get_position()
            if pos is not None:
                self.current_edit.setText(f"{pos:.3f}")
        except Exception as e:
            self._poll.stop()
            self.log(f"Position read failed: {e}")

    def activate_stage(self) -> None:
        from dlab.core.device_registry import REGISTRY
        
        txt = self.motor_id_edit.text().strip()
        if not txt:
            QMessageBox.warning(self, "Error", "Please enter a motor ID.")
            return
        try:
            motor_id = int(txt)
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid motor ID.")
            return
        try:
            self.controller = ThorlabsController(motor_id)
            self.controller.activate(homing=self.home_on_activate_checkbox.isChecked())
            
            self.activate_btn.setEnabled(False)
            self.motor_id_edit.setEnabled(False)
            self.home_btn.setEnabled(True)
            self.ident_btn.setEnabled(True)
            self.move_btn.setEnabled(True)
            self.stage_label.setStyleSheet("background-color: lightgreen;")
            self.log("Activated.")
            
            key_ui = f"stage:{self.stage_number+1}"
            key_ser = f"stage:serial:{motor_id}"
            
            for k in (key_ui, key_ser):
                prev = REGISTRY.get(k)
                if prev and prev is not self.controller:
                    self.log(f"Registry key '{k}' already in use. Replacing.")
                    
            REGISTRY.register(key_ui,  self.controller)
            REGISTRY.register(key_ser, self.controller)
            
            self._poll.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate stage: {e}")
            self.log(f"Activation failed: {e}")
            self.controller = None
            self.stage_label.setStyleSheet("")
            self._poll.stop()

    def home_stage(self) -> None:
        if not self.controller:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return
        try:
            self.controller.home(blocking=False)
            self.log("Homing…")
            self._poll.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to home stage: {e}")
            self.log(f"Home failed: {e}")

    def identify_stage(self) -> None:
        if not self.controller:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return
        try:
            self.controller.identify()
            self.log("Identify blink.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Identify failed: {e}")
            self.log(f"Identify failed: {e}")

    def move_stage(self) -> None:
        if not self.controller:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return
        t = self.target_edit.text().strip()
        if not t:
            QMessageBox.warning(self, "Error", "Please enter a target value.")
            return
        try:
            if self.is_waveplate and self.power_mode_checkbox.isChecked():
                # power→angle conversion
                desired_power = float(t)
                if desired_power > self.amplitude:
                    self.log(f"Desired power {desired_power} exceeds max {self.amplitude:.2f}; capping.")
                    desired_power = self.amplitude
                value = power_to_angle(desired_power, self.amplitude, self.offset)
                # wrap angles for waveplates
                value = value % 360.0
                units = "°"
            else:
                # direct move: angle for waveplates, linear position for translation stages
                value = float(t)
                if self.is_waveplate:
                    value = value % 360.0
                    units = "°"
                else:
                    units = ""  # mm or native units depending on stage; keep generic

            self.controller.move_to(value, blocking=False)
            self.log(f"Moving to {value:.3f}{units} …")
            self._poll.start()
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

        # Group 1: "w vs (2w,3w) mixing" – Stage 1.
        group1 = QGroupBox("w vs (2w,3w) mixing")
        g1_layout = QVBoxLayout()
        activate_all_g1 = QPushButton("Activate All in Group")
        activate_all_g1.clicked.connect(lambda: self.activate_group([0]))
        g1_layout.addWidget(activate_all_g1)
        row1 = StageRow(0, description="w/(2w,3w)", log_callback=self.log_message)
        self.stage_rows.append(row1)
        g1_layout.addWidget(row1)
        group1.setLayout(g1_layout)
        left_layout.addWidget(group1)

        # Group 2: "w, 2w, 3w attenuation" – Stages 2,3,4.
        group2 = QGroupBox("w, 2w, 3w attenuation")
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

        # Group 3: "MPC" – Stages 5,6
        group3 = QGroupBox("MPC")
        g3_layout = QVBoxLayout()
        activate_all_g3 = QPushButton("Activate All in Group")
        activate_all_g3.clicked.connect(lambda: self.activate_group([4, 5]))
        g3_layout.addWidget(activate_all_g3)
        row5 = StageRow(4, description="Before MPC", log_callback=self.log_message)
        row6 = StageRow(5, description="After MPC", log_callback=self.log_message)
        self.stage_rows.extend([row5, row6])
        g3_layout.addWidget(row5)
        g3_layout.addWidget(row6)
        group3.setLayout(g3_layout)
        left_layout.addWidget(group3)

        # Group 4: "Translation Stages" – Stage 7.
        group4 = QGroupBox("Translation Stages")
        g4_layout = QVBoxLayout()
        activate_all_g4 = QPushButton("Activate All in Group")
        activate_all_g4.clicked.connect(lambda: self.activate_group([6, 7, 8]))
        g4_layout.addWidget(activate_all_g4)
        row7 = StageRow(6, description="Focus", log_callback=self.log_message)
        row8 = StageRow(7, description="Delay 1", log_callback=self.log_message)
        row9 = StageRow(8, description="Delay 2", log_callback=self.log_message)
        self.stage_rows.extend([row7, row8, row9])
        g4_layout.addWidget(row7)
        g4_layout.addWidget(row8)
        g4_layout.addWidget(row9)
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
        self.log_text.append(f"[{current_time}] {message}")  # GUI
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        logger.info(message)
        
    def update_stage_calibration(self, wp_index, calibration):
        """
        Update calibration parameters for a single StageRow corresponding to a waveplate.
        Also publish to REGISTRY so GridScan can use it.
        """
        for row in self.stage_rows:
            if row.stage_number < NUM_WAVEPLATES and (row.stage_number + 1) == wp_index:
                row.amplitude, row.offset = calibration
                # Publish (amplitude, offset)
                REGISTRY.register(_reg_key_calib(wp_index), (float(calibration[0]), float(calibration[1])))
                self.log_message(
                    f"Updated calibration for Stage {row.stage_number + 1}: amplitude={calibration[0]:.2f}, offset={calibration[1]:.2f}"
                )


# ----------------------------
# Main Window with Tabbed Interface
# ----------------------------

class StageControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stage Control")
        # Create a tab widget as central widget.
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create the ThorlabsView (first tab).
        self.thorlabs_view = ThorlabsView()
        self.tabs.addTab(self.thorlabs_view, "Thorlabs Control")
        
        self.gc_view = GratingCompressorWindow()
        self.tabs.addTab(self.gc_view, "Grating Compressor")

        self.calib_widget = WaveplateCalibWidget(
            log_callback=self.thorlabs_view.log_message,
            calibration_changed_callback=self.update_stage_calibrations
        )
        self.tabs.addTab(self.calib_widget, "Waveplate Calibration")

        # Create the AutoWavepalteCalib
        self.autocalib_view = AutoWaveplateCalibWindow()
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
    window = StageControlWindow()
    window.show()
    sys.exit(app.exec_())
