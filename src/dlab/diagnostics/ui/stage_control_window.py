from __future__ import annotations

import datetime
import numpy as np

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QTextEdit, QTabWidget, QMainWindow, QCheckBox, QMessageBox
)
from PyQt5.QtGui import QIntValidator

from dlab.boot import get_config
from dlab.hardware.wrappers.thorlabs_controller import ThorlabsController
from dlab.hardware.wrappers.waveplate_calib import WaveplateCalibWidget, NUM_WAVEPLATES
from dlab.diagnostics.ui.auto_waveplate_calib_window import AutoWaveplateCalibWindow
from dlab.diagnostics.ui.grating_compressor import GratingCompressorWindow
from dlab.core.device_registry import REGISTRY

import logging
logger = logging.getLogger("dlab.ui.StageControlWindow")


def _wp_index_from_stage_number(stage_number: int) -> int:
    return stage_number + 1

def _reg_key_powermode(wp_index: int) -> str:
    return f"waveplate:powermode:{wp_index}"  # bool

def _reg_key_calib(wp_index: int) -> str:
    return f"waveplate:calib:{wp_index}"      # (1.0, phase_deg)

def power_to_angle(power_fraction: float, _amp_unused: float, phase_deg: float) -> float:
    y = float(np.clip(power_fraction, 0.0, 1.0))
    return (phase_deg + (45.0 / np.pi) * float(np.arccos(2.0 * y - 1.0))) % 360.0

def load_default_ids() -> dict[int, str]:
    cfg = get_config() or {}
    # We now have 7 waveplates (0..6) + 3 translation stages (7..9) = 10 total.
    defaults = {i: "00000000" for i in range(10)}
    try:
        y = (cfg.get("stages", {}) or {}).get("default_ids", {}) or {}
        for k, v in y.items():
            try:
                defaults[int(k)] = str(v)
            except Exception:
                pass
    except Exception:
        pass
    return defaults



class StageRow(QWidget):
    def __init__(self, stage_number: int, description: str = "", log_callback=None, parent: QWidget = None):
        super().__init__(parent)
        self.stage_number = stage_number
        self.is_waveplate = (stage_number < NUM_WAVEPLATES)
        self.controller: ThorlabsController | None = None
        self.log_callback = log_callback

        self._poll = QTimer(self); self._poll.setInterval(200); self._poll.timeout.connect(self._update_position)

        # phase only; amplitude fixed = 1.0 (fraction)
        self.amplitude = 1.0
        self.offset = 0.0

        layout = QHBoxLayout(self); layout.setSpacing(5)

        title = f"Waveplate {stage_number+1}:" if self.is_waveplate else f"Stage {stage_number+1}:"
        self.stage_label = QLabel(title); self.stage_label.setFixedWidth(120); layout.addWidget(self.stage_label)
        self.desc_label = QLabel(f"{description}"); self.desc_label.setFixedWidth(120); layout.addWidget(self.desc_label)

        self.motor_id_edit = QLineEdit(); self.motor_id_edit.setFixedWidth(90)
        self.motor_id_edit.setPlaceholderText("Motor ID")
        self.motor_id_edit.setValidator(QIntValidator(0, 99999999, self))
        defaults = load_default_ids()
        motor_id = defaults.get(stage_number, "00000000")
        self.motor_id_edit.setText(motor_id)
        layout.addWidget(self.motor_id_edit)

        self.activate_btn = QPushButton("Activate"); self.activate_btn.clicked.connect(self.activate_stage); layout.addWidget(self.activate_btn)
        self.home_on_activate_checkbox = QCheckBox("Home on Activate"); self.home_on_activate_checkbox.setChecked(False); layout.addWidget(self.home_on_activate_checkbox)
        self.home_btn = QPushButton("Home"); self.home_btn.clicked.connect(self.home_stage); self.home_btn.setEnabled(False); layout.addWidget(self.home_btn)
        self.ident_btn = QPushButton("Identify"); self.ident_btn.clicked.connect(self.identify_stage); self.ident_btn.setEnabled(False); layout.addWidget(self.ident_btn)

        self.target_edit = QLineEdit(); self.target_edit.setFixedWidth(100); layout.addWidget(self.target_edit)

        self.power_mode_checkbox = QCheckBox("Power Mode")
        if self.is_waveplate:
            wp_idx = _wp_index_from_stage_number(self.stage_number)
            pm = REGISTRY.get(_reg_key_powermode(wp_idx))
            if isinstance(pm, bool):
                self.power_mode_checkbox.setChecked(pm)
            self.power_mode_checkbox.toggled.connect(lambda chk, wpi=wp_idx: REGISTRY.register(_reg_key_powermode(wpi), bool(chk)))
        else:
            self.power_mode_checkbox.setChecked(False); self.power_mode_checkbox.setEnabled(False); self.power_mode_checkbox.setVisible(False)
        layout.addWidget(self.power_mode_checkbox)

        self.move_btn = QPushButton("Move To"); self.move_btn.clicked.connect(self.move_stage); self.move_btn.setEnabled(False); layout.addWidget(self.move_btn)

        self.current_edit = QLineEdit(); self.current_edit.setPlaceholderText("Current"); self.current_edit.setFixedWidth(100); self.current_edit.setReadOnly(True); layout.addWidget(self.current_edit)
        layout.addStretch(1)

        self._refresh_target_placeholder()
        self.power_mode_checkbox.toggled.connect(lambda _=None: self._refresh_target_placeholder())

    def log(self, message: str) -> None:
        full_msg = f"Stage {self.stage_number+1}: {message}"
        if self.log_callback:
            self.log_callback(full_msg)
        logger.info(full_msg)

    def _update_position(self) -> None:
        if not self.controller:
            self._poll.stop(); return
        try:
            pos = self.controller.get_position()
            if pos is not None:
                self.current_edit.setText(f"{pos:.3f}")
        except Exception as e:
            self._poll.stop(); self.log(f"Position read failed: {e}")

    def _refresh_target_placeholder(self) -> None:
        if not self.is_waveplate:
            self.target_edit.setPlaceholderText("Position")
            self.target_edit.setToolTip("")
            return
        if self.power_mode_checkbox.isChecked():
            self.target_edit.setPlaceholderText("Power fraction (0..1)")
            self.target_edit.setToolTip("Enter fraction of max power; converted to angle via calibration.")
        else:
            self.target_edit.setPlaceholderText("Angle (deg)")
            self.target_edit.setToolTip("Enter target angle in degrees.")

    def activate_stage(self) -> None:
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

            if self.is_waveplate:
                wp_idx = _wp_index_from_stage_number(self.stage_number)
                calib_ui = REGISTRY.get("ui:waveplate_calib_widget")
                if calib_ui and hasattr(calib_ui, "load_waveplate_calibration"):
                    ok = calib_ui.load_waveplate_calibration(wp_idx)
                    if ok:
                        calib = REGISTRY.get(_reg_key_calib(wp_idx))
                        if isinstance(calib, (tuple, list)) and len(calib) >= 2:
                            self.amplitude, self.offset = float(calib[0]), float(calib[1])
                            self.log(f"WP{wp_idx} calibration loaded (phase={self.offset:.2f}°).")
                    else:
                        self.log(f"No calibration found for WP{wp_idx}.")

            self._poll.start()
            self._refresh_target_placeholder()
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
                requested_frac = float(t)
                frac = float(np.clip(requested_frac, 0.0, 1.0))
                if abs(frac - requested_frac) > 1e-12:
                    self.target_edit.blockSignals(True)
                    self.target_edit.setText(f"{frac:.3f}")
                    self.target_edit.blockSignals(False)

                wp_idx = _wp_index_from_stage_number(self.stage_number)
                phase = float(self.offset)

                # prefer user-set max_value; fallback to info value from file
                mv = REGISTRY.get(f"waveplate:max_value:{wp_idx}")
                if mv is None:
                    mv = REGISTRY.get(f"waveplate:max:{wp_idx}")

                angle_deg = power_to_angle(frac, 1.0, phase)
                self.controller.move_to(angle_deg, blocking=False)

                if mv is not None and np.isfinite(float(mv)):
                    self.log(f"Moving to {angle_deg:.3f}° (fraction={frac:.3f}, ~{frac*float(mv):.3g} W)…")
                else:
                    self.log(f"Moving to {angle_deg:.3f}° (fraction={frac:.3f})…")

                self._poll.start()
                return

            # default: direct angle/position
            value = float(t)
            if self.is_waveplate:
                value = value % 360.0
            self.controller.move_to(value, blocking=False)
            self.log(f"Moving to {value:.3f}{'°' if self.is_waveplate else ''} …")
            self._poll.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to move stage: {e}")
            self.log(f"Move failed: {e}")


class ThorlabsView(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thorlabs Stage Control")
        self.stage_rows = []
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)

        # Left panel
        left_layout = QVBoxLayout()

        # Group 1
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

        # Group 2
        group2 = QGroupBox("w, 2w, 3w attenuation")
        g2_layout = QVBoxLayout()
        activate_all_g2 = QPushButton("Activate All in Group")
        activate_all_g2.clicked.connect(lambda: self.activate_group([1, 2, 3]))
        g2_layout.addWidget(activate_all_g2)
        row2 = StageRow(1, description="w", log_callback=self.log_message)
        row3 = StageRow(2, description="2w", log_callback=self.log_message)
        row4 = StageRow(3, description="3w", log_callback=self.log_message)
        self.stage_rows.extend([row2, row3, row4])
        g2_layout.addWidget(row2); g2_layout.addWidget(row3); g2_layout.addWidget(row4)
        group2.setLayout(g2_layout)
        left_layout.addWidget(group2)

        # Group 3
        group3 = QGroupBox("MPC")
        g3_layout = QVBoxLayout()
        activate_all_g3 = QPushButton("Activate All in Group")
        activate_all_g3.clicked.connect(lambda: self.activate_group([4, 5]))
        g3_layout.addWidget(activate_all_g3)
        row5 = StageRow(4, description="Before MPC", log_callback=self.log_message)
        row6 = StageRow(5, description="After MPC", log_callback=self.log_message)
        self.stage_rows.extend([row5, row6])
        g3_layout.addWidget(row5); g3_layout.addWidget(row6)
        group3.setLayout(g3_layout)
        left_layout.addWidget(group3)
        
        # Group 4: SFG c
        group4 = QGroupBox("SFG")
        g4_layout = QVBoxLayout()
        activate_all_g4 = QPushButton("Activate All in Group")
        # The next row we append will be index 6
        activate_all_g4.clicked.connect(lambda: self.activate_group([6]))
        g4_layout.addWidget(activate_all_g4)
        # Waveplate 7 (0-based stage_number=6), named "3w/2w"
        row7_wp = StageRow(6, description="3w/2w", log_callback=self.log_message)
        self.stage_rows.append(row7_wp)
        g4_layout.addWidget(row7_wp)
        group4.setLayout(g4_layout)
        left_layout.addWidget(group4)

        # --- Group 5: Translation Stages (stages 7,8,9) ---------------------
        group5 = QGroupBox("Translation Stages")
        g5_layout = QVBoxLayout()
        activate_all_g5 = QPushButton("Activate All in Group")
        # The next three rows we append will be indices 7,8,9
        activate_all_g5.clicked.connect(lambda: self.activate_group([7, 8, 9]))
        g5_layout.addWidget(activate_all_g5)

        row8 = StageRow(7, description="Focus",   log_callback=self.log_message)
        row9 = StageRow(8, description="Delay 1", log_callback=self.log_message)
        row10 = StageRow(9, description="Delay 2", log_callback=self.log_message)
        self.stage_rows.extend([row8, row9, row10])
        g5_layout.addWidget(row8); g5_layout.addWidget(row9); g5_layout.addWidget(row10)
        group5.setLayout(g5_layout)
        left_layout.addWidget(group5)

        left_layout.addStretch(1)

        # Right panel: log
        right_layout = QVBoxLayout()
        log_label = QLabel("Log:")
        right_layout.addWidget(log_label)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
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
        logger.info(message)

    def update_stage_calibration(self, wp_index, calibration):
        for row in self.stage_rows:
            if row.stage_number < NUM_WAVEPLATES and (row.stage_number + 1) == wp_index:
                row.amplitude, row.offset = calibration
                REGISTRY.register(_reg_key_calib(wp_index), (float(calibration[0]), float(calibration[1])))
                self.log_message(f"Updated calibration for Stage {row.stage_number + 1}: phase={calibration[1]:.2f}°")


class StageControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stage Control")
        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)

        self.thorlabs_view = ThorlabsView()
        self.tabs.addTab(self.thorlabs_view, "Thorlabs Control")

        self.gc_view = GratingCompressorWindow()
        self.tabs.addTab(self.gc_view, "Grating Compressor")

        self.calib_widget = WaveplateCalibWidget(
            log_callback=self.thorlabs_view.log_message,
            calibration_changed_callback=self.update_stage_calibrations
        )
        self.tabs.addTab(self.calib_widget, "Waveplate Calibration")
        REGISTRY.register("ui:waveplate_calib_widget", self.calib_widget)

        self.autocalib_view = AutoWaveplateCalibWindow()
        self.tabs.addTab(self.autocalib_view, "Automatic Waveplate Calibration")

    def update_stage_calibrations(self, wp_index, calibration):
        self.thorlabs_view.update_stage_calibration(wp_index, calibration)
