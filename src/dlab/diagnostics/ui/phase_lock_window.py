from __future__ import annotations

import time
from collections import deque

import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QMessageBox,
    QCheckBox,
    QGroupBox,
)
from PyQt5.QtGui import QDoubleValidator

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from dlab.core.device_registry import REGISTRY
from dlab.utils.log_panel import LogPanel


from dlab.hardware.wrappers.piezojena_controller import NV40

REGISTRY_KEY_DEFAULT = "phaselock:avaspec"


# -----------------------------------------------------------------------------
# Worker Threads
# -----------------------------------------------------------------------------


class ControlThread(QThread):
    """PID control thread for phase locking via piezo stage."""

    update_status = pyqtSignal(float, float)

    def __init__(self, stage: NV40) -> None:
        super().__init__()
        self.stage = stage
        self.kp = 0.05
        self.ki = 0.0
        self.kd = 0.0
        self.gain = -1.0
        self.max_step = 0.05
        self.target = 0.0
        self.unwrap = True
        self.integral = 0.0
        self.last_t = None
        self.last_err = 0.0
        self.q = deque()
        self.vmin = 0.0
        self.vmax = 140.0
        try:
            self.current_v = float(self.stage.get_position())
        except Exception:
            self.current_v = 80
        self.enabled = False

    def run(self) -> None:
        while True:
            if not self.q:
                time.sleep(0.001)
                continue

            phi = self.q.popleft()

            if not self.enabled:
                continue

            now = time.monotonic()
            dt = 0 if self.last_t is None else max(0.0005, now - self.last_t)
            self.last_t = now

            if self.unwrap:
                if not hasattr(self, "_phi_prev"):
                    self._phi_prev = phi
                phi_use = np.unwrap([self._phi_prev, phi])[-1]
                self._phi_prev = phi_use
                err = self.target - phi_use
            else:
                phi_use = phi
                err = (self.target - phi + np.pi) % (2 * np.pi) - np.pi

            d_err = (err - self.last_err) / dt
            self.last_err = err

            self.integral += err * dt
            max_int = 1.0 / (self.ki + 1e-9)
            self.integral = np.clip(self.integral, -max_int, max_int)

            gain = getattr(self, "gain", -1.0)
            u = gain * (self.kp * err + self.ki * self.integral + self.kd * d_err)

            if abs(u) > self.max_step:
                u = np.sign(u) * self.max_step

            new_v = self.current_v + u
            new_v = max(self.vmin, min(self.vmax, new_v))
            new_v = round(new_v / 0.01) * 0.01

            try:
                self.stage.set_position(new_v)
                self.current_v = new_v
            except Exception:
                pass

            self.update_status.emit(phi_use, self.current_v)


class AvaspecThread(QThread):
    """Spectrometer acquisition thread."""

    data_ready = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, ctrl) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.running = True

    def run(self) -> None:
        while self.running and not self.isInterruptionRequested():
            try:
                ts, wl, counts = self.ctrl.measure_once()
                self.data_ready.emit(wl, counts)
            except Exception as e:
                self.error.emit(str(e))
                break

    def stop(self) -> None:
        self.running = False
        self.requestInterruption()


# -----------------------------------------------------------------------------
# AvaspecPhaseLockWindow
# -----------------------------------------------------------------------------


class AvaspecPhaseLockWindow(QWidget):
    """Phase locking control window using Avaspec spectrometer and PiezoJena stage."""

    closed = pyqtSignal()

    def __init__(
        self,
        log_panel: LogPanel | None = None,
        registry_key: str = REGISTRY_KEY_DEFAULT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Phase Locking Control")
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._log = log_panel
        self._registry_key = registry_key
        self._spec_ctrl = None
        self._stage = None
        self._acq_thread = None
        self._ctrl_thread = None

        # Plot state
        self._max_points = 100
        self._hist_phi_raw = deque(maxlen=self._max_points)
        self._hist_phi_unwrapped = deque(maxlen=self._max_points)
        self._last_draw = 0.0
        self._min_draw_dt = 0.05
        self._fft_scatter = None
        self._fft_line = None
        self._phase_plot = None
        self._phase_sp_line = None
        self._bg_fft = None
        self._bg_phase = None
        self._blitting_initialized = False

        # Stability test state
        self._stability_test_active = False
        self._stability_timer = None
        self._stability_step = 0
        self._stability_setpoints = []

        self._init_ui()
        REGISTRY.register(self._registry_key, self)

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)

        # Matplotlib figure
        self._fig, (self._ax_fft, self._ax_phase) = plt.subplots(1, 2, figsize=(14, 4.5))
        self._fig.tight_layout(pad=2.0)
        self._canvas = FigureCanvas(self._fig)
        root.addWidget(self._canvas, 3)

        middle_row = QHBoxLayout()
        left_panel = QVBoxLayout()

        # Device connection group
        left_panel.addWidget(self._create_connection_group())

        # FFT analysis group
        left_panel.addWidget(self._create_fft_group())

        # PID control group
        left_panel.addWidget(self._create_pid_group())

        # Voltage limits group
        left_panel.addWidget(self._create_voltage_group())

        # Stability test group
        left_panel.addWidget(self._create_stability_group())

        # Control row
        left_panel.addLayout(self._create_control_row())

        middle_row.addLayout(left_panel, 1)
        root.addLayout(middle_row, 2)

        # Stability timer
        self._stability_timer = QTimer()
        self._stability_timer.timeout.connect(self._on_stability_step)

    # -------------------------------------------------------------------------
    # UI builders
    # -------------------------------------------------------------------------

    def _create_connection_group(self) -> QGroupBox:
        group = QGroupBox("Devices")
        layout = QGridLayout(group)

        layout.addWidget(QLabel("Spectrometer:"), 0, 0)
        self._spec_key_edit = QLineEdit("spectrometer:avaspec:spec_1")
        self._spec_key_edit.setMaximumWidth(200)
        layout.addWidget(self._spec_key_edit, 0, 1)
        btn_spec = QPushButton("Connect")
        btn_spec.setMaximumWidth(80)
        btn_spec.clicked.connect(self._on_connect_spec)
        layout.addWidget(btn_spec, 0, 2)

        layout.addWidget(QLabel("NV40 Stage:"), 1, 0)
        self._stage_key_edit = QLineEdit("stage:piezojena:nv40")
        self._stage_key_edit.setMaximumWidth(200)
        layout.addWidget(self._stage_key_edit, 1, 1)
        btn_stage = QPushButton("Connect")
        btn_stage.setMaximumWidth(80)
        btn_stage.clicked.connect(self._on_connect_stage)
        layout.addWidget(btn_stage, 1, 2)

        return group

    def _create_fft_group(self) -> QGroupBox:
        group = QGroupBox("FFT Analysis")
        layout = QGridLayout(group)
        layout.setSpacing(5)

        layout.addWidget(QLabel("Center:"), 0, 0)
        self._center_idx_edit = QLineEdit("600")
        self._center_idx_edit.setMaximumWidth(60)
        layout.addWidget(self._center_idx_edit, 0, 1)

        layout.addWidget(QLabel("Window:"), 0, 2)
        self._window_edit = QLineEdit("150")
        self._window_edit.setMaximumWidth(60)
        layout.addWidget(self._window_edit, 0, 3)

        layout.addWidget(QLabel("Plot Every:"), 0, 4)
        self._plot_skip_edit = QLineEdit("5")
        self._plot_skip_edit.setMaximumWidth(40)
        layout.addWidget(self._plot_skip_edit, 0, 5)

        self._weighted_checkbox = QCheckBox("Weighted Avg")
        layout.addWidget(self._weighted_checkbox, 0, 6)

        self._remove_ramp_checkbox = QCheckBox("Remove Ramp")
        self._remove_ramp_checkbox.setChecked(False)
        layout.addWidget(self._remove_ramp_checkbox, 0, 7)

        layout.addWidget(QLabel("FFT Y:"), 1, 0)
        self._fft_ymin_edit = QLineEdit("0")
        self._fft_ymin_edit.setMaximumWidth(60)
        layout.addWidget(self._fft_ymin_edit, 1, 1)
        self._fft_ymax_edit = QLineEdit("200000")
        self._fft_ymax_edit.setMaximumWidth(60)
        layout.addWidget(self._fft_ymax_edit, 1, 2)

        layout.addWidget(QLabel("FFT X:"), 2, 0)
        self._fft_xmin_edit = QLineEdit("450")
        self._fft_xmin_edit.setMaximumWidth(60)
        layout.addWidget(self._fft_xmin_edit, 2, 1)
        self._fft_xmax_edit = QLineEdit("750")
        self._fft_xmax_edit.setMaximumWidth(60)
        layout.addWidget(self._fft_xmax_edit, 2, 2)

        layout.addWidget(QLabel("Phase Y:"), 1, 3)
        self._phase_ymin_edit = QLineEdit("-4")
        self._phase_ymin_edit.setMaximumWidth(50)
        layout.addWidget(self._phase_ymin_edit, 1, 4)
        self._phase_ymax_edit = QLineEdit("4")
        self._phase_ymax_edit.setMaximumWidth(50)
        layout.addWidget(self._phase_ymax_edit, 1, 5)

        layout.addWidget(QLabel("Max Pts:"), 2, 3)
        self._max_points_edit = QLineEdit("100")
        self._max_points_edit.setMaximumWidth(50)
        layout.addWidget(self._max_points_edit, 2, 4)

        btn_update = QPushButton("Update Limits")
        btn_update.setMaximumWidth(100)
        btn_update.clicked.connect(self._on_update_limits)
        layout.addWidget(btn_update, 2, 5, 1, 2)

        return group

    def _create_pid_group(self) -> QGroupBox:
        group = QGroupBox("PID Control")
        layout = QGridLayout(group)
        layout.setSpacing(5)

        layout.addWidget(QLabel("φ₀:"), 0, 0)
        self._setpoint_edit = QLineEdit("0.0")
        self._setpoint_edit.setMaximumWidth(70)
        self._setpoint_edit.setValidator(QDoubleValidator())
        layout.addWidget(self._setpoint_edit, 0, 1)

        layout.addWidget(QLabel("Kp:"), 0, 2)
        self._kp_edit = QLineEdit("0.05")
        self._kp_edit.setMaximumWidth(60)
        layout.addWidget(self._kp_edit, 0, 3)

        layout.addWidget(QLabel("Ki:"), 0, 4)
        self._ki_edit = QLineEdit("0.0")
        self._ki_edit.setMaximumWidth(60)
        layout.addWidget(self._ki_edit, 0, 5)

        layout.addWidget(QLabel("Kd:"), 0, 6)
        self._kd_edit = QLineEdit("0.0")
        self._kd_edit.setMaximumWidth(60)
        layout.addWidget(self._kd_edit, 0, 7)

        layout.addWidget(QLabel("Gain:"), 0, 8)
        self._gain_edit = QLineEdit("-1.0")
        self._gain_edit.setMaximumWidth(60)
        layout.addWidget(self._gain_edit, 0, 9)

        layout.addWidget(QLabel("Max Step:"), 0, 10)
        self._max_step_edit = QLineEdit("0.05")
        self._max_step_edit.setMaximumWidth(60)
        layout.addWidget(self._max_step_edit, 0, 11)

        self._unwrap_checkbox = QCheckBox("Unwrap")
        self._unwrap_checkbox.setChecked(False)
        layout.addWidget(self._unwrap_checkbox, 1, 0)

        self._auto_reset_checkbox = QCheckBox("Auto Reset")
        self._auto_reset_checkbox.setChecked(False)
        layout.addWidget(self._auto_reset_checkbox, 1, 1, 1, 2)

        self._lock_checkbox = QCheckBox("LOCK")
        self._lock_checkbox.setStyleSheet("QCheckBox { font-weight: bold; color: blue; }")
        self._lock_checkbox.stateChanged.connect(self._on_toggle_lock)
        layout.addWidget(self._lock_checkbox, 1, 3)

        btn_update_sp = QPushButton("Update SP")
        btn_update_sp.setMaximumWidth(80)
        btn_update_sp.clicked.connect(self._on_update_setpoint)
        layout.addWidget(btn_update_sp, 1, 4)

        btn_update_pid = QPushButton("Update PID")
        btn_update_pid.setMaximumWidth(80)
        btn_update_pid.clicked.connect(self._on_update_pid)
        layout.addWidget(btn_update_pid, 1, 5)

        return group

    def _create_voltage_group(self) -> QGroupBox:
        group = QGroupBox("Voltage Limits")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Min:"))
        self._vmin_edit = QLineEdit("75.0")
        self._vmin_edit.setMaximumWidth(60)
        layout.addWidget(self._vmin_edit)

        layout.addWidget(QLabel("Max:"))
        self._vmax_edit = QLineEdit("82.0")
        self._vmax_edit.setMaximumWidth(60)
        layout.addWidget(self._vmax_edit)

        layout.addWidget(QLabel("Start:"))
        self._vstart_edit = QLineEdit("80.0")
        self._vstart_edit.setMaximumWidth(60)
        layout.addWidget(self._vstart_edit)

        btn_update = QPushButton("Update Limits")
        btn_update.setMaximumWidth(100)
        btn_update.clicked.connect(self._on_update_voltage_limits)
        layout.addWidget(btn_update)

        layout.addStretch()
        return group

    def _create_stability_group(self) -> QGroupBox:
        group = QGroupBox("Stability Test")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Start:"))
        self._stab_start_edit = QLineEdit("-3.14159")
        self._stab_start_edit.setMaximumWidth(70)
        layout.addWidget(self._stab_start_edit)

        layout.addWidget(QLabel("End:"))
        self._stab_end_edit = QLineEdit("3.14159")
        self._stab_end_edit.setMaximumWidth(70)
        layout.addWidget(self._stab_end_edit)

        layout.addWidget(QLabel("Steps:"))
        self._stab_steps_edit = QLineEdit("10")
        self._stab_steps_edit.setMaximumWidth(40)
        layout.addWidget(self._stab_steps_edit)

        layout.addWidget(QLabel("Wait [s]:"))
        self._stab_wait_edit = QLineEdit("5.0")
        self._stab_wait_edit.setMaximumWidth(50)
        layout.addWidget(self._stab_wait_edit)

        self._stab_test_btn = QPushButton("Start Test")
        self._stab_test_btn.setMaximumWidth(80)
        self._stab_test_btn.clicked.connect(self._on_toggle_stability_test)
        layout.addWidget(self._stab_test_btn)

        layout.addStretch()
        return group

    def _create_control_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        btn_start = QPushButton("Start")
        btn_start.setMaximumWidth(80)
        btn_start.clicked.connect(self._on_start)
        layout.addWidget(btn_start)

        btn_stop = QPushButton("Stop")
        btn_stop.setMaximumWidth(80)
        btn_stop.clicked.connect(self._on_stop)
        layout.addWidget(btn_stop)

        layout.addStretch(1)

        self._phase_label = QLabel("φ = — rad")
        self._phase_label.setStyleSheet("QLabel { font-size: 12pt; font-weight: bold; }")
        layout.addWidget(self._phase_label)

        layout.addWidget(QLabel(" | "))

        self._error_label = QLabel("Error = — rad")
        layout.addWidget(self._error_label)

        layout.addWidget(QLabel(" | "))

        self._slope_label = QLabel("Slope = — rad/index")
        layout.addWidget(self._slope_label)

        return layout

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_message(self, msg: str) -> None:
        if self._log:
            self._log.log(msg, source="PhaseLock")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_current_phase(self) -> float:
        """Get the most recent unwrapped phase value."""
        if len(self._hist_phi_unwrapped) > 0:
            return float(self._hist_phi_unwrapped[-1])
        return float("nan")

    def get_phase_average(self, duration_s: float) -> tuple[float, float]:
        """Get average and std of phase over specified duration."""
        if len(self._hist_phi_unwrapped) == 0:
            return float("nan"), float("nan")
        n_samples = max(1, int(duration_s / self._min_draw_dt))
        samples = list(self._hist_phi_unwrapped)[-n_samples:]
        if len(samples) == 0:
            return float("nan"), float("nan")
        avg = float(np.mean(samples))
        std = float(np.std(samples)) if len(samples) > 1 else 0.0
        return avg, std

    def get_current_voltage(self) -> float:
        """Get current piezo voltage."""
        if self._ctrl_thread and hasattr(self._ctrl_thread, "current_v"):
            return float(self._ctrl_thread.current_v)
        return float("nan")

    def set_target(self, target_rad: float) -> None:
        """Set the phase lock target."""
        if self._ctrl_thread:
            if self._auto_reset_checkbox.isChecked() and self._ctrl_thread.enabled:
                self._ctrl_thread.enabled = False
                self._ctrl_thread.target = float(target_rad)
                self._setpoint_edit.setText(f"{float(target_rad):.6f}")
                self._ctrl_thread.integral = 0.0
                self._ctrl_thread.last_t = None
                self._ctrl_thread.last_err = 0.0
                self._ctrl_thread.enabled = True
            else:
                self._ctrl_thread.target = float(target_rad)
                self._setpoint_edit.setText(f"{float(target_rad):.6f}")

    def is_locked(self) -> bool:
        """Check if phase lock is active."""
        return self._ctrl_thread is not None and self._ctrl_thread.enabled

    # -------------------------------------------------------------------------
    # Device connection
    # -------------------------------------------------------------------------

    def _on_connect_spec(self) -> None:
        key = self._spec_key_edit.text().strip()
        ctrl = REGISTRY.get(key)
        if ctrl is None:
            QMessageBox.critical(self, "Spectrometer", f"Not found: {key}")
            return
        self._spec_ctrl = ctrl
        self._log_message("Connected spectrometer")

    def _on_connect_stage(self) -> None:
        key = self._stage_key_edit.text().strip()
        st = REGISTRY.get(key)
        if st is None:
            QMessageBox.critical(self, "NV40", "Not found")
            return

        self._ctrl_thread = ControlThread(st)
        try:
            self._ctrl_thread.vmin = float(self._vmin_edit.text())
            self._ctrl_thread.vmax = float(self._vmax_edit.text())
            v_start = float(self._vstart_edit.text())
            self._ctrl_thread.current_v = v_start
            try:
                st.set_position(v_start)
                self._log_message(f"Stage set to {v_start:.2f} V")
            except Exception:
                pass
        except Exception:
            pass
        self._ctrl_thread.update_status.connect(self._on_control_update)
        self._ctrl_thread.start()
        self._log_message(
            f"Connected NV40 ({self._ctrl_thread.vmin:.1f}-{self._ctrl_thread.vmax:.1f}V, "
            f"start: {self._ctrl_thread.current_v:.1f}V)"
        )

    # -------------------------------------------------------------------------
    # Parameter updates
    # -------------------------------------------------------------------------

    def _on_update_setpoint(self) -> None:
        if not self._ctrl_thread:
            return
        try:
            new_sp = float(self._setpoint_edit.text())
            if self._auto_reset_checkbox.isChecked() and self._ctrl_thread.enabled:
                self._ctrl_thread.enabled = False
                self._log_message("Auto-reset: Lock disabled")
                self._ctrl_thread.target = new_sp
                self._log_message(f"Setpoint updated to {new_sp:.3f} rad")
                self._ctrl_thread.integral = 0.0
                self._ctrl_thread.last_t = None
                self._ctrl_thread.last_err = 0.0
                self._ctrl_thread.enabled = True
                self._log_message("Auto-reset: Lock re-enabled")
            else:
                self._ctrl_thread.target = new_sp
                self._log_message(f"Setpoint updated to {new_sp:.3f} rad")
        except Exception:
            pass

    def _on_update_pid(self) -> None:
        if not self._ctrl_thread:
            return
        self._ctrl_thread.kp = float(self._kp_edit.text())
        self._ctrl_thread.ki = float(self._ki_edit.text())
        self._ctrl_thread.kd = float(self._kd_edit.text())
        self._ctrl_thread.target = float(self._setpoint_edit.text())
        self._ctrl_thread.max_step = float(self._max_step_edit.text())
        self._ctrl_thread.unwrap = self._unwrap_checkbox.isChecked()
        self._ctrl_thread.gain = float(self._gain_edit.text())
        try:
            self._ctrl_thread.vmin = float(self._vmin_edit.text())
            self._ctrl_thread.vmax = float(self._vmax_edit.text())
        except Exception:
            pass
        self._log_message(f"PID updated (V: {self._ctrl_thread.vmin:.1f}-{self._ctrl_thread.vmax:.1f}V)")

    def _on_update_limits(self) -> None:
        xmin_fft = self._fft_xmin_edit.text().strip()
        xmax_fft = self._fft_xmax_edit.text().strip()
        if xmin_fft and xmax_fft:
            try:
                self._ax_fft.set_xlim(float(xmin_fft), float(xmax_fft))
            except Exception:
                pass

        ymin_fft = self._fft_ymin_edit.text().strip()
        ymax_fft = self._fft_ymax_edit.text().strip()
        if ymin_fft and ymax_fft:
            try:
                self._ax_fft.set_ylim(float(ymin_fft), float(ymax_fft))
            except Exception:
                pass

        ymin_phase = self._phase_ymin_edit.text().strip()
        ymax_phase = self._phase_ymax_edit.text().strip()
        if ymin_phase and ymax_phase:
            try:
                self._ax_phase.set_ylim(float(ymin_phase), float(ymax_phase))
            except Exception:
                pass

        self._blitting_initialized = False
        self._canvas.draw()
        self._log_message("Plot limits updated")

    def _on_update_voltage_limits(self) -> None:
        if not self._ctrl_thread:
            self._log_message("No stage connected")
            return
        try:
            self._ctrl_thread.vmin = float(self._vmin_edit.text())
            self._ctrl_thread.vmax = float(self._vmax_edit.text())
            self._log_message(f"Voltage limits updated: {self._ctrl_thread.vmin:.1f}-{self._ctrl_thread.vmax:.1f}V")
        except Exception:
            self._log_message("Invalid voltage limits")

    # -------------------------------------------------------------------------
    # Acquisition control
    # -------------------------------------------------------------------------

    def _on_start(self) -> None:
        if self._spec_ctrl is None:
            QMessageBox.critical(self, "Run", "No spectrometer")
            return

        try:
            self._max_points = int(self._max_points_edit.text())
        except Exception:
            self._max_points = 100

        self._hist_phi_raw = deque(maxlen=self._max_points)
        self._hist_phi_unwrapped = deque(maxlen=self._max_points)
        self._ax_fft.clear()
        self._ax_phase.clear()
        self._fft_scatter = None
        self._fft_line = None
        self._phase_plot = None
        self._phase_sp_line = None
        self._blitting_initialized = False

        self._acq_thread = AvaspecThread(self._spec_ctrl)
        self._acq_thread.data_ready.connect(self._on_data)
        self._acq_thread.error.connect(self._on_acq_error)
        self._acq_thread.start()
        self._log_message("Acquisition started")

    def _on_stop(self) -> None:
        if self._acq_thread:
            self._acq_thread.stop()
            self._acq_thread.wait()
            self._acq_thread = None
        self._log_message("Acquisition stopped")

    def _on_toggle_lock(self) -> None:
        if self._ctrl_thread is None:
            return
        if self._lock_checkbox.isChecked():
            self._ctrl_thread.integral = 0.0
            self._ctrl_thread.last_t = None
            self._ctrl_thread.last_err = 0.0
            self._ctrl_thread.enabled = True
            self._log_message("Lock ON")
        else:
            self._ctrl_thread.enabled = False
            self._log_message("Lock OFF")
            if self._phase_sp_line:
                self._phase_sp_line.set_visible(False)
                self._canvas.draw_idle()

    # -------------------------------------------------------------------------
    # Stability test
    # -------------------------------------------------------------------------

    def _on_toggle_stability_test(self) -> None:
        if self._stability_test_active:
            self._stability_test_active = False
            self._stability_timer.stop()
            self._stab_test_btn.setText("Start Test")
            self._log_message("Stability test stopped")
        else:
            if not self._ctrl_thread:
                QMessageBox.warning(self, "Stability Test", "NV40 not connected")
                return
            try:
                start_val = float(self._stab_start_edit.text())
                end_val = float(self._stab_end_edit.text())
                num_steps = int(self._stab_steps_edit.text())
                wait_time = float(self._stab_wait_edit.text())
            except ValueError:
                QMessageBox.critical(self, "Stability Test", "Invalid parameters")
                return
            if num_steps < 1:
                QMessageBox.critical(self, "Stability Test", "Steps must be >= 1")
                return

            self._stability_setpoints = np.linspace(start_val, end_val, num_steps).tolist()
            self._stability_step = 0
            self._ctrl_thread.target = self._stability_setpoints[0]
            self._setpoint_edit.setText(f"{self._stability_setpoints[0]:.3f}")
            self._stability_test_active = True
            self._stability_timer.start(int(wait_time * 1000))
            self._stab_test_btn.setText("Stop Test")
            self._log_message(
                f"Stability test: {num_steps} steps, {start_val:.3f} to {end_val:.3f} rad, wait={wait_time}s"
            )

    def _on_stability_step(self) -> None:
        if not self._stability_test_active:
            return
        self._stability_step += 1
        if self._stability_step >= len(self._stability_setpoints):
            self._on_toggle_stability_test()
            return
        new_sp = self._stability_setpoints[self._stability_step]
        self._ctrl_thread.target = new_sp
        self._setpoint_edit.setText(f"{new_sp:.3f}")
        self._log_message(
            f"Step {self._stability_step + 1}/{len(self._stability_setpoints)}: φ0 = {new_sp:.3f} rad"
        )

    # -------------------------------------------------------------------------
    # Data handling
    # -------------------------------------------------------------------------

    def _on_acq_error(self, err: str) -> None:
        self._log_message(f"Thread error: {err}")
        self._on_stop()

    def _on_control_update(self, phi: float, v: float) -> None:
        self._phase_label.setText(f"φ = {phi:+.3f} rad   V={v:+.3f}")

    def _on_data(self, wl, y) -> None:
        if self._acq_thread is None:
            return

        now = time.monotonic()
        if now - self._last_draw < self._min_draw_dt:
            return
        self._last_draw = now

        y = np.asarray(y, float)
        if y.size < 32:
            return

        F = np.fft.fft(y)
        mag = np.abs(F)
        phase = np.angle(F)
        n = len(F)

        try:
            center = int(self._center_idx_edit.text())
        except Exception:
            center = 100
        try:
            W = int(self._window_edit.text())
        except Exception:
            W = 100

        i0 = max(0, center - W)
        i1 = min(n, center + W)

        phase_unwrapped = np.unwrap(phase)
        use_weighted = self._weighted_checkbox.isChecked()
        remove_ramp = self._remove_ramp_checkbox.isChecked()

        window_mag = mag[i0:i1]
        window_phase = phase[i0:i1]

        if remove_ramp and len(window_phase) > 2:
            indices = np.arange(len(window_phase))
            coeffs = np.polyfit(indices, window_phase, 1)
            phase_trend = np.polyval(coeffs, indices)
            phase_detrended = window_phase - phase_trend
        else:
            phase_detrended = window_phase

        phase_slope = 0.0
        if len(phase_detrended) > 2:
            indices = np.arange(len(phase_detrended))
            slope_coeffs = np.polyfit(indices, phase_detrended, 1)
            phase_slope = slope_coeffs[0]

        if use_weighted:
            if np.sum(window_mag) > 0:
                phi = np.sum(window_mag * phase_detrended) / np.sum(window_mag)
            else:
                phi = phase_detrended[len(phase_detrended) // 2] if len(phase_detrended) > 0 else 0.0
        else:
            center_in_window = center - i0
            if 0 <= center_in_window < len(phase_detrended):
                phi = phase_detrended[center_in_window]
            else:
                return

        try:
            plot_skip = max(1, int(self._plot_skip_edit.text()))
        except Exception:
            plot_skip = 5

        x = np.arange(i0, i1, plot_skip)
        mag_decimated = mag[i0:i1:plot_skip]
        phase_unwrapped_decimated = phase_unwrapped[i0:i1:plot_skip]

        if self._fft_scatter is None:
            self._fft_scatter = self._ax_fft.scatter(
                x,
                mag_decimated,
                c=phase_unwrapped_decimated,
                cmap="hsv",
                s=20,
                vmin=-np.pi,
                vmax=np.pi,
                animated=True,
            )

            xmin_fft = self._fft_xmin_edit.text().strip()
            xmax_fft = self._fft_xmax_edit.text().strip()
            if xmin_fft and xmax_fft:
                try:
                    self._ax_fft.set_xlim(float(xmin_fft), float(xmax_fft))
                except Exception:
                    self._ax_fft.set_xlim(i0, i1)
            else:
                self._ax_fft.set_xlim(i0, i1)

            self._ax_fft.set_xlabel("FFT Index")
            self._ax_fft.set_ylabel("Magnitude")
            self._ax_fft.set_ylim(0, 200000)
        else:
            self._fft_scatter.set_offsets(np.c_[x, mag_decimated])
            self._fft_scatter.set_array(phase_unwrapped_decimated)

        if self._fft_line is None:
            if use_weighted:
                self._fft_line = self._ax_fft.axvspan(i0, i1, alpha=0.2, color="orange", animated=True)
            else:
                self._fft_line = self._ax_fft.axvline(center, color="red", linestyle="--", animated=True)
        else:
            self._fft_line.remove()
            if use_weighted:
                self._fft_line = self._ax_fft.axvspan(i0, i1, alpha=0.2, color="orange", animated=True)
            else:
                self._fft_line = self._ax_fft.axvline(center, color="red", linestyle="--", animated=True)

        if self._unwrap_checkbox.isChecked():
            phi_u = phi if not self._hist_phi_unwrapped else np.unwrap([self._hist_phi_unwrapped[-1], phi])[-1]
        else:
            phi_u = phi

        self._hist_phi_raw.append(phi)
        self._hist_phi_unwrapped.append(phi_u)

        x_indices = np.arange(len(self._hist_phi_unwrapped))
        pp = np.array(self._hist_phi_unwrapped if self._unwrap_checkbox.isChecked() else self._hist_phi_raw)

        plot_color = "blue" if (self._ctrl_thread and self._ctrl_thread.enabled) else "black"

        if self._phase_plot is None:
            (self._phase_plot,) = self._ax_phase.plot(x_indices, pp, "o", ms=3, color=plot_color, animated=True)
            self._ax_phase.set_xlim(-1, self._max_points)
            self._ax_phase.set_xlabel("Sample")
            self._ax_phase.set_ylabel("Phase [rad]")
            ymin_phase = self._phase_ymin_edit.text().strip()
            ymax_phase = self._phase_ymax_edit.text().strip()
            if ymin_phase and ymax_phase:
                try:
                    self._ax_phase.set_ylim(float(ymin_phase), float(ymax_phase))
                except Exception:
                    pass
        else:
            self._phase_plot.set_xdata(x_indices)
            self._phase_plot.set_ydata(pp)
            self._phase_plot.set_color(plot_color)

        if self._ctrl_thread and self._ctrl_thread.enabled:
            try:
                sp = float(self._setpoint_edit.text())
                if self._phase_sp_line is None:
                    self._phase_sp_line = self._ax_phase.axhline(sp, color="orange", linestyle="--", animated=True)
                else:
                    self._phase_sp_line.set_ydata([sp, sp])
                    self._phase_sp_line.set_visible(True)
            except Exception:
                pass
        else:
            if self._phase_sp_line:
                self._phase_sp_line.set_visible(False)

        if not self._blitting_initialized:
            self._canvas.draw()
            self._bg_fft = self._canvas.copy_from_bbox(self._ax_fft.bbox)
            self._bg_phase = self._canvas.copy_from_bbox(self._ax_phase.bbox)
            self._blitting_initialized = True

        self._canvas.restore_region(self._bg_fft)
        self._ax_fft.draw_artist(self._fft_scatter)
        self._ax_fft.draw_artist(self._fft_line)
        self._canvas.blit(self._ax_fft.bbox)

        self._canvas.restore_region(self._bg_phase)
        self._ax_phase.draw_artist(self._phase_plot)
        if self._phase_sp_line and self._phase_sp_line.get_visible():
            self._ax_phase.draw_artist(self._phase_sp_line)
        self._canvas.blit(self._ax_phase.bbox)

        if self._ctrl_thread and self._ctrl_thread.enabled:
            try:
                sp = float(self._setpoint_edit.text())
                current_error = abs(phi_u - sp)
                self._error_label.setText(f"Error = {current_error:.4f} rad")
            except Exception:
                self._error_label.setText("Error = — rad")
        else:
            self._error_label.setText("Error = — rad")

        self._slope_label.setText(f"Slope = {phase_slope:.4f} rad/index")

        if self._ctrl_thread and self._ctrl_thread.enabled:
            self._ctrl_thread.q.append(phi)

        self._phase_label.setText(f"φ = {phi_u:+.3f} rad")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._on_stop()
        if self._ctrl_thread:
            self._ctrl_thread.enabled = False
        try:
            REGISTRY.unregister(self._registry_key)
        except Exception:
            pass
        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = AvaspecPhaseLockWindow()
    window.resize(800, 800)
    window.show()
    sys.exit(app.exec_())