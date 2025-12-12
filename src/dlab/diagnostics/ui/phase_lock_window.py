from __future__ import annotations
import sys, time, datetime
import numpy as np
from collections import deque

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QMessageBox,
    QCheckBox, QComboBox, QTabWidget, QGroupBox
)
from PyQt5.QtGui import QDoubleValidator

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import Normalize

from dlab.core.device_registry import REGISTRY
from dlab.hardware.wrappers.piezojena_controller import NV40


class ControlThread(QThread):
    update_status = pyqtSignal(float, float)

    def __init__(self, stage: NV40):
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
        except:
            self.current_v = 80
        self.enabled = False

    def run(self):
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
                err = (self.target - phi + np.pi) % (2*np.pi) - np.pi

            d_err = (err - self.last_err) / dt
            self.last_err = err

            self.integral += err * dt
            max_int = 1.0 / (self.ki + 1e-9)
            self.integral = np.clip(self.integral, -max_int, max_int)
            
            gain = getattr(self, "gain", -1.0)
            u = gain *(self.kp*err + self.ki*self.integral + self.kd*d_err)

            if abs(u) > self.max_step:
                u = np.sign(u) * self.max_step

            new_v = self.current_v + u
            new_v = max(self.vmin, min(self.vmax, new_v))
            new_v = round(new_v / 0.01) * 0.01

            try:
                self.stage.set_position(new_v)
                self.current_v = new_v
            except:
                pass

            self.update_status.emit(phi_use, self.current_v)


class AvaspecThread(QThread):
    data_ready = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self.running = True

    def run(self):
        while self.running and not self.isInterruptionRequested():
            try:
                ts, wl, counts = self.ctrl.measure_once()
                self.data_ready.emit(wl, counts)
            except Exception as e:
                self.error.emit(str(e))
                break

    def stop(self):
        self.running = False
        self.requestInterruption()


class AvaspecPhaseLockTab(QWidget):
    def __init__(self, registry_key: str = "phaselock:avaspec"):
        super().__init__()
        
        self.registry_key = registry_key
        self.spec_ctrl = None
        self.stage = None
        self.th = None
        self.ctrl_th = None
        self.max_points = 100
        self.hist_phi_raw = deque(maxlen=self.max_points)
        self.hist_phi_unwrapped = deque(maxlen=self.max_points)
        self.last_draw = 0.0
        self.min_draw_dt = 0.05
        self.fft_scatter = None
        self.fft_line = None
        self.cbar = None
        self.phase_plot = None
        self.phase_sp_line = None
        self.bg_fft = None
        self.bg_phase = None
        self.blitting_initialized = False
        self.stability_test_active = False
        self.stability_timer = None
        self.stability_step = 0
        self.stability_setpoints = []

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        
        self.fig, (self.ax_fft, self.ax_phase) = plt.subplots(1, 2, figsize=(14,4.5))
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 3)
        
        middle_row = QHBoxLayout()
        left_panel = QVBoxLayout()

        conn_group = QGroupBox("Devices")
        conn_layout = QGridLayout()
        conn_layout.addWidget(QLabel("Spectrometer:"), 0, 0)
        self.ed_spec_key = QLineEdit("spectrometer:avaspec:spec_1")
        self.ed_spec_key.setMaximumWidth(200)
        conn_layout.addWidget(self.ed_spec_key, 0, 1)
        btn_spec = QPushButton("Connect")
        btn_spec.setMaximumWidth(80)
        btn_spec.clicked.connect(self._connect_spec)
        conn_layout.addWidget(btn_spec, 0, 2)
        
        conn_layout.addWidget(QLabel("NV40 Stage:"), 1, 0)
        self.ed_stage_key = QLineEdit("stage:piezojena:nv40")
        self.ed_stage_key.setMaximumWidth(200)
        conn_layout.addWidget(self.ed_stage_key, 1, 1)
        btn_stage = QPushButton("Connect")
        btn_stage.setMaximumWidth(80)
        btn_stage.clicked.connect(self._connect_stage)
        conn_layout.addWidget(btn_stage, 1, 2)
        conn_group.setLayout(conn_layout)
        left_panel.addWidget(conn_group)

        fft_group = QGroupBox("FFT Analysis")
        fft_layout = QGridLayout()
        fft_layout.setSpacing(5)
        
        fft_layout.addWidget(QLabel("Center:"), 0, 0)
        self.ed_center_idx = QLineEdit("600")
        self.ed_center_idx.setMaximumWidth(60)
        fft_layout.addWidget(self.ed_center_idx, 0, 1)
        
        fft_layout.addWidget(QLabel("Window:"), 0, 2)
        self.ed_window = QLineEdit("150")
        self.ed_window.setMaximumWidth(60)
        fft_layout.addWidget(self.ed_window, 0, 3)
        
        fft_layout.addWidget(QLabel("Plot Every:"), 0, 4)
        self.ed_plot_skip = QLineEdit("5")
        self.ed_plot_skip.setMaximumWidth(40)
        fft_layout.addWidget(self.ed_plot_skip, 0, 5)
        
        self.chk_weighted = QCheckBox("Weighted Avg")
        fft_layout.addWidget(self.chk_weighted, 0, 6)
        
        self.chk_remove_ramp = QCheckBox("Remove Ramp")
        self.chk_remove_ramp.setChecked(False)
        fft_layout.addWidget(self.chk_remove_ramp, 0, 7)
        
        fft_layout.addWidget(QLabel("FFT Y:"), 1, 0)
        self.ed_fft_ymin = QLineEdit("0")
        self.ed_fft_ymin.setMaximumWidth(60)
        fft_layout.addWidget(self.ed_fft_ymin, 1, 1)
        self.ed_fft_ymax = QLineEdit("200000")
        self.ed_fft_ymax.setMaximumWidth(60)
        fft_layout.addWidget(self.ed_fft_ymax, 1, 2)
        
        fft_layout.addWidget(QLabel("FFT X:"), 2, 0)
        self.ed_fft_xmin = QLineEdit("450")
        self.ed_fft_xmin.setMaximumWidth(60)
        fft_layout.addWidget(self.ed_fft_xmin, 2, 1)
        self.ed_fft_xmax = QLineEdit("750")
        self.ed_fft_xmax.setMaximumWidth(60)
        fft_layout.addWidget(self.ed_fft_xmax, 2, 2)
        
        fft_layout.addWidget(QLabel("Phase Y:"), 1, 3)
        self.ed_phase_ymin = QLineEdit("-4")
        self.ed_phase_ymin.setMaximumWidth(50)
        fft_layout.addWidget(self.ed_phase_ymin, 1, 4)
        self.ed_phase_ymax = QLineEdit("4")
        self.ed_phase_ymax.setMaximumWidth(50)
        fft_layout.addWidget(self.ed_phase_ymax, 1, 5)
        
        fft_layout.addWidget(QLabel("Max Pts:"), 2, 3)
        self.ed_max_points = QLineEdit("100")
        self.ed_max_points.setMaximumWidth(50)
        fft_layout.addWidget(self.ed_max_points, 2, 4)
        
        self.btn_update_ylim = QPushButton("Update Limits")
        self.btn_update_ylim.setMaximumWidth(100)
        self.btn_update_ylim.clicked.connect(self._update_ylimits)
        fft_layout.addWidget(self.btn_update_ylim, 2, 5, 1, 2)
        
        fft_group.setLayout(fft_layout)
        left_panel.addWidget(fft_group)

        pid_group = QGroupBox("PID Control")
        pid_layout = QGridLayout()
        pid_layout.setSpacing(5)
        
        pid_layout.addWidget(QLabel("φ₀:"), 0, 0)
        self.ed_sp = QLineEdit("0.0")
        self.ed_sp.setMaximumWidth(70)
        self.ed_sp.setValidator(QDoubleValidator())
        pid_layout.addWidget(self.ed_sp, 0, 1)
        
        pid_layout.addWidget(QLabel("Kp:"), 0, 2)
        self.ed_kp = QLineEdit("0.05")
        self.ed_kp.setMaximumWidth(60)
        pid_layout.addWidget(self.ed_kp, 0, 3)
        
        pid_layout.addWidget(QLabel("Ki:"), 0, 4)
        self.ed_ki = QLineEdit("0.0")
        self.ed_ki.setMaximumWidth(60)
        pid_layout.addWidget(self.ed_ki, 0, 5)
        
        pid_layout.addWidget(QLabel("Kd:"), 0, 6)
        self.ed_kd = QLineEdit("0.0")
        self.ed_kd.setMaximumWidth(60)
        pid_layout.addWidget(self.ed_kd, 0, 7)
        
        pid_layout.addWidget(QLabel("Gain:"), 0, 8)
        self.ed_gain = QLineEdit("-1.0")
        self.ed_gain.setMaximumWidth(60)
        pid_layout.addWidget(self.ed_gain, 0, 9)
        
        pid_layout.addWidget(QLabel("Max Step:"), 0, 10)
        self.ed_step = QLineEdit("0.05")
        self.ed_step.setMaximumWidth(60)
        pid_layout.addWidget(self.ed_step, 0, 11)
        
        self.chk_unwrap = QCheckBox("Unwrap")
        self.chk_unwrap.setChecked(False)
        pid_layout.addWidget(self.chk_unwrap, 1, 0)
        
        self.chk_auto_reset = QCheckBox("Auto Reset")
        self.chk_auto_reset.setChecked(False)
        pid_layout.addWidget(self.chk_auto_reset, 1, 1, 1, 2)
        
        self.chk_lock = QCheckBox("LOCK")
        self.chk_lock.setStyleSheet("QCheckBox { font-weight: bold; color: blue; }")
        self.chk_lock.stateChanged.connect(self._toggle_lock)
        pid_layout.addWidget(self.chk_lock, 1, 3)
        
        self.btn_update_sp = QPushButton("Update SP")
        self.btn_update_sp.setMaximumWidth(80)
        self.btn_update_sp.clicked.connect(self._update_setpoint)
        pid_layout.addWidget(self.btn_update_sp, 1, 4)
        
        self.btn_update_pid = QPushButton("Update PID")
        self.btn_update_pid.setMaximumWidth(80)
        self.btn_update_pid.clicked.connect(self._update_pid_params)
        pid_layout.addWidget(self.btn_update_pid, 1, 5)
        
        pid_group.setLayout(pid_layout)
        left_panel.addWidget(pid_group)

        volt_group = QGroupBox("Voltage Limits")
        volt_layout = QHBoxLayout()
        volt_layout.addWidget(QLabel("Min:"))
        self.ed_vmin = QLineEdit("75.0")
        self.ed_vmin.setMaximumWidth(60)
        volt_layout.addWidget(self.ed_vmin)
        volt_layout.addWidget(QLabel("Max:"))
        self.ed_vmax = QLineEdit("82.0")
        self.ed_vmax.setMaximumWidth(60)
        volt_layout.addWidget(self.ed_vmax)
        volt_layout.addWidget(QLabel("Start:"))
        self.ed_vstart = QLineEdit("80.0")
        self.ed_vstart.setMaximumWidth(60)
        volt_layout.addWidget(self.ed_vstart)
        btn_update_volt = QPushButton("Update Limits")
        btn_update_volt.setMaximumWidth(100)
        btn_update_volt.clicked.connect(self._update_voltage_limits)
        volt_layout.addWidget(btn_update_volt)
        volt_layout.addStretch()
        volt_group.setLayout(volt_layout)
        left_panel.addWidget(volt_group)

        stab_group = QGroupBox("Stability Test")
        stab_layout = QHBoxLayout()
        stab_layout.addWidget(QLabel("Start:"))
        self.ed_stab_start = QLineEdit("-3.14159")
        self.ed_stab_start.setMaximumWidth(70)
        stab_layout.addWidget(self.ed_stab_start)
        stab_layout.addWidget(QLabel("End:"))
        self.ed_stab_end = QLineEdit("3.14159")
        self.ed_stab_end.setMaximumWidth(70)
        stab_layout.addWidget(self.ed_stab_end)
        stab_layout.addWidget(QLabel("Steps:"))
        self.ed_stab_steps = QLineEdit("10")
        self.ed_stab_steps.setMaximumWidth(40)
        stab_layout.addWidget(self.ed_stab_steps)
        stab_layout.addWidget(QLabel("Wait [s]:"))
        self.ed_stab_wait = QLineEdit("5.0")
        self.ed_stab_wait.setMaximumWidth(50)
        stab_layout.addWidget(self.ed_stab_wait)
        self.btn_test_stability = QPushButton("Start Test")
        self.btn_test_stability.setMaximumWidth(80)
        self.btn_test_stability.clicked.connect(self._toggle_stability_test)
        stab_layout.addWidget(self.btn_test_stability)
        stab_layout.addStretch()
        stab_group.setLayout(stab_layout)
        left_panel.addWidget(stab_group)

        ctrl_row = QHBoxLayout()
        btn_start = QPushButton("Start")
        btn_start.setMaximumWidth(80)
        btn_start.clicked.connect(self._start)
        ctrl_row.addWidget(btn_start)
        btn_stop = QPushButton("Stop")
        btn_stop.setMaximumWidth(80)
        btn_stop.clicked.connect(self._stop)
        ctrl_row.addWidget(btn_stop)
        ctrl_row.addStretch(1)
        
        self.lbl_phase = QLabel("φ = — rad")
        self.lbl_phase.setStyleSheet("QLabel { font-size: 12pt; font-weight: bold; }")
        ctrl_row.addWidget(self.lbl_phase)
        ctrl_row.addWidget(QLabel(" | "))
        self.lbl_error = QLabel("Error = — rad")
        ctrl_row.addWidget(self.lbl_error)
        ctrl_row.addWidget(QLabel(" | "))
        self.lbl_slope = QLabel("Slope = — rad/index")
        ctrl_row.addWidget(self.lbl_slope)
        left_panel.addLayout(ctrl_row)

        middle_row.addLayout(left_panel, 1)
        root.addLayout(middle_row, 2)

        self.stability_timer = QTimer()
        self.stability_timer.timeout.connect(self._stability_test_step)
        self.log = None
        REGISTRY.register(self.registry_key, self)

    def get_current_phase(self) -> float:
        if len(self.hist_phi_unwrapped) > 0:
            return float(self.hist_phi_unwrapped[-1])
        return float('nan')
    
    def get_phase_average(self, duration_s: float) -> tuple[float, float]:
        if len(self.hist_phi_unwrapped) == 0:
            return float('nan'), float('nan')
        n_samples = max(1, int(duration_s / self.min_draw_dt))
        samples = list(self.hist_phi_unwrapped)[-n_samples:]
        if len(samples) == 0:
            return float('nan'), float('nan')
        avg = float(np.mean(samples))
        std = float(np.std(samples)) if len(samples) > 1 else 0.0
        return avg, std
    
    def get_current_voltage(self) -> float:
        if self.ctrl_th and hasattr(self.ctrl_th, 'current_v'):
            return float(self.ctrl_th.current_v)
        return float('nan')
    
    def set_target(self, target_rad: float) -> None:
        if self.ctrl_th:
            if self.chk_auto_reset.isChecked() and self.ctrl_th.enabled:
                self.ctrl_th.enabled = False
                self.ctrl_th.target = float(target_rad)
                self.ed_sp.setText(f"{float(target_rad):.6f}")
                self.ctrl_th.integral = 0.0
                self.ctrl_th.last_t = None
                self.ctrl_th.last_err = 0.0
                self.ctrl_th.enabled = True
            else:
                self.ctrl_th.target = float(target_rad)
                self.ed_sp.setText(f"{float(target_rad):.6f}")
    
    def is_locked(self) -> bool:
        return self.ctrl_th is not None and self.ctrl_th.enabled

    def _log(self, msg):
        if self.log is None:
            return
        t = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{t}] {msg}")

    def _connect_spec(self):
        key = self.ed_spec_key.text().strip()
        ctrl = REGISTRY.get(key)
        if ctrl is None:
            QMessageBox.critical(self, "Spec", f"Not found: {key}")
            return
        self.spec_ctrl = ctrl
        self._log("Connected spectrometer")

    def _connect_stage(self):
        key = self.ed_stage_key.text().strip()
        st = REGISTRY.get(key)
        if st is None:
            QMessageBox.critical(self, "NV40", "Not found")
            return

        self.ctrl_th = ControlThread(st)
        try:
            self.ctrl_th.vmin = float(self.ed_vmin.text())
            self.ctrl_th.vmax = float(self.ed_vmax.text())
            v_start = float(self.ed_vstart.text())
            self.ctrl_th.current_v = v_start
            try:
                st.set_position(v_start)
                self._log(f"Stage set to {v_start:.2f} V")
            except:
                pass
        except:
            pass
        self.ctrl_th.update_status.connect(self._on_control_update)
        self.ctrl_th.start()
        self._log(f"Connected NV40 ({self.ctrl_th.vmin:.1f}-{self.ctrl_th.vmax:.1f}V, start: {self.ctrl_th.current_v:.1f}V)")

    def _update_setpoint(self):
        if not self.ctrl_th:
            return
        try:
            new_sp = float(self.ed_sp.text())
            if self.chk_auto_reset.isChecked() and self.ctrl_th.enabled:
                self.ctrl_th.enabled = False
                self._log(f"Auto-reset: Lock disabled")
                self.ctrl_th.target = new_sp
                self._log(f"Setpoint updated to {new_sp:.3f} rad")
                self.ctrl_th.integral = 0.0
                self.ctrl_th.last_t = None
                self.ctrl_th.last_err = 0.0
                self.ctrl_th.enabled = True
                self._log(f"Auto-reset: Lock re-enabled")
            else:
                self.ctrl_th.target = new_sp
                self._log(f"Setpoint updated to {new_sp:.3f} rad")
        except:
            pass

    def _toggle_stability_test(self):
        if self.stability_test_active:
            self.stability_test_active = False
            self.stability_timer.stop()
            self.btn_test_stability.setText("Start Test")
            self._log("Stability test stopped")
        else:
            if not self.ctrl_th:
                QMessageBox.warning(self, "Stability Test", "NV40 not connected")
                return
            try:
                start_val = float(self.ed_stab_start.text())
                end_val = float(self.ed_stab_end.text())
                num_steps = int(self.ed_stab_steps.text())
                wait_time = float(self.ed_stab_wait.text())
            except ValueError:
                QMessageBox.critical(self, "Stability Test", "Invalid parameters")
                return
            if num_steps < 1:
                QMessageBox.critical(self, "Stability Test", "Steps must be >= 1")
                return
            self.stability_setpoints = np.linspace(start_val, end_val, num_steps).tolist()
            self.stability_step = 0
            self.ctrl_th.target = self.stability_setpoints[0]
            self.ed_sp.setText(f"{self.stability_setpoints[0]:.3f}")
            self.stability_test_active = True
            self.stability_timer.start(int(wait_time * 1000))
            self.btn_test_stability.setText("Stop Test")
            self._log(f"Stability test: {num_steps} steps, {start_val:.3f} to {end_val:.3f} rad, wait={wait_time}s")

    def _stability_test_step(self):
        if not self.stability_test_active:
            return
        self.stability_step += 1
        if self.stability_step >= len(self.stability_setpoints):
            self._toggle_stability_test()
            return
        new_sp = self.stability_setpoints[self.stability_step]
        self.ctrl_th.target = new_sp
        self.ed_sp.setText(f"{new_sp:.3f}")
        self._log(f"Step {self.stability_step+1}/{len(self.stability_setpoints)}: φ0 = {new_sp:.3f} rad")

    def _update_pid_params(self):
        if not self.ctrl_th:
            return
        self.ctrl_th.kp = float(self.ed_kp.text())
        self.ctrl_th.ki = float(self.ed_ki.text())
        self.ctrl_th.kd = float(self.ed_kd.text())
        self.ctrl_th.target = float(self.ed_sp.text())
        self.ctrl_th.max_step = float(self.ed_step.text())
        self.ctrl_th.unwrap = self.chk_unwrap.isChecked()
        self.ctrl_th.gain = float(self.ed_gain.text())
        try:
            self.ctrl_th.vmin = float(self.ed_vmin.text())
            self.ctrl_th.vmax = float(self.ed_vmax.text())
        except:
            pass
        self._log(f"PID updated (V: {self.ctrl_th.vmin:.1f}-{self.ctrl_th.vmax:.1f}V)")

    def _update_ylimits(self):
        xmin_fft = self.ed_fft_xmin.text().strip()
        xmax_fft = self.ed_fft_xmax.text().strip()
        if xmin_fft and xmax_fft:
            try:
                self.ax_fft.set_xlim(float(xmin_fft), float(xmax_fft))
            except:
                pass
        
        ymin_fft = self.ed_fft_ymin.text().strip()
        ymax_fft = self.ed_fft_ymax.text().strip()
        if ymin_fft and ymax_fft:
            try:
                self.ax_fft.set_ylim(float(ymin_fft), float(ymax_fft))
            except:
                pass
        
        ymin_phase = self.ed_phase_ymin.text().strip()
        ymax_phase = self.ed_phase_ymax.text().strip()
        if ymin_phase and ymax_phase:
            try:
                self.ax_phase.set_ylim(float(ymin_phase), float(ymax_phase))
            except:
                pass
        
        self.blitting_initialized = False
        self.canvas.draw()
        self._log("Plot limits updated")

    def _update_voltage_limits(self):
        if not self.ctrl_th:
            self._log("No stage connected")
            return
        try:
            self.ctrl_th.vmin = float(self.ed_vmin.text())
            self.ctrl_th.vmax = float(self.ed_vmax.text())
            self._log(f"Voltage limits updated: {self.ctrl_th.vmin:.1f}-{self.ctrl_th.vmax:.1f}V")
        except:
            self._log("Invalid voltage limits")

    def _start(self):
        if self.spec_ctrl is None:
            QMessageBox.critical(self, "Run", "No spectrometer")
            return
        try:
            self.max_points = int(self.ed_max_points.text())
        except:
            self.max_points = 100
        self.hist_phi_raw = deque(maxlen=self.max_points)
        self.hist_phi_unwrapped = deque(maxlen=self.max_points)
        self.ax_fft.clear()
        self.ax_phase.clear()
        self.fft_scatter = None
        self.fft_line = None
        self.phase_plot = None
        self.phase_sp_line = None
        self.blitting_initialized = False
        self.th = AvaspecThread(self.spec_ctrl)
        self.th.data_ready.connect(self._on_data)
        self.th.error.connect(self._on_err)
        self.th.start()
        self._log("Acquisition started")

    def _stop(self):
        if self.th:
            self.th.stop()
            self.th.wait()
            self.th = None
        self._log("Acquisition stopped")

    def _toggle_lock(self):
        if self.ctrl_th is None:
            return
        if self.chk_lock.isChecked():
            self.ctrl_th.integral = 0.0
            self.ctrl_th.last_t = None
            self.ctrl_th.last_err = 0.0
            self.ctrl_th.enabled = True
            self._log("Lock ON")
        else:
            self.ctrl_th.enabled = False
            self._log("Lock OFF")
            if self.phase_sp_line:
                self.phase_sp_line.set_visible(False)
                self.canvas.draw_idle()

    def _on_err(self, err):
        self._log(f"Thread error: {err}")
        self._stop()

    def _on_data(self, wl, y):
        if self.th is None:
            return
        now = time.monotonic()
        if now - self.last_draw < self.min_draw_dt:
            return
        self.last_draw = now

        y = np.asarray(y, float)
        if y.size < 32:
            return

        F = np.fft.fft(y)
        mag = np.abs(F)
        phase = np.angle(F)
        n = len(F)

        try:
            center = int(self.ed_center_idx.text())
        except:
            center = 100
        try:
            W = int(self.ed_window.text())
        except:
            W = 100

        i0 = max(0, center - W)
        i1 = min(n, center + W)

        phase_unwrapped = np.unwrap(phase)
        use_weighted = self.chk_weighted.isChecked()
        remove_ramp = self.chk_remove_ramp.isChecked()
        
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
                phi = phase_detrended[len(phase_detrended)//2] if len(phase_detrended) > 0 else 0.0
        else:
            center_in_window = center - i0
            if 0 <= center_in_window < len(phase_detrended):
                phi = phase_detrended[center_in_window]
            else:
                return

        try:
            plot_skip = max(1, int(self.ed_plot_skip.text()))
        except:
            plot_skip = 5
        
        x = np.arange(i0, i1, plot_skip)
        mag_decimated = mag[i0:i1:plot_skip]
        phase_unwrapped_decimated = phase_unwrapped[i0:i1:plot_skip]
        
        if self.fft_scatter is None:
            self.fft_scatter = self.ax_fft.scatter(
                x, mag_decimated, c=phase_unwrapped_decimated, cmap='hsv', 
                s=20, vmin=-np.pi, vmax=np.pi, animated=True
            )
            
            xmin_fft = self.ed_fft_xmin.text().strip()
            xmax_fft = self.ed_fft_xmax.text().strip()
            if xmin_fft and xmax_fft:
                try:
                    self.ax_fft.set_xlim(float(xmin_fft), float(xmax_fft))
                except:
                    self.ax_fft.set_xlim(i0, i1)
            else:
                self.ax_fft.set_xlim(i0, i1)
            
            self.ax_fft.set_xlabel("FFT Index")
            self.ax_fft.set_ylabel("Magnitude")
            self.ax_fft.set_ylim(0, 200000)
        else:
            self.fft_scatter.set_offsets(np.c_[x, mag_decimated])
            self.fft_scatter.set_array(phase_unwrapped_decimated)

        if self.fft_line is None:
            if use_weighted:
                ymax = np.max(mag[i0:i1]) if len(mag[i0:i1]) > 0 else 1
                self.fft_line = self.ax_fft.axvspan(i0, i1, alpha=0.2, color='orange', animated=True)
            else:
                self.fft_line = self.ax_fft.axvline(center, color='red', linestyle='--', animated=True)
        else:
            self.fft_line.remove()
            if use_weighted:
                ymax = np.max(mag[i0:i1]) if len(mag[i0:i1]) > 0 else 1
                self.fft_line = self.ax_fft.axvspan(i0, i1, alpha=0.2, color='orange', animated=True)
            else:
                self.fft_line = self.ax_fft.axvline(center, color='red', linestyle='--', animated=True)

        if self.chk_unwrap.isChecked():
            phi_u = phi if not self.hist_phi_unwrapped else np.unwrap([self.hist_phi_unwrapped[-1], phi])[-1]
        else:
            phi_u = phi

        self.hist_phi_raw.append(phi)
        self.hist_phi_unwrapped.append(phi_u)

        x_indices = np.arange(len(self.hist_phi_unwrapped))
        pp = np.array(self.hist_phi_unwrapped if self.chk_unwrap.isChecked() else self.hist_phi_raw)

        plot_color = "blue" if (self.ctrl_th and self.ctrl_th.enabled) else "black"
        
        if self.phase_plot is None:
            self.phase_plot, = self.ax_phase.plot(x_indices, pp, "o", ms=3, color=plot_color, animated=True)
            self.ax_phase.set_xlim(-1, self.max_points)
            self.ax_phase.set_xlabel("Sample")
            self.ax_phase.set_ylabel("Phase [rad]")
            ymin_phase = self.ed_phase_ymin.text().strip()
            ymax_phase = self.ed_phase_ymax.text().strip()
            if ymin_phase and ymax_phase:
                try:
                    self.ax_phase.set_ylim(float(ymin_phase), float(ymax_phase))
                except:
                    pass
        else:
            self.phase_plot.set_xdata(x_indices)
            self.phase_plot.set_ydata(pp)
            self.phase_plot.set_color(plot_color)

        if self.ctrl_th and self.ctrl_th.enabled:
            try:
                sp = float(self.ed_sp.text())
                if self.phase_sp_line is None:
                    self.phase_sp_line = self.ax_phase.axhline(sp, color="orange", linestyle="--", animated=True)
                else:
                    self.phase_sp_line.set_ydata([sp, sp])
                    self.phase_sp_line.set_visible(True)
            except:
                pass
        else:
            if self.phase_sp_line:
                self.phase_sp_line.set_visible(False)

        if not self.blitting_initialized:
            self.canvas.draw()
            self.bg_fft = self.canvas.copy_from_bbox(self.ax_fft.bbox)
            self.bg_phase = self.canvas.copy_from_bbox(self.ax_phase.bbox)
            self.blitting_initialized = True

        self.canvas.restore_region(self.bg_fft)
        self.ax_fft.draw_artist(self.fft_scatter)
        self.ax_fft.draw_artist(self.fft_line)
        self.canvas.blit(self.ax_fft.bbox)

        self.canvas.restore_region(self.bg_phase)
        self.ax_phase.draw_artist(self.phase_plot)
        if self.phase_sp_line and self.phase_sp_line.get_visible():
            self.ax_phase.draw_artist(self.phase_sp_line)
        self.canvas.blit(self.ax_phase.bbox)

        if self.ctrl_th and self.ctrl_th.enabled:
            try:
                sp = float(self.ed_sp.text())
                current_error = abs(phi_u - sp)
                self.lbl_error.setText(f"Error = {current_error:.4f} rad")
            except:
                self.lbl_error.setText("Error = — rad")
        else:
            self.lbl_error.setText("Error = — rad")
        
        self.lbl_slope.setText(f"Slope = {phase_slope:.4f} rad/index")

        if self.ctrl_th and self.ctrl_th.enabled:
            self.ctrl_th.q.append(phi)

        self.lbl_phase.setText(f"φ = {phi_u:+.3f} rad")

    def _on_control_update(self, phi, v):
        self.lbl_phase.setText(f"φ = {phi:+.3f} rad   V={v:+.3f}")


class PhaseLockApp(QWidget):
    def __init__(self, registry_key: str = "phaselock:avaspec"):
        super().__init__()
        self.setWindowTitle("Phase Locking Control")
        self.resize(800, 800)
        
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        
        self.phase_tab = AvaspecPhaseLockTab(registry_key=registry_key)
        tabs.addTab(self.phase_tab, "Control")
        
        self.log_tab = QWidget()
        log_layout = QVBoxLayout(self.log_tab)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        log_layout.addWidget(self.log)
        tabs.addTab(self.log_tab, "Log")
        
        self.phase_tab.log = self.log
        
        layout.addWidget(tabs)


def main():
    app = QApplication(sys.argv)
    w = PhaseLockApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()