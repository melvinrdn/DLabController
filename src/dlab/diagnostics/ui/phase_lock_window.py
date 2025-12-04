from __future__ import annotations
import sys, time, datetime
import numpy as np
from collections import deque

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QMessageBox,
    QCheckBox, QComboBox, QTabWidget
)
from PyQt5.QtGui import QDoubleValidator

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from dlab.core.device_registry import REGISTRY
from dlab.hardware.wrappers.piezojena_controller import NV40


class ControlThread(QThread):
    update_status = pyqtSignal(float, float)

    def __init__(self, stage: NV40):
        super().__init__()
        self.stage = stage

        self.kp = 0.0
        self.ki = 0.0
        self.kd = 0.0
        self.gain = 1.0
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
            self.current_v = 130.0

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
            
            gain = getattr(self, "gain", 1.0)

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

        self.fft_plot = None
        self.fft_line = None
        self.fft_label = None

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

        row = QHBoxLayout()
        self.ed_spec_key = QLineEdit("spectrometer:avaspec:spec_1")
        self.ed_stage_key = QLineEdit("stage:piezojena:nv40")
        row.addWidget(QLabel("Spec:"))
        row.addWidget(self.ed_spec_key)
        b1 = QPushButton("Connect Spec")
        b1.clicked.connect(self._connect_spec)
        row.addWidget(b1)

        row.addWidget(QLabel("NV40:"))
        row.addWidget(self.ed_stage_key)
        b2 = QPushButton("Connect NV40")
        b2.clicked.connect(self._connect_stage)
        row.addWidget(b2)
        root.addLayout(row)

        row = QHBoxLayout()
        self.ed_center_idx = QLineEdit("305")
        self.ed_window = QLineEdit("150")
        self.ed_fft_ymin = QLineEdit("")
        self.ed_fft_ymax = QLineEdit("")
        row.addWidget(QLabel("FFT center:"))
        row.addWidget(self.ed_center_idx)
        row.addWidget(QLabel("Half window:"))
        row.addWidget(self.ed_window)
        row.addWidget(QLabel("FFT Ymin:"))
        row.addWidget(self.ed_fft_ymin)
        row.addWidget(QLabel("FFT Ymax:"))
        row.addWidget(self.ed_fft_ymax)
        root.addLayout(row)

        row = QHBoxLayout()
        self.ed_phase_ymin = QLineEdit("-4")
        self.ed_phase_ymax = QLineEdit("4")
        self.ed_max_points = QLineEdit("100")
        row.addWidget(QLabel("Phase Ymin:"))
        row.addWidget(self.ed_phase_ymin)
        row.addWidget(QLabel("Phase Ymax:"))
        row.addWidget(self.ed_phase_ymax)
        row.addWidget(QLabel("Max points:"))
        row.addWidget(self.ed_max_points)
        root.addLayout(row)

        row = QHBoxLayout()
        self.ed_vmin = QLineEdit("0.0")
        self.ed_vmax = QLineEdit("140.0")
        self.ed_vstart = QLineEdit("130.0")
        row.addWidget(QLabel("V min [V]:"))
        row.addWidget(self.ed_vmin)
        row.addWidget(QLabel("V max [V]:"))
        row.addWidget(self.ed_vmax)
        row.addWidget(QLabel("V start [V]:"))
        row.addWidget(self.ed_vstart)
        root.addLayout(row)

        row = QHBoxLayout()
        self.ed_sp = QLineEdit("0.0")
        self.ed_sp.setValidator(QDoubleValidator())
        self.ed_kp = QLineEdit("0.05")
        self.ed_ki = QLineEdit("0.0")
        self.ed_kd = QLineEdit("0.0")
        self.ed_gain = QLineEdit("1.0")
        self.ed_step = QLineEdit("0.03")
        self.chk_unwrap = QCheckBox("Unwrapped")
        self.chk_unwrap.setChecked(True)
        self.chk_lock = QCheckBox("Lock")

        row.addWidget(QLabel("φ0:")); row.addWidget(self.ed_sp)
        row.addWidget(QLabel("Kp:")); row.addWidget(self.ed_kp)
        row.addWidget(QLabel("Ki:")); row.addWidget(self.ed_ki)
        row.addWidget(QLabel("Kd:")); row.addWidget(self.ed_kd)
        row.addWidget(QLabel("Gain:")); row.addWidget(self.ed_gain)
        row.addWidget(QLabel("Max step:")); row.addWidget(self.ed_step)
        row.addWidget(self.chk_unwrap)
        row.addWidget(self.chk_lock)
        root.addLayout(row)

        row = QHBoxLayout()
        self.btn_update_sp = QPushButton("Update Setpoint")
        self.btn_update_sp.clicked.connect(self._update_setpoint)
        self.btn_update_pid = QPushButton("Update PID Params")
        self.btn_update_pid.clicked.connect(self._update_pid_params)
        self.btn_update_ylim = QPushButton("Update Y Limits")
        self.btn_update_ylim.clicked.connect(self._update_ylimits)
        self.btn_test_stability = QPushButton("Test Stability")
        self.btn_test_stability.clicked.connect(self._toggle_stability_test)
        row.addWidget(self.btn_update_sp)
        row.addWidget(self.btn_update_pid)
        row.addWidget(self.btn_update_ylim)
        row.addWidget(self.btn_test_stability)
        root.addLayout(row)

        row = QHBoxLayout()
        self.lbl_error = QLabel("Error = — rad")
        row.addWidget(self.lbl_error)
        root.addLayout(row)

        row = QHBoxLayout()
        self.ed_stab_start = QLineEdit("-3.14159")
        self.ed_stab_end = QLineEdit("3.14159")
        self.ed_stab_steps = QLineEdit("10")
        self.ed_stab_wait = QLineEdit("5.0")
        row.addWidget(QLabel("Stability test - Start [rad]:"))
        row.addWidget(self.ed_stab_start)
        row.addWidget(QLabel("End [rad]:"))
        row.addWidget(self.ed_stab_end)
        row.addWidget(QLabel("Steps:"))
        row.addWidget(self.ed_stab_steps)
        row.addWidget(QLabel("Wait [s]:"))
        row.addWidget(self.ed_stab_wait)
        root.addLayout(row)

        row = QHBoxLayout()
        b3 = QPushButton("Start")
        b4 = QPushButton("Stop")
        b3.clicked.connect(self._start)
        b4.clicked.connect(self._stop)
        self.lbl_phase = QLabel("φ = — rad")
        row.addWidget(b3)
        row.addWidget(b4)
        row.addStretch(1)
        row.addWidget(self.lbl_phase)
        root.addLayout(row)

        self.fig, (self.ax_fft, self.ax_phase) = plt.subplots(1, 2, figsize=(14,5))
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 10)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        root.addWidget(self.log)

        self.chk_lock.stateChanged.connect(self._toggle_lock)
        
        self.stability_timer = QTimer()
        self.stability_timer.timeout.connect(self._stability_test_step)
        
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
            self.ctrl_th.target = float(target_rad)
            self.ed_sp.setText(f"{float(target_rad):.6f}")
    
    def is_locked(self) -> bool:
        return self.ctrl_th is not None and self.ctrl_th.enabled

    def _log(self, msg):
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
        self._log(f"Connected NV40 (limits: {self.ctrl_th.vmin:.1f}-{self.ctrl_th.vmax:.1f}V, start: {self.ctrl_th.current_v:.1f}V)")

    def _update_setpoint(self):
        if not self.ctrl_th:
            return
        try:
            new_sp = float(self.ed_sp.text())
            self.ctrl_th.target = new_sp
            self._log(f"Setpoint updated to {new_sp:.3f} rad")
        except:
            pass

    def _toggle_stability_test(self):
        if self.stability_test_active:
            self.stability_test_active = False
            self.stability_timer.stop()
            self.btn_test_stability.setText("Test Stability")
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
            self.btn_test_stability.setText("Stop Stability Test")
            self._log(f"Stability test started: {num_steps} steps from {start_val:.3f} to {end_val:.3f} rad, wait={wait_time}s")

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
        self._log(f"Stability test step {self.stability_step+1}/{len(self.stability_setpoints)}: φ0 = {new_sp:.3f} rad")

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

        self._log(f"PID params updated (V limits: {self.ctrl_th.vmin:.1f}-{self.ctrl_th.vmax:.1f}V)")

    def _update_ylimits(self):
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
        self._log("Y-axis limits updated")

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

        self.fft_plot = None
        self.fft_line = None
        self.fft_label = None

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

        x = np.arange(n)

        if self.fft_plot is None:
            self.fft_plot, = self.ax_fft.plot(x, mag, color='black', animated=True)
            self.ax_fft.set_xlim(i0, i1)
            
            ymin_fft = self.ed_fft_ymin.text().strip()
            ymax_fft_txt = self.ed_fft_ymax.text().strip()
            if ymin_fft and ymax_fft_txt:
                try:
                    self.ax_fft.set_ylim(float(ymin_fft), float(ymax_fft_txt))
                except:
                    pass
        else:
            self.fft_plot.set_ydata(mag)

        if self.fft_line is None:
            self.fft_line = self.ax_fft.axvline(center, color='red', linestyle='--', animated=True)
        else:
            self.fft_line.set_xdata([center, center])

        ymax_fft = max(1e-9, np.max(mag[i0:i1]))
        if self.fft_label is None:
            self.fft_label = self.ax_fft.text(
                center, ymax_fft, f"{center}",
                color='red', ha='center', va='bottom', fontsize=8, animated=True
            )
        else:
            self.fft_label.set_position((center, ymax_fft))

        if 0 <= center < n:
            phi = np.angle(F[center])
        else:
            return

        if self.chk_unwrap.isChecked():
            phi_u = phi if not self.hist_phi_unwrapped else np.unwrap([self.hist_phi_unwrapped[-1], phi])[-1]
        else:
            phi_u = phi

        self.hist_phi_raw.append(phi)
        self.hist_phi_unwrapped.append(phi_u)

        x_indices = np.arange(len(self.hist_phi_unwrapped))
        pp = np.array(self.hist_phi_unwrapped if self.chk_unwrap.isChecked()
                      else self.hist_phi_raw)

        if self.phase_plot is None:
            self.phase_plot, = self.ax_phase.plot(x_indices, pp, "o", ms=3, color="black", animated=True)
            self.ax_phase.set_xlim(-1, self.max_points)
            
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
        self.ax_fft.draw_artist(self.fft_plot)
        self.ax_fft.draw_artist(self.fft_line)
        self.ax_fft.draw_artist(self.fft_label)
        self.canvas.blit(self.ax_fft.bbox)

        self.canvas.restore_region(self.bg_phase)
        self.ax_phase.draw_artist(self.phase_plot)
        if self.phase_sp_line and self.phase_sp_line.get_visible():
            self.ax_phase.draw_artist(self.phase_sp_line)
        self.canvas.blit(self.ax_phase.bbox)

        if self.ctrl_th and self.ctrl_th.enabled:
            try:
                sp = float(self.ed_sp.text())
                errors = [abs(p - sp) for p in self.hist_phi_unwrapped]
                if len(errors) > 0:
                    mean_error = np.mean(errors)
                    self.lbl_error.setText(f"Mean Error = {mean_error:.4f} rad")
                else:
                    self.lbl_error.setText("Mean Error = — rad")
            except:
                self.lbl_error.setText("Mean Error = — rad")
        else:
            self.lbl_error.setText("Mean Error = — rad")

        if self.ctrl_th and self.ctrl_th.enabled:
            self.ctrl_th.q.append(phi)

        self.lbl_phase.setText(f"φ = {phi_u:+.3f} rad")

    def _on_control_update(self, phi, v):
        self.lbl_phase.setText(f"φ = {phi:+.3f} rad   V={v:+.3f}")


class PhaseLockApp(QWidget):
    def __init__(self, registry_key: str = "phaselock:avaspec"):
        super().__init__()
        self.setWindowTitle("Phase Locking")
        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        self.phase_tab = AvaspecPhaseLockTab(registry_key=registry_key)
        tabs.addTab(self.phase_tab, "Avaspec Spectrometer Lock")

        layout.addWidget(tabs)


def main():
        app = QApplication(sys.argv)
        w = PhaseLockApp()
        w.resize(1700, 950)
        w.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()