from __future__ import annotations
import sys
import time
import datetime
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QMessageBox,
    QCheckBox,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from dlab.core.device_registry import REGISTRY


class PhaseMeasureThread(QThread):
    data_ready = pyqtSignal(float, object, object)
    error = pyqtSignal(str)

    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self._running = True

    def run(self):
        while self._running and not self.isInterruptionRequested():
            try:
                ts, wl, counts = self.ctrl.measure_once()
                self.data_ready.emit(ts, wl, counts)
            except Exception as e:
                self.error.emit(str(e))
                break

    def stop(self):
        self._running = False
        self.requestInterruption()


class AvaspecPhaseLock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Avaspec Phase Lock")
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.spec_ctrl = None
        self.stage = None
        self.th = None

        self._fft_freq = None
        self._fft_peak_idx = None
        self._fft_peak_freq = None

        self._peak_hist_t = []
        self._peak_hist_phi = []
        self._t0 = None

        self._fft_line = None
        self._peak_scatter = None
        self._fft_peak_vline = None

        self._lock_enabled = False
        self._last_ctrl_t = None
        self._integral = 0.0

        self._min_draw_dt = 0.05
        self._last_draw = 0.0

        self._mpl_cid_click = None

        self._build_ui()
        self.resize(1100, 700)

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Registry keys row
        row_keys = QHBoxLayout()
        self.ed_spec_key = QLineEdit("spectrometer:avaspec:spec_1")
        self.ed_stage_key = QLineEdit("stage:delay")
        self.btn_spec_connect = QPushButton("Connect spectrometer")
        self.btn_stage_connect = QPushButton("Connect stage")
        self.btn_spec_connect.clicked.connect(self._connect_spec)
        self.btn_stage_connect.clicked.connect(self._connect_stage)
        row_keys.addWidget(QLabel("Spec key:"))
        row_keys.addWidget(self.ed_spec_key, 1)
        row_keys.addWidget(self.btn_spec_connect)
        row_keys.addSpacing(10)
        row_keys.addWidget(QLabel("Stage key:"))
        row_keys.addWidget(self.ed_stage_key, 1)
        row_keys.addWidget(self.btn_stage_connect)
        root.addLayout(row_keys)

        # Phase / FFT controls
        row_ctrl = QHBoxLayout()
        self.ed_fft_peak = QLineEdit("")
        self.ed_fft_peak.setPlaceholderText("ν₀ (1/nm)")
        self.ed_fft_peak.setFixedWidth(90)
        self.btn_set_peak = QPushButton("Set ν₀")
        self.btn_set_peak.clicked.connect(self._on_set_peak_freq)

        self.ed_twin = QLineEdit("30")
        self.ed_twin.setFixedWidth(60)

        self.chk_fft_zoom = QCheckBox("FFT zoom")
        self.ed_fft_c = QLineEdit("")
        self.ed_fft_c.setPlaceholderText("νc")
        self.ed_fft_c.setFixedWidth(80)
        self.ed_fft_pm = QLineEdit("0.01")
        self.ed_fft_pm.setFixedWidth(60)

        row_ctrl.addWidget(QLabel("Track:"))
        row_ctrl.addWidget(self.ed_fft_peak)
        row_ctrl.addWidget(self.btn_set_peak)
        row_ctrl.addSpacing(15)
        row_ctrl.addWidget(QLabel("Time window (s):"))
        row_ctrl.addWidget(self.ed_twin)
        row_ctrl.addSpacing(15)
        row_ctrl.addWidget(self.chk_fft_zoom)
        row_ctrl.addWidget(QLabel("νc:"))
        row_ctrl.addWidget(self.ed_fft_c)
        row_ctrl.addWidget(QLabel("±"))
        row_ctrl.addWidget(self.ed_fft_pm)
        row_ctrl.addStretch(1)
        root.addLayout(row_ctrl)

        # PID controls
        row_pid = QHBoxLayout()
        self.ed_setpoint = QLineEdit("0.0")
        self.ed_setpoint.setFixedWidth(70)
        self.ed_kp = QLineEdit("0.1")
        self.ed_kp.setFixedWidth(60)
        self.ed_ki = QLineEdit("0.0")
        self.ed_ki.setFixedWidth(60)
        self.ed_gain = QLineEdit("1.0")
        self.ed_gain.setFixedWidth(70)
        self.ed_max_step = QLineEdit("0.1")
        self.ed_max_step.setFixedWidth(70)

        self.chk_lock = QCheckBox("Lock enabled")
        self.chk_lock.stateChanged.connect(self._on_lock_toggle)

        row_pid.addWidget(QLabel("Setpoint φ₀ (rad):"))
        row_pid.addWidget(self.ed_setpoint)
        row_pid.addSpacing(10)
        row_pid.addWidget(QLabel("Kp:"))
        row_pid.addWidget(self.ed_kp)
        row_pid.addWidget(QLabel("Ki:"))
        row_pid.addWidget(self.ed_ki)
        row_pid.addSpacing(10)
        row_pid.addWidget(QLabel("Gain (stage_units/rad):"))
        row_pid.addWidget(self.ed_gain)
        row_pid.addSpacing(10)
        row_pid.addWidget(QLabel("Max step:"))
        row_pid.addWidget(self.ed_max_step)
        row_pid.addSpacing(15)
        row_pid.addWidget(self.chk_lock)
        row_pid.addStretch(1)
        root.addLayout(row_pid)

        # Start/stop
        row_run = QHBoxLayout()
        self.btn_start = QPushButton("Start loop")
        self.btn_stop = QPushButton("Stop loop")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._start_loop)
        self.btn_stop.clicked.connect(self._stop_loop)
        self.lbl_phase = QLabel("φ = — rad, err = — rad")
        self.lbl_phase.setMinimumWidth(260)
        self.lbl_phase.setAlignment(Qt.AlignCenter)
        self.lbl_phase.setStyleSheet("QLabel { border: 1px solid #888; padding: 4px; }")

        row_run.addWidget(self.btn_start)
        row_run.addWidget(self.btn_stop)
        row_run.addStretch(1)
        row_run.addWidget(self.lbl_phase)
        root.addLayout(row_run)

        # Figure: FFT and phase vs time
        self.fig, (self.ax_fft, self.ax_phase) = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [2, 2]}
        )
        self.ax_fft.set_xlabel("Frequency (1/nm)")
        self.ax_fft.set_ylabel("|FFT|")
        self.ax_fft.grid(True)

        self.ax_phase.set_xlabel("Time (s)")
        self.ax_phase.set_ylabel("arg(FFT(ν₀)) (rad)")
        self.ax_phase.grid(True)
        self.ax_phase.set_ylim(-np.pi, np.pi)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            self.canvas.sizePolicy().Expanding,
            self.canvas.sizePolicy().Expanding,
        )
        root.addWidget(self.canvas, 10)

        self._mpl_cid_click = self.canvas.mpl_connect(
            "button_press_event", self._on_canvas_click
        )

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        self.log.setStyleSheet(
            "QTextEdit { font-family: 'Fira Mono', 'Consolas', monospace; font-size: 11px; }"
        )
        root.addWidget(self.log, 0)

    # ---------- registry / connect ----------

    def _connect_spec(self):
        key = self.ed_spec_key.text().strip()
        if not key:
            QMessageBox.warning(self, "Phase lock", "Empty spectrometer key.")
            return
        ctrl = REGISTRY.get(key)
        if ctrl is None:
            QMessageBox.critical(self, "Phase lock", f"No object in registry at '{key}'.")
            return
        self.spec_ctrl = ctrl
        self._log(f"Connected spectrometer from registry key '{key}'.")

    def _connect_stage(self):
        key = self.ed_stage_key.text().strip()
        if not key:
            QMessageBox.warning(self, "Phase lock", "Empty stage key.")
            return
        st = REGISTRY.get(key)
        if st is None:
            QMessageBox.critical(self, "Phase lock", f"No object in registry at '{key}'.")
            return
        self.stage = st
        self._log(f"Connected stage from registry key '{key}'.")

    # ---------- UI helpers ----------

    def _on_lock_toggle(self, state):
        self._lock_enabled = state == Qt.Checked
        if self._lock_enabled:
            self._log("Lock enabled.")
            self._integral = 0.0
            self._last_ctrl_t = None
        else:
            self._log("Lock disabled.")

    def _on_canvas_click(self, event):
        if event.button != 1 or event.inaxes is not self.ax_fft:
            return
        if self._fft_freq is None or self._fft_freq.size == 0:
            return
        if event.xdata is None:
            return
        x = float(event.xdata)
        idx = int(np.argmin(np.abs(self._fft_freq - x)))
        if idx < 0 or idx >= self._fft_freq.size:
            return
        self._set_peak_idx(idx)

    def _on_set_peak_freq(self):
        if self._fft_freq is None or self._fft_freq.size == 0:
            self._log("FFT not available yet.")
            return
        try:
            f0 = float(self.ed_fft_peak.text())
        except Exception:
            self._log("Invalid ν₀ value.")
            return
        idx = int(np.argmin(np.abs(self._fft_freq - f0)))
        if idx < 0 or idx >= self._fft_freq.size:
            self._log("ν₀ out of FFT range.")
            return
        self._set_peak_idx(idx)

    def _set_peak_idx(self, idx):
        self._fft_peak_idx = idx
        self._fft_peak_freq = float(self._fft_freq[idx])
        self.ed_fft_peak.setText(f"{self._fft_peak_freq:.6g}")

        self._peak_hist_t = []
        self._peak_hist_phi = []
        self._t0 = time.monotonic()

        if self._fft_peak_vline is None:
            self._fft_peak_vline = self.ax_fft.axvline(
                self._fft_peak_freq, color="r", linestyle="--", linewidth=1.0
            )
        else:
            self._fft_peak_vline.set_xdata(
                [self._fft_peak_freq, self._fft_peak_freq]
            )

        self._log(f"Tracking FFT phase at ν₀ ≈ {self._fft_peak_freq:.4g} 1/nm")
        self.canvas.draw_idle()

    def _log(self, msg):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{t}] {msg}")

    # ---------- loop control ----------

    def _start_loop(self):
        if self.spec_ctrl is None:
            QMessageBox.critical(self, "Phase lock", "No spectrometer connected.")
            return
        if self.th is not None:
            self._stop_loop()

        self._peak_hist_t = []
        self._peak_hist_phi = []
        self._t0 = time.monotonic()
        self._last_ctrl_t = None
        self._integral = 0.0

        self.th = PhaseMeasureThread(self.spec_ctrl)
        self.th.data_ready.connect(self._on_data, Qt.QueuedConnection)
        self.th.error.connect(self._on_thread_error, Qt.QueuedConnection)
        self.th.finished.connect(self._on_thread_finished, Qt.QueuedConnection)
        self.th.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._log("Loop started.")

    def _stop_loop(self):
        th = self.th
        self.th = None
        if th is not None:
            try:
                th.stop()
                th.wait(2000)
            finally:
                try:
                    th.finished.disconnect(self._on_thread_finished)
                except Exception:
                    pass
                try:
                    th.error.disconnect(self._on_thread_error)
                except Exception:
                    pass
                try:
                    th.data_ready.disconnect(self._on_data)
                except Exception:
                    pass

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._log("Loop stopped.")

    def _on_thread_finished(self):
        if self.th is None:
            self._log("Thread finished.")

    def _on_thread_error(self, err):
        self._log(f"Thread error: {err}")
        QMessageBox.critical(self, "Acquisition error", err)
        self._stop_loop()

    # ---------- FFT / phase / control ----------

    def _apply_fft_zoom(self, freq):
        if not self.chk_fft_zoom.isChecked():
            self.ax_fft.autoscale_view()
            return
        try:
            fc = float(self.ed_fft_c.text())
            pm = float(self.ed_fft_pm.text())
        except Exception:
            self.ax_fft.autoscale_view()
            return
        if pm <= 0:
            self.ax_fft.autoscale_view()
            return
        freq = np.asarray(freq, float).ravel()
        if freq.size < 2:
            return
        fmin, fmax = float(freq.min()), float(freq.max())
        x0 = max(fmin, fc - pm)
        x1 = min(fmax, fc + pm)
        if x1 <= x0:
            self.ax_fft.set_xlim(fmin, fmax)
        else:
            self.ax_fft.set_xlim(x0, x1)

    @staticmethod
    def _wrap_phase(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def _update_fft_and_phase(self, wl, y):
        wl = np.asarray(wl, float).ravel()
        y = np.asarray(y, float).ravel()
        if wl.size < 4 or y.size != wl.size:
            return

        dlam = np.mean(np.diff(wl))
        if not np.isfinite(dlam) or dlam == 0:
            return

        N = wl.size
        freq = np.fft.rfftfreq(N, d=dlam)
        Y = np.fft.rfft(y)
        mag = np.abs(Y)
        phase = np.angle(Y)

        self._fft_freq = freq

        # FFT amplitude
        if self._fft_line is None:
            self.ax_fft.cla()
            self.ax_fft.set_xlabel("Frequency (1/nm)")
            self.ax_fft.set_ylabel("|FFT|")
            self.ax_fft.grid(True)
            (self._fft_line,) = self.ax_fft.plot(freq, mag, lw=1.0)
        else:
            self._fft_line.set_xdata(freq)
            self._fft_line.set_ydata(mag)

        self._apply_fft_zoom(freq)

        # Phase tracking
        if self._fft_peak_idx is None or self._fft_peak_idx < 0 or self._fft_peak_idx >= freq.size:
            self.canvas.draw_idle()
            return

        self._fft_peak_freq = float(freq[self._fft_peak_idx])

        if self._fft_peak_vline is None:
            self._fft_peak_vline = self.ax_fft.axvline(
                self._fft_peak_freq, color="r", linestyle="--", linewidth=1.0
            )
        else:
            self._fft_peak_vline.set_xdata(
                [self._fft_peak_freq, self._fft_peak_freq]
            )

        if self._t0 is None:
            self._t0 = time.monotonic()
        t_rel = time.monotonic() - self._t0

        phi_val = float(phase[self._fft_peak_idx])
        phi_val = self._wrap_phase(phi_val)

        self._peak_hist_t.append(t_rel)
        self._peak_hist_phi.append(phi_val)

        try:
            twin = float(self.ed_twin.text())
        except Exception:
            twin = 30.0
        if twin <= 0:
            twin = 30.0

        tt = np.asarray(self._peak_hist_t, float)
        pp = np.asarray(self._peak_hist_phi, float)
        if tt.size == 0:
            return

        t_max = tt.max()
        t_min_win = max(0.0, t_max - twin)
        mask = tt >= t_min_win
        tt = tt[mask]
        pp = pp[mask]
        self._peak_hist_t = tt.tolist()
        self._peak_hist_phi = pp.tolist()

        if tt.size:
            if t_max > twin:
                t_win = tt - (t_max - twin)
            else:
                t_win = tt
        else:
            t_win = tt

        if self._peak_scatter is None:
            self.ax_phase.cla()
            self.ax_phase.set_xlabel("Time (s)")
            self.ax_phase.set_ylabel("arg(FFT(ν₀)) (rad)")
            self.ax_phase.grid(True)
            self.ax_phase.set_ylim(-np.pi, np.pi)
            self._peak_scatter = self.ax_phase.plot(
                t_win,
                pp,
                linestyle="None",
                marker=".",
                markersize=4,
            )[0]
        else:
            self._peak_scatter.set_xdata(t_win)
            self._peak_scatter.set_ydata(pp)

        self.ax_phase.set_xlim(0.0, twin)
        self.ax_phase.set_ylim(-np.pi, np.pi)

        self._apply_control(phi_val)
        self.canvas.draw_idle()

    def _apply_control(self, phi_val):
        try:
            sp = float(self.ed_setpoint.text())
            kp = float(self.ed_kp.text())
            ki = float(self.ed_ki.text())
            gain = float(self.ed_gain.text())
            max_step = float(self.ed_max_step.text())
        except Exception:
            return

        err = self._wrap_phase(sp - phi_val)

        now = time.monotonic()
        if self._last_ctrl_t is None:
            dt = 0.0
        else:
            dt = max(0.0, now - self._last_ctrl_t)
        self._last_ctrl_t = now

        self._integral += err * dt
        u = kp * err + ki * self._integral
        step = gain * u

        if max_step > 0:
            if step > max_step:
                step = max_step
            elif step < -max_step:
                step = -max_step

        self.lbl_phase.setText(f"φ = {phi_val:+.3f} rad, err = {err:+.3f} rad")

        if not self._lock_enabled:
            return
        if self.stage is None:
            return

        try:
            self._move_stage(step)
        except Exception as e:
            self._log(f"Stage move error: {e}")

    def _move_stage(self, step):
        st = self.stage
        if hasattr(st, "move_relative"):
            st.move_relative(step)
        elif hasattr(st, "move_rel"):
            st.move_rel(step)
        elif hasattr(st, "move_to"):
            pos = None
            if hasattr(st, "get_position"):
                pos = st.get_position()
            elif hasattr(st, "position"):
                pos = st.position
            if pos is not None:
                st.move_to(pos + step)

    def _on_data(self, ts, wl, counts):
        now = time.monotonic()
        wl = np.asarray(wl, float).ravel()
        counts = np.asarray(counts, float).ravel()

        if (now - self._last_draw) < self._min_draw_dt:
            return
        self._last_draw = now

        y = counts
        self._update_fft_and_phase(wl, y)

    # ---------- shutdown ----------

    def _safe_shutdown(self):
        self._stop_loop()
        try:
            if self._mpl_cid_click is not None:
                self.canvas.mpl_disconnect(self._mpl_cid_click)
        except Exception:
            pass
        try:
            self.ax_fft.cla()
            self.ax_phase.cla()
            self.canvas.draw_idle()
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            self._safe_shutdown()
        finally:
            super().closeEvent(event)

    def __del__(self):
        try:
            self._safe_shutdown()
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    w = AvaspecPhaseLock()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
