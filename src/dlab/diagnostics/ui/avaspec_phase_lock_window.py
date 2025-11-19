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
from dlab.hardware.wrappers.piezojena_controller import NV40


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
        self.stage: NV40 | None = None
        self.th = None

        self._fft_freq = None
        self._fft_peak_idx = None
        self._fft_peak_freq = None

        self._peak_hist_t: list[float] = []
        self._peak_hist_phi: list[float] = []
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

        # NV40
        self._vmin, self._vmax = NV40.get_voltage_limits()
        self._stage_v: float | None = None
        self._phi_last: float | None = None  # unwrapped last phase

        self._build_ui()
        self.resize(1100, 700)

    def _build_ui(self):
        root = QVBoxLayout(self)

        row_keys = QHBoxLayout()
        self.ed_spec_key = QLineEdit("spectrometer:avaspec:spec_1")
        self.ed_stage_key = QLineEdit("stage:piezojena:nv40")
        self.btn_spec_connect = QPushButton("Connect spectrometer")
        self.btn_stage_connect = QPushButton("Connect NV40")
        self.btn_spec_connect.clicked.connect(self._connect_spec)
        self.btn_stage_connect.clicked.connect(self._connect_stage)
        row_keys.addWidget(QLabel("Spec key:"))
        row_keys.addWidget(self.ed_spec_key, 1)
        row_keys.addWidget(self.btn_spec_connect)
        row_keys.addSpacing(10)
        row_keys.addWidget(QLabel("NV40 key:"))
        row_keys.addWidget(self.ed_stage_key, 1)
        row_keys.addWidget(self.btn_stage_connect)
        root.addLayout(row_keys)

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

        row_pid = QHBoxLayout()
        self.ed_setpoint = QLineEdit("0.0")
        self.ed_setpoint.setFixedWidth(70)
        self.ed_kp = QLineEdit("0.1")
        self.ed_kp.setFixedWidth(60)
        self.ed_ki = QLineEdit("0.0")
        self.ed_ki.setFixedWidth(60)
        self.ed_gain = QLineEdit("1.0")      # V/rad
        self.ed_gain.setFixedWidth(70)
        self.ed_max_step = QLineEdit("0.1")  # V
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
        row_pid.addWidget(QLabel("Gain (V/rad):"))
        row_pid.addWidget(self.ed_gain)
        row_pid.addSpacing(10)
        row_pid.addWidget(QLabel("Max step (V):"))
        row_pid.addWidget(self.ed_max_step)
        row_pid.addSpacing(15)
        row_pid.addWidget(self.chk_lock)
        row_pid.addStretch(1)
        root.addLayout(row_pid)

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

        self.fig, (self.ax_fft, self.ax_phase) = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [2, 2]}
        )
        self.ax_fft.set_xlabel("Frequency (1/nm)")
        self.ax_fft.set_ylabel("|FFT|")
        self.ax_fft.grid(True)

        self.ax_phase.set_xlabel("Time (s)")
        self.ax_phase.set_ylabel("arg(FFT(ν₀)) (rad)")
        self.ax_phase.grid(True)

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            self.canvas.sizePolicy().Expanding,
            self.canvas.sizePolicy().Expanding,
        )
        root.addWidget(self.canvas, 10)

        self._mpl_cid_click = self.canvas.mpl_connect(
            "button_press_event", self._on_canvas_click
        )

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
            QMessageBox.warning(self, "Phase lock", "Empty NV40 key.")
            return
        st = REGISTRY.get(key)
        if st is None:
            QMessageBox.critical(self, "Phase lock", f"No object in registry at '{key}'.")
            return
        if not hasattr(st, "set_position"):
            QMessageBox.critical(
                self,
                "Phase lock",
                f"Object at '{key}' has no set_position().",
            )
            return
        self.stage = st

        # Optionnel : une seule read au connect pour initialiser self._stage_v
        v0 = 0.0
        if hasattr(st, "get_position"):
            try:
                v0 = float(st.get_position())
            except Exception:
                v0 = 0.0
        self._stage_v = v0
        self._log(f"Connected NV40 stage from registry key '{key}', V ≈ {v0:.3f}.")

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
        if 0 <= idx < self._fft_freq.size:
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
        self._phi_last = None

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

    def _log(self, msg: str):
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
        self._phi_last = None

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
                for sig, slot in (
                    (th.finished, self._on_thread_finished),
                    (th.error, self._on_thread_error),
                    (th.data_ready, self._on_data),
                ):
                    try:
                        sig.disconnect(slot)
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

    def _apply_fft_zoom(self, freq, mag):
        freq = np.asarray(freq, float).ravel()
        mag = np.asarray(mag, float).ravel()
        if freq.size == 0 or mag.size != freq.size:
            return

        if not self.chk_fft_zoom.isChecked():
            fmin, fmax = float(freq.min()), float(freq.max())
            self.ax_fft.set_xlim(fmin, fmax)
            self.ax_fft.relim()
            self.ax_fft.autoscale_view(scalex=False, scaley=True)
            return

        try:
            fc = float(self.ed_fft_c.text())
            pm = float(self.ed_fft_pm.text())
        except Exception:
            fmin, fmax = float(freq.min()), float(freq.max())
            self.ax_fft.set_xlim(fmin, fmax)
            self.ax_fft.relim()
            self.ax_fft.autoscale_view(scalex=False, scaley=True)
            return

        if pm <= 0:
            fmin, fmax = float(freq.min()), float(freq.max())
            self.ax_fft.set_xlim(fmin, fmax)
            self.ax_fft.relim()
            self.ax_fft.autoscale_view(scalex=False, scaley=True)
            return

        fmin, fmax = float(freq.min()), float(freq.max())
        x0 = max(fmin, fc - pm)
        x1 = min(fmax, fc + pm)
        if x1 <= x0:
            self.ax_fft.set_xlim(fmin, fmax)
            self.ax_fft.relim()
            self.ax_fft.autoscale_view(scalex=False, scaley=True)
            return

        self.ax_fft.set_xlim(x0, x1)

        mask = (freq >= x0) & (freq <= x1)
        m_sel = mag[mask]
        if m_sel.size == 0:
            self.ax_fft.relim()
            self.ax_fft.autoscale_view(scalex=False, scaley=True)
            return

        ymax = float(m_sel.max())
        if ymax <= 0:
            self.ax_fft.set_ylim(0.0, 1.0)
        else:
            self.ax_fft.set_ylim(0.0, 1.05 * ymax)


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

        if self._fft_line is None:
            self.ax_fft.cla()
            self.ax_fft.set_xlabel("Frequency (1/nm)")
            self.ax_fft.set_ylabel("|FFT|")
            self.ax_fft.grid(True)
            (self._fft_line,) = self.ax_fft.plot(freq, mag, lw=1.0)
        else:
            self._fft_line.set_xdata(freq)
            self._fft_line.set_ydata(mag)

        self._apply_fft_zoom(freq, mag)

        if (
            self._fft_peak_idx is None
            or self._fft_peak_idx < 0
            or self._fft_peak_idx >= freq.size
        ):
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

        phi_raw = float(phase[self._fft_peak_idx])

        # Unwrap phase incrementally
        if self._phi_last is None:
            phi_val = phi_raw
        else:
            phi_val = phi_raw
            delta = phi_val - self._phi_last
            while delta > np.pi:
                phi_val -= 2 * np.pi
                delta -= 2 * np.pi
            while delta < -np.pi:
                phi_val += 2 * np.pi
                delta += 2 * np.pi
        self._phi_last = phi_val

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
        self.ax_phase.relim()
        self.ax_phase.autoscale_view(scalex=False, scaley=True)

        self._apply_control(phi_val)
        self.canvas.draw_idle()

    def _apply_control(self, phi_val: float):
        try:
            sp = float(self.ed_setpoint.text())
            kp = float(self.ed_kp.text())
            ki = float(self.ed_ki.text())
            gain = float(self.ed_gain.text())
            max_step = float(self.ed_max_step.text())
        except Exception:
            return

        err = sp - phi_val  # unwrapped error

        now = time.monotonic()
        if self._last_ctrl_t is None:
            dt = 0.0
        else:
            dt = max(0.0, now - self._last_ctrl_t)
        self._last_ctrl_t = now

        self._integral += err * dt
        u = kp * err + ki * self._integral
        step = gain * u  # Volts

        # Quantification + seuil à 0.1 V
        quant = 0.1
        if abs(step) < quant:
            step = 0.0
        else:
            step = round(step / quant) * quant

        if max_step > 0 and abs(step) > max_step:
            step = np.sign(step) * max_step

        self.lbl_phase.setText(f"φ = {phi_val:+.3f} rad, err = {err:+.3f} rad")

        if not self._lock_enabled:
            return
        if self.stage is None:
            return
        if step == 0.0:
            return

        try:
            self._move_stage(step)
        except Exception as e:
            self._log(f"Stage move error: {e}")

    def _move_stage(self, step: float):
        st = self.stage
        if st is None:
            return

        if self._stage_v is None:
            self._stage_v = 0.0

        tgt = self._stage_v + step
        if tgt < self._vmin:
            tgt = self._vmin
        if tgt > self._vmax:
            tgt = self._vmax

        quant = 0.1
        tgt = round(tgt / quant) * quant

        try:
            st.set_position(tgt)
            self._stage_v = tgt
        except Exception as e:
            self._log(f"NV40 set_position failed: {e}")

    def _on_data(self, ts, wl, counts):
        now = time.monotonic()
        wl = np.asarray(wl, float).ravel()
        counts = np.asarray(counts, float).ravel()

        if (now - self._last_draw) < self._min_draw_dt:
            return
        self._last_draw = now

        self._update_fft_and_phase(wl, counts)

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
