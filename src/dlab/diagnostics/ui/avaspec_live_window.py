from __future__ import annotations
import sys, os, datetime, time, numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QTextEdit, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from dlab.hardware.wrappers.avaspec_controller import AvaspecController, AvaspecError
from dlab.core.device_registry import REGISTRY
from dlab.boot import ROOT, get_config

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _data_root():
    cfg = get_config() or {}
    return str((ROOT / (cfg.get("paths", {}).get("data_root", "C:/data"))).resolve())

def _append_avaspec_log(folder, spec_name, fn, int_ms, averages, comment):
    log_path = os.path.join(folder, f"{spec_name}_log_{datetime.datetime.now():%Y-%m-%d}.log")
    exists = os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write("File Name\tIntegration_ms\tAverages\tComment\n")
        f.write(f"{fn}\t{int_ms}\t{averages}\t{comment}\n")

class MeasureThread(QThread):
    data_ready = pyqtSignal(float, object, object)
    error = pyqtSignal(str)
    def __init__(self, ctrl, int_ms, avg, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self.int_ms = float(max(1.0, float(int_ms)))
        self.avg = int(max(1, int(avg)))
        self._running = True
        self._need_apply = True
    def update_params(self, int_ms, avg):
        self.int_ms = float(max(1.0, float(int_ms)))
        self.avg = int(max(1, int(avg)))
        self._need_apply = True
    def run(self):
        while self._running and not self.isInterruptionRequested():
            try:
                if self._need_apply:
                    self.ctrl.set_params(self.int_ms, self.avg)
                    self._need_apply = False
                ts, wl, counts = self.ctrl.measure_once()
                self.data_ready.emit(ts, wl, counts)
            except Exception as e:
                self.error.emit(str(e))
                break
    def stop(self):
        self._running = False
        self.requestInterruption()

class AvaspecLive(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Avaspec Live")
        self.ctrl = None
        self.th = None
        self._handles = []
        self._line = None
        self._last_wl = None
        self._last_counts = None
        self._ref_wl = None
        self._reg_key = None
        self._min_draw_dt = 0.05
        self._last_draw = 0.0
        self._build_ui()
        self.resize(1120, 780)
        try:
            REGISTRY.register("ui:avaspec_live", self)
        except Exception:
            pass
    def _build_ui(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.cmb = QComboBox()
        self.btn_search = QPushButton("Search")
        self.btn_search.clicked.connect(self.search_specs)
        self.btn_act = QPushButton("Activate")
        self.btn_act.clicked.connect(self.activate)
        self.btn_deact = QPushButton("Deactivate")
        self.btn_deact.clicked.connect(self.deactivate)
        self.btn_deact.setEnabled(False)
        top.addWidget(QLabel("Spectrometer:"))
        top.addWidget(self.cmb, 1)
        top.addWidget(self.btn_search)
        top.addWidget(self.btn_act)
        top.addWidget(self.btn_deact)
        root.addLayout(top)
        row = QHBoxLayout()
        self.ed_int = QLineEdit("100")
        self.ed_avg = QLineEdit("1")
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self.apply_params)
        row.addWidget(QLabel("Integration (ms):"))
        row.addWidget(self.ed_int)
        row.addWidget(QLabel("Averages:"))
        row.addWidget(self.ed_avg)
        row.addWidget(self.btn_apply)
        self.lbl_ftl = QLabel("FTL: — fs")
        self.lbl_ftl.setMinimumWidth(140)
        self.lbl_ftl.setAlignment(Qt.AlignCenter)
        self.lbl_ftl.setStyleSheet("QLabel { border: 1px solid #888; padding: 4px; }")
        row.addStretch(1)
        row.addWidget(self.lbl_ftl)
        root.addLayout(row)
        row2 = QHBoxLayout()
        self.btn_bg_set = QPushButton("Set BG")
        self.btn_bg_set.clicked.connect(self.set_background)
        self.btn_bg_clear = QPushButton("Clear BG")
        self.btn_bg_clear.clicked.connect(self.clear_background)
        self.btn_cal_toggle = QPushButton("Calibration: OFF")
        self.btn_cal_toggle.setCheckable(True)
        self.btn_cal_toggle.clicked.connect(self.toggle_calibration)
        row2.addWidget(self.btn_bg_set)
        row2.addWidget(self.btn_bg_clear)
        row2.addWidget(self.btn_cal_toggle)
        row2.addStretch(1)
        root.addLayout(row2)
        row4 = QHBoxLayout()
        self.ed_comment = QLineEdit("")
        self.ed_thresh = QLineEdit("0.02")
        row4.addWidget(QLabel("Comment:"))
        row4.addWidget(self.ed_comment, 1)
        row4.addWidget(QLabel("FTL thresh (0..1):"))
        row4.addWidget(self.ed_thresh)
        root.addLayout(row4)
        row3 = QHBoxLayout()
        self.btn_start = QPushButton("Start Live")
        self.btn_stop = QPushButton("Stop Live")
        self.btn_stop.setEnabled(False)
        self.btn_save = QPushButton("Save Spectrum")
        self.btn_start.clicked.connect(self.start_live)
        self.btn_stop.clicked.connect(self.stop_live)
        self.btn_save.clicked.connect(self.save_spectrum)
        row3.addWidget(self.btn_start)
        row3.addWidget(self.btn_stop)
        row3.addStretch(1)
        row3.addWidget(self.btn_save)
        root.addLayout(row3)
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Counts")
        self.ax.grid(True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(self.canvas.sizePolicy().Expanding, self.canvas.sizePolicy().Expanding)
        root.addWidget(self.canvas, 10)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        self.log.setStyleSheet("QTextEdit { font-family: 'Fira Mono', 'Consolas', monospace; font-size: 11px; }")
        root.addWidget(self.log, 0)
        self.search_specs()
    def log_msg(self, msg):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{t}] {msg}")
    def search_specs(self):
        self.cmb.clear()
        self._handles = AvaspecController.list_spectrometers()
        if not self._handles:
            self.log_msg("No spectrometers found.")
        else:
            self.log_msg(f"Found {len(self._handles)} spectrometer(s).")
        for i in range(len(self._handles)):
            self.cmb.addItem(f"Spec {i+1}")
    def activate(self):
        if self.ctrl is not None:
            self.log_msg("Already activated.")
            return
        idx = self.cmb.currentIndex()
        if idx < 0 or idx >= len(self._handles):
            QMessageBox.critical(self, "Error", "No spectrometer selected.")
            return
        try:
            self.ctrl = AvaspecController(self._handles[idx])
            self.ctrl.activate()
            self.apply_params()
            key = f"spectrometer:avaspec:spec_{idx+1}"
            try:
                for k, v in REGISTRY.items(prefix="spectrometer:avaspec:"):
                    if k == key or v is self.ctrl:
                        REGISTRY.unregister(k)
            except Exception:
                pass
            try:
                REGISTRY.register(key, self.ctrl)
                self._reg_key = key
            except Exception:
                self._reg_key = None
            self.log_msg("Activated.")
            self.btn_act.setEnabled(False)
            self.btn_deact.setEnabled(True)
        except Exception as e:
            self.ctrl = None
            QMessageBox.critical(self, "Error", f"Activate failed: {e}")
            self.log_msg(f"Activate failed: {e}")
    def deactivate(self):
        if self.th is not None:
            QMessageBox.warning(self, "Warning", "Stop live first.")
            return
        try:
            if self.ctrl is not None:
                self.ctrl.deactivate()
        finally:
            try:
                if self._reg_key:
                    REGISTRY.unregister(self._reg_key)
            except Exception:
                pass
            self._reg_key = None
            self.ctrl = None
            self.btn_act.setEnabled(True)
            self.btn_deact.setEnabled(False)
            self.log_msg("Deactivated.")
    def apply_params(self):
        if self.ctrl is None:
            return
        try:
            it = float(self.ed_int.text())
            av = int(self.ed_avg.text())
            self.ctrl.set_params(it, av)
            if self.th is not None and self.th.isRunning():
                self.th.update_params(it, av)
            self.log_msg(f"Params: int={it} ms, avg={av}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid params: {e}")
    def start_live(self):
        if self.ctrl is None:
            QMessageBox.critical(self, "Error", "Activate first.")
            return
        if self.th is not None:
            self.stop_live()
        try:
            it = float(self.ed_int.text())
            av = int(self.ed_avg.text())
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid params.")
            return
        self._ref_wl = None
        self._last_draw = 0.0
        self.th = MeasureThread(self.ctrl, it, av)
        self.th.data_ready.connect(self._on_data, Qt.QueuedConnection)
        self.th.error.connect(self._on_thread_error, Qt.QueuedConnection)
        self.th.finished.connect(self._on_thread_finished, Qt.QueuedConnection)
        self.th.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_deact.setEnabled(False)
        self.log_msg("Live started.")
    def stop_live(self):
        th = self.th
        self.th = None
        if th is not None:
            try:
                th.stop()
                th.wait(2000)
            finally:
                try: th.finished.disconnect(self._on_thread_finished)
                except Exception: pass
                try: th.error.disconnect(self._on_thread_error)
                except Exception: pass
                try: th.data_ready.disconnect(self._on_data)
                except Exception: pass
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self.ctrl is not None:
            self.btn_deact.setEnabled(True)
        self.log_msg("Live stopped.")
    def _on_thread_finished(self):
        if self.th is None:
            self.log_msg("Thread finished.")
    def _on_thread_error(self, err):
        self.log_msg(f"Thread error: {err}")
        QMessageBox.critical(self, "Acquisition error", err)
        self.stop_live()
    def _apply_threshold_for_display(self, y):
        y = np.asarray(y, float)
        y_pos = np.clip(y, 0, None)
        thr = self._get_threshold_fraction()
        ymax = float(np.nanmax(y_pos)) if y_pos.size else 0.0
        if not np.isfinite(ymax) or ymax <= 0:
            return np.zeros_like(y_pos)
        cutoff = thr * ymax
        return np.where(y_pos >= cutoff, y_pos, 0.0)
    def _on_data(self, ts, wl, counts):
        now = time.monotonic()
        wl = np.asarray(wl, float).ravel()
        counts = np.asarray(counts, float).ravel()
        y = self.ctrl.process_counts(wl, counts) if self.ctrl else counts
        if self._ref_wl is None:
            self._ref_wl = wl.copy()
            self.ax.set_xlim(self._ref_wl[0], self._ref_wl[-1])
        if wl.shape != self._ref_wl.shape or np.max(np.abs(wl - self._ref_wl)) > 1e-9:
            y = np.interp(self._ref_wl, wl, y)
            wl = self._ref_wl
        self._last_wl = wl
        self._last_counts = counts
        if (now - self._last_draw) < self._min_draw_dt:
            return
        self._last_draw = now
        disp = self._apply_threshold_for_display(y)
        ftl_s, _, _ = self._compute_ftl_from_spectrum(wl, disp, level=0.5)
        ftl_fs = ftl_s * 1e15 if np.isfinite(ftl_s) else float("nan")
        self.lbl_ftl.setText(f"FTL: {ftl_fs:.0f} fs" if np.isfinite(ftl_fs) else "FTL: — fs")
        if self._line is None:
            self.ax.cla()
            self.ax.set_xlabel("Wavelength (nm)")
            self.ax.set_ylabel("Counts")
            self.ax.grid(True)
            (self._line,) = self.ax.plot(wl, y, lw=1.2)
            self.ax.set_xlim(self._ref_wl[0], self._ref_wl[-1])
        else:
            self._line.set_xdata(wl)
            self._line.set_ydata(y)
        self.ax.relim()
        self.ax.autoscale(enable=True, axis='y', tight=False)
        self.ax.autoscale_view(scalex=False, scaley=True)
        self.canvas.draw_idle()
    def _get_threshold_fraction(self):
        try:
            v = float(self.ed_thresh.text())
        except Exception:
            v = 0.02
        if not np.isfinite(v):
            v = 0.02
        return float(min(max(v, 0.0), 1.0))
    @staticmethod
    def _fwxm(time_v, intens, x=0.5):
        t = np.asarray(time_v, float).ravel()
        y = np.asarray(intens, float).ravel()
        if t.size != y.size or t.size < 3:
            return float("nan")
        m = float(np.nanmax(y)) if y.size else float("nan")
        if not np.isfinite(m) or m <= 0:
            return float("nan")
        P = y / m
        idx = np.where(P >= x)[0]
        if idx.size == 0:
            return float("nan")
        i0, i1 = int(idx[0]), int(idx[-1])
        if i0 == 0:
            t1 = t[0]
        else:
            m1 = (P[i0] - P[i0 - 1]) / (t[i0] - t[i0 - 1] if t[i0] != t[i0 - 1] else 1e-300)
            n1 = P[i0] - m1 * t[i0]
            t1 = (x - n1) / m1
        if i1 >= t.size - 1:
            t2 = t[-1]
        else:
            m2 = (P[i1 + 1] - P[i1]) / (t[i1 + 1] - t[i1] if t[i1 + 1] != t[i1] else 1e-300)
            n2 = P[i1] - m2 * t[i1]
            t2 = (x - n2) / m2
        return float(t2 - t1)
    @staticmethod
    def _compute_ftl_from_spectrum(wl_nm, spec, level=0.5):
        c = 299_792_458.0
        wl = np.asarray(wl_nm, float).ravel()
        S = np.asarray(spec, float).ravel()
        if wl.size != S.size or wl.size < 3:
            return float("nan"), np.array([]), np.array([])
        freqs = c / (wl * 1e-9)
        N = freqs.size
        freqs_i = np.linspace(freqs[-1], freqs[0], N)
        specs_f = np.abs(np.interp(freqs_i, np.flip(freqs), np.flip(S)))
        if N < 2:
            return float("nan"), np.array([]), np.array([])
        dnu = freqs_i[1] - freqs_i[0]
        if not np.isfinite(dnu) or dnu == 0:
            return float("nan"), np.array([]), np.array([])
        amp_f = np.sqrt(np.clip(specs_f, 0, None))
        E_t = np.fft.fftshift(np.fft.ifft(amp_f))
        I_t = np.abs(E_t) ** 2
        time_v = np.fft.fftshift(np.fft.fftfreq(N, d=dnu))
        ftl_s = AvaspecLive._fwxm(time_v, I_t, x=level)
        I_t_n = I_t / (np.max(I_t) if np.max(I_t) > 0 else 1.0)
        return ftl_s, time_v, I_t_n
    def save_spectrum(self):
        if self._last_wl is None or self._last_counts is None or self.ctrl is None:
            QMessageBox.warning(self, "Save", "No spectrum to save.")
            return
        now = datetime.datetime.now()
        folder = os.path.join(_data_root(), f"{now:%Y-%m-%d}", "avaspec")
        _ensure_dir(folder)
        safe_ts = now.strftime("%Y-%m-%d_%H-%M-%S")
        base = "Avaspec"
        fn = os.path.join(folder, f"{base}_Spectrum_{safe_ts}.txt")
        comment = self.ed_comment.text() or ""
        try:
            wl = np.asarray(self._last_wl, float)
            raw = np.asarray(self._last_counts, float)
            bgsub = self.ctrl._apply_background(wl, raw)
            use_cal = bool(self.ctrl._cal_enabled)
            divcal = None
            cal_applied = False
            cal_path_str = ""
            if use_cal:
                cal_applied = self.ctrl._cal_wl is not None and self.ctrl._cal_vals is not None
                if cal_applied:
                    cal_path = get_config().get("avaspec", {}).get("calibration_file", "")
                    cal_path_str = str((ROOT / cal_path).resolve()) if cal_path else ""
                    divcal = self.ctrl._apply_calibration(wl, bgsub)
            thr = self._get_threshold_fraction()
            disp_for_ftl = np.where(bgsub >= thr * max(1.0, float(np.nanmax(np.clip(bgsub, 0, None)))), bgsub, 0.0)
            ftl_s, _, _ = self._compute_ftl_from_spectrum(wl, disp_for_ftl, level=0.5)
            ftl_fs = ftl_s * 1e15 if np.isfinite(ftl_s) else float("nan")
            with open(fn, "w", encoding="utf-8") as f:
                if comment:
                    f.write(f"# Comment: {comment}\n")
                f.write(f"# Timestamp: {now:%Y-%m-%d %H:%M:%S}\n")
                f.write(f"# IntegrationTime_ms: {self.ed_int.text()}\n")
                f.write(f"# Averages: {self.ed_avg.text()}\n")
                f.write(f"# BackgroundApplied: {self.ctrl._bg_counts is not None}\n")
                f.write(f"# CalibrationApplied: {bool(cal_applied)}\n")
                if cal_applied and cal_path_str:
                    f.write(f"# CalibrationFile: {cal_path_str}\n")
                f.write(f"# DisplayThreshold: {thr:.4f}\n")
                if np.isfinite(ftl_fs):
                    f.write(f"# FTL_fs: {ftl_fs:.3f}\n")
                if cal_applied and divcal is not None:
                    f.write("Wavelength_nm;Counts_raw;Counts_bgsub;Counts_bgsub_divcal\n")
                    for x, y_raw, y_bg, y_cal in zip(wl, raw, bgsub, divcal):
                        f.write(f"{float(x):.6f};{float(y_raw):.6f};{float(y_bg):.6f};{float(y_cal):.6f}\n")
                else:
                    f.write("Wavelength_nm;Counts_raw;Counts_bgsub\n")
                    for x, y_raw, y_bg in zip(wl, raw, bgsub):
                        f.write(f"{float(x):.6f};{float(y_raw):.6f};{float(y_bg):.6f}\n")
            try:
                int_ms = float(self.ed_int.text())
            except Exception:
                int_ms = float("nan")
            try:
                averages = int(self.ed_avg.text())
            except Exception:
                averages = 1
            _append_avaspec_log(folder, base, os.path.basename(fn), int_ms, averages, comment)
            self.log_msg(f"Spectrum saved to {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")
            self.log_msg(f"Save failed: {e}")
    def set_background(self):
        if self._last_wl is None or self._last_counts is None or self.ctrl is None:
            self.log_msg("No spectrum to set as BG.")
            return
        try:
            self.ctrl.set_background(self._last_wl, self._last_counts)
            self.log_msg("Background set.")
        except Exception as e:
            self.log_msg(f"Set BG failed: {e}")
            QMessageBox.warning(self, "Background", f"Failed: {e}")
    def clear_background(self):
        if self.ctrl is None:
            return
        try:
            self.ctrl.clear_background()
            self.log_msg("Background cleared.")
        except Exception as e:
            self.log_msg(f"Clear BG failed: {e}")
            QMessageBox.warning(self, "Background", f"Failed: {e}")
    def toggle_calibration(self):
        enabled = self.btn_cal_toggle.isChecked()
        self.btn_cal_toggle.setText("Calibration: ON" if enabled else "Calibration: OFF")
        if self.ctrl is None:
            return
        try:
            self.ctrl.enable_calibration(enabled)
            self.log_msg(f"Calibration {'enabled' if enabled else 'disabled'}.")
        except Exception as e:
            self.btn_cal_toggle.setChecked(False)
            self.btn_cal_toggle.setText("Calibration: OFF")
            self.log_msg(f"Calibration failed: {e}")
            QMessageBox.warning(self, "Calibration", f"Failed: {e}")

def main():
    app = QApplication(sys.argv)
    w = AvaspecLive()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
