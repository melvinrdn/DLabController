# src/dlab/diagnostics/view/AvaspecLive.py
from __future__ import annotations
import os, datetime, time
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QTextEdit, QMessageBox, QSplitter, QCheckBox, QApplication
)
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from dlab.boot import ROOT, get_config
from dlab.hardware.wrappers.avaspec_controller import AvaspecController, AvaspecError
from dlab.core.device_registry import REGISTRY

def _data_root() -> str:
    cfg = get_config() or {}
    return str((ROOT / (cfg.get("paths", {}).get("data_root", "C:/data"))).resolve())

def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def _append_avaspec_log(folder: str, spec_name: str, fn: str, int_ms: float, averages: int, comment: str) -> None:
    log_path = os.path.join(folder, f"{spec_name}_log_{datetime.datetime.now():%Y-%m-%d}.log")
    exists = os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write("File Name\tIntegration_ms\tAverages\tComment\n")
        f.write(f"{fn}\t{int_ms}\t{averages}\t{comment}\n")

class LiveMeasurementThread(QThread):
    spectrum_signal = pyqtSignal(float, object)
    error_signal = pyqtSignal(str)
    def __init__(self, ctrl, int_time_ms: float, no_avg: int):
        super().__init__()
        self.ctrl = ctrl
        self.int_time = float(int_time_ms)
        self.no_avg = int(no_avg)
        self._running = True
    def update_params(self, int_time_ms: float, no_avg: int) -> None:
        self.int_time = float(int_time_ms)
        self.no_avg = int(no_avg)
    def run(self) -> None:
        while self._running:
            try:
                ts, data = self.ctrl.measure_spectrum(self.int_time, self.no_avg)
                self.spectrum_signal.emit(ts, data)
                time.sleep(max(0.02, self.int_time / 1000.0))
            except Exception as e:
                self.error_signal.emit(str(e))
                break
    def stop(self) -> None:
        self._running = False

class AvaspecLiveWindow(QWidget):
    closed = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Avaspec Live")
        self.ctrl: AvaspecController | None = None
        self.thread: LiveMeasurementThread | None = None
        self.handles = []
        self.line = None
        self.last_data = None
        self._last_raw: np.ndarray | None = None
        self.registry_key = None
        self._build_ui()
        try:
            REGISTRY.register("ui:avaspec_live", self)
        except Exception:
            pass
    def _build_ui(self):
        main = QHBoxLayout(self)
        splitter = QSplitter()
        left = QWidget(); left_l = QVBoxLayout(left)
        self.spec_combo = QComboBox()
        btn_search = QPushButton("Search Spectrometers")
        btn_search.clicked.connect(self.search_spectrometers)
        left_l.addWidget(QLabel("Select Spectrometer:"))
        left_l.addWidget(self.spec_combo)
        left_l.addWidget(btn_search)
        btn_act = QPushButton("Activate")
        btn_deact = QPushButton("Deactivate"); btn_deact.setEnabled(False)
        btn_act.clicked.connect(self.activate_hardware)
        btn_deact.clicked.connect(self.deactivate_hardware)
        self.btn_act, self.btn_deact = btn_act, btn_deact
        left_l.addWidget(btn_act); left_l.addWidget(btn_deact)
        self.int_edit = QLineEdit("100")
        self.avg_edit = QLineEdit("1")
        left_l.addWidget(QLabel("Integration Time (ms):")); left_l.addWidget(self.int_edit)
        left_l.addWidget(QLabel("Number of Averages:"));     left_l.addWidget(self.avg_edit)
        self.cb_autoscale = QCheckBox("Autoscale"); self.cb_autoscale.setChecked(True)
        left_l.addWidget(self.cb_autoscale)
        bg_row = QHBoxLayout()
        self.btn_bg = QPushButton("Update Background")
        self.btn_bg_reset = QPushButton("Reset Background")
        self.btn_bg.clicked.connect(self.update_background)
        self.btn_bg_reset.clicked.connect(self.reset_background)
        bg_row.addWidget(self.btn_bg); bg_row.addWidget(self.btn_bg_reset)
        left_l.addLayout(bg_row)
        self.lbl_bg = QLabel("Background: none")
        left_l.addWidget(self.lbl_bg)
        self.cb_calibration = QCheckBox("Calibration ((counts-bg)/cal)")
        self.cb_calibration.setChecked(False)
        left_l.addWidget(self.cb_calibration)
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Threshold (0..1):"))
        self.threshold_edit = QLineEdit("0.02")
        thr_row.addWidget(self.threshold_edit)
        left_l.addLayout(thr_row)
        self.lbl_ftl = QLabel("FTL: — fs")
        left_l.addWidget(self.lbl_ftl)
        btn_start = QPushButton("Start Live"); btn_start.clicked.connect(self.start_live)
        btn_stop  = QPushButton("Stop Live");  btn_stop.setEnabled(False); btn_stop.clicked.connect(self.stop_live)
        self.btn_start, self.btn_stop = btn_start, btn_stop
        left_l.addWidget(btn_start); left_l.addWidget(btn_stop)
        self.comment_edit = QLineEdit("")
        self.btn_save = QPushButton("Save Spectrum"); self.btn_save.clicked.connect(self.save_spectrum)
        left_l.addWidget(QLabel("Comment:")); left_l.addWidget(self.comment_edit); left_l.addWidget(self.btn_save)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        left_l.addWidget(self.log)
        splitter.addWidget(left)
        right = QWidget(); right_l = QVBoxLayout(right)
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Wavelength (nm)"); self.ax.set_ylabel("Counts"); self.ax.grid(True)
        self.canvas = FigureCanvas(self.fig)
        right_l.addWidget(NavigationToolbar(self.canvas, self))
        right_l.addWidget(self.canvas)
        splitter.addWidget(right)
        main.addWidget(splitter)
        self.resize(1080, 720)
    def _ui_log(self, msg: str):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{t}] {msg}")
    @staticmethod
    def _fwxm(time_v: np.ndarray, intens: np.ndarray, x: float = 0.5) -> float:
        t = np.asarray(time_v, float)
        y = np.asarray(intens, float)
        if t.ndim != 1 or y.ndim != 1 or t.size != y.size or t.size < 3:
            return float("nan")
        m = np.nanmax(y)
        if not np.isfinite(m) or m <= 0:
            return float("nan")
        P = y / m
        ihw = np.where(P >= x)[0]
        if ihw.size == 0:
            return float("nan")
        i0, i1 = int(ihw[0]), int(ihw[-1])
        if i0 == 0:
            t1 = t[0]
        else:
            m1 = (P[i0] - P[i0-1]) / (t[i0] - t[i0-1] if t[i0] != t[i0-1] else 1e-300)
            n1 = P[i0] - m1 * t[i0]
            t1 = (x - n1) / m1
        if i1 >= t.size - 1:
            t2 = t[-1]
        else:
            m2 = (P[i1+1] - P[i1]) / (t[i1+1] - t[i1] if t[i1+1] != t[i1] else 1e-300)
            n2 = P[i1] - m2 * t[i1]
            t2 = (x - n2) / m2
        return float(t2 - t1)
    @staticmethod
    def _compute_ftl_from_spectrum(wl_nm: np.ndarray, spec: np.ndarray, level: float = 0.5) -> tuple[float, np.ndarray, np.ndarray]:
        c = 299_792_458.0
        wl = np.asarray(wl_nm, float).ravel()
        S  = np.asarray(spec,  float).ravel()
        if wl.size != S.size or wl.size < 3:
            return float("nan"), np.array([]), np.array([])
        freqs = c / (wl * 1e-9)
        N = freqs.size
        freqs_interp = np.linspace(freqs[-1], freqs[0], N)
        specs_f = np.abs(np.interp(freqs_interp, np.flip(freqs), np.flip(S)))
        if N < 2:
            return float("nan"), np.array([]), np.array([])
        dnu = freqs_interp[1] - freqs_interp[0]
        if not np.isfinite(dnu) or dnu == 0:
            return float("nan"), np.array([]), np.array([])
        amp_f = np.sqrt(np.clip(specs_f, 0, None))
        E_t   = np.fft.fftshift(np.fft.ifft(amp_f))
        I_t   = np.abs(E_t) ** 2
        time_v = np.fft.fftshift(np.fft.fftfreq(N, d=dnu))
        ftl_s = AvaspecLiveWindow._fwxm(time_v, I_t, x=level)
        I_t_n = I_t / (np.max(I_t) if np.max(I_t) > 0 else 1.0)
        return ftl_s, time_v, I_t_n
    def _get_threshold_fraction(self) -> float:
        try:
            v = float(self.threshold_edit.text())
        except Exception:
            v = 0.05
        if not np.isfinite(v):
            v = 0.05
        return float(min(max(v, 0.0), 1.0))
    def _apply_threshold_for_display(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, float)
        y_pos = np.clip(y, 0, None)
        thr = self._get_threshold_fraction()
        ymax = float(np.nanmax(y_pos)) if y_pos.size else 0.0
        if not np.isfinite(ymax) or ymax <= 0:
            return np.zeros_like(y_pos)
        cutoff = thr * ymax
        y_disp = np.where(y_pos >= cutoff, y_pos, 0.0)
        return y_disp
    def update_background(self):
        if not self.ctrl:
            QMessageBox.critical(self, "Error", "Spectrometer not activated.")
            return
        if self._last_raw is None or self.ctrl.wavelength is None:
            QMessageBox.warning(self, "Warning", "No spectrum available to set as background.")
            return
        wl = np.asarray(self.ctrl.wavelength, dtype=float)
        bg = np.asarray(self._last_raw, dtype=float)
        self.ctrl.set_background(wl, bg)
        self.lbl_bg.setText(f"Background: set ({bg.size} px)")
        self._ui_log("Background updated from last displayed spectrum.")
    def reset_background(self):
        if self.ctrl:
            self.ctrl.reset_background()
        self.lbl_bg.setText("Background: none")
        self._ui_log("Background reset.")
    def search_spectrometers(self):
        self.spec_combo.clear()
        lst = AvaspecController.list_spectrometers()
        if not lst:
            QMessageBox.critical(self, "Error", "No spectrometer found.")
            self._ui_log("No spectrometer found.")
            return
        self.handles = lst
        self.spec_combo.addItems([f"Spectrometer {i+1}" for i in range(len(lst))])
        self._ui_log(f"Found {len(lst)} spectrometer(s).")
    def activate_hardware(self):
        idx = self.spec_combo.currentIndex()
        if idx < 0 or idx >= len(self.handles):
            QMessageBox.critical(self, "Error", "No spectrometer selected.")
            return
        try:
            self.ctrl = AvaspecController(self.handles[idx])
            self.ctrl.activate()
            self._ui_log("Spectrometer activated.")
            key = f"spectrometer:avaspec:spec_{idx+1}"
            try:
                for k, v in REGISTRY.items(prefix="spectrometer:avaspec:"):
                    if k == key or v is self.ctrl:
                        REGISTRY.unregister(k)
            except Exception:
                pass
            REGISTRY.register(key, self.ctrl)
            self.registry_key = key
            self._ui_log(f"Registered '{key}' in device registry.")
            self.btn_act.setEnabled(False)
            self.btn_deact.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate: {e}")
            self._ui_log(f"Activate failed: {e}")
    def deactivate_hardware(self):
        try:
            if self.ctrl:
                self.ctrl.deactivate()
                self._ui_log("Spectrometer deactivated.")
        finally:
            try:
                if self.registry_key:
                    REGISTRY.unregister(self.registry_key)
                    self._ui_log(f"Unregistered '{self.registry_key}' from device registry.")
            except Exception:
                pass
            self.registry_key = None
            self.ctrl = None
            self.btn_act.setEnabled(True)
            self.btn_deact.setEnabled(False)
    def start_live(self):
        if not self.ctrl:
            QMessageBox.critical(self, "Error", "Spectrometer not activated.")
            return
        try:
            it = float(self.int_edit.text())
            av = int(self.avg_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid integration/averages.")
            return
        self.thread = LiveMeasurementThread(self.ctrl, it, av)
        self.thread.spectrum_signal.connect(self.update_spectrum)
        self.thread.error_signal.connect(lambda e: self._ui_log(f"Error: {e}"))
        self.thread.start()
        self._ui_log("Live started.")
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
    def stop_live(self):
        if self.thread:
            self.thread.stop(); self.thread.wait()
            self.thread = None
            self._ui_log("Live stopped.")
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
    def update_spectrum(self, ts: float, data: np.ndarray):
        if self.ctrl is None:
            return
        raw = np.asarray(data, dtype=float)
        self._last_raw = raw.copy()
        wl = np.asarray(self.ctrl.wavelength, dtype=float)
        use_cal = self.cb_calibration.isChecked()
        if use_cal and not self.ctrl.has_calibration():
            try:
                p = self.ctrl.load_calibration_from_config(silent=True)
                if p is None:
                    self._ui_log("Calibration file not found; proceeding without calibration.")
                else:
                    self._ui_log(f"Calibration loaded: {p}")
            except Exception as e:
                self._ui_log(f"Calibration load failed: {e}")
        proc = self.ctrl.apply_processing(wl, raw, use_calibration=use_cal)
        self.last_data = proc
        disp = self._apply_threshold_for_display(proc)
        if self.line is None:
            (self.line,) = self.ax.plot(wl, disp, label="Data")
        else:
            self.line.set_xdata(wl)
            self.line.set_ydata(disp)
        ftl_s, _, _ = self._compute_ftl_from_spectrum(wl, disp, level=0.5)
        ftl_fs = ftl_s * 1e15 if np.isfinite(ftl_s) else float("nan")
        if np.isfinite(ftl_fs):
            self.lbl_ftl.setText(f"FTL: {ftl_fs:.0f} fs")
            self.ax.set_title(f"FTL ≈ {ftl_fs:.0f} fs")
        else:
            self.lbl_ftl.setText("FTL: — fs")
            self.ax.setTitle("")
        self.ax.set_yscale("linear")
        if self.cb_autoscale.isChecked():
            self.ax.relim(); self.ax.autoscale_view()
        has_bg = (self.ctrl._bg_counts is not None)
        if use_cal and self.ctrl.has_calibration():
            self.ax.set_ylabel("Counts ((bg-sub)/cal)")
        else:
            self.ax.set_ylabel("Counts (bg-subtracted)" if has_bg else "Counts")
        self.canvas.draw_idle()
    def set_spectrum_from_scan(self, wl, counts):
        try:
            wl = np.asarray(wl, dtype=float)
            raw = np.asarray(counts, dtype=float)
            self._last_raw = raw.copy()
            use_cal = self.ctrl is not None and self.cb_calibration.isChecked()
            if self.ctrl is not None and use_cal:
                if not self.ctrl.has_calibration():
                    try:
                        p = self.ctrl.load_calibration_from_config(silent=True)
                        if p is None:
                            self._ui_log("Calibration file not found; proceeding without calibration.")
                        else:
                            self._ui_log(f"Calibration loaded: {p}")
                    except Exception as e:
                        self._ui_log(f"Calibration load failed: {e}")
                proc = self.ctrl.apply_processing(wl, raw, use_calibration=True)
            else:
                proc = np.asarray(raw, dtype=float)
            self.last_data = proc
            disp = self._apply_threshold_for_display(proc)
            if self.line is None:
                self.ax.cla()
                self.ax.set_xlabel("Wavelength (nm)")
                self.ax.set_ylabel("Counts")
                self.ax.grid(True)
                (self.line,) = self.ax.plot(wl, disp, label="Data")
            else:
                self.line.set_xdata(wl)
                self.line.set_ydata(disp)
            ftl_s, _, _ = self._compute_ftl_from_spectrum(wl, disp, level=0.5)
            ftl_fs = ftl_s * 1e15 if np.isfinite(ftl_s) else float("nan")
            if np.isfinite(ftl_fs):
                self.lbl_ftl.setText(f"FTL: {ftl_fs:.0f} fs")
                self.ax.set_title(f"FTL ≈ {ftl_fs:.0f} fs")
            else:
                self.lbl_ftl.setText("FTL: — fs")
                self.ax.set_title("")
            self.ax.set_yscale("linear")
            if self.cb_autoscale.isChecked():
                self.ax.relim(); self.ax.autoscale_view()
            self.canvas.draw_idle()
        except Exception as e:
            self._ui_log(f"Scan-plot update failed: {e}")
    def save_spectrum(self):
        if self.last_data is None or self.ctrl is None:
            QMessageBox.warning(self, "Warning", "No spectrum to save.")
            return
        now = datetime.datetime.now()
        folder = os.path.join(_data_root(), f"{now:%Y-%m-%d}", "avaspec")
        _ensure_dir(folder)
        safe_ts = now.strftime("%Y-%m-%d_%H-%M-%S")
        base = "Avaspec"
        fn = os.path.join(folder, f"{base}_Spectrum_{safe_ts}.txt")
        comment = self.comment_edit.text() or ""
        try:
            wl = np.asarray(self.ctrl.wavelength, dtype=float)
            if self._last_raw is not None and self._last_raw.shape == np.asarray(self.last_data).shape:
                raw = self._last_raw
            else:
                raw = np.asarray(self.last_data, dtype=float)
                if self.ctrl._bg_counts is not None and self.ctrl._bg_wavelength is not None:
                    bg = np.interp(wl, self.ctrl._bg_wavelength, self.ctrl._bg_counts, left=np.nan, right=np.nan)
                    if np.isnan(bg[0]):
                        i0 = np.flatnonzero(~np.isnan(bg))
                        if i0.size: bg[:i0[0]] = bg[i0[0]]
                    if np.isnan(bg[-1]):
                        i1 = np.flatnonzero(~np.isnan(bg))
                        if i1.size: bg[i1[-1]:] = bg[i1[-1]]
                    raw = raw + bg
            bgsub = self.ctrl._apply_background(wl, raw)
            use_cal = self.cb_calibration.isChecked()
            divcal = None
            cal_applied = False
            cal_path_str = ""
            if use_cal:
                if not self.ctrl.has_calibration():
                    try:
                        p = self.ctrl.load_calibration_from_config(silent=True)
                        if p is not None:
                            cal_path_str = str(p)
                    except Exception as e:
                        self._ui_log(f"Calibration load failed for save: {e}")
                else:
                    cal_path_str = str(self.ctrl._cal_path) if self.ctrl._cal_path else ""
                divcal = self.ctrl._apply_calibration(wl, bgsub)
                cal_applied = self.ctrl.has_calibration()
            disp_for_ftl = self._apply_threshold_for_display(self.last_data)
            ftl_s, _, _ = self._compute_ftl_from_spectrum(wl, disp_for_ftl, level=0.5)
            ftl_fs = ftl_s * 1e15 if np.isfinite(ftl_s) else float("nan")
            with open(fn, "w", encoding="utf-8") as f:
                if comment:
                    f.write(f"# Comment: {comment}\n")
                f.write(f"# Timestamp: {now:%Y-%m-%d %H:%M:%S}\n")
                f.write(f"# IntegrationTime_ms: {self.int_edit.text()}\n")
                f.write(f"# Averages: {self.avg_edit.text()}\n")
                f.write(f"# BackgroundApplied: {self.ctrl._bg_counts is not None}\n")
                f.write(f"# CalibrationApplied: {bool(cal_applied)}\n")
                if cal_applied and cal_path_str:
                    f.write(f"# CalibrationFile: {cal_path_str}\n")
                f.write(f"# DisplayThreshold: {self._get_threshold_fraction():.4f}\n")
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
                int_ms = float(self.int_edit.text())
            except Exception:
                int_ms = float("nan")
            try:
                averages = int(self.avg_edit.text())
            except Exception:
                averages = 1
            _append_avaspec_log(folder, base, os.path.basename(fn), int_ms, averages, comment)
            self._ui_log(f"Spectrum saved to {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")
            self._ui_log(f"Save failed: {e}")
    def closeEvent(self, event):
        try:
            self.stop_live()
        finally:
            try:
                if self.ctrl:
                    self.ctrl.deactivate()
            finally:
                try:
                    REGISTRY.unregister(self.registry_key or "")
                    self._ui_log(f"Unregistered '{self.registry_key}' from device registry.")
                except Exception:
                    pass
                try:
                    REGISTRY.unregister("ui:avaspec_live")
                except Exception:
                    pass
                self.registry_key = None
                self.ctrl = None
        self.closed.emit()
        super().closeEvent(event)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = AvaspecLiveWindow()
    w.show()
    sys.exit(app.exec_())
