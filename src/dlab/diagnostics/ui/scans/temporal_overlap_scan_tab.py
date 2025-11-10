from __future__ import annotations
import sys, os, time, datetime
from pathlib import Path
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QComboBox, QDoubleSpinBox, QGroupBox, QMessageBox,
    QCheckBox, QProgressBar
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from dlab.boot import ROOT, get_config
from dlab.core.device_registry import REGISTRY
import cmasher as cmr 
def _ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _data_root():
    cfg = get_config() or {}
    base = cfg.get("paths", {}).get("data_root", "C:/data")
    return (ROOT / base).resolve()

def _append_avaspec_log(folder, spec_name, fn, int_ms, averages, comment):
    log_path = folder / f"{spec_name}_log_{datetime.datetime.now():%Y-%m-%d}.log"
    exists = log_path.exists()
    with open(log_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write("File Name\tIntegration_ms\tAverages\tComment\n")
        f.write(f"{fn}\t{int_ms}\t{averages}\t{comment}\n")

class TOverlapWorker(QObject):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    partial = pyqtSignal(float, object, object)
    finished = pyqtSignal(str)
    def __init__(self, stage_key, spec_key, positions, settle_s, int_ms, averages, use_processed, scan_name, comment, parent=None):
        super().__init__(parent)
        self.stage_key = stage_key
        self.spec_key = spec_key
        self.positions = [float(p) for p in positions]
        self.settle_s = float(settle_s)
        self.int_ms = float(max(1.0, float(int_ms)))
        self.averages = int(max(1, int(averages)))
        self.use_processed = bool(use_processed)
        self.scan_name = (scan_name or "temporal_overlap").strip()
        self.comment = comment or ""
        self.abort = False
    def _emit(self, s):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.emit(f"[{ts}] {s}")
    def _save_spectrum_like_ui(self, wl_nm, counts_raw, ctrl):
        now = datetime.datetime.now()
        avaspec_folder = _ensure_dir(_data_root() / f"{now:%Y-%m-%d}" / "avaspec")
        safe_ts = now.strftime("%Y-%m-%d_%H-%M-%S")
        base = "Avaspec"
        fn = avaspec_folder / f"{base}_Spectrum_{safe_ts}.txt"
        wl = np.asarray(wl_nm, float).ravel()
        raw = np.asarray(counts_raw, float).ravel()
        bgsub = ctrl._apply_background(wl, raw) if hasattr(ctrl, "_apply_background") else raw
        use_cal = bool(getattr(ctrl, "_cal_enabled", False))
        divcal = None
        cal_applied = False
        cal_path_str = ""
        if use_cal:
            cal_applied = getattr(ctrl, "_cal_wl", None) is not None and getattr(ctrl, "_cal_vals", None) is not None
            if cal_applied:
                cal_path = (get_config() or {}).get("avaspec", {}).get("calibration_file", "")
                cal_path_str = str((ROOT / cal_path).resolve()) if cal_path else ""
                if hasattr(ctrl, "_apply_calibration"):
                    divcal = ctrl._apply_calibration(wl, bgsub)
        display_thr = 0.02
        with open(fn, "w", encoding="utf-8") as f:
            if self.comment:
                f.write(f"# Comment: {self.comment}\n")
            f.write(f"# Timestamp: {now:%Y-%m-%d %H:%M:%S}\n")
            f.write(f"# IntegrationTime_ms: {self.int_ms}\n")
            f.write(f"# Averages: {self.averages}\n")
            f.write(f"# BackgroundApplied: {getattr(ctrl, '_bg_counts', None) is not None}\n")
            f.write(f"# CalibrationApplied: {bool(cal_applied)}\n")
            if cal_applied and cal_path_str:
                f.write(f"# CalibrationFile: {cal_path_str}\n")
            f.write(f"# DisplayThreshold: {display_thr:.4f}\n")
            if cal_applied and divcal is not None:
                f.write("Wavelength_nm;Counts_raw;Counts_bgsub;Counts_bgsub_divcal\n")
                for x, y_raw, y_bg, y_cal in zip(wl, raw, bgsub, divcal):
                    f.write(f"{float(x):.6f};{float(y_raw):.6f};{float(y_bg):.6f};{float(y_cal):.6f}\n")
            else:
                f.write("Wavelength_nm;Counts_raw;Counts_bgsub\n")
                for x, y_raw, y_bg in zip(wl, raw, bgsub):
                    f.write(f"{float(x):.6f};{float(y_raw):.6f};{float(y_bg):.6f}\n")
        _append_avaspec_log(avaspec_folder, base, fn.name, self.int_ms, self.averages, self.comment)
        return fn
    def run(self):
        stage = REGISTRY.get(self.stage_key)
        ctrl = REGISTRY.get(self.spec_key)
        if stage is None:
            self._emit(f"Stage '{self.stage_key}' not found in registry.")
            self.finished.emit("")
            return
        if ctrl is None or not hasattr(ctrl, "measure_once"):
            self._emit(f"Spectrometer '{self.spec_key}' not found or invalid.")
            self.finished.emit("")
            return
        try:
            if hasattr(ctrl, "set_params"):
                ctrl.set_params(self.int_ms, self.averages)
        except Exception as e:
            self._emit(f"Failed to set spectrometer params: {e}")
        now = datetime.datetime.now()
        scan_dir = _ensure_dir(_data_root() / f"{now:%Y-%m-%d}" / "Scans" / self.scan_name)
        date_str = f"{now:%Y-%m-%d}"
        idx = 1
        while True:
            candidate = scan_dir / f"{self.scan_name}_log_{date_str}_{idx}.log"
            if not candidate.exists():
                break
            idx += 1
        scan_log = candidate
        with open(scan_log, "w", encoding="utf-8") as lf:
            lf.write("SpecFile\tStageKey\tPosition_mm\tIntegration_ms\tAverages\tComment\n")
            lf.write(f"# {self.comment}\n")
        wl_ref = None
        n = len(self.positions)
        for i, pos in enumerate(self.positions, 1):
            if self.abort:
                self._emit("Scan aborted.")
                break
            try:
                stage.move_to(float(pos), blocking=True)
                self._emit(f"Moved {self.stage_key} to {pos:.3f} mm.")
            except Exception as e:
                self._emit(f"Move failed @ {pos:.3f}: {e}")
                self.progress.emit(i, n)
                continue
            time.sleep(self.settle_s)
            try:
                ts, wl, counts = ctrl.measure_once()
                wl = np.asarray(wl, float).ravel()
                y_raw = np.asarray(counts, float).ravel()
                if self.use_processed and hasattr(ctrl, "process_counts"):
                    _ = ctrl.process_counts(wl, y_raw)
            except Exception as e:
                self._emit(f"Acquisition failed @ {pos:.3f}: {e}")
                self.progress.emit(i, n)
                continue
            if wl_ref is None:
                wl_ref = wl.copy()
                y_disp = y_raw
                wl_disp = wl
            else:
                if wl.shape != wl_ref.shape or np.max(np.abs(wl - wl_ref)) > 1e-9:
                    y_disp = np.interp(wl_ref, wl, y_raw)
                    wl_disp = wl_ref
                else:
                    y_disp = y_raw
                    wl_disp = wl
            try:
                out_file = self._save_spectrum_like_ui(wl, y_raw, ctrl)
            except Exception as e:
                self._emit(f"Save spectrum failed @ {pos:.3f}: {e}")
                self.progress.emit(i, n)
                continue
            try:
                with open(scan_log, "a", encoding="utf-8") as lf:
                    lf.write(f"{Path(out_file).name}\t{self.stage_key}\t{pos:.6f}\t{self.int_ms:.3f}\t{self.averages}\t{self.comment}\n")
            except Exception as e:
                self._emit(f"Scan log write failed: {e}")
            self.partial.emit(float(pos), wl_disp.copy(), y_disp.copy())
            self.progress.emit(i, n)
            self._emit(f"Saved {Path(out_file).name} @ {pos:.3f} mm.")
        self.finished.emit(scan_log.as_posix())

class TOverlapTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Temporal Overlap Scan")
        self.resize(1000, 660)
        self._thread = None
        self._worker = None
        self._wl = None
        self._stack = None
        self._pos_list = []
        self._build_ui()
        self._refresh_devices()
    def _build_ui(self):
        main = QVBoxLayout(self)
        dev = QGroupBox("Devices")
        d = QHBoxLayout(dev)
        self.stage_combo = QComboBox()
        self.spec_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        d.addWidget(QLabel("Stage:")); d.addWidget(self.stage_combo, 1)
        d.addWidget(QLabel("Spectrometer:")); d.addWidget(self.spec_combo, 1)
        d.addWidget(self.refresh_btn)
        main.addWidget(dev)
        prm = QGroupBox("Scan parameters")
        p = QHBoxLayout(prm)
        self.start_sb = QDoubleSpinBox(); self.start_sb.setDecimals(3); self.start_sb.setRange(-1e6, 1e6); self.start_sb.setValue(0.0)
        self.end_sb   = QDoubleSpinBox(); self.end_sb.setDecimals(3);   self.end_sb.setRange(-1e6, 1e6);   self.end_sb.setValue(6.0)
        self.step_sb  = QDoubleSpinBox(); self.step_sb.setDecimals(3);  self.step_sb.setRange(1e-6, 1e6);  self.step_sb.setValue(0.05)
        self.settle_sb= QDoubleSpinBox(); self.settle_sb.setDecimals(2);self.settle_sb.setRange(0.0, 60.0);self.settle_sb.setValue(0.20)
        self.int_sb   = QDoubleSpinBox(); self.int_sb.setDecimals(1);   self.int_sb.setRange(1.0, 600000.0); self.int_sb.setValue(50.0)
        self.avg_sb   = QDoubleSpinBox(); self.avg_sb.setDecimals(0);   self.avg_sb.setRange(1, 1000);     self.avg_sb.setValue(1)
        self.proc_cb  = QCheckBox("Use processed counts (display only)")
        self.proc_cb.setChecked(True)
        self.name_edit= QLineEdit("temporal_overlap")
        p.addWidget(QLabel("Start (mm)"));  p.addWidget(self.start_sb)
        p.addWidget(QLabel("End (mm)"));    p.addWidget(self.end_sb)
        p.addWidget(QLabel("Step (mm)"));   p.addWidget(self.step_sb)
        p.addWidget(QLabel("Settle (s)"));  p.addWidget(self.settle_sb)
        p.addWidget(QLabel("Int (ms)"));    p.addWidget(self.int_sb)
        p.addWidget(QLabel("Avg"));         p.addWidget(self.avg_sb)
        p.addWidget(self.proc_cb)
        main.addWidget(prm)
        xlim_grp = QGroupBox("Display options")
        xl = QHBoxLayout(xlim_grp)
        self.xlim_cb = QCheckBox("Limit wavelength axis")
        self.xlim_center = QDoubleSpinBox(); self.xlim_center.setDecimals(2); self.xlim_center.setRange(0.0, 1e6); self.xlim_center.setValue(515.0)
        self.xlim_half  = QDoubleSpinBox(); self.xlim_half.setDecimals(2);   self.xlim_half.setRange(0.0, 1e6);   self.xlim_half.setValue(40.0)
        xl.addWidget(self.xlim_cb)
        xl.addWidget(QLabel("Center (nm)")); xl.addWidget(self.xlim_center)
        xl.addWidget(QLabel("± Halfwidth (nm)")); xl.addWidget(self.xlim_half)
        xl.addStretch(1)
        main.addWidget(xlim_grp)
        row = QHBoxLayout()
        self.comment_edit = QLineEdit("")
        self.start_btn = QPushButton("Start")
        self.abort_btn = QPushButton("Abort"); self.abort_btn.setEnabled(False)
        self.prog = QProgressBar(); self.prog.setMinimum(0); self.prog.setValue(0)
        self.start_btn.clicked.connect(self._start)
        self.abort_btn.clicked.connect(self._abort)
        row.addWidget(QLabel("Comment:")); row.addWidget(self.comment_edit, 1)
        row.addWidget(self.start_btn); row.addWidget(self.abort_btn); row.addWidget(self.prog, 1)
        main.addLayout(row)
        self.figure = Figure(figsize=(8.8, 5.0), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        main.addWidget(self.canvas, 1)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Position (mm)")
        self.im = None
        self.log = QTextEdit(); self.log.setReadOnly(True)
        main.addWidget(self.log, 0)
    def _refresh_devices(self):
        self.stage_combo.clear()
        for k in REGISTRY.keys("stage:"):
            if k.startswith("stage:serial:"):
                continue
            self.stage_combo.addItem(k)
        self.spec_combo.clear()
        for k in REGISTRY.keys("spectrometer:avaspec:"):
            self.spec_combo.addItem(k)
    @staticmethod
    def _linspace_positions(a0, a1, step):
        if step <= 0:
            raise ValueError("Step must be > 0.")
        if a1 >= a0:
            nsteps = int(np.floor((a1 - a0) / step))
            pos = [a0 + i * step for i in range(nsteps + 1)]
            if pos[-1] < a1 - 1e-12:
                pos.append(a1)
        else:
            nsteps = int(np.floor((a0 - a1) / step))
            pos = [a0 - i * step for i in range(nsteps + 1)]
            if pos[-1] > a1 + 1e-12:
                pos.append(a1)
        return pos
    def _start(self):
        try:
            stage_key = self.stage_combo.currentText().strip()
            spec_key  = self.spec_combo.currentText().strip()
            if not stage_key or not spec_key:
                raise ValueError("Select a stage and a spectrometer.")
            a0 = float(self.start_sb.value())
            a1 = float(self.end_sb.value())
            step = float(self.step_sb.value())
            positions = self._linspace_positions(a0, a1, step)
            settle = float(self.settle_sb.value())
            int_ms = float(self.int_sb.value())
            avg = int(self.avg_sb.value())
            use_processed = bool(self.proc_cb.isChecked())
            scan_name = self.name_edit.text().strip() or "temporal_overlap"
            comment = self.comment_edit.text()
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return
        self._wl = None
        self._stack = None
        self._pos_list = []
        self._thread = QThread(self)
        self._worker = TOverlapWorker(
            stage_key=stage_key, spec_key=spec_key, positions=positions,
            settle_s=settle, int_ms=int_ms, averages=avg, use_processed=use_processed,
            scan_name=scan_name, comment=comment
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.partial.connect(self._on_partial, Qt.QueuedConnection)
        self._worker.finished.connect(self._finished, Qt.QueuedConnection)
        self._thread.finished.connect(self._thread.deleteLater)
        self.prog.setMaximum(len(positions))
        self.prog.setValue(0)
        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self._log("Scan started…")
        self._thread.start()
    def _abort(self):
        if self._worker:
            self._worker.abort = True
            self._log("Abort requested.")
            self.abort_btn.setEnabled(False)
    def _on_progress(self, i, n):
        self.prog.setMaximum(n)
        self.prog.setValue(i)
    def _on_partial(self, pos_mm, wl, y):
        wl = np.asarray(wl, float).ravel()
        y = np.asarray(y, float).ravel()
        if self._wl is None:
            self._wl = wl
            self._stack = np.zeros((0, wl.size), dtype=float)
        self._stack = np.vstack([self._stack, y])
        self._pos_list.append(float(pos_mm))
        self._update_image()
    def _finished(self, scan_log_path):
        if self._thread and self._thread.isRunning():
            self._thread.quit(); self._thread.wait()
        self._thread = None
        self._worker = None
        self.abort_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        if scan_log_path:
            self._log(f"Scan finished. Log: {scan_log_path}")
        else:
            self._log("Scan finished with errors.")
    def _update_image(self):
        if self._wl is None or self._stack is None or self._stack.size == 0 or not self._pos_list:
            return
        wl_min = float(self._wl[0])
        wl_max = float(self._wl[-1])
        y_min = float(min(self._pos_list))
        y_max = float(max(self._pos_list))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            y_min, y_max = 0.0, 1.0
        if abs(y_max - y_min) < 1e-12:
            y_max = y_min + 1e-9
        self.ax.cla()
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Position (mm)")
        extent = [wl_min, wl_max, y_min, y_max]
        self.im = self.ax.imshow(self._stack, aspect="auto", origin="lower", extent=extent, interpolation="nearest", cmap='cmr.amethyst_r')
        if self.xlim_cb.isChecked():
            c = float(self.xlim_center.value())
            hw = float(self.xlim_half.value())
            self.ax.set_xlim(c - hw, c + hw)
        self.canvas.draw_idle()
    def _log(self, s):
        self.log.append(s)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

def main():
    app = QApplication(sys.argv)
    w = TOverlapTab()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
