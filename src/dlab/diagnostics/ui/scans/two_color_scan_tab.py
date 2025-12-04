from __future__ import annotations

import datetime, time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image, PngImagePlugin

from PyQt5.QtCore import QTimer, QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QPushButton,
    QDoubleSpinBox, QTextEdit, QProgressBar, QMessageBox, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QLineEdit
)

from dlab.boot import ROOT, get_config
from dlab.core.device_registry import REGISTRY

import logging
logger = logging.getLogger("dlab.scans.two_color_scan_tab")


def _data_root() -> Path:
    cfg = get_config() or {}
    base = cfg.get("paths", {}).get("data_root", "C:/data")
    return (ROOT / base).resolve()


def _save_png_with_meta(folder: Path, filename: str, frame_u16: np.ndarray, meta: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    img = Image.fromarray(frame_u16, mode="I;16")
    pnginfo = PngImagePlugin.PngInfo()
    for k, v in meta.items():
        pnginfo.add_text(str(k), str(v))
    img.save(path.as_posix(), format="PNG", pnginfo=pnginfo)
    return path


class TwoColorScanWorker(QObject):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(
        self,
        phase_ctrl_key: str,
        setpoints: List[float],
        detector_params: Dict[str, tuple],
        settle_s: float,
        phase_avg_s: float,
        scan_name: str,
        comment: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.phase_ctrl_key = phase_ctrl_key
        self.setpoints = setpoints
        self.detector_params = detector_params
        self.settle_s = float(settle_s)
        self.phase_avg_s = float(phase_avg_s)
        self.scan_name = scan_name
        self.comment = comment
        self.abort = False

    def _emit(self, msg: str) -> None:
        self.log.emit(msg)
        logger.info(msg)

    def run(self) -> None:
        phase_ctrl = REGISTRY.get(self.phase_ctrl_key)
        if phase_ctrl is None:
            self._emit(f"Phase controller '{self.phase_ctrl_key}' not found.")
            self.finished.emit("")
            return

        if not hasattr(phase_ctrl, 'set_target'):
            self._emit(f"Phase controller doesn't have required API methods.")
            self.finished.emit("")
            return
        
        if not phase_ctrl.is_locked():
            self._emit(f"Phase controller is not locked. Please enable lock first.")
            self.finished.emit("")
            return

        detectors = {}
        for det_key, params in self.detector_params.items():
            dev = REGISTRY.get(det_key)
            if dev is None:
                self._emit(f"Detector '{det_key}' not found.")
                self.finished.emit("")
                return

            is_camera = hasattr(dev, "grab_frame_for_scan")
            is_spectro = hasattr(dev, "measure_spectrum") or hasattr(dev, "grab_spectrum_for_scan")
            is_pow = hasattr(dev, "fetch_power") or hasattr(dev, "read_power")

            if not (is_camera or is_spectro or is_pow):
                self._emit(f"Detector '{det_key}' doesn't expose a scan API.")
                self.finished.emit("")
                return

            try:
                if is_camera:
                    exposure = int(params[0])
                    if hasattr(dev, "set_exposure_us"):
                        dev.set_exposure_us(exposure)
                    elif hasattr(dev, "setExposureUS"):
                        dev.setExposureUS(exposure)
                    elif hasattr(dev, "set_exposure"):
                        dev.set_exposure(exposure)
            except Exception as e:
                self._emit(f"Warning: failed to preset on '{det_key}': {e}")

            detectors[det_key] = dev

        now = datetime.datetime.now()
        root = _data_root()

        scan_dir = root / f"{now:%Y-%m-%d}" / "TwoColorScans" / self.scan_name
        scan_dir.mkdir(parents=True, exist_ok=True)

        date_str = f"{now:%Y-%m-%d}"
        idx = 1
        while True:
            candidate = scan_dir / f"{self.scan_name}_log_{date_str}_{idx}.log"
            if not candidate.exists():
                break
            idx += 1
        scan_log = candidate

        with open(scan_log, "w", encoding="utf-8") as f:
            header = ["Setpoint_rad", "MeasuredPhase_rad", "PhaseStd_rad", "PhaseError_rad", 
                     "DetectorKey", "DataFile", "Exposure_or_IntTime", "Averages"]
            f.write("\t".join(header) + "\n")
            f.write(f"# {self.comment}\n")
            f.write(f"# Phase controller: {self.phase_ctrl_key}\n")
            f.write(f"# Settle time: {self.settle_s} s\n")
            f.write(f"# Phase averaging: {self.phase_avg_s} s\n")

        total_points = len(self.setpoints) * len(detectors)
        done = 0

        try:
            for sp in self.setpoints:
                if self.abort:
                    self._emit("Scan aborted.")
                    self.finished.emit("")
                    return

                phase_ctrl.set_target(float(sp))
                self._emit(f"Moving to setpoint: {sp:.4f} rad")

                time.sleep(self.settle_s)

                avg_phase, std_phase = phase_ctrl.get_phase_average(self.phase_avg_s)
                phase_error = avg_phase - sp

                for det_key, dev in detectors.items():
                    if self.abort:
                        self._emit("Scan aborted.")
                        self.finished.emit("")
                        return

                    params = self.detector_params.get(det_key, (0, 1))

                    try:
                        if hasattr(dev, "grab_frame_for_scan"):
                            exposure_or_int = int(params[0]) if len(params) >= 1 else 0
                            averages = int(params[1]) if len(params) >= 2 else 1
                            
                            try:
                                frame_u16, meta = dev.grab_frame_for_scan(
                                    averages=int(averages),
                                    background=False,
                                    dead_pixel_cleanup=True,
                                    exposure_us=int(exposure_or_int),
                                )
                            except TypeError:
                                frame_u16, meta = dev.grab_frame_for_scan(
                                    averages=int(averages),
                                    background=False,
                                    dead_pixel_cleanup=True,
                                )
                            
                            exp_meta = int((meta or {}).get("Exposure_us", exposure_or_int))
                            
                            det_name = det_key.split(":")[-1]
                            det_day = root / f"{now:%Y-%m-%d}" / det_name
                            ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            fn = f"{det_name}_sp{sp:.4f}_{ts_ms}.png"
                            
                            meta_dict = {
                                "Exposure_us": exp_meta, 
                                "Setpoint_rad": sp,
                                "MeasuredPhase_rad": avg_phase,
                                "PhaseStd_rad": std_phase,
                                "Comment": self.comment
                            }
                            _save_png_with_meta(det_day, fn, frame_u16, meta_dict)
                            
                            data_fn = fn
                            saved_label = f"exp {exp_meta} µs"

                        elif hasattr(dev, "measure_spectrum") or hasattr(dev, "grab_spectrum_for_scan"):
                            exposure_or_int = float(params[0]) if len(params) >= 1 else 0.0
                            averages = int(params[1]) if len(params) >= 2 else 1
                            
                            if hasattr(dev, "get_wavelengths"):
                                wl = np.asarray(dev.get_wavelengths(), dtype=float)
                            else:
                                wl = np.asarray(getattr(dev, "wavelength", None), dtype=float)

                            if wl is None or wl.size == 0:
                                self._emit(f"{det_key}: wavelength array empty.")
                                continue

                            if hasattr(dev, "grab_spectrum_for_scan"):
                                counts, meta = dev.grab_spectrum_for_scan(
                                    int_ms=float(exposure_or_int),
                                    averages=int(averages)
                                )
                                counts = np.asarray(counts, dtype=float)
                                int_ms = float((meta or {}).get("Integration_ms", float(exposure_or_int)))
                            else:
                                _buf = []
                                for _ in range(int(averages)):
                                    _ts, _data = dev.measure_spectrum(float(exposure_or_int), 1)
                                    _buf.append(np.asarray(_data, dtype=float))
                                    time.sleep(0.01)
                                counts = np.mean(np.stack(_buf, axis=0), axis=0)
                                int_ms = float(exposure_or_int)

                            det_day = root / f"{now:%Y-%m-%d}" / "Avaspec"
                            safe_name = det_key.split(":")[-1].replace(" ", "")
                            ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            fn = f"{safe_name}_sp{sp:.4f}_{ts_ms}.txt"
                            
                            header = {
                                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "IntegrationTime_ms": int_ms,
                                "Averages": averages,
                                "Setpoint_rad": sp,
                                "MeasuredPhase_rad": avg_phase,
                                "PhaseStd_rad": std_phase,
                                "Comment": self.comment,
                            }
                            det_day.mkdir(parents=True, exist_ok=True)
                            path = det_day / fn
                            lines = [f"# {k}: {v}" for k, v in header.items()]
                            lines.append("Wavelength_nm;Counts")
                            with open(path, "w", encoding="utf-8") as f:
                                f.write("\n".join(lines) + "\n")
                                for xv, yv in zip(wl, counts):
                                    f.write(f"{float(xv):.6f};{float(yv):.6f}\n")
                            
                            data_fn = fn
                            saved_label = f"int {int_ms:.0f} ms"

                        else:
                            period_ms = float(params[0]) if len(params) >= 1 else 100.0
                            averages = int(params[1]) if len(params) >= 2 else 1

                            vals = []
                            n_avg = max(1, int(averages))
                            for i in range(n_avg):
                                if hasattr(dev, "read_power"):
                                    v = float(dev.read_power())
                                else:
                                    v = float(dev.fetch_power())
                                vals.append(v)
                                if i + 1 < n_avg:
                                    time.sleep(period_ms / 1000.0)

                            power = float(np.mean(vals)) if vals else float("nan")
                            data_fn = f"{power:.9f}"
                            saved_label = f"P={power:.3e} W"

                        row = [
                            f"{float(sp):.9f}",
                            f"{float(avg_phase):.9f}",
                            f"{float(std_phase):.9f}",
                            f"{float(phase_error):.9f}",
                            det_key,
                            data_fn,
                            str(params[0] if len(params) >= 1 else ""),
                            str(params[1] if len(params) >= 2 else ""),
                        ]

                        with open(scan_log, "a", encoding="utf-8") as f:
                            f.write("\t".join(row) + "\n")

                        self._emit(
                            f"Saved {data_fn} @ SP={sp:.4f} rad, "
                            f"Phase={avg_phase:.4f}±{std_phase:.4f} rad, Error={phase_error:.4f} rad "
                            f"({saved_label})"
                        )

                    except Exception as e:
                        self._emit(f"Capture failed @ SP={sp:.4f} rad on {det_key}: {e}")

                    done += 1
                    self.progress.emit(done, total_points)

        except Exception as e:
            self._emit(f"Fatal error: {e}")
            self.finished.emit("")
            return

        self.finished.emit(scan_log.as_posix())


class TwoColorScanTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._worker = None
        self._build_ui()
        self._refresh_devices()

    def _build_ui(self):
        main = QVBoxLayout(self)

        ctrl_box = QGroupBox("Phase Lock Controller")
        ctrl_l = QHBoxLayout(ctrl_box)
        self.phase_ctrl_picker = QComboBox()
        ctrl_l.addWidget(QLabel("Controller:"))
        ctrl_l.addWidget(self.phase_ctrl_picker, 1)
        main.addWidget(ctrl_box)

        sp_box = QGroupBox("Setpoints")
        sp_l = QVBoxLayout(sp_box)
        
        sp_params = QHBoxLayout()
        self.sp_start = QLineEdit("-3.14159")
        self.sp_end = QLineEdit("3.14159")
        self.sp_step = QLineEdit("0.5")
        sp_params.addWidget(QLabel("Start [rad]:"))
        sp_params.addWidget(self.sp_start)
        sp_params.addWidget(QLabel("End [rad]:"))
        sp_params.addWidget(self.sp_end)
        sp_params.addWidget(QLabel("Step [rad]:"))
        sp_params.addWidget(self.sp_step)
        sp_l.addLayout(sp_params)
        main.addWidget(sp_box)

        det_box = QGroupBox("Detectors")
        det_l = QVBoxLayout(det_box)

        det_pick = QHBoxLayout()
        self.det_picker = QComboBox()
        self.add_det_btn = QPushButton("Add Detector")
        self.add_det_btn.clicked.connect(self._add_det_row)
        det_pick.addWidget(QLabel("Detector:"))
        det_pick.addWidget(self.det_picker, 1)
        det_pick.addWidget(self.add_det_btn)
        det_l.addLayout(det_pick)

        self.det_tbl = QTableWidget(0, 3)
        self.det_tbl.setHorizontalHeaderLabels(["DetectorKey", "Exposure_us/Int_ms", "Averages"])
        self.det_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.det_tbl.setEditTriggers(QAbstractItemView.AllEditTriggers)
        det_l.addWidget(self.det_tbl)

        rm_det = QHBoxLayout()
        self.rm_det_btn = QPushButton("Remove Selected")
        self.rm_det_btn.clicked.connect(self._remove_det_row)
        rm_det.addStretch(1)
        rm_det.addWidget(self.rm_det_btn)
        det_l.addLayout(rm_det)

        main.addWidget(det_box)

        params = QGroupBox("Scan Parameters")
        p = QHBoxLayout(params)
        self.settle_sb = QDoubleSpinBox()
        self.settle_sb.setDecimals(2)
        self.settle_sb.setRange(0.0, 60.0)
        self.settle_sb.setValue(2.0)
        
        self.phase_avg_sb = QDoubleSpinBox()
        self.phase_avg_sb.setDecimals(2)
        self.phase_avg_sb.setRange(0.1, 60.0)
        self.phase_avg_sb.setValue(1.0)
        
        self.scan_name = QLineEdit("")
        self.comment = QLineEdit("")
        
        p.addWidget(QLabel("Settle (s)"))
        p.addWidget(self.settle_sb)
        p.addWidget(QLabel("Phase avg (s)"))
        p.addWidget(self.phase_avg_sb)
        p.addWidget(QLabel("Scan name"))
        p.addWidget(self.scan_name, 1)
        p.addWidget(QLabel("Comment"))
        p.addWidget(self.comment, 2)
        main.addWidget(params)

        ctl = QHBoxLayout()
        self.start_btn = QPushButton("Start Scan")
        self.start_btn.clicked.connect(self._start)
        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self._abort)
        self.prog = QProgressBar()
        self.prog.setMinimum(0)
        self.prog.setValue(0)
        ctl.addWidget(self.start_btn)
        ctl.addWidget(self.abort_btn)
        ctl.addWidget(self.prog, 1)
        main.addLayout(ctl)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        main.addWidget(self.log, 1)

        rr = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Devices")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        rr.addStretch(1)
        rr.addWidget(self.refresh_btn)
        main.addLayout(rr)

    def _add_det_row(self):
        det_key = self.det_picker.currentText().strip()
        if not det_key:
            QMessageBox.warning(self, "Pick a detector", "Select a detector to add.")
            return
        r = self.det_tbl.rowCount()
        self.det_tbl.insertRow(r)
        self.det_tbl.setItem(r, 0, QTableWidgetItem(det_key))
        self.det_tbl.setItem(r, 1, QTableWidgetItem("5000"))
        self.det_tbl.setItem(r, 2, QTableWidgetItem("1"))

    def _remove_det_row(self):
        rows = sorted({i.row() for i in self.det_tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self.det_tbl.removeRow(r)

    def _refresh_devices(self):
        self.phase_ctrl_picker.clear()
        for k in REGISTRY.keys("phaselock:"):
            self.phase_ctrl_picker.addItem(k)
        
        self.det_picker.clear()
        for prefix in ("camera:daheng:", "camera:andor:", "spectrometer:avaspec:", "powermeter:"):
            for k in REGISTRY.keys(prefix):
                if ":index:" not in k:
                    self.det_picker.addItem(k)

    def _positions(self, start, end, step):
        if step <= 0:
            raise ValueError("Step must be > 0.")
        if end >= start:
            n = int((end - start) / step)
            vals = [start + i * step for i in range(n + 1)]
            if abs(vals[-1] - end) > 1e-9:
                vals.append(end)
        else:
            n = int((start - end) / step)
            vals = [start - i * step for i in range(n + 1)]
            if abs(vals[-1] - end) > 1e-9:
                vals.append(end)
        return vals

    def _start(self):
        phase_ctrl_key = self.phase_ctrl_picker.currentText().strip()
        if not phase_ctrl_key:
            QMessageBox.critical(self, "Error", "Select a phase lock controller.")
            return

        try:
            start = float(self.sp_start.text())
            end = float(self.sp_end.text())
            step = float(self.sp_step.text())
            setpoints = self._positions(start, end, step)
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid setpoint parameters: {e}")
            return

        if self.det_tbl.rowCount() == 0:
            QMessageBox.critical(self, "Error", "Add at least one detector.")
            return

        detector_params = {}
        for r in range(self.det_tbl.rowCount()):
            det = (self.det_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            if not det:
                QMessageBox.critical(self, "Error", f"Empty detector key at row {r+1}.")
                return
            p1 = (self.det_tbl.item(r, 1) or QTableWidgetItem("0")).text()
            p2 = (self.det_tbl.item(r, 2) or QTableWidgetItem("1")).text()
            detector_params[det] = (int(float(p1)), int(float(p2)))

        settle = float(self.settle_sb.value())
        phase_avg = float(self.phase_avg_sb.value())
        name = self.scan_name.text().strip()
        if not name:
            QMessageBox.critical(self, "Error", "Enter a scan name.")
            return
        comment = self.comment.text()

        self._thread = QThread(self)
        self._worker = TwoColorScanWorker(
            phase_ctrl_key=phase_ctrl_key,
            setpoints=setpoints,
            detector_params=detector_params,
            settle_s=settle,
            phase_avg_s=phase_avg,
            scan_name=name,
            comment=comment,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)

        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._finished)

        self._thread.finished.connect(self._thread.deleteLater)

        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)

        total = len(setpoints) * len(detector_params)
        self.prog.setMaximum(total)
        self.prog.setValue(0)

        self._thread.start()
        self._log("Two-color scan started...")

    def _abort(self):
        if self._worker:
            self._worker.abort = True
            self.abort_btn.setEnabled(False)

    def _on_progress(self, i, n):
        self.prog.setMaximum(n)
        self.prog.setValue(i)

    def _finished(self, log_path):
        if log_path:
            self._log(f"Scan finished: {log_path}")
        else:
            self._log("Scan finished with errors.")

        self.abort_btn.setEnabled(False)
        self.start_btn.setEnabled(True)

        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())