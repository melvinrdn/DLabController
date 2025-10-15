# src/dlab/diagnostics/ui/scans/stability_scan.py
from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, QThread, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QPushButton,
    QDoubleSpinBox, QSpinBox, QTextEdit, QProgressBar, QMessageBox, QLineEdit, QDialog
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from dlab.boot import ROOT, get_config
from dlab.core.device_registry import REGISTRY
import logging

logger = logging.getLogger("dlab.scans.stability_scan")


def _data_root() -> Path:
    cfg = get_config() or {}
    base = cfg.get("paths", {}).get("data_root", "C:/data")
    return (ROOT / base).resolve()


def _open_new_log(scan_dir: Path, scan_name: str, comment: str, det_key: str,
                  params_text: str, background: bool, mcp_voltage: str) -> Path:
    now = datetime.datetime.now()
    scan_dir.mkdir(parents=True, exist_ok=True)
    date_str = f"{now:%Y-%m-%d}"
    idx = 1
    while True:
        candidate = scan_dir / f"{scan_name}_stability_{date_str}_{idx}.log"
        if not candidate.exists():
            break
        idx += 1
    with open(candidate, "w", encoding="utf-8") as f:
        f.write("# Stability scan\n")
        f.write(f"# TimestampStart: {now:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"# Comment: {comment}\n")
        f.write(f"# Background: {bool(background)}\n")
        f.write(f"# MCP_Voltage: {mcp_voltage}\n")
        f.write(f"# Detector: {det_key}\n")
        f.write(f"# Params: {params_text}\n")
        f.write("Timestamp\tElapsed_s\tDetectorKey\tMetric\tValue\tExposure_or_Int_or_Period\t"
                "Averages_or_None\tMCP_Voltage\n")
    return candidate


def _append_row(log_path: Path, det_key: str, metric: str, value: float,
                expo_or_int_or_period: str, averages: str, mcp_voltage: str,
                t0: float) -> None:
    ts = time.time()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("{ts}\t{elapsed:.6f}\t{det}\t{metric}\t{val:.9g}\t{eip}\t{avg}\t{mcp}\n".format(
            ts=datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
            elapsed=float(ts - float(t0)),
            det=det_key,
            metric=metric,
            val=float(value),
            eip=str(expo_or_int_or_period),
            avg=str(averages),
            mcp=str(mcp_voltage),
        ))


def _detector_kind(dev) -> str:
    if hasattr(dev, "grab_frame_for_scan"):
        return "camera"
    if hasattr(dev, "measure_spectrum") or hasattr(dev, "grab_spectrum_for_scan"):
        return "spectro"
    if hasattr(dev, "read_power") or hasattr(dev, "fetch_power"):
        return "powermeter"
    return "unknown"


class StabilityScanWorker(QObject):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self,
                 det_key: str,
                 det_params: Tuple,
                 sample_ms: float,
                 duration_s: float,
                 metric: str,
                 scan_name: str,
                 comment: str,
                 mcp_voltage: str,
                 background: bool = False,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.det_key = str(det_key)
        self.det_params = det_params
        self.sample_ms = float(sample_ms)
        self.duration_s = float(duration_s)
        self.metric = str(metric).lower().strip()
        self.scan_name = str(scan_name)
        self.comment = str(comment)
        self.mcp_voltage = str(mcp_voltage)
        self.background = bool(background)
        self.abort = False

    def _emit(self, msg: str) -> None:
        self.log.emit(msg)
        logger.info(msg)

    def _preset_device(self, dev, kind: str) -> None:
        p = self.det_params
        try:
            if kind == "camera":
                exposure = int(p[0]) if len(p) >= 1 else 0
                if hasattr(dev, "set_exposure_us"):
                    dev.set_exposure_us(exposure)
                elif hasattr(dev, "setExposureUS"):
                    dev.setExposureUS(exposure)
                elif hasattr(dev, "set_exposure"):
                    dev.set_exposure(exposure)
            elif kind == "powermeter":
                if len(p) >= 3 and p[2] not in ("", None) and hasattr(dev, "set_wavelength"):
                    try:
                        dev.set_wavelength(float(p[2]))
                    except Exception:
                        pass
                if len(p) >= 2 and hasattr(dev, "set_avg"):
                    try:
                        dev.set_avg(int(p[1]))
                    except Exception:
                        pass
        except Exception as e:
            self._emit(f"Warning: failed device preset on '{self.det_key}': {e}")

    def run(self) -> None:
        dev = REGISTRY.get(self.det_key)
        if dev is None:
            self._emit(f"Detector '{self.det_key}' not found.")
            self.finished.emit("")
            return

        kind = _detector_kind(dev)
        if kind == "unknown":
            self._emit(f"Detector '{self.det_key}' has no supported API.")
            self.finished.emit("")
            return

        now = datetime.datetime.now()
        root = _data_root()
        scan_dir = root / f"{now:%Y-%m-%d}" / "Scans" / self.scan_name

        if kind == "camera":
            e = int(self.det_params[0]) if len(self.det_params) >= 1 else 0
            a = int(self.det_params[1]) if len(self.det_params) >= 2 else 1
            params_text = f"Exposure_us={e}; Averages={a}; Sample_ms={self.sample_ms}"
            averages_text = str(a)
            expo_or_int_or_period_text = str(e)
        elif kind == "spectro":
            integ = float(self.det_params[0]) if len(self.det_params) >= 1 else 0.0
            a = int(self.det_params[1]) if len(self.det_params) >= 2 else 1
            params_text = f"Integration_ms={integ}; Averages={a}; Sample_ms={self.sample_ms}"
            averages_text = str(a)
            expo_or_int_or_period_text = f"{integ:.0f}"
        else:
            a = int(self.det_params[1]) if len(self.det_params) >= 2 else 1
            wl = self.det_params[2] if len(self.det_params) >= 3 else ""
            params_text = f"Sample_ms={self.sample_ms}; Averages={a}; Wavelength_nm={wl}"
            averages_text = str(a)
            expo_or_int_or_period_text = f"{self.sample_ms:.0f}"

        try:
            log_path = _open_new_log(
                scan_dir=scan_dir,
                scan_name=self.scan_name,
                comment=self.comment,
                det_key=self.det_key,
                params_text=params_text,
                background=self.background,
                mcp_voltage=self.mcp_voltage,
            )
        except Exception as e:
            self._emit(f"Failed to open log file: {e}")
            self.finished.emit("")
            return

        self._preset_device(dev, kind)

        if self.sample_ms <= 0:
            self.sample_ms = 1.0
        dt = self.sample_ms / 1000.0
        total_steps = max(1, int(self.duration_s // dt))

        self.progress.emit(0, total_steps)
        self._emit(f"Stability scan running for {self.duration_s:.2f} s (period {self.sample_ms:.0f} ms)…")

        t0 = time.time()
        step = 0

        try:
            while (time.time() - t0) < self.duration_s:
                if self.abort:
                    self._emit("Scan aborted.")
                    self.finished.emit("")
                    return

                t_start = time.time()

                try:
                    if kind == "camera":
                        exposure = int(self.det_params[0]) if len(self.det_params) >= 1 else 0
                        averages = int(self.det_params[1]) if len(self.det_params) >= 2 else 1
                        try:
                            frame_u16, _meta = dev.grab_frame_for_scan(
                                averages=int(averages),
                                background=self.background,
                                dead_pixel_cleanup=True,
                                exposure_us=int(exposure),
                            )
                        except TypeError:
                            frame_u16, _meta = dev.grab_frame_for_scan(
                                averages=int(averages),
                                background=self.background,
                                dead_pixel_cleanup=True,
                            )
                        if self.metric == "sum":
                            val = float(np.asarray(frame_u16, dtype=np.uint64).sum())
                        else:
                            val = float(np.asarray(frame_u16, dtype=np.uint32).max())

                    elif kind == "spectro":
                        integ_ms = float(self.det_params[0]) if len(self.det_params) >= 1 else 0.0
                        averages = int(self.det_params[1]) if len(self.det_params) >= 2 else 1
                        if hasattr(dev, "grab_spectrum_for_scan"):
                            counts, _meta = dev.grab_spectrum_for_scan(int_ms=float(integ_ms), averages=int(averages))
                            y = np.asarray(counts, dtype=float)
                        else:
                            buf = []
                            for _ in range(max(1, averages)):
                                _ts, _data = dev.measure_spectrum(float(integ_ms), 1)
                                buf.append(np.asarray(_data, dtype=float))
                                time.sleep(0.01)
                            y = np.mean(np.stack(buf, axis=0), axis=0)
                        if self.metric == "sum":
                            val = float(np.asarray(y, dtype=np.float64).sum())
                        else:
                            val = float(np.asarray(y, dtype=np.float64).max())

                    else:
                        averages = int(self.det_params[1]) if len(self.det_params) >= 2 else 1
                        inner_period = dt
                        vals = []
                        for i in range(max(1, averages)):
                            if hasattr(dev, "read_power"):
                                v = float(dev.read_power())
                            else:
                                v = float(dev.fetch_power())
                            vals.append(v)
                            if i + 1 < averages and inner_period > 0:
                                time.sleep(inner_period)
                        val = float(np.mean(vals)) if vals else float("nan")

                    metric_txt = self.metric if kind != "powermeter" else "power"
                    _append_row(
                        log_path=log_path,
                        det_key=self.det_key,
                        metric=metric_txt,
                        value=val,
                        expo_or_int_or_period=expo_or_int_or_period_text,
                        averages=averages_text,
                        mcp_voltage=self.mcp_voltage,
                        t0=t0,
                    )

                    step += 1
                    self.progress.emit(step, total_steps)
                    self._emit(f"t={time.time()-t0:.1f}s -> {metric_txt}={val:.6g}")

                except Exception as e:
                    self._emit(f"Acquisition failed on {self.det_key}: {e}")

                elapsed = time.time() - t_start
                sleep_left = dt - elapsed
                if sleep_left > 0:
                    time.sleep(sleep_left)

        except Exception as e:
            self._emit(f"Fatal error: {e}")
            self.finished.emit("")
            return

        self._emit(f"Stability scan finished. Log: {log_path.as_posix()}")
        self.finished.emit(log_path.as_posix())


class _MetricPlotDialog(QDialog):
    def __init__(self, log_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stability metric over time")
        self.resize(800, 500)
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(6, 3.5), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        try:
            t, y, metric = self._load_log(log_path)
            ax = self.fig.add_subplot(111)
            ax.plot(t, y)
            ax.set_xlabel("Elapsed time (s)")
            ax.set_ylabel(metric)
            ax.set_title(Path(log_path).name)
            ax.grid(True, which="both", alpha=0.3)
            self.canvas.draw()
        except Exception as e:
            msg = QTextEdit(str(e)); msg.setReadOnly(True)
            layout.addWidget(msg)

    def _load_log(self, path: str):
        t = []
        y = []
        metric = "Value"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                elapsed = float(parts[1])
                met = parts[3].strip()
                val = float(parts[4])
                t.append(elapsed)
                y.append(val)
                metric = met
        if not t:
            raise RuntimeError("No data rows found in log.")
        return np.asarray(t, dtype=float), np.asarray(y, dtype=float), metric


class StabilityScanTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[StabilityScanWorker] = None
        self._last_log: Optional[str] = None

        self._build_ui()
        self._refresh_devices()

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(5000)
        self._refresh_timer.timeout.connect(self._refresh_devices)
        self._refresh_timer.start()

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)

        det_box = QGroupBox("Detector")
        det_l = QHBoxLayout(det_box)
        self.det_picker = QComboBox()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        det_l.addWidget(QLabel("Key:")); det_l.addWidget(self.det_picker, 1); det_l.addWidget(self.refresh_btn)
        main.addWidget(det_box)

        p_box = QGroupBox("Acquisition parameters")
        p = QHBoxLayout(p_box)
        self.duration_s = QDoubleSpinBox(); self.duration_s.setDecimals(1); self.duration_s.setRange(0.1, 24 * 3600.0); self.duration_s.setValue(60.0)
        self.sample_ms = QDoubleSpinBox(); self.sample_ms.setDecimals(0); self.sample_ms.setRange(1.0, 3600_000.0); self.sample_ms.setValue(500.0)
        self.metric_cb = QComboBox(); self.metric_cb.addItems(["Max pixel", "Sum of pixels"])
        p.addWidget(QLabel("Duration (s)")); p.addWidget(self.duration_s)
        p.addWidget(QLabel("Period (ms)"));  p.addWidget(self.sample_ms)
        p.addWidget(QLabel("Metric"));       p.addWidget(self.metric_cb)
        main.addWidget(p_box)

        dev_box = QGroupBox("Detector-specific")
        d = QHBoxLayout(dev_box)
        self.exposure_us = QSpinBox(); self.exposure_us.setRange(0, 10_000_000); self.exposure_us.setValue(5000)
        self.integration_ms = QDoubleSpinBox(); self.integration_ms.setDecimals(1); self.integration_ms.setRange(0.0, 60_000.0); self.integration_ms.setValue(10.0)
        self.averages = QSpinBox(); self.averages.setRange(1, 10_000); self.averages.setValue(1)
        self.wavelength_nm = QDoubleSpinBox(); self.wavelength_nm.setDecimals(1); self.wavelength_nm.setRange(0.0, 10_000.0); self.wavelength_nm.setValue(1030.0)
        d.addWidget(QLabel("Exposure (us)")); d.addWidget(self.exposure_us)
        d.addWidget(QLabel("Integration (ms)")); d.addWidget(self.integration_ms)
        d.addWidget(QLabel("Averages")); d.addWidget(self.averages)
        d.addWidget(QLabel("Wavelength (nm, PM)")); d.addWidget(self.wavelength_nm)
        main.addWidget(dev_box)

        meta_box = QGroupBox("Scan metadata")
        m = QHBoxLayout(meta_box)
        self.scan_name = QLineEdit("stability_scan")
        self.scan_name.setEnabled(False)
        self.comment = QLineEdit("")
        self.mcp_voltage = QLineEdit("")
        m.addWidget(QLabel("Scan name")); m.addWidget(self.scan_name, 1)
        m.addWidget(QLabel("Comment"));   m.addWidget(self.comment, 2)
        m.addWidget(QLabel("MCP voltage")); m.addWidget(self.mcp_voltage, 1)
        main.addWidget(meta_box)

        ctl = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.abort_btn = QPushButton("Abort"); self.abort_btn.setEnabled(False)
        self.plot_btn = QPushButton("Open Plot"); self.plot_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        self.abort_btn.clicked.connect(self._abort)
        self.plot_btn.clicked.connect(self._open_plot)
        self.prog = QProgressBar(); self.prog.setMinimum(0); self.prog.setValue(0)
        ctl.addWidget(self.start_btn); ctl.addWidget(self.abort_btn); ctl.addWidget(self.plot_btn); ctl.addWidget(self.prog, 1)
        main.addLayout(ctl)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        main.addWidget(self.log, 1)

        self.det_picker.currentTextChanged.connect(self._on_detector_changed)

    def _refresh_devices(self) -> None:
        cur = self.det_picker.currentText().strip()
        keys = []
        for prefix in ("camera:daheng:", "camera:andor:", "spectrometer:avaspec:", "powermeter:"):
            for k in REGISTRY.keys(prefix):
                if ":index:" in k:
                    continue
                keys.append(k)
        keys = sorted(set(keys))
        self.det_picker.blockSignals(True)
        self.det_picker.clear()
        for k in keys:
            self.det_picker.addItem(k)
        self.det_picker.blockSignals(False)
        if cur and (idx := self.det_picker.findText(cur)) >= 0:
            self.det_picker.setCurrentIndex(idx)
        self._on_detector_changed(self.det_picker.currentText())

    def _on_detector_changed(self, det_key: str) -> None:
        dev = REGISTRY.get(det_key)
        kind = _detector_kind(dev) if dev is not None else "unknown"
        is_cam = (kind == "camera"); is_spec = (kind == "spectro"); is_pm = (kind == "powermeter")
        self.exposure_us.setEnabled(is_cam)
        self.integration_ms.setEnabled(is_spec)
        self.wavelength_nm.setEnabled(is_pm)
        self.metric_cb.setEnabled(is_cam or is_spec)
        if is_pm:
            self.metric_cb.setCurrentIndex(0)

    def _collect_params(self):
        det_key = self.det_picker.currentText().strip()
        if not det_key:
            raise ValueError("Pick a detector.")
        duration_s = float(self.duration_s.value())
        sample_ms = float(self.sample_ms.value())
        if duration_s <= 0:
            raise ValueError("Duration must be > 0.")
        if sample_ms <= 0:
            raise ValueError("Period must be > 0 ms.")
        metric = self.metric_cb.currentText().strip().lower()
        metric = "sum" if "sum" in metric else "max"
        dev = REGISTRY.get(det_key)
        kind = _detector_kind(dev) if dev is not None else "unknown"
        if kind == "camera":
            det_params = (int(self.exposure_us.value()), int(self.averages.value()))
        elif kind == "spectro":
            det_params = (float(self.integration_ms.value()), int(self.averages.value()))
        elif kind == "powermeter":
            det_params = (float(sample_ms), int(self.averages.value()), float(self.wavelength_nm.value()))
        else:
            raise ValueError("Selected device is not supported by Stability scan.")
        scan_name = "stability_scan"
        comment = self.comment.text()
        mcp_voltage = self.mcp_voltage.text().strip()
        return dict(
            det_key=det_key,
            det_params=det_params,
            sample_ms=sample_ms,
            duration_s=duration_s,
            metric=metric,
            scan_name=scan_name,
            comment=comment,
            mcp_voltage=mcp_voltage,
        )

    def _start(self) -> None:
        try:
            p = self._collect_params()
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return
        self._thread = QThread(self)
        self._worker = StabilityScanWorker(
            det_key=p["det_key"],
            det_params=p["det_params"],
            sample_ms=p["sample_ms"],
            duration_s=p["duration_s"],
            metric=p["metric"],
            scan_name=p["scan_name"],
            comment=p["comment"],
            mcp_voltage=p["mcp_voltage"],
            background=False,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._finished)
        self._thread.finished.connect(self._thread.deleteLater)
        n_steps = max(1, int(p["duration_s"] // (p["sample_ms"] / 1000.0)))
        self.prog.setMaximum(n_steps); self.prog.setValue(0)
        self.start_btn.setEnabled(False); self.abort_btn.setEnabled(True); self.plot_btn.setEnabled(False)
        self._thread.start()
        self._log("Stability scan started…")

    def _abort(self) -> None:
        if self._worker:
            self._worker.abort = True
            self._log("Abort requested.")
            self.abort_btn.setEnabled(False)

    def _on_progress(self, i: int, n: int) -> None:
        self.prog.setMaximum(n); self.prog.setValue(i)
        self._log(f"{i}/{n}")

    def _finished(self, log_path: str) -> None:
        if log_path:
            self._last_log = log_path
            self._log(f"Stability scan finished. Log: {log_path}")
            self.plot_btn.setEnabled(True)
        else:
            self._log("Scan finished with errors or aborted.")
            self._last_log = None
            self.plot_btn.setEnabled(False)
        self.abort_btn.setEnabled(False); self.start_btn.setEnabled(True)
        if self._thread and self._thread.isRunning():
            self._thread.quit(); self._thread.wait()
        self._thread = None; self._worker = None

    def _open_plot(self) -> None:
        if not self._last_log:
            QMessageBox.information(self, "No log", "No finished log to plot yet.")
            return
        dlg = _MetricPlotDialog(self._last_log, self)
        dlg.exec_()

    def _log(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    w = StabilityScanTab()
    w.show()
    sys.exit(app.exec_())
