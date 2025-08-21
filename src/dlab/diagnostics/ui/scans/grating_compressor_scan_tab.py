# src/dlab/diagnostics/ui/scans/grating_compressor_scan_tab.py
from __future__ import annotations
import datetime, time
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image, PngImagePlugin

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit,
    QGroupBox, QMessageBox, QDoubleSpinBox, QComboBox, QProgressBar, QCheckBox
)

from dlab.boot import ROOT, get_config
from dlab.core.device_registry import REGISTRY

# ---------- utils ----------
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

def _append_cam_log(cam_folder: Path, cam_name: str, fn: str, exposure_us: int, comment: str) -> None:
    import datetime
    log_path = cam_folder / f"{cam_name}_log_{datetime.datetime.now():%Y-%m-%d}.log"
    header = "File Name\tExposure_us\tGain\tComment\n"
    exists = log_path.exists()
    with open(log_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(header)
        f.write(f"{fn}\t{exposure_us}\t\t{comment}\n")


# ---------- worker ----------
import logging
logger = logging.getLogger("dlab.scans.gc_scan")

class GCWorker(QObject):
    progress = pyqtSignal(int, int)      # (i, n)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)           # scan log path

    def __init__(
        self,
        stage_key: str,                  # ex: "stage:zaber:grating_compressor"
        andor_key: str,                  # ex: "camera:andor:andorcam_1"
        positions: Iterable[float],
        exposure_us: int,
        averages: int,
        settle_s: float,
        scan_name: str,
        comment: str,
        do_background: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.stage_key = stage_key
        self.andor_key = andor_key
        self.positions = list(float(p) for p in positions)
        self.exposure_us = int(exposure_us)
        self.averages = int(max(1, averages))
        self.settle_s = float(settle_s)
        self.scan_name = scan_name
        self.comment = comment
        self.do_background = bool(do_background)
        self.abort = False

    def _emit(self, msg: str) -> None:
        self.log.emit(msg)
        logger.info(msg)

    def run(self) -> None:
        import datetime
        stage = REGISTRY.get(self.stage_key)
        camwin = REGISTRY.get(self.andor_key)

        if stage is None:
            self._emit(f"Stage '{self.stage_key}' not found.")
            self.finished.emit("")
            return
        if camwin is None:
            self._emit(f"Camera '{self.andor_key}' not found.")
            self.finished.emit("")
            return
        if not hasattr(camwin, "grab_frame_for_scan"):
            self._emit("Selected camera does not expose grab_frame_for_scan().")
            self.finished.emit("")
            return

        # try preset exposure on Andor
        try:
            if hasattr(camwin, "set_exposure_us"):
                camwin.set_exposure_us(self.exposure_us)
            elif hasattr(camwin, "setExposureUS"):
                camwin.setExposureUS(self.exposure_us)
            elif hasattr(camwin, "set_exposure"):
                # some wrappers expect µs, some s; your Andor wrapper uses µs
                camwin.set_exposure(self.exposure_us)
        except Exception as e:
            self._emit(f"Warning: failed to preset exposure on '{self.andor_key}': {e}")

        now = datetime.datetime.now()
        root = _data_root()

        # scan log (under Scans/scan_name)
        scan_dir = root / f"{now:%Y-%m-%d}" / "Scans" / self.scan_name
        scan_dir.mkdir(parents=True, exist_ok=True)
        scan_log = scan_dir / f"{self.scan_name}_log_{now:%Y-%m-%d}.log"
        if not scan_log.exists():
            with open(scan_log, "w", encoding="utf-8") as f:
                f.write("ImageFile\tStageKey\tPosition_mm\tExposure_us\tAverages\tComment\n")

        total = len(self.positions)
        done = 0

        for pos in self.positions:
            if self.abort:
                self._emit("Scan aborted.")
                self.finished.emit("")
                return

            # move stage
            try:
                stage.move_to(float(pos), blocking=True)
                self._emit(f"Moved {self.stage_key} to {pos:.3f} mm.")
            except Exception as e:
                self._emit(f"Move to {pos:.3f} failed: {e}")
                done += 1
                self.progress.emit(done, total)
                continue

            time.sleep(self.settle_s)

            # capture Andor
            try:
                frame_u16, meta = camwin.grab_frame_for_scan(
                    averages=self.averages,
                    background=False,                # main pass
                    dead_pixel_cleanup=True,
                    exposure_us=self.exposure_us,
                )
            except TypeError:
                frame_u16, meta = camwin.grab_frame_for_scan(
                    averages=self.averages,
                    background=False,
                    dead_pixel_cleanup=True,
                )
            except Exception as e:
                self._emit(f"Capture failed at {pos:.3f}: {e}")
                done += 1
                self.progress.emit(done, total)
                continue

            cam_name = str((meta or {}).get("CameraName", "AndorCam")).strip() or "AndorCam"
            exp_meta = int((meta or {}).get("Exposure_us", self.exposure_us))
            ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            tag = "Image"
            cam_day = root / f"{now:%Y-%m-%d}" / cam_name
            cam_fn = f"{cam_name}_{tag}_{ts_ms}.png"

            try:
                _save_png_with_meta(
                    cam_day, cam_fn, frame_u16,
                    {"Exposure_us": exp_meta, "Gain": "", "Comment": self.comment}
                )
                _append_cam_log(cam_day, cam_name, cam_fn, exp_meta, self.comment)
            except Exception as e:
                self._emit(f"Save failed at {pos:.3f}: {e}")
                done += 1
                self.progress.emit(done, total)
                continue

            # append scan log
            try:
                with open(scan_log, "a", encoding="utf-8") as f:
                    f.write(f"{cam_fn}\t{self.stage_key}\t{pos:.6f}\t{exp_meta}\t{self.averages}\t{self.comment}\n")
            except Exception as e:
                self._emit(f"Scan log write failed: {e}")

            self._emit(f"Saved {cam_fn} @ {pos:.3f} mm (exp {exp_meta} µs, avg {self.averages}).")
            done += 1
            self.progress.emit(done, total)

        # Optional background pass
        if self.do_background and (not self.abort):
            # ask user handled in UI; here we only run if requested
            try:
                # single background frame at current position
                frame_u16, meta = camwin.grab_frame_for_scan(
                    averages=self.averages,
                    background=True,
                    dead_pixel_cleanup=True,
                    exposure_us=self.exposure_us,
                )
            except TypeError:
                frame_u16, meta = camwin.grab_frame_for_scan(
                    averages=self.averages,
                    background=True,
                    dead_pixel_cleanup=True,
                )
            except Exception as e:
                self._emit(f"Background capture failed: {e}")
            else:
                cam_name = str((meta or {}).get("CameraName", "AndorCam")).strip() or "AndorCam"
                exp_meta = int((meta or {}).get("Exposure_us", self.exposure_us))
                ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                tag = "Background"
                cam_day = root / f"{now:%Y-%m-%d}" / cam_name
                cam_fn = f"{cam_name}_{tag}_{ts_ms}.png"
                try:
                    _save_png_with_meta(
                        cam_day, cam_fn, frame_u16,
                        {"Exposure_us": exp_meta, "Gain": "", "Comment": self.comment}
                    )
                    _append_cam_log(cam_day, cam_name, cam_fn, exp_meta, self.comment)
                    with open(scan_log, "a", encoding="utf-8") as f:
                        f.write(f"{cam_fn}\t{self.stage_key}\tBG\t{exp_meta}\t{self.averages}\t{self.comment}\n")
                    self._emit(f"Saved background: {cam_fn}")
                except Exception as e:
                    self._emit(f"Background save/log failed: {e}")

        self.finished.emit(scan_log.as_posix())


# ---------- tab ----------
class GCScanTab(QWidget):
    """
    Scan the Zaber grating compressor stage and capture an Andor image at each position.
    """
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: GCWorker | None = None
        self._build_ui()
        self._refresh_devices()

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)

        # Devices
        dev_box = QGroupBox("Devices")
        dl = QHBoxLayout(dev_box)
        self.stage_label = QLabel("Stage:"); self.stage_val = QLabel("stage:zaber:grating_compressor")
        self.cam_combo = QComboBox()
        self.btn_refresh = QPushButton("Refresh"); self.btn_refresh.clicked.connect(self._refresh_devices)
        dl.addWidget(self.stage_label); dl.addWidget(self.stage_val, 1)
        dl.addWidget(QLabel("Andor camera:")); dl.addWidget(self.cam_combo, 1)
        dl.addWidget(self.btn_refresh)
        main.addWidget(dev_box)

        # Parameters
        p_box = QGroupBox("Scan parameters")
        pl = QHBoxLayout(p_box)
        self.start_sb  = QDoubleSpinBox(); self.start_sb.setDecimals(3); self.start_sb.setRange(-1e6, 1e6); self.start_sb.setValue(0.0)
        self.end_sb    = QDoubleSpinBox(); self.end_sb.setDecimals(3);   self.end_sb.setRange(-1e6, 1e6);   self.end_sb.setValue(10.0)
        self.step_sb   = QDoubleSpinBox(); self.step_sb.setDecimals(3);  self.step_sb.setRange(1e-6, 1e6);  self.step_sb.setValue(0.5)
        self.settle_sb = QDoubleSpinBox(); self.settle_sb.setDecimals(2);self.settle_sb.setRange(0.0, 60.0);self.settle_sb.setValue(0.30)
        self.exp_sb    = QDoubleSpinBox(); self.exp_sb.setDecimals(0);   self.exp_sb.setRange(1, 5_000_000);self.exp_sb.setValue(5_000)  # µs
        self.avg_sb    = QDoubleSpinBox(); self.avg_sb.setDecimals(0);   self.avg_sb.setRange(1, 1000);     self.avg_sb.setValue(1)
        self.name_edit = QLineEdit("grating_compressor")
        self.comment   = QLineEdit("")
        pl.addWidget(QLabel("Start (mm)"));   pl.addWidget(self.start_sb)
        pl.addWidget(QLabel("End (mm)"));     pl.addWidget(self.end_sb)
        pl.addWidget(QLabel("Step (mm)"));    pl.addWidget(self.step_sb)
        pl.addWidget(QLabel("Settle (s)"));   pl.addWidget(self.settle_sb)
        pl.addWidget(QLabel("Exposure (µs)"));pl.addWidget(self.exp_sb)
        pl.addWidget(QLabel("Averages"));     pl.addWidget(self.avg_sb)
        main.addWidget(p_box)

        # Options
        opt = QGroupBox("Options")
        ol = QHBoxLayout(opt)
        self.bg_cb = QCheckBox("Record background after scan"); self.bg_cb.setChecked(False)
        ol.addWidget(self.bg_cb); ol.addStretch(1)
        main.addWidget(opt)

        # Controls
        ctl = QHBoxLayout()
        self.btn_start = QPushButton("Start"); self.btn_start.clicked.connect(self._start)
        self.btn_abort = QPushButton("Abort"); self.btn_abort.setEnabled(False); self.btn_abort.clicked.connect(self._abort)
        self.prog = QProgressBar(); self.prog.setMinimum(0); self.prog.setValue(0)
        ctl.addWidget(self.btn_start); ctl.addWidget(self.btn_abort); ctl.addWidget(self.prog, 1)
        main.addLayout(ctl)

        # Log
        self.log = QTextEdit(); self.log.setReadOnly(True)
        main.addWidget(self.log, 1)

    def _refresh_devices(self) -> None:
        # Andor only
        self.cam_combo.clear()
        for k in REGISTRY.keys("camera:andor:"):
            if ":index:" in k:
                continue
            self.cam_combo.addItem(k)

    def _positions(self, a0: float, a1: float, step: float) -> List[float]:
        if step <= 0:
            raise ValueError("Step must be > 0.")
        if a1 >= a0:
            n = int(np.floor((a1 - a0) / step))
            pos = [a0 + i*step for i in range(n+1)]
            if pos[-1] < a1 - 1e-12: pos.append(a1)
        else:
            n = int(np.floor((a0 - a1) / step))
            pos = [a0 - i*step for i in range(n+1)]
            if pos[-1] > a1 + 1e-12: pos.append(a1)
        return pos

    def _start(self) -> None:
        try:
            stage_key = "stage:zaber:grating_compressor"
            if REGISTRY.get(stage_key) is None:
                raise ValueError("Grating compressor stage not found in registry. Open the Grating Compressor window and Activate it first.")
            cam_key = self.cam_combo.currentText().strip()
            if not cam_key:
                raise ValueError("Select an Andor camera.")

            start = float(self.start_sb.value())
            end   = float(self.end_sb.value())
            step  = float(self.step_sb.value())
            pos   = self._positions(start, end, step)

            settle = float(self.settle_sb.value())
            expo   = int(self.exp_sb.value())
            avg    = int(self.avg_sb.value())
            name   = self.name_edit.text().strip() or "grating_compressor"
            comment= self.comment.text()

        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        # thread/worker
        self._thread = QThread(self)
        self._worker = GCWorker(
            stage_key=stage_key,
            andor_key=cam_key,
            positions=pos,
            exposure_us=expo,
            averages=avg,
            settle_s=settle,
            scan_name=name,
            comment=comment,
            do_background=False,   # handled by UI prompt below
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._finished)
        self._thread.finished.connect(self._thread.deleteLater)

        self.prog.setMaximum(len(pos)); self.prog.setValue(0)
        self.btn_start.setEnabled(False); self.btn_abort.setEnabled(True)
        self._thread.start()
        self._log("Grating compressor scan started…")

    def _abort(self) -> None:
        if self._worker:
            self._worker.abort = True
            self._log("Abort requested.")
            self.btn_abort.setEnabled(False)

    def _on_progress(self, i: int, n: int) -> None:
        self.prog.setMaximum(n)
        self.prog.setValue(i)
        self._log(f"{i}/{n}")

    def _finished(self, log_path: str) -> None:
        if log_path:
            self._log(f"Scan finished. Log: {log_path}")
        else:
            self._log("Scan finished with errors or aborted.")

        # optional background capture after scan
        if self.bg_cb.isChecked():
            reply = QMessageBox.information(
                self, "Background",
                "Block the beam, then click OK to record one background image.",
                QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok
            )
            if reply == QMessageBox.Ok and self._worker and not self._worker.abort:
                # reuse same worker in a tiny 1-step pass at current position
                try:
                    cam_key = self.cam_combo.currentText().strip()
                    stage_key = "stage:zaber:grating_compressor"
                    self._log("Recording background…")
                    w = GCWorker(
                        stage_key=stage_key,
                        andor_key=cam_key,
                        positions=[],
                        exposure_us=int(self.exp_sb.value()),
                        averages=int(self.avg_sb.value()),
                        settle_s=0.0,
                        scan_name=self.name_edit.text().strip() or "grating_compressor",
                        comment=self.comment.text(),
                        do_background=True,
                    )
                    t = QThread(self)
                    w.moveToThread(t)
                    t.started.connect(w.run)
                    w.log.connect(self._log)
                    # no progress bar for bg
                    def _bg_done(_path): 
                        self._log("Background captured.")
                        t.quit(); t.wait()
                    w.finished.connect(_bg_done)
                    t.start()
                except Exception as e:
                    self._log(f"Background failed: {e}")

        self.btn_abort.setEnabled(False)
        self.btn_start.setEnabled(True)
        if self._thread and self._thread.isRunning():
            self._thread.quit(); self._thread.wait()
        self._thread = None; self._worker = None

    def _log(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
