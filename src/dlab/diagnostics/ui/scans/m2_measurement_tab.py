from __future__ import annotations
import os, datetime, time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, PngImagePlugin

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QLineEdit, QTextEdit, QComboBox, QTabWidget, QDoubleSpinBox,
    QGroupBox, QMessageBox, QSpinBox, QCheckBox, QProgressBar
)

from dlab.boot import ROOT, get_config
from dlab.core.device_registry import REGISTRY


def _data_root() -> Path:
    cfg = get_config() or {}
    base = cfg.get("paths", {}).get("data_root", "C:/data")
    return (ROOT / base).resolve()


# ---------- worker ----------
import logging
logger = logging.getLogger("dlab.scans.m2_measurement_tab")

class M2Worker(QObject):
    progress = pyqtSignal(int, int)          # i, n
    log = pyqtSignal(str)
    finished = pyqtSignal(str)               # log path

    def __init__(
        self,
        stage_key: str,
        camera_key: str,
        positions: Iterable[float],
        settle_s: float,
        comment: str,
        scan_name: str,
        averages: int = 1,
        adaptive: dict | None = None,
        background: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.stage_key = stage_key
        self.camera_key = camera_key
        self.positions = list(float(p) for p in positions)
        self.settle_s = float(settle_s)
        self.comment = comment
        self.scan_name = scan_name
        self.averages = int(max(1, averages))
        self.adaptive = adaptive or {}
        self.background = bool(background)
        self.abort = False

    def _emit(self, msg: str) -> None:
        self.log.emit(msg)
        logger.info(msg)

    def _save_png_with_meta(self, folder: Path, filename: str, frame_u16: np.ndarray, meta: dict) -> Path:
        path = folder / filename
        img = Image.fromarray(frame_u16, mode="I;16")
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in meta.items():
            pnginfo.add_text(str(k), str(v))
        img.save(path.as_posix(), format="PNG", pnginfo=pnginfo)
        return path

    def _append_daheng_log(self, cam_folder: Path, cam_name: str, fn: str, exposure_us: int, comment: str) -> None:
        log_path = cam_folder / f"{cam_name}_log_{datetime.datetime.now():%Y-%m-%d}.log"
        header = "File Name\tExposure_us\tGain\tComment\n"
        exists = log_path.exists()
        with open(log_path, "a", encoding="utf-8") as f:
            if not exists:
                f.write(header)
            # Gain intentionally left blank for scans
            f.write(f"{fn}\t{exposure_us}\t\t{comment}\n")

    def run(self) -> None:
        stage = REGISTRY.get(self.stage_key)
        camwin = REGISTRY.get(self.camera_key)

        if stage is None:
            self._emit(f"Stage '{self.stage_key}' not found in registry.")
            self.finished.emit("")
            return
        if camwin is None:
            self._emit(f"Camera '{self.camera_key}' not found in registry.")
            self.finished.emit("")
            return
        if not hasattr(camwin, "grab_frame_for_scan"):
            self._emit("Camera window does not expose grab_frame_for_scan().")
            self.finished.emit("")
            return

        now = datetime.datetime.now()
        root = _data_root()

        # ---- Scan daily log (append) ----
        scan_dir = root / f"{now:%Y-%m-%d}" / "Scans" / self.scan_name
        scan_dir.mkdir(parents=True, exist_ok=True)
        scan_log = scan_dir / f"{self.scan_name}_log_{now:%Y-%m-%d}.log"
        scan_header = "ImageFile\tStageKey\tPosition\tExposure_us\tComment\n"
        if not scan_log.exists():
            with open(scan_log, "w", encoding="utf-8") as lf:
                lf.write(scan_header)

        n = len(self.positions)
        for i, pos in enumerate(self.positions, 1):
            if self.abort:
                self._emit("Scan aborted.")
                break

            # move stage
            try:
                stage.move_to(float(pos), blocking=True)
                self._emit(f"Moved {self.stage_key} to {pos:.3f}.")
            except Exception as e:
                self._emit(f"Move to {pos:.3f} failed: {e}")
                continue

            time.sleep(self.settle_s)

            # capture (averaging + adaptive handled by camera window)
            try:
                frame_u16, meta = camwin.grab_frame_for_scan(
                    averages=self.averages,
                    adaptive=self.adaptive,
                    dead_pixel_cleanup=True,
                    background=self.background,
                )
            except Exception as e:
                self._emit(f"Capture failed at {pos:.3f}: {e}")
                continue

            cam_name  = meta.get("CameraName", "DahengCam")
            exposure  = int(meta.get("Exposure_us", 0))
            tag       = "Background" if self.background else "Image"

            # ---- Save ONLY in camera folder ----
            cam_day = root / f"{now:%Y-%m-%d}" / cam_name
            cam_day.mkdir(parents=True, exist_ok=True)

            # unique name (no _1/_2 suffix here — averaging returns one image)
            ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cam_fn = f"{cam_name}_{tag}_{ts_ms}.png"

            try:
                self._save_png_with_meta(
                    cam_day, cam_fn, frame_u16,
                    {"Exposure_us": exposure, "Gain": "", "Comment": self.comment}
                )
                # Daheng-style daily log (append)
                self._append_daheng_log(cam_day, cam_name, cam_fn, exposure, self.comment)
            except Exception as e:
                self._emit(f"Save to camera folder failed at {pos:.3f}: {e}")
                continue

            # ---- Append the (single) scan daily log ----
            try:
                with open(scan_log, "a", encoding="utf-8") as lf:
                    lf.write(f"{cam_fn}\t{self.stage_key}\t{pos:.6f}\t{exposure}\t{self.comment}\n")
            except Exception as e:
                self._emit(f"Scan log write failed: {e}")

            self._emit(f"Saved to camera: {cam_fn}")
            self.progress.emit(i, n)

        self.finished.emit(scan_log.as_posix())



# ---------- tab widget ----------
class M2Tab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: M2Worker | None = None
        self._last_params: dict | None = None
        self._doing_background = False
        self._build_ui()

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)

        # Devices
        pick = QGroupBox("Devices")
        pick_l = QHBoxLayout(pick)
        self.stage_combo = QComboBox()
        self.cam_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        pick_l.addWidget(QLabel("Stage:"));  pick_l.addWidget(self.stage_combo, 1)
        pick_l.addWidget(QLabel("Camera:")); pick_l.addWidget(self.cam_combo, 1)
        pick_l.addWidget(self.refresh_btn)
        main.addWidget(pick)

        # Scan parameters
        params = QGroupBox("Scan parameters")
        p = QHBoxLayout(params)
        self.start_sb = QDoubleSpinBox(); self.start_sb.setDecimals(3); self.start_sb.setRange(-1e6, 1e6); self.start_sb.setValue(0.0)
        self.end_sb   = QDoubleSpinBox(); self.end_sb.setDecimals(3);   self.end_sb.setRange(-1e6, 1e6);   self.end_sb.setValue(10.0)
        self.step_sb  = QDoubleSpinBox(); self.step_sb.setDecimals(3);  self.step_sb.setRange(1e-6, 1e6);  self.step_sb.setValue(1.0)
        self.settle_sb= QDoubleSpinBox(); self.settle_sb.setDecimals(2);self.settle_sb.setRange(0.0, 60.0);self.settle_sb.setValue(0.50)
        self.avg_sb   = QDoubleSpinBox(); self.avg_sb.setDecimals(0);   self.avg_sb.setRange(1, 1000);     self.avg_sb.setValue(1)   # NEW

        self.scan_name_edit = QLineEdit("m_squared")   # forced later, but visible
        self.comment_edit = QLineEdit("")
        p.addWidget(QLabel("Start"));     p.addWidget(self.start_sb)
        p.addWidget(QLabel("End"));       p.addWidget(self.end_sb)
        p.addWidget(QLabel("Step"));      p.addWidget(self.step_sb)
        p.addWidget(QLabel("Settle (s)"));p.addWidget(self.settle_sb)
        p.addWidget(QLabel("Avg"));       p.addWidget(self.avg_sb)       # NEW
        main.addWidget(params)

        # Adaptive exposure box
        adapt = QGroupBox("Adaptive exposure")
        a = QHBoxLayout(adapt)
        self.adapt_cb     = QCheckBox("Enable"); self.adapt_cb.setChecked(True)
        self.adapt_target = QDoubleSpinBox(); self.adapt_target.setRange(0.1, 0.99); self.adapt_target.setSingleStep(0.05); self.adapt_target.setValue(0.75)
        self.adapt_low    = QDoubleSpinBox(); self.adapt_low.setRange(0.1, 0.99);    self.adapt_low.setSingleStep(0.05);    self.adapt_low.setValue(0.60)
        self.adapt_high   = QDoubleSpinBox(); self.adapt_high.setRange(0.1, 0.999);  self.adapt_high.setSingleStep(0.05);   self.adapt_high.setValue(0.90)
        self.adapt_min    = QDoubleSpinBox(); self.adapt_min.setRange(1, 2_000_000); self.adapt_min.setDecimals(0);          self.adapt_min.setValue(50)
        self.adapt_max    = QDoubleSpinBox(); self.adapt_max.setRange(1, 2_000_000); self.adapt_max.setDecimals(0);          self.adapt_max.setValue(2_000_000)
        self.bg_cb        = QCheckBox("Do background after scan")   # NEW
        a.addWidget(self.adapt_cb)
        a.addWidget(QLabel("target")); a.addWidget(self.adapt_target)
        a.addWidget(QLabel("low"));    a.addWidget(self.adapt_low)
        a.addWidget(QLabel("high"));   a.addWidget(self.adapt_high)
        a.addWidget(QLabel("exp_min (us)")); a.addWidget(self.adapt_min)
        a.addWidget(QLabel("exp_max (us)")); a.addWidget(self.adapt_max)
        a.addStretch(1); a.addWidget(self.bg_cb)
        main.addWidget(adapt)

        # Comment row
        c = QHBoxLayout()
        c.addWidget(QLabel("Comment:")); c.addWidget(self.comment_edit, 1)
        main.addLayout(c)

        # Controls + progress
        ctl = QHBoxLayout()
        self.start_btn = QPushButton("Start"); self.start_btn.clicked.connect(self._start)
        self.abort_btn = QPushButton("Abort"); self.abort_btn.setEnabled(False); self.abort_btn.clicked.connect(self._abort)
        self.prog = QProgressBar(); self.prog.setMinimum(0); self.prog.setValue(0)   # NEW
        ctl.addWidget(self.start_btn); ctl.addWidget(self.abort_btn); ctl.addWidget(self.prog, 1)
        main.addLayout(ctl)

        # Log
        self.log = QTextEdit(); self.log.setReadOnly(True)
        main.addWidget(self.log, 1)

        self._refresh_devices()


    def _refresh_devices(self) -> None:
        self.stage_combo.clear()
        for k in REGISTRY.keys("stage:"):
            if k.startswith("stage:serial:"):
                continue # hide serial keys
            self.stage_combo.addItem(k)

        self.cam_combo.clear()
        for k in REGISTRY.keys("camera:daheng:"):
            if ":index:" in k:
                continue  # hide index keys
            self.cam_combo.addItem(k)


    def _start(self) -> None:
        try:
            stage_key = self.stage_combo.currentText().strip()
            cam_key   = self.cam_combo.currentText().strip()
            if not stage_key or not cam_key:
                raise ValueError("Please select a stage and a camera.")

            a0 = float(self.start_sb.value()); a1 = float(self.end_sb.value()); step = float(self.step_sb.value())
            if step <= 0:
                raise ValueError("Step must be > 0.")

            # positions inclusive of end
            if a1 >= a0:
                nsteps = int(np.floor((a1 - a0) / step))
                positions = [a0 + i*step for i in range(nsteps+1)]
                if positions[-1] < a1 - 1e-12: positions.append(a1)
            else:
                nsteps = int(np.floor((a0 - a1) / step))
                positions = [a0 - i*step for i in range(nsteps+1)]
                if positions[-1] > a1 + 1e-12: positions.append(a1)

            settle   = float(self.settle_sb.value())
            avg      = int(self.avg_sb.value())
            scan_name= "m_squared"   # enforced
            comment  = self.comment_edit.text()

            adaptive = {
                "enabled": self.adapt_cb.isChecked(),
                "target_frac": float(self.adapt_target.value()),
                "low_frac":    float(self.adapt_low.value()),
                "high_frac":   float(self.adapt_high.value()),
                "min_us":      int(self.adapt_min.value()),
                "max_us":      int(self.adapt_max.value()),
            }
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        # save for background pass
        self._last_params = dict(stage_key=stage_key, cam_key=cam_key,
                                positions=positions, settle=settle,
                                avg=avg, scan_name=scan_name,
                                comment=comment, adaptive=adaptive)
        self._doing_background = False

        self._launch_worker(background=False)
        self._log("Scan started…")
        
    def _launch_worker(self, background: bool):
        p = self._last_params
        if not p:
            return
        self._thread = QThread(self)
        self._worker = M2Worker(
            stage_key=p["stage_key"],
            camera_key=p["cam_key"],
            positions=p["positions"],
            settle_s=p["settle"],
            comment=p["comment"],
            scan_name=p["scan_name"],
            averages=p["avg"],
            adaptive=p["adaptive"],
            background=background,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._finished)
        self._thread.finished.connect(self._thread.deleteLater)

        # progress bar maximum for this pass
        self.prog.setMaximum(len(p["positions"]))
        self.prog.setValue(0)

        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self._thread.start()



    def _abort(self) -> None:
        if self._worker:
            self._worker.abort = True
            self._log("Abort requested.")
            self.abort_btn.setEnabled(False)

    def _log(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _on_progress(self, i: int, n: int) -> None:
        self.prog.setMaximum(n)
        self.prog.setValue(i)
        self._log(f"{i}/{n}")

    def _finished(self, log_path: str) -> None:
        if log_path:
            self._log(f"Scan finished. Log: {log_path}")
        else:
            self._log("Scan finished with errors.")

        # Chain a background pass if requested and not done yet
        if self._last_params and self.bg_cb.isChecked() and not self._doing_background:
            self._doing_background = True
            reply = QMessageBox.information(
                self, "Background scan",
                "Please block the laser now, then click OK to record the background.",
                QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok
            )
            if reply == QMessageBox.Ok:
                self._log("Starting background pass…")
                if self._thread and self._thread.isRunning():
                    self._thread.quit(); self._thread.wait()
                self._thread = None; self._worker = None
                self._launch_worker(background=True)
                return  # keep UI in scanning mode

        # done
        self.abort_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        if self._thread and self._thread.isRunning():
            self._thread.quit(); self._thread.wait()
        self._thread = None
        self._worker = None
