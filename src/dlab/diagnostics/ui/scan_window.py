# src/dlab/diagnostics/ui/scan_window.py
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
    QGroupBox, QMessageBox, QSpinBox
)

from dlab.boot import ROOT, get_config
from dlab.core.device_registry import REGISTRY


def _data_root() -> Path:
    cfg = get_config() or {}
    base = cfg.get("paths", {}).get("data_root", "C:/data")
    return (ROOT / base).resolve()


# ---------- worker ----------
import logging

class StageCameraScanWorker(QObject):
    progress = pyqtSignal(int, int)      # i, n (images)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)           # scan log path

    def __init__(
        self,
        stage_key: str,
        camera_key: str,
        positions: Iterable[float],
        settle_s: float,
        comment: str,
        scan_name: str,
        averages: int = 1,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.stage_key = stage_key
        self.camera_key = camera_key
        self.positions = list(float(p) for p in positions)
        self.settle_s = float(settle_s)
        self.comment = comment
        self.scan_name = "m_squared"  # enforce
        self.averages = max(1, int(averages))
        self.abort = False
        self._logger = logging.getLogger("dlab.scan.StageCameraScanWorker")

    def _emit_log(self, msg: str) -> None:
        self.log.emit(msg)
        self._logger.info(msg)

    def _save_png_with_meta(self, folder: Path, filename: str, frame_u16: np.ndarray, meta: dict) -> Path:
        path = folder / filename
        img = Image.fromarray(frame_u16, mode="I;16")
        png = PngImagePlugin.PngInfo()
        for k, v in meta.items():
            png.add_text(str(k), str(v))
        img.save(path.as_posix(), format="PNG", pnginfo=png)
        return path

    def run(self) -> None:
        # Devices
        stage = REGISTRY.get(self.stage_key)
        camwin = REGISTRY.get(self.camera_key)
        if stage is None:
            self._emit_log(f"Stage '{self.stage_key}' not found in registry.")
            self.finished.emit(""); return
        if camwin is None:
            self._emit_log(f"Camera '{self.camera_key}' not found in registry.")
            self.finished.emit(""); return
        if not hasattr(camwin, "grab_frame_for_scan"):
            self._emit_log("Camera window does not expose grab_frame_for_scan().")
            self.finished.emit(""); return

        # Paths
        now = datetime.datetime.now()
        day_dir  = _data_root() / f"{now:%Y-%m-%d}"
        scan_dir = day_dir / "Scans" / self.scan_name
        scan_dir.mkdir(parents=True, exist_ok=True)

        # Scan TSV (no timestamp column)
        scan_log = scan_dir / f"{self.scan_name}_{now:%Y%m%d_%H%M%S}.log"
        with open(scan_log, "w", encoding="utf-8") as lf:
            lf.write("ImageFile\tStageKey\tPosition\tCamera\tExposure_us\tGain\tComment\n")

            total_imgs = len(self.positions) * self.averages
            done = 0

            for pos in self.positions:
                if self.abort:
                    self._emit_log("Scan aborted."); break

                # Move stage
                try:
                    stage.move_to(float(pos), blocking=True)
                    self._emit_log(f"Moved {self.stage_key} to {pos:.3f}.")
                except Exception as e:
                    self._emit_log(f"Move to {pos:.3f} failed: {e}")
                    continue

                time.sleep(self.settle_s)

                # For each average
                for j in range(1, self.averages + 1):
                    if self.abort:
                        break

                    # Capture
                    try:
                        frame_u16, meta = camwin.grab_frame_for_scan()
                    except Exception as e:
                        self._emit_log(f"Capture failed at {pos:.3f}: {e}")
                        continue

                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    cam_name = (meta.get("CameraName") or "Camera").replace(" ", "")

                    # Filenames:
                    #   Scan folder: <Cam>_m_squared_<ts>_pos{pos}[ _<j>.png if averages>1]
                    #   Camera folder (like live): <Cam>_Image_<ts>[ _<j>.png if averages>1]
                    scan_stem = f"{cam_name}_{self.scan_name}_{ts}_pos{pos:.3f}"
                    cam_stem  = f"{cam_name}_{self.scan_name}_{ts}"
                    scan_fn = f"{scan_stem}.png" if self.averages == 1 else f"{scan_stem}_{j}.png"
                    cam_fn  = f"{cam_stem}.png"  if self.averages == 1 else f"{cam_stem}_{j}.png"

                    # Meta enrich
                    meta_out = dict(meta)
                    meta_out.update({
                        "ScanName": self.scan_name,
                        "StageKey": self.stage_key,
                        "StagePosition": f"{pos:.6f}",
                        "Comment": self.comment,
                    })

                    try:
                        # Save in SCAN folder
                        scan_img_path = self._save_png_with_meta(scan_dir, scan_fn, frame_u16, meta_out)

                        # Save ALSO in CAMERA folder
                        cam_dir = day_dir / cam_name
                        cam_dir.mkdir(parents=True, exist_ok=True)
                        cam_img_path = self._save_png_with_meta(cam_dir, cam_fn, frame_u16, meta_out)

                        # Append one line in SCAN LOG (no timestamp; leave Gain blank)
                        lf.write(
                            f"{scan_img_path.name}\t{self.stage_key}\t{pos:.6f}\t{cam_name}\t"
                            f"{meta.get('Exposure_us','')}\t\t{self.comment}\n"
                        )

                        # Append to Daheng LIVE log (same file as live captures)
                        # Header: "ImageFile\tExposure_us\tGain\tComment\n"
                        cam_day_log = cam_dir / f"{cam_name}_log_{now:%Y-%m-%d}.log"
                        if not cam_day_log.exists():
                            with open(cam_day_log, "w", encoding="utf-8") as cf:
                                cf.write("ImageFile\tExposure_us\tGain\tComment\n")
                        with open(cam_day_log, "a", encoding="utf-8") as cf:
                            cf.write(f"{cam_img_path.name}\t{meta.get('Exposure_us','')}\t\t{self.comment}\n")

                        self._emit_log(f"Saved {scan_img_path.name} (and {cam_img_path.name})")
                    except Exception as e:
                        self._emit_log(f"Save failed at {pos:.3f}: {e}")

                    done += 1
                    self.progress.emit(done, total_imgs)

        self.finished.emit(scan_log.as_posix())


# ---------- tab widget ----------
class StageCameraScanTab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: StageCameraScanWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)

        # Device selection
        pick = QGroupBox("Devices")
        pick_l = QHBoxLayout(pick)
        self.stage_combo = QComboBox()
        self.cam_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        pick_l.addWidget(QLabel("Stage:")); pick_l.addWidget(self.stage_combo, 1)
        pick_l.addWidget(QLabel("Camera:")); pick_l.addWidget(self.cam_combo, 1)
        pick_l.addWidget(self.refresh_btn)
        main.addWidget(pick)

        # Scan params
        params = QGroupBox("Scan parameters")
        p = QHBoxLayout(params)
        self.start_sb = QDoubleSpinBox(); self.start_sb.setDecimals(3); self.start_sb.setRange(-1e6, 1e6); self.start_sb.setValue(0.0)
        self.end_sb   = QDoubleSpinBox(); self.end_sb.setDecimals(3);   self.end_sb.setRange(-1e6, 1e6);   self.end_sb.setValue(10.0)
        self.step_sb  = QDoubleSpinBox(); self.step_sb.setDecimals(3);  self.step_sb.setRange(1e-6, 1e6);  self.step_sb.setValue(1.0)
        self.settle_sb= QDoubleSpinBox(); self.settle_sb.setDecimals(2);self.settle_sb.setRange(0.0, 60.0);self.settle_sb.setValue(0.50)
        self.avg_sb = QSpinBox(); self.avg_sb.setRange(1, 999); self.avg_sb.setValue(1)
        self.scan_name_edit = QLineEdit("m_squared")
        self.scan_name_edit.setReadOnly(True)
        self.scan_name_edit.setToolTip("Fixed scan name")
        self.comment_edit = QLineEdit("")
        p.addWidget(QLabel("Start")); p.addWidget(self.start_sb)
        p.addWidget(QLabel("End"));   p.addWidget(self.end_sb)
        p.addWidget(QLabel("Step"));  p.addWidget(self.step_sb)
        p.addWidget(QLabel("Settle (s)")); p.addWidget(self.settle_sb)
        p.addWidget(QLabel("Averages")); p.addWidget(self.avg_sb)
        p.addWidget(QLabel("Name"));  p.addWidget(self.scan_name_edit, 1)
        main.addWidget(params)

        # Comment
        c = QHBoxLayout()
        c.addWidget(QLabel("Comment:")); c.addWidget(self.comment_edit, 1)
        main.addLayout(c)

        # Controls
        ctl = QHBoxLayout()
        self.start_btn = QPushButton("Start"); self.start_btn.clicked.connect(self._start)
        self.abort_btn = QPushButton("Abort"); self.abort_btn.setEnabled(False); self.abort_btn.clicked.connect(self._abort)
        ctl.addWidget(self.start_btn); ctl.addWidget(self.abort_btn); ctl.addStretch(1)
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
            cam_key = self.cam_combo.currentText().strip()
            if not stage_key or not cam_key:
                raise ValueError("Please select a stage and a camera.")
            a0 = float(self.start_sb.value()); a1 = float(self.end_sb.value()); step = float(self.step_sb.value())
            if step <= 0:
                raise ValueError("Step must be > 0.")
            # build positions inclusive of end (last step may overshoot; clamp)
            nsteps = int(np.floor((a1 - a0) / step)) if a1 >= a0 else int(np.floor((a0 - a1) / step))
            if nsteps < 0: nsteps = 0
            if a1 >= a0:
                positions = [a0 + i*step for i in range(nsteps+1)]
                if positions[-1] < a1 - 1e-12: positions.append(a1)
            else:
                positions = [a0 - i*step for i in range(nsteps+1)]
                if positions[-1] > a1 + 1e-12: positions.append(a1)

            settle = float(self.settle_sb.value())
            comment = self.comment_edit.text()
            avg = int(self.avg_sb.value())
            scan_name = "m_squared"
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        # worker + thread
        self._thread = QThread(self)
        self._worker = StageCameraScanWorker(
            stage_key=stage_key,
            camera_key=cam_key,
            positions=positions,
            settle_s=settle,
            comment=comment,
            scan_name=scan_name,
            averages=avg,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._finished)
        self._thread.finished.connect(self._thread.deleteLater)

        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self._thread.start()
        self._log("Scan started…")

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
        self._log(f"{i}/{n}")

    def _finished(self, log_path: str) -> None:
        if log_path:
            self._log(f"Scan finished. Log: {log_path}")
        else:
            self._log("Scan finished with errors.")
        self.abort_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        if self._thread and self._thread.isRunning():
            self._thread.quit(); self._thread.wait()
        self._thread = None
        self._worker = None


# ---------- main window ----------
class ScanWindow(QMainWindow):
    closed = pyqtSignal()
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Scan")
        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)

        self.tabs.addTab(StageCameraScanTab(), "M2 measurement")

        # Future: add more tabs easily
        # self.tabs.addTab(SomeOtherScanTab(), "Whatever")
        
    def closeEvent(self, event):
        try:
            self.closed.emit()
        finally:
            super().closeEvent(event)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = ScanWindow()
    w.show()
    sys.exit(app.exec_())
