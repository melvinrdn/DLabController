from __future__ import annotations

import datetime
import time
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QMessageBox,
    QCheckBox,
    QProgressBar,
)

from dlab.core.device_registry import REGISTRY
from dlab.utils.log_panel import LogPanel
from dlab.utils.paths_utils import data_dir


# -----------------------------------------------------------------------------
# Worker thread
# -----------------------------------------------------------------------------


class M2Worker(QObject):
    """Worker for M² measurement scan."""

    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(
        self,
        stage_key: str,
        camera_key: str,
        positions: list[float],
        settle_s: float,
        comment: str,
        scan_name: str,
        averages: int = 1,
        background: bool = False,
        existing_scan_log: str | None = None,
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
        self.background = bool(background)
        self.abort = False
        self.existing_scan_log = existing_scan_log

    def _emit(self, msg: str) -> None:
        self.log.emit(msg)

    def _save_png_with_meta(
        self, folder: Path, filename: str, frame_u8: np.ndarray, meta: dict
    ) -> Path:
        """Save an 8-bit grayscale PNG with metadata."""
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        f8 = np.asarray(frame_u8, dtype=np.uint8, copy=False)
        img = Image.fromarray(f8, mode="L")
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in meta.items():
            pnginfo.add_text(str(k), str(v))
        img.save(path.as_posix(), format="PNG", pnginfo=pnginfo)
        return path

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
        root = data_dir()
        scan_dir = root / f"{now:%Y-%m-%d}" / "Scans" / self.scan_name
        scan_dir.mkdir(parents=True, exist_ok=True)

        # Create or use existing scan log
        if self.existing_scan_log:
            scan_log = Path(self.existing_scan_log)
            if not scan_log.exists():
                with open(scan_log, "w", encoding="utf-8") as lf:
                    lf.write("ImageFile\tStageKey\tPosition\tExposure_us\n")
                    lf.write(f"# {self.comment}\n")
                    step = (self.positions[1] - self.positions[0]) if len(self.positions) > 1 else 0.0
                    lf.write(f"# Start={self.positions[0]:.6f}; End={self.positions[-1]:.6f}; Step={step:.6f}\n")
        else:
            date_str = f"{now:%Y-%m-%d}"
            idx = 1
            while True:
                candidate = scan_dir / f"{self.scan_name}_log_{date_str}_{idx}.log"
                if not candidate.exists():
                    break
                idx += 1
            scan_log = candidate
            with open(scan_log, "w", encoding="utf-8") as lf:
                lf.write("ImageFile\tStageKey\tPosition\tExposure_us\n")
                lf.write(f"# {self.comment}\n")
                step = (self.positions[1] - self.positions[0]) if len(self.positions) > 1 else 0.0
                lf.write(f"# Start={self.positions[0]:.6f}; End={self.positions[-1]:.6f}; Step={step:.6f}\n")

        # Background capture
        if self.background:
            try:
                try:
                    pos = float(stage.get_position())
                except Exception:
                    pos = 0.0
                frame_u8, meta = camwin.grab_frame_for_scan(
                    averages=1,
                    dead_pixel_cleanup=False,
                    background=True,
                    force_roi=True,
                )
            except Exception as e:
                self._emit(f"Background capture failed: {e}")
                self.finished.emit(scan_log.as_posix())
                return

            cam_name = str(meta.get("CameraName", "DahengCam")).strip() or "DahengCam"
            exposure = int(meta.get("Exposure_us", 0))
            cam_day = root / f"{now:%Y-%m-%d}" / cam_name
            cam_day.mkdir(parents=True, exist_ok=True)
            ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cam_fn = f"{cam_name}_Background_{ts_ms}.png"

            try:
                self._save_png_with_meta(
                    cam_day,
                    cam_fn,
                    frame_u8,
                    {"Exposure_us": exposure, "Gain": "", "Comment": self.comment},
                )
            except Exception as e:
                self._emit(f"Save background failed: {e}")
                self.finished.emit(scan_log.as_posix())
                return

            try:
                with open(scan_log, "a", encoding="utf-8") as lf:
                    lf.write(f"{cam_fn}\t{self.stage_key}\t{pos:.6f}\t{exposure}\n")
            except Exception as e:
                self._emit(f"Background log write failed: {e}")

            self._emit(f"Saved background {cam_fn} (exp {exposure} µs).")
            self.progress.emit(1, 1)
            self.finished.emit(scan_log.as_posix())
            return

        # Main scan
        n = len(self.positions)
        for i, pos in enumerate(self.positions, 1):
            if self.abort:
                self._emit("Scan aborted.")
                break

            try:
                stage.move_to(float(pos), blocking=True)
                self._emit(f"Moved {self.stage_key} to {pos:.3f}.")
            except Exception as e:
                self._emit(f"Move to {pos:.3f} failed: {e}")
                self.progress.emit(i, n)
                continue

            time.sleep(self.settle_s)

            try:
                frame_u8, meta = camwin.grab_frame_for_scan(
                    averages=self.averages,
                    dead_pixel_cleanup=False,
                    background=False,
                    force_roi=True,
                )
            except Exception as e:
                self._emit(f"Capture failed at {pos:.3f}: {e}")
                self.progress.emit(i, n)
                continue

            cam_name = str(meta.get("CameraName", "DahengCam")).strip() or "DahengCam"
            exposure = int(meta.get("Exposure_us", 0))
            cam_day = root / f"{now:%Y-%m-%d}" / cam_name
            cam_day.mkdir(parents=True, exist_ok=True)
            ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cam_fn = f"{cam_name}_Image_{ts_ms}.png"

            try:
                self._save_png_with_meta(
                    cam_day,
                    cam_fn,
                    frame_u8,
                    {"Exposure_us": exposure, "Gain": "", "Comment": self.comment},
                )
            except Exception as e:
                self._emit(f"Save to camera folder failed at {pos:.3f}: {e}")
                self.progress.emit(i, n)
                continue

            try:
                with open(scan_log, "a", encoding="utf-8") as lf:
                    lf.write(f"{cam_fn}\t{self.stage_key}\t{pos:.6f}\t{exposure}\n")
            except Exception as e:
                self._emit(f"Scan log write failed: {e}")

            self._emit(f"Saved {cam_fn} @ {pos:.3f} (exp {exposure} µs, avg {self.averages}).")
            self.progress.emit(i, n)

        self.finished.emit(scan_log.as_posix())


# -----------------------------------------------------------------------------
# M2Tab
# -----------------------------------------------------------------------------


class M2Tab(QWidget):
    """Tab for M² beam measurement scan."""

    def __init__(
        self, log_panel: LogPanel | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)

        self._log = log_panel
        self._thread: QThread | None = None
        self._worker: M2Worker | None = None
        self._last_params: dict | None = None
        self._doing_background = False
        self._last_scan_log_path: str | None = None

        self._init_ui()
        self._refresh_devices()

    def _init_ui(self) -> None:
        main = QVBoxLayout(self)

        # Devices group
        main.addWidget(self._create_devices_group())

        # Parameters group
        main.addWidget(self._create_parameters_group())

        # Options group
        main.addWidget(self._create_options_group())

        # Comment row
        comment_row = QHBoxLayout()
        comment_row.addWidget(QLabel("Comment:"))
        self._comment_edit = QLineEdit("")
        comment_row.addWidget(self._comment_edit, 1)
        main.addLayout(comment_row)

        # Controls row
        main.addLayout(self._create_controls_row())

    def _create_devices_group(self) -> QGroupBox:
        group = QGroupBox("Devices")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Stage:"))
        self._stage_combo = QComboBox()
        layout.addWidget(self._stage_combo, 1)

        layout.addWidget(QLabel("Camera:"))
        self._cam_combo = QComboBox()
        layout.addWidget(self._cam_combo, 1)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_devices)
        layout.addWidget(btn_refresh)

        return group

    def _create_parameters_group(self) -> QGroupBox:
        group = QGroupBox("Scan Parameters")
        layout = QHBoxLayout(group)

        layout.addWidget(QLabel("Start"))
        self._start_sb = QDoubleSpinBox()
        self._start_sb.setDecimals(3)
        self._start_sb.setRange(-1e6, 1e6)
        self._start_sb.setValue(10.0)
        layout.addWidget(self._start_sb)

        layout.addWidget(QLabel("End"))
        self._end_sb = QDoubleSpinBox()
        self._end_sb.setDecimals(3)
        self._end_sb.setRange(-1e6, 1e6)
        self._end_sb.setValue(16.0)
        layout.addWidget(self._end_sb)

        layout.addWidget(QLabel("Step"))
        self._step_sb = QDoubleSpinBox()
        self._step_sb.setDecimals(3)
        self._step_sb.setRange(1e-6, 1e6)
        self._step_sb.setValue(0.2)
        layout.addWidget(self._step_sb)

        layout.addWidget(QLabel("Settle (s)"))
        self._settle_sb = QDoubleSpinBox()
        self._settle_sb.setDecimals(2)
        self._settle_sb.setRange(0.0, 60.0)
        self._settle_sb.setValue(0.50)
        layout.addWidget(self._settle_sb)

        layout.addWidget(QLabel("Avg"))
        self._avg_sb = QDoubleSpinBox()
        self._avg_sb.setDecimals(0)
        self._avg_sb.setRange(1, 1000)
        self._avg_sb.setValue(1)
        layout.addWidget(self._avg_sb)

        return group

    def _create_options_group(self) -> QGroupBox:
        group = QGroupBox("Options")
        layout = QHBoxLayout(group)

        layout.addStretch(1)
        self._bg_checkbox = QCheckBox("Do background after scan")
        layout.addWidget(self._bg_checkbox)

        return group

    def _create_controls_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self._on_start)
        layout.addWidget(self._start_btn)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setEnabled(False)
        self._abort_btn.clicked.connect(self._on_abort)
        layout.addWidget(self._abort_btn)

        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setValue(0)
        layout.addWidget(self._progress, 1)

        return layout

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_message(self, msg: str) -> None:
        if self._log:
            self._log.log(msg, source="M2Scan")

    # -------------------------------------------------------------------------
    # Device management
    # -------------------------------------------------------------------------

    def _refresh_devices(self) -> None:
        self._stage_combo.clear()
        for k in REGISTRY.keys("stage:"):
            if k.startswith("stage:serial:"):
                continue
            self._stage_combo.addItem(k)

        self._cam_combo.clear()
        for k in REGISTRY.keys("camera:daheng:"):
            if ":index:" in k:
                continue
            self._cam_combo.addItem(k)

    # -------------------------------------------------------------------------
    # Scan control
    # -------------------------------------------------------------------------

    def _on_start(self) -> None:
        try:
            stage_key = self._stage_combo.currentText().strip()
            cam_key = self._cam_combo.currentText().strip()
            if not stage_key or not cam_key:
                raise ValueError("Please select a stage and a camera.")

            a0 = float(self._start_sb.value())
            a1 = float(self._end_sb.value())
            step = float(self._step_sb.value())
            if step <= 0:
                raise ValueError("Step must be > 0.")

            if a1 >= a0:
                nsteps = int(np.floor((a1 - a0) / step))
                positions = [a0 + i * step for i in range(nsteps + 1)]
                if positions[-1] < a1 - 1e-12:
                    positions.append(a1)
            else:
                nsteps = int(np.floor((a0 - a1) / step))
                positions = [a0 - i * step for i in range(nsteps + 1)]
                if positions[-1] > a1 + 1e-12:
                    positions.append(a1)

            settle = float(self._settle_sb.value())
            avg = int(self._avg_sb.value())
            comment = self._comment_edit.text()

        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        self._last_params = dict(
            stage_key=stage_key,
            cam_key=cam_key,
            positions=positions,
            settle=settle,
            avg=avg,
            scan_name="m_squared",
            comment=comment,
        )
        self._doing_background = False
        self._last_scan_log_path = None
        self._launch_worker(background=False, existing_scan_log=None)
        self._log_message("Scan started…")

    def _launch_worker(self, background: bool, existing_scan_log: str | None) -> None:
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
            background=background,
            existing_scan_log=existing_scan_log,
        )

        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log_message)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._thread.finished.connect(self._thread.deleteLater)

        self._progress.setMaximum(len(p["positions"]) if not background else 1)
        self._progress.setValue(0)
        self._start_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)
        self._thread.start()

    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort = True
            self._log_message("Abort requested.")
            self._abort_btn.setEnabled(False)

    def _on_progress(self, i: int, n: int) -> None:
        self._progress.setMaximum(n)
        self._progress.setValue(i)

    def _on_finished(self, log_path: str) -> None:
        if log_path:
            self._last_scan_log_path = log_path
            self._log_message(f"Scan finished. Log: {log_path}")
        else:
            self._log_message("Scan finished with errors.")

        # Background pass
        if self._last_params and self._bg_checkbox.isChecked() and not self._doing_background:
            self._doing_background = True
            reply = QMessageBox.information(
                self,
                "Background scan",
                "Please block the laser now, then click OK to record the background.",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Ok,
            )
            if reply == QMessageBox.Ok:
                self._log_message("Starting background pass…")
                if self._thread and self._thread.isRunning():
                    self._thread.quit()
                    self._thread.wait()
                self._thread = None
                self._worker = None
                self._launch_worker(background=True, existing_scan_log=self._last_scan_log_path)
                return

        self._abort_btn.setEnabled(False)
        self._start_btn.setEnabled(True)
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None