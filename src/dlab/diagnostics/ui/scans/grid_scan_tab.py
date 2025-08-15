# src/dlab/diagnostics/ui/scans/grid_scan_tab.py
from __future__ import annotations
import datetime, time
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

import numpy as np
from PIL import Image, PngImagePlugin

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QPushButton,
    QDoubleSpinBox, QTextEdit, QProgressBar, QMessageBox, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QLineEdit, QSpinBox
)

from dlab.boot import ROOT, get_config
from dlab.core.device_registry import REGISTRY

import logging
logger = logging.getLogger("dlab.scans.grid_scan_tab_fixedexpo")


# ---------- helpers ----------
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

def _append_daheng_log(cam_folder: Path, cam_name: str, fn: str, exposure_us: int, comment: str) -> None:
    log_path = cam_folder / f"{cam_name}_log_{datetime.datetime.now():%Y-%m-%d}.log"
    header = "File Name\tExposure_us\tGain\tComment\n"
    exists = log_path.exists()
    with open(log_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(header)
        f.write(f"{fn}\t{exposure_us}\t\t{comment}\n")


# ---------- worker ----------
class GridScanWorker(QObject):
    progress = pyqtSignal(int, int)   # i, total images
    log = pyqtSignal(str)
    finished = pyqtSignal(str)        # scan log path (or "")

    def __init__(
        self,
        axes: List[Tuple[str, List[float]]],      # [(stage_key, [pos...]), ...]
        camera_params: Dict[str, Tuple[int,int]], # camera_key -> (exposure_us, averages)
        settle_s: float,
        scan_name: str,
        comment: str,
        mcp_voltage: str,                          # logged only
        background: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.axes = axes
        self.camera_params = camera_params
        self.settle_s = float(settle_s)
        self.scan_name = scan_name
        self.comment = comment
        self.mcp_voltage = mcp_voltage
        self.background = bool(background)
        self.abort = False

    def _emit(self, msg: str) -> None:
        self.log.emit(msg)
        logger.info(msg)

    def _cartesian_positions(self) -> Iterable[List[Tuple[str, float]]]:
        pos_lists = [[(ax, p) for p in pos] for ax, pos in self.axes]
        def rec(idx: int, acc: List[Tuple[str, float]]):
            if idx == len(pos_lists):
                yield list(acc); return
            for pair in pos_lists[idx]:
                acc.append(pair)
                yield from rec(idx+1, acc)
                acc.pop()
        yield from rec(0, [])

    def _apply_exposure_if_possible(self, camwin, exposure_us: int) -> None:
        try:
            if hasattr(camwin, "set_exposure_us"):
                camwin.set_exposure_us(int(exposure_us)); return
            if hasattr(camwin, "setExposureUS"):
                camwin.setExposureUS(int(exposure_us)); return
            if hasattr(camwin, "set_exposure"):  # seconds
                camwin.set_exposure(exposure_us / 1e6); return
        except Exception as e:
            self._emit(f"Warning: failed to set exposure on '{getattr(camwin, 'name', 'Camera')}'. ({e})")

    def run(self) -> None:
        # Stages
        stages = {}
        for stage_key, _ in self.axes:
            stg = REGISTRY.get(stage_key)
            if stg is None:
                self._emit(f"Stage '{stage_key}' not found."); self.finished.emit(""); return
            stages[stage_key] = stg

        # Cameras
        cameras = {}
        for cam_key, (expo_us, _avg) in self.camera_params.items():
            camwin = REGISTRY.get(cam_key)
            if camwin is None:
                self._emit(f"Camera '{cam_key}' not found."); self.finished.emit(""); return
            if not hasattr(camwin, "grab_frame_for_scan"):
                self._emit(f"Camera '{cam_key}' missing grab_frame_for_scan()."); self.finished.emit(""); return
            cameras[cam_key] = camwin
            self._apply_exposure_if_possible(camwin, int(expo_us))

        now = datetime.datetime.now()
        root = _data_root()
        scan_dir = root / f"{now:%Y-%m-%d}" / "Scans" / self.scan_name
        scan_dir.mkdir(parents=True, exist_ok=True)

        # Log header: axes + CameraKey + ImageFile + Exposure_us + Averages + MCP_Voltage + Comment
        header_cols = []
        for i, (ax, _) in enumerate(self.axes, 1):
            header_cols += [f"Stage_{i}", f"Pos_{i}"]
        header_cols += ["CameraKey", "ImageFile", "Exposure_us", "Averages", "MCP_Voltage", "Comment"]
        scan_log = scan_dir / f"{self.scan_name}_log_{now:%Y-%m-%d}.log"
        if not scan_log.exists():
            with open(scan_log, "w", encoding="utf-8") as f:
                f.write("\t".join(header_cols) + "\n")

        # Progress: points * cameras  (on compte 1 image sauvée par combo, l'average est interne)
        total_points = 1
        for _, pos in self.axes:
            total_points *= max(1, len(pos))
        total_images = total_points * max(1, len(self.camera_params))
        done = 0

        # -- helpers (à mettre en haut du fichier, près des autres helpers) --
        def _pretty_from_key(cam_key: str) -> str:
            # ex: "camera:daheng:dahengcam_1" -> "DahengCam_1"
            suffix = cam_key.split(":")[-1]  # "dahengcam_1"
            # camel-case simple
            base, *rest = suffix.split("_")
            camel = "".join([base[:1].upper(), base[1:]])
            # insert camel-case for "cam" -> "Cam"
            camel = "".join([camel[0].upper(), camel[1:3], camel[3:4].upper(), camel[4:]]) if len(camel) >= 4 else camel
            if rest:
                camel += "_" + "_".join(rest)
            return camel

        def _camera_display_name(cam_key: str, camwin, meta: dict | None) -> str:
            if meta and "CameraName" in meta and str(meta["CameraName"]).strip():
                return str(meta["CameraName"]).strip()
            for attr in ("name", "camera_name", "model_name"):
                val = getattr(camwin, attr, None)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return _pretty_from_key(cam_key)


        def save_one(cam_key: str, camwin, frame_u16: np.ndarray, exposure_us: int, tag: str, meta: dict | None) -> str:
            cam_name = _camera_display_name(cam_key, camwin, meta)  # << nom d'affichage fiable
            cam_day = root / f"{now:%Y-%m-%d}" / cam_name
            ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fn = f"{cam_name}_{tag}_{ts_ms}.png"                    # << prefixe = cam_name
            _save_png_with_meta(cam_day, fn, frame_u16, {"Exposure_us": exposure_us, "Gain": "", "Comment": self.comment})
            _append_daheng_log(cam_day, cam_name, fn, exposure_us, self.comment)
            return fn


        try:
            for combo in self._cartesian_positions():
                if self.abort:
                    self._emit("Scan aborted."); self.finished.emit(""); return

                # Move all axes
                for ax, pos in combo:
                    try:
                        stages[ax].move_to(float(pos), blocking=True)
                    except Exception as e:
                        self._emit(f"Move {ax} -> {pos:.6f} failed: {e}")
                        break
                time.sleep(self.settle_s)

                # Capture on each camera with fixed exposure + per-camera averages
                for cam_key, camwin in cameras.items():
                    if self.abort:
                        self._emit("Scan aborted."); self.finished.emit(""); return

                    exposure_us, averages = self.camera_params.get(cam_key, (0, 1))
                    try:
                        # Pass averages when possible
                        try:
                            frame_u16, meta = camwin.grab_frame_for_scan(
                                averages=int(averages),
                                background=self.background,
                                dead_pixel_cleanup=True,
                                exposure_us=int(exposure_us),
                            )
                        except TypeError:
                            frame_u16, meta = camwin.grab_frame_for_scan(
                                averages=int(averages),
                                background=self.background,
                                dead_pixel_cleanup=True,
                            )

                        exp_meta = int(meta.get("Exposure_us", int(exposure_us)))
                        tag = "Background" if self.background else "Image"
                        cam_fn = save_one(cam_key, camwin, frame_u16, exp_meta, tag, meta)

                        # Append scan log line
                        row = []
                        for ax, pos in combo:
                            row += [ax, f"{pos:.9f}"]
                        row += [cam_key, cam_fn, str(exp_meta), str(int(averages)), str(self.mcp_voltage), self.comment]
                        with open(scan_log, "a", encoding="utf-8") as f:
                            f.write("\t".join(row) + "\n")

                        self._emit(
                            f"Saved {cam_fn} @ " +
                            ", ".join([f"{ax}={pos:.6f}" for ax, pos in combo]) +
                            f" on {cam_key} (exp {exp_meta} µs, avg {int(averages)})"
                        )
                    except Exception as e:
                        self._emit(
                            f"Capture failed @ " +
                            ", ".join([f"{ax}={pos:.6f}" for ax, pos in combo]) +
                            f" on {cam_key}: {e}"
                        )

                    done += 1
                    self.progress.emit(done, total_images)

        except Exception as e:
            self._emit(f"Fatal error: {e}")
            self.finished.emit("")
            return

        self.finished.emit(scan_log.as_posix())


# ---------- UI ----------
class GridScanTab(QWidget):
    """
    Multiple axes (stages) + multiple cameras with fixed exposure & per-camera averages.
    Background pass optional.
    """
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: GridScanWorker | None = None
        self._doing_background = False
        self._cached_params = None
        self._build_ui()
        self._refresh_devices()

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)

        # Axes (stages)
        axes_box = QGroupBox("Axes (stages)")
        axes_l = QVBoxLayout(axes_box)

        picker = QHBoxLayout()
        self.stage_picker = QComboBox()
        self.add_axis_btn = QPushButton("Add Axis")
        self.add_axis_btn.clicked.connect(self._add_axis_row)
        picker.addWidget(QLabel("Stage:"))
        picker.addWidget(self.stage_picker, 1)
        picker.addWidget(self.add_axis_btn)
        axes_l.addLayout(picker)

        self.axes_tbl = QTableWidget(0, 4)
        self.axes_tbl.setHorizontalHeaderLabels(["Stage", "Start", "End", "Step"])
        self.axes_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.axes_tbl.setEditTriggers(QAbstractItemView.AllEditTriggers)
        axes_l.addWidget(self.axes_tbl)

        rm_row = QHBoxLayout()
        self.rm_axis_btn = QPushButton("Remove Selected Axis")
        self.rm_axis_btn.clicked.connect(self._remove_axis_row)
        rm_row.addStretch(1); rm_row.addWidget(self.rm_axis_btn)
        axes_l.addLayout(rm_row)
        main.addWidget(axes_box)

        # Cameras with fixed exposure & averages
        cams_box = QGroupBox("Cameras")
        cams_l = QVBoxLayout(cams_box)

        cam_pick_row = QHBoxLayout()
        self.cam_picker = QComboBox()
        self.add_cam_btn = QPushButton("Add Camera")
        self.add_cam_btn.clicked.connect(self._add_cam_row)
        cam_pick_row.addWidget(QLabel("Camera:"))
        cam_pick_row.addWidget(self.cam_picker, 1)
        cam_pick_row.addWidget(self.add_cam_btn)
        cams_l.addLayout(cam_pick_row)

        # Camera table: CameraKey | Exposure_us | Averages
        self.cam_tbl = QTableWidget(0, 3)
        self.cam_tbl.setHorizontalHeaderLabels(["CameraKey", "Exposure_us", "Averages"])
        self.cam_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cam_tbl.setEditTriggers(QAbstractItemView.AllEditTriggers)
        cams_l.addWidget(self.cam_tbl)

        rm_cam_row = QHBoxLayout()
        self.rm_cam_btn = QPushButton("Remove Selected Camera")
        self.rm_cam_btn.clicked.connect(self._remove_cam_row)
        rm_cam_row.addStretch(1); rm_cam_row.addWidget(self.rm_cam_btn)
        cams_l.addLayout(rm_cam_row)
        main.addWidget(cams_box)

        # Scan parameters (+ MCP voltage)
        params = QGroupBox("Scan parameters")
        p = QHBoxLayout(params)
        self.settle_sb = QDoubleSpinBox(); self.settle_sb.setDecimals(2); self.settle_sb.setRange(0.0, 60.0); self.settle_sb.setValue(0.50)
        self.scan_name = QLineEdit("grid_scan")
        self.comment   = QLineEdit("")
        self.mcp_voltage = QLineEdit("")   # logged only
        p.addWidget(QLabel("Settle (s)")); p.addWidget(self.settle_sb)
        p.addWidget(QLabel("Scan name"));  p.addWidget(self.scan_name, 1)
        p.addWidget(QLabel("Comment"));    p.addWidget(self.comment, 2)
        p.addWidget(QLabel("MCP voltage"));p.addWidget(self.mcp_voltage, 1)
        main.addWidget(params)

        # Background toggle
        bg_box = QGroupBox("Background")
        b = QHBoxLayout(bg_box)
        self.bg_cb = QPushButton("Do background after scan")
        self.bg_cb.setCheckable(True); self.bg_cb.setChecked(True)
        b.addWidget(self.bg_cb)
        b.addStretch(1)
        main.addWidget(bg_box)

        # Controls
        ctl = QHBoxLayout()
        self.start_btn = QPushButton("Start"); self.start_btn.clicked.connect(self._start)
        self.abort_btn = QPushButton("Abort"); self.abort_btn.setEnabled(False); self.abort_btn.clicked.connect(self._abort)
        self.prog = QProgressBar(); self.prog.setMinimum(0); self.prog.setValue(0)
        ctl.addWidget(self.start_btn); ctl.addWidget(self.abort_btn); ctl.addWidget(self.prog, 1)
        main.addLayout(ctl)

        # Log + refresh
        self.log = QTextEdit(); self.log.setReadOnly(True)
        main.addWidget(self.log, 1)
        ref_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh devices")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        ref_row.addStretch(1); ref_row.addWidget(self.refresh_btn)
        main.addLayout(ref_row)

    # ----- UI helpers -----
    def _add_axis_row(self) -> None:
        stage_key = self.stage_picker.currentText().strip()
        if not stage_key:
            QMessageBox.warning(self, "Pick a stage", "Select a stage to add.")
            return
        r = self.axes_tbl.rowCount()
        self.axes_tbl.insertRow(r)
        self.axes_tbl.setItem(r, 0, QTableWidgetItem(stage_key))
        self.axes_tbl.setItem(r, 1, QTableWidgetItem("0.0"))
        self.axes_tbl.setItem(r, 2, QTableWidgetItem("10.0"))
        self.axes_tbl.setItem(r, 3, QTableWidgetItem("1.0"))

    def _remove_axis_row(self) -> None:
        rows = sorted({idx.row() for idx in self.axes_tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self.axes_tbl.removeRow(r)

    def _add_cam_row(self) -> None:
        cam_key = self.cam_picker.currentText().strip()
        if not cam_key:
            QMessageBox.warning(self, "Pick a camera", "Select a camera to add.")
            return
        r = self.cam_tbl.rowCount()
        self.cam_tbl.insertRow(r)
        self.cam_tbl.setItem(r, 0, QTableWidgetItem(cam_key))
        self.cam_tbl.setItem(r, 1, QTableWidgetItem("5000"))  # default 5 ms
        self.cam_tbl.setItem(r, 2, QTableWidgetItem("1"))     # default averages = 1

    def _remove_cam_row(self) -> None:
        rows = sorted({idx.row() for idx in self.cam_tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self.cam_tbl.removeRow(r)

    def _refresh_devices(self) -> None:
        # stages
        self.stage_picker.clear()
        for k in REGISTRY.keys("stage:"):
            if k.startswith("stage:serial:"):
                continue
            self.stage_picker.addItem(k)
        # cameras (daheng + andor)
        self.cam_picker.clear()
        for prefix in ("camera:daheng:", "camera:andor:"):
            for k in REGISTRY.keys(prefix):
                if ":index:" in k:
                    continue
                self.cam_picker.addItem(k)

    def _positions_from_row(self, start: float, end: float, step: float) -> List[float]:
        if step <= 0:
            raise ValueError("Step must be > 0.")
        if end >= start:
            nsteps = int(np.floor((end - start) / step))
            pos = [start + i*step for i in range(nsteps+1)]
            if pos[-1] < end - 1e-12:
                pos.append(end)
            return pos
        else:
            nsteps = int(np.floor((start - end) / step))
            pos = [start - i*step for i in range(nsteps+1)]
            if pos[-1] > end + 1e-12:
                pos.append(end)
            return pos

    def _collect_params(self):
        # axes
        axes: List[Tuple[str, List[float]]] = []
        for r in range(self.axes_tbl.rowCount()):
            stage_key = (self.axes_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            try:
                start = float((self.axes_tbl.item(r, 1) or QTableWidgetItem("0")).text())
                end   = float((self.axes_tbl.item(r, 2) or QTableWidgetItem("0")).text())
                step  = float((self.axes_tbl.item(r, 3) or QTableWidgetItem("1")).text())
            except ValueError:
                raise ValueError(f"Invalid number in axis row {r+1}.")
            if not stage_key:
                raise ValueError(f"Empty stage key in row {r+1}.")
            positions = self._positions_from_row(start, end, step)
            axes.append((stage_key, positions))
        if not axes:
            raise ValueError("Add at least one axis.")

        # cameras + exposure + averages
        cam_params: Dict[str, Tuple[int,int]] = {}
        if self.cam_tbl.rowCount() == 0:
            raise ValueError("Add at least one camera.")
        for r in range(self.cam_tbl.rowCount()):
            cam_key = (self.cam_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            if not cam_key:
                raise ValueError(f"Empty camera key in row {r+1}.")
            try:
                expo = int(float((self.cam_tbl.item(r, 1) or QTableWidgetItem("0")).text()))
                avg  = int(float((self.cam_tbl.item(r, 2) or QTableWidgetItem("1")).text()))
            except ValueError:
                raise ValueError(f"Invalid exposure/averages in camera row {r+1}.")
            if expo <= 0:
                raise ValueError(f"Exposure must be > 0 in camera row {r+1}.")
            if avg <= 0:
                raise ValueError(f"Averages must be > 0 in camera row {r+1}.")
            cam_params[cam_key] = (expo, avg)

        settle = float(self.settle_sb.value())
        scan_name = self.scan_name.text().strip() or "grid_scan"
        comment = self.comment.text()
        mcp_voltage = self.mcp_voltage.text().strip()

        return dict(
            axes=axes,
            camera_params=cam_params,
            settle=settle,
            scan_name=scan_name,
            comment=comment,
            mcp_voltage=mcp_voltage,
        )

    # ----- control flow -----
    def _start(self) -> None:
        try:
            p = self._collect_params()
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        self._cached_params = p
        self._doing_background = False
        self._launch(background=False)
        self._log("Grid scan (fixed exposure, per-camera averages) started…")

    def _launch(self, background: bool) -> None:
        p = self._cached_params
        if not p:
            return
        self._thread = QThread(self)
        self._worker = GridScanWorker(
            axes=p["axes"],
            camera_params=p["camera_params"],
            settle_s=p["settle"],
            scan_name=p["scan_name"],
            comment=p["comment"],
            mcp_voltage=p["mcp_voltage"],
            background=background,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._finished)
        self._thread.finished.connect(self._thread.deleteLater)

        # progress: #points * #cameras (1 image sauvegardée par caméra et par point)
        total_points = 1
        for _, pos in p["axes"]:
            total_points *= max(1, len(pos))
        total_images = total_points * max(1, len(p["camera_params"]))
        self.prog.setMaximum(total_images)
        self.prog.setValue(0)

        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self._thread.start()

    def _abort(self) -> None:
        if self._worker:
            self._worker.abort = True
            self._log("Abort requested.")
            self.abort_btn.setEnabled(False)

    def _on_progress(self, i: int, n: int) -> None:
        self.prog.setMaximum(n)
        self.prog.setValue(i)
        self._log(f"{i}/{n}")

    def _finished(self, log_path: str) -> None:
        if log_path:
            self._log(f"Grid scan finished. Log: {log_path}")
        else:
            self._log("Scan finished with errors or aborted.")

        # Optional background pass
        if self._cached_params and self.bg_cb.isChecked() and not self._doing_background:
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
                self._launch(background=True)
                return

        self.abort_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        if self._thread and self._thread.isRunning():
            self._thread.quit(); self._thread.wait()
        self._thread = None
        self._worker = None

    def _log(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
