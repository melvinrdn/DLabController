# src/dlab/diagnostics/ui/scans/grid_scan_tab.py
from __future__ import annotations
import datetime, time
from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Optional
import sip
import numpy as np
from PIL import Image, PngImagePlugin

from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QPushButton,
    QDoubleSpinBox, QTextEdit, QProgressBar, QMessageBox, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QLineEdit, QSpinBox
)

from dlab.boot import ROOT, get_config
from dlab.core.device_registry import REGISTRY
from dlab.diagnostics.ui.scans.grid_scan_live_view import AndorGridScanLiveView
import logging
logger = logging.getLogger("dlab.scans.grid_scan_tab")


from dlab.hardware.wrappers.waveplate_calib import NUM_WAVEPLATES

def power_to_angle(power: float, amplitude: float, offset: float) -> float:
    """
    Converts a power value to the corresponding angle using a cosine fit.
    This function assumes a fit of the form:
         y = amplitude * cos(2*pi/90 * x - 2*pi/90 * offset) + amplitude
    and returns the corresponding angle for a given power.
    """
    A = amplitude / 2.0
    try:
        angle = -(45 * np.arccos(power / A - 1)) / np.pi + offset
    except Exception:
        angle = offset
    return angle

def _wp_index_from_stage_key(stage_key: str) -> Optional[int]:
    # stage keys look like "stage:6"
    try:
        if not stage_key.startswith("stage:"):
            return None
        n = int(stage_key.split(":")[1])
        if 1 <= n <= NUM_WAVEPLATES:
            return n
    except Exception:
        pass
    return None

def _reg_key_powermode(wp_index: int) -> str:
    return f"waveplate:powermode:{wp_index}"

def _reg_key_calib(wp_index: int) -> str:
    return f"waveplate:calib:{wp_index}"

def _reg_key_calib_path(wp_index: int) -> str:
    return f"waveplate:calib_path:{wp_index}"          # str path to calib file (if known)

def _wp_index_from_stage_key(stage_key: str) -> int | None:
    try:
        if not stage_key.startswith("stage:"):
            return None
        i = int(stage_key.split(":")[1])
        return i if 1 <= i <= NUM_WAVEPLATES else None
    except Exception:
        return None


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

# ---------- worker ----------
class GridScanWorker(QObject):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)
    andor_frame = pyqtSignal(object, object, str)  # (axis_indices, frame_u16, camera_key)
    spec_updated = pyqtSignal(object, object)      # (wavelength, counts)
    
    def __init__(
        self,
        axes: List[Tuple[str, List[float]]],      # [(stage_key, [pos...]), ...]
        camera_params: Dict[str, Tuple[int,int]], # camera_key -> (exposure_us, averages) OR (int_ms, averages)
        settle_s: float,
        scan_name: str,
        comment: str,
        mcp_voltage: str,                          # logged only
        background: bool = False,
        existing_scan_log: Optional[str] = None,   # NEW: append to this log if provided
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
        self.existing_scan_log = existing_scan_log
        self.abort = False

    def _emit(self, msg: str) -> None:
        self.log.emit(msg)
        logger.info(msg)

    def _cartesian_indices(self):
        lengths = [len(pos) for _, pos in self.axes]
        def rec(level: int, idxs: list[int]):
            if level == len(lengths):
                yield list(idxs); return
            for i in range(lengths[level]):
                idxs.append(i)
                yield from rec(level + 1, idxs)
                idxs.pop()
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
        import datetime, time, numpy as np
        from dlab.core.device_registry import REGISTRY
        
        
        def _write_scan_log_header(scan_log: Path, axes: List[Tuple[str, List[float]]], comment: str) -> None:
            # header line
            header_cols = []
            for i, (ax, _) in enumerate(axes, 1):
                header_cols += [f"Stage_{i}", f"Pos_{i}"]
            header_cols += ["DetectorKey", "ImageFile", "Exposure_or_IntTime", "Averages", "MCP_Voltage"]
            with open(scan_log, "w", encoding="utf-8") as f:
                f.write("\t".join(header_cols) + "\n")
                # base user comment
                f.write(f"# {comment}\n")
                # append PowerMode context lines (sync with StageControl window)
                from dlab.core.device_registry import REGISTRY
                for ax, _ in axes:
                    wp = _wp_index_from_stage_key(ax)
                    if wp is None:
                        continue
                    pm = bool(REGISTRY.get(_reg_key_powermode(wp)) or False)
                    if not pm:
                        continue
                    calib = REGISTRY.get(_reg_key_calib(wp)) or (None, None)
                    calib_path = REGISTRY.get(_reg_key_calib_path(wp)) or "unknown"
                    f.write(f"# PowerMode ON for {ax} (WP{wp}) using calibration: {calib_path}\n")

        # --- resolve stages ---
        stages = {}
        for stage_key, _ in self.axes:
            stg = REGISTRY.get(stage_key)
            if stg is None:
                self._emit(f"Stage '{stage_key}' not found.")
                self.finished.emit("")
                return
            stages[stage_key] = stg

        # --- resolve detectors ---
        detectors = {}
        for det_key, (expo_us_or_int_ms, _avg) in self.camera_params.items():
            dev = REGISTRY.get(det_key)
            if dev is None:
                self._emit(f"Detector '{det_key}' not found.")
                self.finished.emit("")
                return

            is_camera = hasattr(dev, "grab_frame_for_scan")
            is_spectro = (hasattr(dev, "measure_spectrum") or hasattr(dev, "grab_spectrum_for_scan"))

            if not (is_camera or is_spectro):
                self._emit(f"Detector '{det_key}' doesn't expose a scan API.")
                self.finished.emit("")
                return

            # best-effort preset
            try:
                if is_camera:
                    if hasattr(dev, "set_exposure_us"):
                        dev.set_exposure_us(int(expo_us_or_int_ms))
                    elif hasattr(dev, "setExposureUS"):
                        dev.setExposureUS(int(expo_us_or_int_ms))
                    elif hasattr(dev, "set_exposure"):
                        dev.set_exposure(int(expo_us_or_int_ms))
                else:
                    pass  # Avaspec: no preset needed
            except Exception as e:
                self._emit(f"Warning: failed to preset on '{det_key}': {e}")

            detectors[det_key] = dev

        now = datetime.datetime.now()
        root = _data_root()

        # --- per-scan log selection/creation ---
        scan_dir = root / f"{now:%Y-%m-%d}" / "Scans" / self.scan_name
        scan_dir.mkdir(parents=True, exist_ok=True)

        if self.existing_scan_log:
            scan_log = Path(self.existing_scan_log)
            if not scan_log.exists():
                _write_scan_log_header(scan_log, self.axes, self.comment)
                
        else:
            date_str = f"{now:%Y-%m-%d}"
            idx = 1
            while True:
                candidate = scan_dir / f"{self.scan_name}_log_{date_str}_{idx}.log"
                if not candidate.exists():
                    break
                idx += 1
            scan_log = candidate
            _write_scan_log_header(scan_log, self.axes, self.comment)


        # --- helpers for names/saving ---
        def _pretty_from_key(cam_key: str) -> str:
            suffix = cam_key.split(":")[-1]
            base, *rest = suffix.split("_")
            if base.lower().endswith("cam"):
                vendor = base[:-3]
                camel = (vendor[:1].upper() + vendor[1:]) + "Cam"
            else:
                camel = base[:1].upper() + base[1:]
            return camel + (("_" + "_".join(rest)) if rest else "")

        def _detector_display_name(det_key: str, dev, meta: dict | None) -> str:
            if meta and str(meta.get("DeviceName", "")).strip():
                return str(meta["DeviceName"]).strip()
            for attr in ("name", "camera_name", "model_name"):
                v = getattr(dev, attr, None)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return _pretty_from_key(det_key)
        
        def _save_txt_with_meta(folder: Path, filename: str, x: np.ndarray, y: np.ndarray, header: dict) -> Path:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / filename
            lines = []
            for k, v in header.items():
                lines.append(f"# {k}: {v}")
            lines.append("Wavelength_nm;Counts")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
                for xv, yv in zip(x, y):
                    f.write(f"{float(xv):.6f};{float(yv):.6f}\n")
            return path

        def _save_image(det_key: str, dev, frame_u16: np.ndarray, exposure_us: int, tag: str, meta: dict | None) -> str:
            det_name = _detector_display_name(det_key, dev, meta)
            det_day = root / f"{now:%Y-%m-%d}" / det_name
            ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fn = f"{det_name}_{tag}_{ts_ms}.png"
            _save_png_with_meta(det_day, fn, frame_u16, {"Exposure_us": exposure_us, "Gain": "", "Comment": self.comment})
            return fn
        
        def _save_spectrum(det_key: str, dev, wl_nm: np.ndarray, counts: np.ndarray, int_ms: float, averages: int) -> str:
            det_day = root / f"{now:%Y-%m-%d}" / "Avaspec"
            safe_name = _detector_display_name(det_key, dev, None).replace(" ", "")
            ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            tag = "Background" if self.background else "Spectrum"
            fn = f"{safe_name}_{tag}_{ts_ms}.txt"
            header = {
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "IntegrationTime_ms": int_ms,
                "Averages": averages,
                "Comment": self.comment,
            }
            _save_txt_with_meta(det_day, fn, wl_nm, counts, header)
            return fn

        # --- totals for progress ---
        lengths = [len(pos) for _, pos in self.axes]
        total_points = 1
        for L in lengths:
            total_points *= max(1, L)
        total_images = total_points * max(1, len(self.camera_params))
        done = 0

        # --- main loop ---
        try:
            for idxs in self._cartesian_indices():
                if self.abort:
                    self._emit("Scan aborted.")
                    self.finished.emit("")
                    return
                
                ui_combo = [(self.axes[k][0], self.axes[k][1][idxs[k]]) for k in range(len(self.axes))]

                # Build:
                #   - move_targets: the actual values sent to stages (angles for WP in PowerMode)
                #   - log_combo   : what we write into Pos_i (power if PowerMode, else the same as move)
                move_targets: List[Tuple[str, float]] = []
                log_combo:    List[Tuple[str, float]] = []

                violation_found = False
                violation_msg = ""

                for ax, pos in ui_combo:
                    wp = _wp_index_from_stage_key(ax)
                    if wp is not None:
                        pm = bool(REGISTRY.get(_reg_key_powermode(wp)) or False)
                        if pm:
                            amp_off = REGISTRY.get(_reg_key_calib(wp)) or (None, None)
                            if amp_off[0] is None:
                                self._emit(f"{ax}: Power Mode ON but no calibration; aborting.")
                                self.finished.emit("")
                                return
                            amp, off = float(amp_off[0]), float(amp_off[1])
                            # clip power to [0, amp]
                            orig = float(pos)
                            p = max(0.0, min(amp, orig))
                            if p != orig and not violation_found:
                                violation_found = True
                                violation_msg = (f"{ax}: requested power {orig:.6f} exceeds bounds [0, {amp:.6f}]. "
                                                f"Capped to {p:.6f}. Scan aborted — values were adjusted in the table.")
                            angle = power_to_angle(p, amp, off)
                            move_targets.append((ax, angle))    # move by ANGLE
                            log_combo.append((ax, p))           # log POWER (clipped)
                            continue

                    # default: direct move and log
                    move_targets.append((ax, float(pos)))
                    log_combo.append((ax, float(pos)))

                if violation_found:
                    self._emit(violation_msg)
                    self.finished.emit("")
                    return

                # move all stages (using move_targets)
                move_ok = True
                for ax, move_val in move_targets:
                    try:
                        stages[ax].move_to(float(move_val), blocking=True)
                    except Exception as e:
                        self._emit(f"Move {ax} -> {move_val:.6f} failed: {e}")
                        move_ok = False
                        break
                if not move_ok:
                    continue

                time.sleep(self.settle_s)

                # capture on each detector
                for det_key, dev in detectors.items():
                    if self.abort:
                        self._emit("Scan aborted.")
                        self.finished.emit("")
                        return

                    exposure_or_int, averages = self.camera_params.get(det_key, (0, 1))
                    try:
                        if hasattr(dev, "grab_frame_for_scan"):
                            # CAMERA PATH
                            try:
                                frame_u16, meta = dev.grab_frame_for_scan(
                                    averages=int(averages),
                                    background=self.background,
                                    dead_pixel_cleanup=True,
                                    exposure_us=int(exposure_or_int),
                                )
                            except TypeError:
                                frame_u16, meta = dev.grab_frame_for_scan(
                                    averages=int(averages),
                                    background=self.background,
                                    dead_pixel_cleanup=True,
                                )

                            exp_meta = int((meta or {}).get("Exposure_us", int(exposure_or_int)))
                            tag = "Background" if self.background else "Image"
                            data_fn = _save_image(det_key, dev, frame_u16, exp_meta, tag, meta)

                            if det_key.startswith("camera:andor:"):
                                self.andor_frame.emit(list(idxs), frame_u16, det_key)

                            saved_label = f"exp {exp_meta} µs"

                        else:
                            # SPECTRO PATH (Avaspec)
                            wl = getattr(dev, "wavelength", None)
                            if wl is None:
                                wl = np.asarray(dev.get_wavelengths(), dtype=float)
                            else:
                                wl = np.asarray(wl, dtype=float)

                            if hasattr(dev, "grab_spectrum_for_scan"):
                                counts, meta = dev.grab_spectrum_for_scan(int_ms=float(exposure_or_int), averages=int(averages))
                                counts = np.asarray(counts, dtype=float)
                                int_ms = float((meta or {}).get("Integration_ms", float(exposure_or_int)))
                            else:
                                _buf = []
                                for _ in range(int(averages)):
                                    _ts, _data = dev.measure_spectrum(float(exposure_or_int), 1)
                                    _buf.append(np.asarray(_data, dtype=float))
                                counts = np.mean(np.stack(_buf, axis=0), axis=0)
                                int_ms = float(exposure_or_int)

                            data_fn = _save_spectrum(det_key, dev, wl, counts, int_ms, int(averages))
                            saved_label = f"int {int_ms:.0f} ms"
                            
                            try:
                                self.spec_updated.emit(wl, counts)
                            except Exception:
                                pass    

                        # append scan log row (no Comment column)
                        row = []
                        for ax, logged_pos in log_combo:
                            row += [ax, f"{logged_pos:.9f}"]
                        row += [det_key, data_fn, str(int(exposure_or_int)), str(int(averages)), str(self.mcp_voltage)]
                        with open(scan_log, "a", encoding="utf-8") as f:
                            f.write("\t".join(row) + "\n")

                        self._emit(
                            "Saved {fn} @ {axes} on {det} ({label}, avg {avg})".format(
                                fn=data_fn,
                                axes=", ".join([f"{ax}={val:.6f}" for ax, val in log_combo]),
                                det=det_key,
                                label=saved_label,
                                avg=int(averages),
                            )
                        )

                    except Exception as e:
                        self._emit(
                            f"Capture failed @ " +
                            ", ".join([f"{ax}={val:.6f}" for ax, val in log_combo]) +
                            f" on {det_key}: {e}"
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
    Background pass optional (appended to the same scan log).
    """
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: GridScanWorker | None = None
        self._live_view: AndorGridScanLiveView | None = None
        self._doing_background = False
        self._cached_params = None
        self._last_scan_log_path: Optional[str] = None  # NEW: remember last log for background append
        self._build_ui()
        self._refresh_devices()
        self._pm_sync = QTimer(self)
        self._pm_sync.setInterval(400)  # ms
        self._pm_sync.timeout.connect(self._sync_power_mode_from_registry)
        self._pm_sync.start()

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

        self.axes_tbl = QTableWidget(0, 5)
        self.axes_tbl.setHorizontalHeaderLabels(["Stage", "Start", "End", "Step size", "Power mode"])
        self.axes_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.axes_tbl.setEditTriggers(QAbstractItemView.AllEditTriggers)
        axes_l.addWidget(self.axes_tbl)
        
        move_row = QHBoxLayout()
        self.up_axis_btn = QPushButton("Move Up")
        self.down_axis_btn = QPushButton("Move Down")
        self.up_axis_btn.clicked.connect(lambda: self._move_axis_row(-1))
        self.down_axis_btn.clicked.connect(lambda: self._move_axis_row(+1))
        move_row.addStretch(1)
        move_row.addWidget(self.up_axis_btn)
        move_row.addWidget(self.down_axis_btn)
        axes_l.addLayout(move_row)

        order_hint = QLabel("Traversal order: top row = outer loop → bottom row = inner loop")
        axes_l.addWidget(order_hint)

        rm_row = QHBoxLayout()
        self.rm_axis_btn = QPushButton("Remove Selected Axis")
        self.rm_axis_btn.clicked.connect(self._remove_axis_row)
        rm_row.addStretch(1); rm_row.addWidget(self.rm_axis_btn)
        axes_l.addLayout(rm_row)
        main.addWidget(axes_box)

        # Detectors
        cams_box = QGroupBox("Detectors")
        cams_l = QVBoxLayout(cams_box)

        cam_pick_row = QHBoxLayout()
        self.cam_picker = QComboBox()
        self.add_cam_btn = QPushButton("Add detector")
        self.add_cam_btn.clicked.connect(self._add_cam_row)
        cam_pick_row.addWidget(QLabel("Detector:"))
        cam_pick_row.addWidget(self.cam_picker, 1)
        cam_pick_row.addWidget(self.add_cam_btn)
        cams_l.addLayout(cam_pick_row)

        # Camera table: DetectorKey | Exposure_us/Int_ms | Averages
        self.cam_tbl = QTableWidget(0, 3)
        self.cam_tbl.setHorizontalHeaderLabels(["DetectorsKey", "Exposure_us/Int_ms", "Averages"])
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
        self.scan_name = QLineEdit("")   # NO DEFAULT NAME
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
        self.live_btn = QPushButton("Live Matrix View (Andor)")
        self.live_btn.clicked.connect(self._open_live_view)
        ctl.addWidget(self.live_btn)
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
    def _sync_power_mode_from_registry(self) -> None:
        # Pull current power-mode states from REGISTRY into the table checkboxes (non-invasive)
        for r in range(self.axes_tbl.rowCount()):
            stage_key = (self.axes_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            wp_idx = _wp_index_from_stage_key(stage_key)
            if wp_idx is None:
                continue
            w = self.axes_tbl.cellWidget(r, 4)
            if not hasattr(w, "setChecked"):
                continue
            val = REGISTRY.get(_reg_key_powermode(wp_idx))
            if isinstance(val, bool) and val != w.isChecked():
                w.blockSignals(True)
                w.setChecked(val)
                w.blockSignals(False)


    def _move_axis_row(self, delta: int) -> None:
        sel = self.axes_tbl.selectedIndexes()
        if not sel:
            return
        rows = sorted({i.row() for i in sel})
        if len(rows) != 1:
            return
        r = rows[0]
        dest = r + delta
        if dest < 0 or dest >= self.axes_tbl.rowCount():
            return
        self._swap_axis_rows(r, dest)
        self.axes_tbl.selectRow(dest)

    def _swap_axis_rows(self, r1: int, r2: int) -> None:
        for c in range(self.axes_tbl.columnCount()):
            i1 = self.axes_tbl.item(r1, c)
            i2 = self.axes_tbl.item(r2, c)
            t1 = i1.text() if i1 else ""
            t2 = i2.text() if i2 else ""
            if i1: i1.setText(t2)
            else:  self.axes_tbl.setItem(r1, c, QTableWidgetItem(t2))
            if i2: i2.setText(t1)
            else:  self.axes_tbl.setItem(r2, c, QTableWidgetItem(t1))

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

        # Power mode checkbox per axis (only meaningful for waveplates; disabled otherwise)
        from PyQt5.QtWidgets import QCheckBox
        pm_cb = QCheckBox()
        pm_cb.setTristate(False)
        wp_idx = _wp_index_from_stage_key(stage_key)
        if wp_idx is None:
            pm_cb.setChecked(False)
            pm_cb.setEnabled(False)
            pm_cb.setToolTip("Power mode only available for Thorlabs waveplates")
        else:
            existing = REGISTRY.get(_reg_key_powermode(wp_idx))
            pm_cb.setChecked(bool(existing) if isinstance(existing, bool) else False)

            def _on_local_pm_changed(state: int, wp_index=wp_idx):
                REGISTRY.register(_reg_key_powermode(wp_index), bool(state == 2 or state == 1))
            pm_cb.stateChanged.connect(_on_local_pm_changed)

        self.axes_tbl.setCellWidget(r, 4, pm_cb)


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
        # cameras (daheng + andor + avaspec)
        self.cam_picker.clear()
        for prefix in ("camera:daheng:", "camera:andor:", "spectrometer:avaspec:"):
            for k in REGISTRY.keys(prefix):
                if ":index:" in k:
                    continue
                self.cam_picker.addItem(k)

    def _positions_from_row(self, start: float, end: float, step: float) -> List[float]:
        if step <= 0:
            raise ValueError("Step must be > 0.")
        if end >= start:
            n = int(np.floor((end - start) / step))
            vals = [start + i * step for i in range(n + 1)]
            if vals[-1] < end - 1e-12:
                vals.append(end)
        else:
            n = int(np.floor((start - end) / step))
            vals = [start - i * step for i in range(n + 1)]
            if vals[-1] > end + 1e-12:
                vals.append(end)
        return vals



    def _collect_params(self):
        # axes
        axes: List[Tuple[str, List[float]]] = []

        # small local helper
        def _clip_power(v: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, v))

        corrected_rows: List[int] = []
        row_caps_info: List[str] = []

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

            # If this row is a waveplate in Power Mode, enforce power bounds [0, amplitude]
            pm = self._pm_checked_for_row(r, stage_key)
            wp_idx = _wp_index_from_stage_key(stage_key)
            if pm and wp_idx is not None:
                calib = REGISTRY.get(_reg_key_calib(wp_idx))
                if not calib:
                    raise ValueError(f"{stage_key}: Power Mode is ON but no calibration loaded for WP{wp_idx}.")
                amplitude, offset = float(calib[0]), float(calib[1])
                lo, hi = 0.0, amplitude

                new_start = _clip_power(start, lo, hi)
                new_end   = _clip_power(end,   lo, hi)

                if (new_start != start) or (new_end != end):
                    # Update the UI cells to the capped values
                    (self.axes_tbl.item(r, 1) or QTableWidgetItem()).setText(f"{new_start:.6f}")
                    (self.axes_tbl.item(r, 2) or QTableWidgetItem()).setText(f"{new_end:.6f}")
                    corrected_rows.append(r + 1)
                    row_caps_info.append(f"Row {r+1} ({stage_key}) capped to [{lo:.3f}, {hi:.3f}]")

                start, end = new_start, new_end  # continue with capped values

            # normal positions list (these are powers if PM was ON, angles/positions otherwise)
            pos = self._positions_from_row(start, end, step)
            axes.append((stage_key, pos))

        if not axes:
            raise ValueError("Add at least one axis.")

        # If any row was corrected, abort now with a clear message.
        if corrected_rows:
            raise ValueError("Power value(s) out of range were capped.\n"
                            + "\n".join(row_caps_info)
                            + "\nReview the updated values and press Start again.")

        # cameras + exposure + averages
        cam_params: Dict[str, Tuple[int,int]] = {}
        if self.cam_tbl.rowCount() == 0:
            raise ValueError("Add at least one detector.")
        for r in range(self.cam_tbl.rowCount()):
            cam_key = (self.cam_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            if not cam_key:
                raise ValueError(f"Empty detector key in row {r+1}.")
            try:
                expo = int(float((self.cam_tbl.item(r, 1) or QTableWidgetItem("0")).text()))
                avg  = int(float((self.cam_tbl.item(r, 2) or QTableWidgetItem("1")).text()))
            except ValueError:
                raise ValueError(f"Invalid exposure/averages in detector row {r+1}.")
            if expo <= 0:
                raise ValueError(f"Exposure must be > 0 in detector row {r+1}.")
            if avg <= 0:
                raise ValueError(f"Averages must be > 0 in detector row {r+1}.")
            cam_params[cam_key] = (expo, avg)

        settle = float(self.settle_sb.value())
        scan_name = self.scan_name.text().strip()
        if not scan_name:
            raise ValueError("Please enter a scan name before starting.")
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

        
        
    def _pm_checked_for_row(self, row: int, stage_key: str) -> bool:
        w = self.axes_tbl.cellWidget(row, 4)
        if hasattr(w, "isChecked"):
            # Also force-sync from REGISTRY if it’s a waveplate, so both UIs stay together
            wp_idx = _wp_index_from_stage_key(stage_key)
            if wp_idx is not None:
                external = REGISTRY.get(_reg_key_powermode(wp_idx))
                if isinstance(external, bool) and external != w.isChecked():
                    w.blockSignals(True)
                    w.setChecked(external)
                    w.blockSignals(False)
        return bool(w.isChecked()) if w else False


    def _read_axes_from_table(self):
        """Return [(stage_key, [positions...]), ...] using current table values.
        If 'Power mode' is ON for a Thorlabs waveplate, Start/End/Step are POWER,
        converted here to ANGLES using the calibration in REGISTRY."""
        axes = []
        for r in range(self.axes_tbl.rowCount()):
            stage_key = (self.axes_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            if not stage_key:
                continue
            try:
                start = float((self.axes_tbl.item(r, 1) or QTableWidgetItem("0")).text())
                end   = float((self.axes_tbl.item(r, 2) or QTableWidgetItem("0")).text())
                step  = float((self.axes_tbl.item(r, 3) or QTableWidgetItem("1")).text())
                if step <= 0:
                    continue

                pm = self._pm_checked_for_row(r, stage_key)

                pos = self._positions_from_row(
                    start=start, end=end, step=step,
                    stage_key=stage_key, power_mode=pm
                )
                axes.append((stage_key, pos))
            except Exception:
                pass
        return axes


    def _read_andor_keys_from_cam_table(self):
        """Return the list of selected Andor camera registry keys from the camera table."""
        andor = []
        for r in range(self.cam_tbl.rowCount()):
            cam_key = (self.cam_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            if cam_key.startswith("camera:andor:"):
                andor.append(cam_key)
        return andor

    def _open_live_view(self) -> None:
        axes = self._read_axes_from_table()
        andor_keys = self._read_andor_keys_from_cam_table()

        need_new = (self._live_view is None) or (sip.isdeleted(self._live_view)) or (not self._live_view.isVisible())
        if need_new:
            from dlab.diagnostics.ui.scans.grid_scan_live_view import AndorGridScanLiveView
            self._live_view = AndorGridScanLiveView(None)
            self._live_view.destroyed.connect(lambda: setattr(self, "_live_view", None))
            self._live_view.preconfigure(axes, andor_keys)
        else:
            self._live_view.set_context(axes, andor_keys, preserve=True)

        self._live_view.show(); self._live_view.raise_(); self._live_view.activateWindow()

    # ----- control flow -----
    def _start(self) -> None:
        try:
            p = self._collect_params()
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        self._cached_params = p
        self._doing_background = False
        self._last_scan_log_path = None
        self._launch(background=False, existing_scan_log=None)
        self._log("Grid scan started…")

    def _launch(self, background: bool, existing_scan_log: Optional[str]) -> None:
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
            existing_scan_log=existing_scan_log,  # <— append to same log if provided
        )
        
        # Live view wiring (Andor only)
        if self._live_view is not None and self._live_view.isVisible():
            andor_keys = [k for k in p["camera_params"].keys() if k.startswith("camera:andor:")]
            self._live_view.set_context(p["axes"], andor_keys, preserve=True)
            self._live_view.prepare_for_run()
            try:
                from PyQt5.QtCore import Qt
                self._worker.andor_frame.connect(self._live_view.on_andor_frame, Qt.QueuedConnection)
            except Exception:
                pass

        try:
            from PyQt5.QtCore import Qt
            ui_spec = REGISTRY.get("ui:avaspec_live")
            if ui_spec is not None and hasattr(ui_spec, "set_spectrum_from_scan"):
                self._worker.spec_updated.connect(ui_spec.set_spectrum_from_scan, Qt.QueuedConnection)
        except Exception:
            pass
    
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._finished)
        self._thread.finished.connect(self._thread.deleteLater)
        
        # progress: #points * #cameras
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
            self._last_scan_log_path = log_path  # remember for background
            self._log(f"Grid scan finished. Log: {log_path}")
        else:
            self._log("Scan finished with errors or aborted.")
            self._last_scan_log_path = None

        # Optional background pass — append to the SAME log
        if self._cached_params and self.bg_cb.isChecked() and not self._doing_background:
            self._doing_background = True
            reply = QMessageBox.information(
                self, "Background scan",
                "Please block the laser now, then click OK to record the background.",
                QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok
            )
            if reply == QMessageBox.Ok:
                self._log("Starting background pass…")
                # restart worker on same params, with existing_scan_log set
                if self._thread and self._thread.isRunning():
                    self._thread.quit(); self._thread.wait()
                self._thread = None; self._worker = None
                self._launch(background=True, existing_scan_log=self._last_scan_log_path)
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
