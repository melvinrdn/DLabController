from __future__ import annotations

import datetime
import time
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from PyQt5.QtCore import QTimer, QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox, QPushButton,
    QDoubleSpinBox, QTextEdit, QProgressBar, QMessageBox, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QLineEdit, QCheckBox
)

from dlab.boot import ROOT, get_config
from dlab.core.device_registry import REGISTRY
from dlab.hardware.wrappers.phase_settings import PhaseSettings

import logging
logger = logging.getLogger("dlab.scans.grid_scan_tab")

from dlab.hardware.wrappers.waveplate_calib import NUM_WAVEPLATES


POWER_MODE_SYNC_INTERVAL_MS = 400
SPECTRUM_MEASUREMENT_DELAY_S = 0.01


def power_to_angle(power_fraction, _amp_unused, phase_deg):
    y = float(np.clip(power_fraction, 0.0, 1.0))
    return (phase_deg + (45.0 / np.pi) * float(np.arccos(2.0 * y - 1.0))) % 360.0


def angle_to_power(angle_deg, phase_deg):
    y = 0.5 * (1.0 + float(np.cos(2.0 * np.pi / 90.0 * (float(angle_deg) - float(phase_deg)))))
    return float(np.clip(y, 0.0, 1.0))


def _wp_index_from_stage_key(stage_key):
    try:
        if not stage_key.startswith("stage:"):
            return None
        n = int(stage_key.split(":")[1])
        if 1 <= n <= NUM_WAVEPLATES:
            return n
    except (ValueError, IndexError):
        pass
    return None


def _reg_key_powermode(wp_index):
    return f"waveplate:powermode:{wp_index}"


def _reg_key_calib(wp_index):
    return f"waveplate:calib:{wp_index}"


def _reg_key_calib_path(wp_index):
    return f"waveplate:calib_path:{wp_index}"


def _reg_key_maxvalue(wp_index):
    return f"waveplate:max_value:{wp_index}"


def _data_root():
    cfg = get_config() or {}
    base = cfg.get("paths", {}).get("data_root", "C:/data")
    return (ROOT / base).resolve()


def _save_png_with_meta(folder, filename, frame_u16, meta):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    img = Image.fromarray(frame_u16, mode="I;16")
    pnginfo = PngImagePlugin.PngInfo()
    for k, v in meta.items():
        pnginfo.add_text(str(k), str(v))
    img.save(path.as_posix(), format="PNG", pnginfo=pnginfo)
    return path

def _save_png_with_meta_8bit(folder, filename, frame_u8, meta):
    """Save 8-bit image for Daheng cameras"""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    f8 = np.asarray(frame_u8, dtype=np.uint8, copy=False)
    img = Image.fromarray(f8, mode="L")
    pnginfo = PngImagePlugin.PngInfo()
    for k, v in meta.items():
        pnginfo.add_text(str(k), str(v))
    img.save(path.as_posix(), format="PNG", pnginfo=pnginfo)
    return path

def _detector_display_name(det_key, dev, meta):
    if meta and str(meta.get("DeviceName", "")).strip():
        return str(meta["DeviceName"]).strip()
    for attr in ("name", "camera_name", "model_name"):
        v = getattr(dev, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    suffix = det_key.split(":")[-1]
    base, *rest = suffix.split("_")
    if base.lower().endswith("cam"):
        vendor = base[:-3]
        camel = (vendor[:1].upper() + vendor[1:]) + "Cam"
    else:
        camel = base[:1].upper() + base[1:]
    return camel + (("_" + "_".join(rest)) if rest else "")


class GridScanWorker(QObject):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(
        self,
        axes,
        camera_params,
        settle_s,
        scan_name,
        comment,
        mcp_voltage,
        background=False,
        existing_scan_log=None,
        axes_meta=None,
        parent=None,
    ):
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
        self.axes_meta = axes_meta or {}
        self.data_root = _data_root()
        self.timestamp = datetime.datetime.now()

    def _emit(self, msg):
        self.log.emit(msg)
        logger.info(msg)

    def _cartesian_indices(self):
        lengths = [len(pos) for _, pos in self.axes]
        def rec(level, idxs):
            if level == len(lengths):
                yield list(idxs)
                return
            for i in range(lengths[level]):
                idxs.append(i)
                yield from rec(level + 1, idxs)
                idxs.pop()
        yield from rec(0, [])

    def _write_scan_log_header(self, scan_log, axes, comment):
        header_cols = []
        for i, (ax, _) in enumerate(axes, 1):
            header_cols += [f"Stage_{i}", f"pos_{i}", f"power_{i}"]
        header_cols += ["DetectorKey", "ImageFile", "Exposure_or_IntTime_or_Period",
                        "Averages_or_None", "MCP_Voltage"]

        with open(scan_log, "w", encoding="utf-8") as f:
            f.write("\t".join(header_cols) + "\n")
            f.write(f"# {comment}\n")
            for ax, _ in axes:
                wp = _wp_index_from_stage_key(ax)
                meta = self.axes_meta.get(ax, {})
                pm_on = bool(meta.get("pm", False))
                if wp is not None and pm_on:
                    calib_path = meta.get("calib_path",
                        REGISTRY.get(_reg_key_calib_path(wp)) or "unknown")
                    mv = meta.get("max_value_W",
                        REGISTRY.get(_reg_key_maxvalue(wp)))
                    mv_txt = "none" if mv is None else f"{float(mv):.6g} W"
                    f.write("# PowerMode ON for {ax} (WP{wp}) | calib={calib} | max_value={mv}\n"
                            .format(ax=ax, wp=wp, calib=calib_path, mv=mv_txt))
                    f.write("#   Start fraction={sf:.6f} | Start angle={sa:.3f} deg | "
                            "Rotation={de:.3f} deg | Step={st:.3f} deg\n"
                            .format(
                                sf=float(meta.get("start_fraction", float("nan"))),
                                sa=float(meta.get("start_angle_deg", float("nan"))),
                                de=float(meta.get("delta_deg", float("nan"))),
                                st=float(meta.get("step_deg", float("nan")))
                            ))
                else:
                    f.write("# PowerMode OFF for {ax} | Start={st:.6g} | End={en:.6g} | Step={sp:.6g}\n"
                            .format(
                                ax=ax,
                                st=float(meta.get("start", float("nan"))),
                                en=float(meta.get("end", float("nan"))),
                                sp=float(meta.get("step", float("nan")))
                            ))

    def _save_image(self, det_key, dev, frame, exposure_us, tag, meta, is_8bit=False):
        det_name = _detector_display_name(det_key, dev, meta)
        det_day = self.data_root / f"{self.timestamp:%Y-%m-%d}" / det_name
        ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fn = f"{det_name}_{tag}_{ts_ms}.png"
        
        if is_8bit:
            _save_png_with_meta_8bit(det_day, fn, frame,
                                    {"Exposure_us": exposure_us, "Gain": "", "Comment": self.comment})
        else:
            _save_png_with_meta(det_day, fn, frame,
                            {"Exposure_us": exposure_us, "Gain": "", "Comment": self.comment})
        return fn

    def _save_spectrum(self, det_key, dev, wl_nm, counts, int_ms, averages):
        det_day = self.data_root / f"{self.timestamp:%Y-%m-%d}" / "Avaspec"
        safe_name = _detector_display_name(det_key, dev, None).replace(" ", "")
        ts_ms = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tag = "Background" if self.background else "Spectrum"
        fn = f"{safe_name}_{tag}_{ts_ms}.txt"
        header = {
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "IntegrationTime_ms": int_ms,
            "Averages": averages,
            "Comment": self.comment,
            "CalibrationApplied": bool(getattr(dev, "has_calibration", lambda: False)())
        }
        det_day.mkdir(parents=True, exist_ok=True)
        path = det_day / fn
        lines = [f"# {k}: {v}" for k, v in header.items()]
        lines.append("Wavelength_nm;Counts")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            for xv, yv in zip(wl_nm, counts):
                f.write(f"{float(xv):.6f};{float(yv):.6f}\n")
        return fn

    def _capture_camera(self, det_key, dev, params):
        exposure_or_int = int(params[0]) if len(params) >= 1 else 0
        averages = int(params[1]) if len(params) >= 2 else 1
        
        is_daheng = "daheng" in det_key.lower()
        
        try:
            frame, meta = dev.grab_frame_for_scan(
                averages=int(averages),
                background=self.background,
                dead_pixel_cleanup=True,
                exposure_us=int(exposure_or_int),
                force_roi=True,
            )
        except TypeError:
            frame, meta = dev.grab_frame_for_scan(
                averages=int(averages),
                background=self.background,
                dead_pixel_cleanup=True,
                force_roi=True,
            )
        
        exp_meta = int((meta or {}).get("Exposure_us", exposure_or_int))
        tag = "Background" if self.background else "Image"
        data_fn = self._save_image(det_key, dev, frame, exp_meta, tag, meta=None, is_8bit=is_daheng)
        saved_label = f"exp {exp_meta} µs"
        
        return data_fn, saved_label

    def _capture_spectrometer(self, det_key, dev, params):
        exposure_or_int = float(params[0]) if len(params) >= 1 else 0.0
        averages = int(params[1]) if len(params) >= 2 else 1
        
        if hasattr(dev, "get_wavelengths"):
            wl = np.asarray(dev.get_wavelengths(), dtype=float)
        else:
            wl = np.asarray(getattr(dev, "wavelength", None), dtype=float)

        if wl is None or wl.size == 0:
            raise ValueError(f"{det_key}: wavelength array empty")

        if hasattr(dev, "grab_spectrum_for_scan"):
            counts, meta = dev.grab_spectrum_for_scan(
                int_ms=float(exposure_or_int),
                averages=int(averages)
            )
            counts = np.asarray(counts, dtype=float)
            int_ms = float((meta or {}).get("Integration_ms", float(exposure_or_int)))
        else:
            buf = []
            for _ in range(int(averages)):
                _ts, _data = dev.measure_spectrum(float(exposure_or_int), 1)
                buf.append(np.asarray(_data, dtype=float))
                time.sleep(SPECTRUM_MEASUREMENT_DELAY_S)
            counts = np.mean(np.stack(buf, axis=0), axis=0)
            int_ms = float(exposure_or_int)

        if counts.size != wl.size:
            raise ValueError(f"{det_key}: spectrum length mismatch")

        data_fn = f"{int_ms:.1f}ms_spectrum"
        saved_label = f"int {int_ms:.0f} ms"
        
        return data_fn, saved_label

    def _capture_powermeter(self, det_key, dev, params):
        period_ms = float(params[0]) if len(params) >= 1 else 100.0
        averages = int(params[1]) if len(params) >= 2 else 1
        wavelength_nm = float(params[2]) if len(params) >= 3 else None

        if wavelength_nm is not None and hasattr(dev, "set_wavelength"):
            try:
                dev.set_wavelength(float(wavelength_nm))
            except Exception as e:
                logger.warning(f"Failed to set wavelength: {e}")

        vals = []
        n_avg = max(1, int(averages))
        for i in range(n_avg):
            v = float(dev.read_power())
            vals.append(v)
            if i + 1 < n_avg:
                time.sleep(period_ms / 1000.0)

        power = float(np.mean(vals)) if vals else float("nan")
        data_fn = f"{power:.9f}"
        saved_label = f"P={power:.3e} W"
        
        return data_fn, saved_label

    def _format_position_log(self, log_combo):
        return ", ".join([
            f"{ax}: pos={float(pv):.6f}" +
            ("" if (powv == "") else f", power={float(powv):.6f}")
            for ax, pv, powv in log_combo
        ])

    def _move_slm_axis(self, ax, pos):
        parts = ax.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid SLM axis format '{ax}'. Expected slm:ClassName:FieldName")

        _, class_name, field_name = parts

        active_classes = REGISTRY.get("slm:red:active_classes") or []
        widgets = REGISTRY.get("slm:red:widgets") or []

        if class_name not in active_classes:
            raise ValueError(
                f"SLM class '{class_name}' is not active on the red SLM.\n"
                f"Active classes: {active_classes}"
            )

        phase_widget = None
        for w in widgets:
            if getattr(w, "name_", lambda: "")() == class_name:
                phase_widget = w
                break

        if phase_widget is None:
            raise ValueError(f"SLM widget for '{class_name}' not found in registry.")

        if not hasattr(phase_widget, field_name):
            candidate_fields = [
                name for name in dir(phase_widget)
                if name.startswith("le_") or name.startswith("cb_")
            ]
            raise ValueError(
                f"Field '{field_name}' does not exist in SLM class '{class_name}'.\n"
                f"Available fields: {candidate_fields}"
            )

        widget = getattr(phase_widget, field_name)
        widget.setText(str(pos))

        slm_window = REGISTRY.get("slm:red:window")
        if slm_window is None:
            raise RuntimeError("SLM window not registered. Cannot compose multi-layer patterns.")

        levels = slm_window.compose_levels()

        slm_red = REGISTRY.get("slm:red:controller")
        if slm_red is None:
            raise RuntimeError(
                "Red SLM is not active. Open the SLM window and publish a pattern first."
            )

        screen_num = self.axes_meta[ax].get("screen", 3)
        slm_red.publish(levels, screen_num=screen_num)
        self._emit(f"SLM {class_name}:{field_name} = {pos}")

    def _prepare_move_targets(self, ui_combo):
        move_targets = []
        log_combo = []

        for ax, pos in ui_combo:
            wp = _wp_index_from_stage_key(ax)
            meta = self.axes_meta.get(ax, {})
            pm_on = bool(meta.get("pm", False))

            if wp is not None and pm_on:
                angle = float(pos)
                amp_off = REGISTRY.get(_reg_key_calib(wp)) or (None, None)
                if amp_off[1] is None:
                    raise ValueError(f"{ax}: Power Mode ON but no calibration phase")
                phase = float(amp_off[1])
                frac = angle_to_power(angle, phase)
                mv = REGISTRY.get(_reg_key_maxvalue(wp))
                if mv is None:
                    power_val = frac
                else:
                    power_val = frac * float(mv)

                move_targets.append((ax, angle))
                log_combo.append((ax, angle, power_val))
            else:
                move_targets.append((ax, float(pos)))
                log_combo.append((ax, float(pos), ""))

        return move_targets, log_combo

    def _initialize_stages(self):
        stages = {}
        for stage_key, _ in self.axes:
            if stage_key.startswith("slm:"):
                stages[stage_key] = "VIRTUAL_SLM"
            else:
                stg = REGISTRY.get(stage_key)
                if stg is None:
                    raise ValueError(f"Stage '{stage_key}' not found")
                stages[stage_key] = stg
        return stages

    def _initialize_detectors(self):
        detectors = {}
        for det_key, params in self.camera_params.items():
            dev = REGISTRY.get(det_key)
            if dev is None:
                raise ValueError(f"Detector '{det_key}' not found")

            is_camera = hasattr(dev, "grab_frame_for_scan")
            is_spectro = hasattr(dev, "measure_spectrum") or hasattr(dev, "grab_spectrum_for_scan")
            is_pow = hasattr(dev, "fetch_power")

            if not (is_camera or is_spectro or is_pow):
                raise ValueError(f"Detector '{det_key}' doesn't expose a scan API")

            try:
                if is_camera:
                    exposure = int(params[0])
                    if hasattr(dev, "set_exposure_us"):
                        dev.set_exposure_us(exposure)
                    elif hasattr(dev, "setExposureUS"):
                        dev.setExposureUS(exposure)
                    elif hasattr(dev, "set_exposure"):
                        dev.set_exposure(exposure)
                elif is_pow:
                    if len(params) >= 2 and hasattr(dev, "set_avg"):
                        try:
                            dev.set_avg(int(params[1]))
                        except Exception as e:
                            logger.warning(f"Failed to set avg on {det_key}: {e}")
                    if len(params) >= 3 and hasattr(dev, "set_wavelength"):
                        try:
                            dev.set_wavelength(float(params[2]))
                        except Exception as e:
                            logger.warning(f"Failed to set wavelength on {det_key}: {e}")
            except Exception as e:
                self._emit(f"Warning: failed to preset on '{det_key}': {e}")

            detectors[det_key] = dev
        return detectors

    def _create_scan_log(self):
        scan_dir = self.data_root / f"{self.timestamp:%Y-%m-%d}" / "Scans" / self.scan_name
        scan_dir.mkdir(parents=True, exist_ok=True)

        if self.existing_scan_log:
            return Path(self.existing_scan_log)

        date_str = f"{self.timestamp:%Y-%m-%d}"
        idx = 1
        while True:
            candidate = scan_dir / f"{self.scan_name}_log_{date_str}_{idx}.log"
            if not candidate.exists():
                break
            idx += 1
        
        self._write_scan_log_header(candidate, self.axes, self.comment)
        return candidate

    def run(self):
        try:
            stages = self._initialize_stages()
            detectors = self._initialize_detectors()
            scan_log = self._create_scan_log()
        except ValueError as e:
            self._emit(str(e))
            self.finished.emit("")
            return

        lengths = [len(pos) for _, pos in self.axes]
        total_points = 1
        for L in lengths:
            total_points *= max(1, L)
        total_images = total_points * max(1, len(self.camera_params))
        done = 0

        try:
            for idxs in self._cartesian_indices():
                if self.abort:
                    self._emit("Scan aborted.")
                    self.finished.emit("")
                    return

                ui_combo = [(self.axes[k][0], self.axes[k][1][idxs[k]])
                            for k in range(len(self.axes))]

                try:
                    move_targets, log_combo = self._prepare_move_targets(ui_combo)
                except ValueError as e:
                    self._emit(str(e))
                    self.finished.emit("")
                    return

                move_ok = True
                for ax, move_val in move_targets:
                    try:
                        if ax.startswith("slm:"):
                            self._move_slm_axis(ax, move_val)
                        else:
                            stages[ax].move_to(float(move_val), blocking=True)
                    except Exception as e:
                        self._emit(f"Move {ax} -> {move_val:.6f} failed: {e}")
                        move_ok = False
                        break
                
                if not move_ok:
                    continue

                time.sleep(float(self.settle_s))

                for det_key, dev in detectors.items():
                    if self.abort:
                        self._emit("Scan aborted.")
                        self.finished.emit("")
                        return

                    params = self.camera_params.get(det_key, (0, 1))

                    try:
                        if hasattr(dev, "grab_frame_for_scan"):
                            data_fn, saved_label = self._capture_camera(det_key, dev, params)
                        elif hasattr(dev, "measure_spectrum") or hasattr(dev, "grab_spectrum_for_scan"):
                            data_fn, saved_label = self._capture_spectrometer(det_key, dev, params)
                        else:
                            data_fn, saved_label = self._capture_powermeter(det_key, dev, params)

                        row = []
                        for ax, pos_val, power_val in log_combo:
                            row += [
                                ax,
                                f"{float(pos_val):.9f}",
                                ("" if power_val == "" else f"{float(power_val):.9f}")
                            ]

                        row += [
                            det_key,
                            data_fn,
                            str(params[0] if len(params) >= 1 else ""),
                            str(params[1] if len(params) >= 2 else ""),
                            str(self.mcp_voltage)
                        ]

                        with open(scan_log, "a", encoding="utf-8") as f:
                            f.write("\t".join(row) + "\n")

                        pos_log = self._format_position_log(log_combo)
                        avg = int(params[1] if len(params) >= 2 else 1)
                        self._emit(f"Saved {data_fn} @ {pos_log} on {det_key} ({saved_label}, avg {avg})")

                    except Exception as e:
                        pos_log = self._format_position_log(log_combo)
                        self._emit(f"Capture failed @ {pos_log} on {det_key}: {e}")

                    done += 1
                    self.progress.emit(done, total_images)

        except Exception as e:
            self._emit(f"Fatal error: {e}")
            self.finished.emit("")
            return

        self.finished.emit(scan_log.as_posix())


class GridScanTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._worker = None
        self._doing_background = False
        self._cached_params = None
        self._last_scan_log_path = None
        self._build_ui()
        self._refresh_devices()
        self._pm_sync = QTimer(self)
        self._pm_sync.setInterval(POWER_MODE_SYNC_INTERVAL_MS)
        self._pm_sync.timeout.connect(self._sync_power_mode_from_registry)
        self._pm_sync.start()

    def _build_ui(self):
        main = QVBoxLayout(self)

        axes_box = QGroupBox("Axes")
        axes_l = QVBoxLayout(axes_box)

        picker = QHBoxLayout()
        self.stage_picker = QComboBox()
        self.add_axis_btn = QPushButton("Add Axis")
        self.add_axis_btn.clicked.connect(self._add_axis_row)
        picker.addWidget(QLabel("Stage or slm:<Class>:"))
        picker.addWidget(self.stage_picker, 1)
        picker.addWidget(self.add_axis_btn)
        axes_l.addLayout(picker)

        self.axes_tbl = QTableWidget(0, 9)
        self.axes_tbl.setHorizontalHeaderLabels([
            "Stage", "Param", "Start", "End", "Step", "Screen", "Power mode", "Max value (W)", "Go max"
        ])
        self.axes_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.axes_tbl.setEditTriggers(QAbstractItemView.AllEditTriggers)
        axes_l.addWidget(self.axes_tbl)

        mv = QHBoxLayout()
        self.up_axis_btn = QPushButton("Up")
        self.down_axis_btn = QPushButton("Down")
        self.up_axis_btn.clicked.connect(lambda: self._move_axis_row(-1))
        self.down_axis_btn.clicked.connect(lambda: self._move_axis_row(+1))
        mv.addStretch(1)
        mv.addWidget(self.up_axis_btn)
        mv.addWidget(self.down_axis_btn)
        axes_l.addLayout(mv)

        rm = QHBoxLayout()
        self.rm_axis_btn = QPushButton("Remove")
        self.rm_axis_btn.clicked.connect(self._remove_axis_row)
        rm.addStretch(1)
        rm.addWidget(self.rm_axis_btn)
        axes_l.addLayout(rm)

        main.addWidget(axes_box)

        cams_box = QGroupBox("Detectors")
        cams_l = QVBoxLayout(cams_box)

        cam_pick = QHBoxLayout()
        self.cam_picker = QComboBox()
        self.add_cam_btn = QPushButton("Add detector")
        self.add_cam_btn.clicked.connect(self._add_cam_row)
        cam_pick.addWidget(QLabel("Detector:"))
        cam_pick.addWidget(self.cam_picker, 1)
        cam_pick.addWidget(self.add_cam_btn)
        cams_l.addLayout(cam_pick)

        self.cam_tbl = QTableWidget(0, 4)
        self.cam_tbl.setHorizontalHeaderLabels(["DetectorsKey", "Exposure_us/Int_ms", "Wavelength_nm", "Averages"])
        self.cam_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cam_tbl.setEditTriggers(QAbstractItemView.AllEditTriggers)
        cams_l.addWidget(self.cam_tbl)

        rm_cam = QHBoxLayout()
        self.rm_cam_btn = QPushButton("Remove Selected Camera")
        self.rm_cam_btn.clicked.connect(self._remove_cam_row)
        rm_cam.addStretch(1)
        rm_cam.addWidget(self.rm_cam_btn)
        cams_l.addLayout(rm_cam)

        main.addWidget(cams_box)

        params = QGroupBox("Scan parameters")
        p = QHBoxLayout(params)
        self.settle_sb = QDoubleSpinBox()
        self.settle_sb.setDecimals(2)
        self.settle_sb.setRange(0.0, 60.0)
        self.settle_sb.setValue(0.5)
        self.scan_name = QLineEdit("")
        self.comment = QLineEdit("")
        self.mcp_voltage = QLineEdit("")
        p.addWidget(QLabel("Settle (s)"))
        p.addWidget(self.settle_sb)
        p.addWidget(QLabel("Scan name"))
        p.addWidget(self.scan_name, 1)
        p.addWidget(QLabel("Comment"))
        p.addWidget(self.comment, 2)
        p.addWidget(QLabel("MCP voltage"))
        p.addWidget(self.mcp_voltage, 1)
        main.addWidget(params)

        ctl = QHBoxLayout()
        self.start_btn = QPushButton("Start")
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
        self.refresh_btn = QPushButton("Refresh devices")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        rr.addStretch(1)
        rr.addWidget(self.refresh_btn)
        main.addLayout(rr)

    def _sync_power_mode_from_registry(self):
        for r in range(self.axes_tbl.rowCount()):
            ax = (self.axes_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            if ax.startswith("slm:"):
                continue
            wp = _wp_index_from_stage_key(ax)
            if wp is None:
                continue
            w = self.axes_tbl.cellWidget(r, 6)
            if not hasattr(w, "setChecked"):
                continue
            val = REGISTRY.get(_reg_key_powermode(wp))
            if isinstance(val, bool) and val != w.isChecked():
                w.blockSignals(True)
                w.setChecked(val)
                w.blockSignals(False)

    def _move_axis_row(self, delta):
        sel = self.axes_tbl.selectedIndexes()
        if not sel:
            return
        rows = sorted({i.row() for i in sel})
        if len(rows) != 1:
            return
        r = rows[0]
        d = r + delta
        if 0 <= d < self.axes_tbl.rowCount():
            self._swap_axis_rows(r, d)
            self.axes_tbl.selectRow(d)

    def _swap_axis_rows(self, r1, r2):
        for c in range(self.axes_tbl.columnCount()):
            x = self.axes_tbl.item(r1, c)
            y = self.axes_tbl.item(r2, c)
            t1 = x.text() if x else ""
            t2 = y.text() if y else ""
            if x:
                x.setText(t2)
            else:
                self.axes_tbl.setItem(r1, c, QTableWidgetItem(t2))
            if y:
                y.setText(t1)
            else:
                self.axes_tbl.setItem(r2, c, QTableWidgetItem(t1))

    def _add_axis_row(self):
        ax = self.stage_picker.currentText().strip()
        if not ax:
            QMessageBox.warning(self, "Pick an axis", "Select a stage or slm:<Class>.")
            return

        r = self.axes_tbl.rowCount()
        self.axes_tbl.insertRow(r)
        self.axes_tbl.setItem(r, 0, QTableWidgetItem(ax))

        if ax.startswith("slm:"):
            parts = ax.split(":")
            if len(parts) < 2:
                QMessageBox.critical(self, "Invalid SLM axis", "SLM axis must be slm:ClassName")
                return

            class_name = parts[1]

            try:
                phase_ref = PhaseSettings.new_type(None, class_name)
            except Exception:
                valid = ", ".join(sorted(PhaseSettings.types.keys()))
                QMessageBox.critical(
                    self,
                    "Unknown SLM class",
                    f"Class '{class_name}' not found.\nValid classes:\n{valid}"
                )
                return

            self.axes_tbl.setItem(r, 1, QTableWidgetItem(""))
            self.axes_tbl.setItem(r, 2, QTableWidgetItem("0.0"))
            self.axes_tbl.setItem(r, 3, QTableWidgetItem("1.0"))
            self.axes_tbl.setItem(r, 4, QTableWidgetItem("0.1"))

            self.axes_tbl.setItem(r, 5, QTableWidgetItem("1"))
            pm = QCheckBox()
            pm.setEnabled(False)
            self.axes_tbl.setCellWidget(r, 6, pm)

            self.axes_tbl.setItem(r, 7, QTableWidgetItem(""))
            gm = QCheckBox()
            gm.setEnabled(False)
            self.axes_tbl.setCellWidget(r, 8, gm)

            def on_item_changed(item):
                if item.row() != r:
                    return
                if item.column() != 1:
                    return

                param = item.text().strip()
                if not param:
                    return

                if not hasattr(phase_ref, param):
                    valid = sorted([k for k in dir(phase_ref) if k.startswith("le_")])
                    QMessageBox.critical(
                        self,
                        "Invalid SLM parameter",
                        f"Parameter '{param}' does not exist for class '{class_name}'.\n"
                        f"Valid parameters:\n" + "\n".join(valid)
                    )
                    return

                new_name = f"slm:{class_name}:{param}"
                self.axes_tbl.item(r, 0).setText(new_name)

            self.axes_tbl.itemChanged.connect(on_item_changed)
            return

        self.axes_tbl.setItem(r, 1, QTableWidgetItem(""))
        self.axes_tbl.setItem(r, 2, QTableWidgetItem("0.0"))
        self.axes_tbl.setItem(r, 3, QTableWidgetItem("1.0"))
        self.axes_tbl.setItem(r, 4, QTableWidgetItem("0.1"))

        pm = QCheckBox()
        self.axes_tbl.setCellWidget(r, 6, pm)
        self.axes_tbl.setItem(r, 5, QTableWidgetItem(""))

        self.axes_tbl.setItem(r, 7, QTableWidgetItem(""))
        gm = QCheckBox()
        self.axes_tbl.setCellWidget(r, 8, gm)

    def _remove_axis_row(self):
        rows = sorted({idx.row() for idx in self.axes_tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self.axes_tbl.removeRow(r)

    def _add_cam_row(self):
        cam_key = self.cam_picker.currentText().strip()
        if not cam_key:
            QMessageBox.warning(self, "Pick a camera", "Select a camera to add.")
            return
        r = self.cam_tbl.rowCount()
        self.cam_tbl.insertRow(r)
        self.cam_tbl.setItem(r, 0, QTableWidgetItem(cam_key))
        self.cam_tbl.setItem(r, 1, QTableWidgetItem("5000"))
        self.cam_tbl.setItem(r, 2, QTableWidgetItem(""))
        self.cam_tbl.setItem(r, 3, QTableWidgetItem("1"))

    def _remove_cam_row(self):
        rows = sorted({i.row() for i in self.cam_tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self.cam_tbl.removeRow(r)

    def _refresh_devices(self):
        self.stage_picker.clear()
        for k in REGISTRY.keys("stage:"):
            if not k.startswith("stage:serial:"):
                self.stage_picker.addItem(k)
        for t in PhaseSettings.types:
            self.stage_picker.addItem(f"slm:{t}")
        self.cam_picker.clear()
        for prefix in ("camera:daheng:", "camera:andor:", "spectrometer:avaspec:", "powermeter:"):
            for k in REGISTRY.keys(prefix):
                if ":index:" not in k:
                    self.cam_picker.addItem(k)

    def _positions(self, start, end, step):
        if step <= 0:
            raise ValueError("Step must be > 0.")
        if end >= start:
            n = int((end - start) // step)
            vals = [start + i * step for i in range(n + 1)]
            if vals[-1] < end:
                vals.append(end)
        else:
            n = int((start - end) // step)
            vals = [start - i * step for i in range(n + 1)]
            if vals[-1] > end:
                vals.append(end)
        return vals

    def _collect_params(self):
        axes = []
        axes_meta = {}
        if self.axes_tbl.rowCount() == 0:
            raise ValueError("Add at least one axis.")

        for r in range(self.axes_tbl.rowCount()):
            ax = (self.axes_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            param = (self.axes_tbl.item(r, 1) or QTableWidgetItem("")).text().strip()
            try:
                start = float((self.axes_tbl.item(r, 2) or QTableWidgetItem("0")).text())
                end = float((self.axes_tbl.item(r, 3) or QTableWidgetItem("0")).text())
                step = float((self.axes_tbl.item(r, 4) or QTableWidgetItem("1")).text())
            except ValueError:
                raise ValueError(f"Invalid numeric value in axis row {r+1}.")

            if ax.startswith("slm:"):
                if param == "":
                    raise ValueError(f"SLM axis {ax}: missing parameter name.")
                vals = self._positions(start, end, step)
                axes.append((ax, vals))
                axes_meta[ax] = {
                    "param": param,
                    "screen": int((self.axes_tbl.item(r, 5) or QTableWidgetItem("1")).text()),
                }
                continue

            wp = _wp_index_from_stage_key(ax)
            pm = False
            if wp is not None:
                w = self.axes_tbl.cellWidget(r, 6)
                pm = bool(w.isChecked()) if w else False

            if pm and wp is not None:
                sf = max(0.0, min(1.0, start))
                amp_off = REGISTRY.get(_reg_key_calib(wp)) or (None, None)
                if amp_off[1] is None:
                    raise ValueError(f"{ax}: Power mode ON but no calibration.")
                phase = float(amp_off[1])
                start_angle = power_to_angle(sf, 1.0, phase)
                end_angle_abs = start_angle + end
                pos = self._positions(start_angle, end_angle_abs, step)
                axes.append((ax, pos))

                max_item = self.axes_tbl.item(r, 7)
                try:
                    mv = float((max_item.text() if max_item else "").strip())
                except:
                    mv = float("nan")
                if not (np.isfinite(mv) and mv > 0):
                    raise ValueError(f"{ax}: invalid max power value.")
                REGISTRY.register(_reg_key_maxvalue(wp), float(mv))

                axes_meta[ax] = {
                    "pm": True,
                    "start_fraction": float(sf),
                    "start_angle_deg": float(start_angle),
                    "delta_deg": float(end),
                    "step_deg": float(step),
                    "max_value_W": float(mv),
                }
            else:
                pos = self._positions(start, end, step)
                axes.append((ax, pos))
                axes_meta[ax] = {
                    "pm": False,
                    "start": float(start),
                    "end": float(end),
                    "step": float(step),
                }

        cam_params = {}
        if self.cam_tbl.rowCount() == 0:
            raise ValueError("Add at least one detector.")
        for r in range(self.cam_tbl.rowCount()):
            cam = (self.cam_tbl.item(r, 0) or QTableWidgetItem("")).text().strip()
            if not cam:
                raise ValueError(f"Empty detector key row {r+1}.")
            p1 = (self.cam_tbl.item(r, 1) or QTableWidgetItem("0")).text()
            p2 = (self.cam_tbl.item(r, 2) or QTableWidgetItem("")).text()
            p3 = (self.cam_tbl.item(r, 3) or QTableWidgetItem("1")).text()
            if cam.startswith("powermeter:"):
                cam_params[cam] = (float(p1), int(p3), float(p2 or 1030))
            else:
                cam_params[cam] = (int(float(p1)), int(float(p3)))

        settle = float(self.settle_sb.value())
        name = self.scan_name.text().strip()
        if not name:
            raise ValueError("Missing scan name.")
        comment = self.comment.text()
        mcp = self.mcp_voltage.text().strip()

        return {
            "axes": axes,
            "axes_meta": axes_meta,
            "camera_params": cam_params,
            "settle": settle,
            "scan_name": name,
            "comment": comment,
            "mcp_voltage": mcp,
        }

    def _start(self):
        try:
            p = self._collect_params()
        except Exception as e:
            QMessageBox.critical(self, "Invalid parameters", str(e))
            return
        self._cached_params = p
        self._doing_background = False
        self._last_scan_log_path = None
        self._launch(False, None)
        self._log("Scan started…")

    def _launch(self, background, existing):
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
            existing_scan_log=existing,
            axes_meta=p.get("axes_meta", {}),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)

        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._finished)

        self._thread.finished.connect(self._thread.deleteLater)

        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)

        total = 1
        for _, pos in p["axes"]:
            total *= max(1, len(pos))
        total *= max(1, len(p["camera_params"]))
        self.prog.setMaximum(total)
        self.prog.setValue(0)

        self._thread.start()

    def _abort(self):
        if self._worker:
            self._worker.abort = True
            self.abort_btn.setEnabled(False)

    def _on_progress(self, i, n):
        self.prog.setMaximum(n)
        self.prog.setValue(i)
        self._log(f"{i}/{n}")

    def _finished(self, log_path):
        if log_path:
            self._last_scan_log_path = log_path
            self._log(f"Scan finished: {log_path}")
        else:
            self._log("Scan finished with errors.")
            self._last_scan_log_path = None

        self.abort_btn.setEnabled(False)
        self.start_btn.setEnabled(True)

        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

        if not self._doing_background and self._last_scan_log_path is not None:
            reply = QMessageBox.question(
                self,
                "Run Background Scan?",
                "The scan finished.\n\nDo you want to run the BACKGROUND scan now?\n"
                "If yes, cut the gas and wait 3-5min before continuing.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self._doing_background = True
                self._log("Launching background scan…")
                self._launch(background=True, existing=self._last_scan_log_path)
                return

        self._doing_background = False

    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())