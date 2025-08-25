from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QFileDialog, QLineEdit
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
        # backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from dlab.boot import ROOT, get_config

# ----------------------------
# Waveplate Calibration Widget
# ----------------------------

NUM_WAVEPLATES = 6
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']


def _calibration_root() -> Path:
    cfg = get_config() or {}
    rel = (cfg.get("paths", {}) or {}).get("calibration", "ressources/calibration")
    return (ROOT / rel).resolve()


def _wp_default_yaml() -> Path:
    return _calibration_root() / "wp_default.yaml"


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (_calibration_root() / p).resolve()


def _rel_to_root(p: Path) -> str:
    try:
        return p.resolve().relative_to(_calibration_root()).as_posix()
    except Exception:
        return p.as_posix()


def _load_xy_file(path: Path) -> tuple[np.ndarray, np.ndarray]:

    xs: list[float] = []
    ys: list[float] = []

    def _try_float(tok: str) -> float | None:
        try:
            return float(tok)
        except Exception:
            return None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            for sep in (";", ",", "\t"):
                s = s.replace(sep, " ")
            parts = [p for p in s.split(" ") if p]
            if len(parts) < 2:
                continue
            x = _try_float(parts[0]); y = _try_float(parts[1])
            if x is None or y is None:
                continue
            xs.append(float(x)); ys.append(float(y))

    if not xs:
        return np.array([]), np.array([])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


class WaveplateCalibWidget(QWidget):

    def __init__(self, log_callback=None, calibration_changed_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Waveplate Calibration")

        self.log_callback = log_callback
        self.calibration_changed_callback = calibration_changed_callback

        # {wp_index (int): absolute Path to file}
        self.default_calib: Dict[int, Path] = {}
        # {wp_index (int): (1.0, phase_deg)}
        self.calibration_params: Dict[int, Tuple[float, float]] = {}
        # {wp_index (int): max power seen in file (W) — informational only}
        self.max_abs_power: Dict[int, float] = {}

        # per-WP UI
        self.wp_entries: Dict[str, Dict[str, QLineEdit]] = {}

        self._init_ui()
        self._load_defaults_from_yaml()

    # ---------- UI ----------

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # Left: controls
        options_group = QGroupBox("Calibration Options")
        options_layout = QVBoxLayout(options_group)

        for i in range(1, NUM_WAVEPLATES + 1):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"WP{i}:"), stretch=0)

            row.addWidget(QLabel("Max (norm):"), stretch=0)
            max_edit = QLineEdit("1.00"); max_edit.setFixedWidth(70); max_edit.setReadOnly(True)
            row.addWidget(max_edit, stretch=0)

            row.addWidget(QLabel("Phase (deg):"), stretch=0)
            off_edit = QLineEdit("0"); off_edit.setFixedWidth(70); off_edit.setReadOnly(True)
            row.addWidget(off_edit, stretch=0)

            self.wp_entries[str(i)] = {"max": max_edit, "offset": off_edit}
            options_layout.addLayout(row)

        choose_row = QHBoxLayout()
        choose_row.addWidget(QLabel("Select WP:"), stretch=0)
        self.wp_dropdown = QComboBox()
        self.wp_dropdown.addItems([str(i) for i in range(1, NUM_WAVEPLATES + 1)])
        choose_row.addWidget(self.wp_dropdown, stretch=1)
        options_layout.addLayout(choose_row)

        self.update_btn = QPushButton("Update Calibration File")
        self.update_btn.clicked.connect(self._update_selected_calibration_file)
        options_layout.addWidget(self.update_btn)

        main_layout.addWidget(options_group, 1)

        # Right: plot
        self.fig = Figure(figsize=(5, 6), dpi=100)
        rows = int(np.ceil(np.sqrt(NUM_WAVEPLATES)))
        cols = int(np.ceil(NUM_WAVEPLATES / rows))
        self.axes = [self.fig.add_subplot(rows, cols, i + 1) for i in range(NUM_WAVEPLATES)]
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, 2)

    # ---------- YAML I/O ----------

    def _load_defaults_from_yaml(self) -> None:
        yaml_path = _wp_default_yaml()
        if not yaml_path.exists():
            self.log(f"`{yaml_path}` not found. No defaults loaded.")
            return
        try:
            import yaml
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            files_map = (data.get("wp_defaults") or {}) if isinstance(data, dict) else {}
            for k, v in files_map.items():
                try:
                    idx = int(k) if not isinstance(k, int) else k
                    self.default_calib[idx] = _resolve_path(v)
                except Exception:
                    pass
            self.log("Loaded defaults from wp_default.yaml.")
        except Exception as e:
            self.log(f"Failed to read wp_default.yaml: {e}")

    def _save_defaults_to_yaml(self) -> None:
        yaml_path = _wp_default_yaml()
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml
            out = {"wp_defaults": {str(k): _rel_to_root(v) for k, v in self.default_calib.items()}}
            yaml_path.write_text(yaml.safe_dump(out, sort_keys=True), encoding="utf-8")
            self.log("Saved defaults to wp_default.yaml.")
        except Exception as e:
            self.log(f"Failed to write wp_default.yaml: {e}")

    # ---------- Model: normalized cos + phase-only fit ----------

    @staticmethod
    def _cos01(x_deg: np.ndarray | float, phase_deg: float) -> np.ndarray | float:
        # y ∈ [0,1] with period 180° for a half-wave plate
        return 0.5 * (1.0 + np.cos(2.0 * np.pi / 90.0 * (np.asarray(x_deg) - phase_deg)))

    def _fit_phase_only(self, x: np.ndarray, y01: np.ndarray) -> float:
        def f(xx, phase):
            return self._cos01(xx, phase)
        popt, _ = curve_fit(f, x, y01, p0=(0.0,))
        return float(popt[0])

    # ---------- Public API used by StageControl ----------

    def load_waveplate_calibration(self, wp_index: int) -> bool:
        p = self.default_calib.get(wp_index)
        if not p or not p.exists():
            self.log(f"No default calibration file for WP{wp_index}.")
            return False

        self._open_calibration_file(wp_index, p)
        try:
            from dlab.core.device_registry import REGISTRY
            REGISTRY.register(f"waveplate:calib:{wp_index}",
                              tuple(self.calibration_params[wp_index]))
            REGISTRY.register(f"waveplate:calib_path:{wp_index}",
                              p.as_posix())
        except Exception:
            pass
        return True

    # ---------- Calibration loading ----------

    def _open_calibration_file(self, wp_index: int, filepath: Path) -> None:
        try:
            angles, powers = _load_xy_file(filepath)
            if angles.size == 0:
                raise ValueError("Empty calibration file")

            pmax = float(np.nanmax(powers))
            if not np.isfinite(pmax) or pmax <= 0:
                raise ValueError("Invalid max power in calibration file")

            # Normalize to 0..1
            y01 = np.clip(powers / pmax, 0.0, 1.0)
            phase = self._fit_phase_only(angles, y01)

            # Persist
            self.calibration_params[wp_index] = (1.0, phase)   # amplitude=1.0 (fraction)
            self.max_abs_power[wp_index] = pmax

            # UI
            self.wp_entries[str(wp_index)]["max"].setText("1.00")
            self.wp_entries[str(wp_index)]["offset"].setText(f"{phase:.2f}")

            # Plot normalized data + fit
            ax = self.axes[wp_index - 1]
            ax.clear()
            color = COLORS[(wp_index - 1) % len(COLORS)]
            ax.plot(angles, y01, marker='o', linestyle='None', color=color, label="data (norm)")
            xs = np.linspace(0, 360, 721)
            ax.plot(xs, self._cos01(xs, phase), color=color, label="fit (norm)")
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Relative power (0..1)")
            ax.legend(loc="best")

            rel = _rel_to_root(filepath)
            self.log(f"WP{wp_index} loaded from ./{rel} (max={pmax:.3g} W, normalized to 0..1)")
            if self.calibration_changed_callback:
                self.calibration_changed_callback(wp_index, self.calibration_params[wp_index])

            # Publish to REGISTRY
            try:
                from dlab.core.device_registry import REGISTRY
                REGISTRY.register(f"waveplate:calib:{wp_index}", (1.0, phase))
                REGISTRY.register(f"waveplate:calib_path:{wp_index}", filepath.as_posix())
                REGISTRY.register(f"waveplate:max:{wp_index}", pmax)  # info only
            except Exception:
                pass

            self.canvas.draw_idle()

        except Exception as e:
            self.log(f"Failed to load calibration for WP{wp_index}: {e}")

    def _update_selected_calibration_file(self) -> None:
        wp_txt = self.wp_dropdown.currentText()
        wp_index = int(wp_txt)
        start_dir = _calibration_root()
        fname, _ = QFileDialog.getOpenFileName(
            self, f"Select calibration file for WP{wp_index}",
            str(start_dir),
            "Data Files (*.twt *.txt *.csv);;All Files (*)"
        )
        if not fname:
            return
        p = Path(fname)
        self.default_calib[wp_index] = p
        self._save_defaults_to_yaml()
        self._open_calibration_file(wp_index, p)

    # ---------- Logging ----------

    def log(self, message: str):
        msg = f"[WaveplateCalib] {message}"
        if self.log_callback:
            self.log_callback(msg)
        else:
            now = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] {msg}")
