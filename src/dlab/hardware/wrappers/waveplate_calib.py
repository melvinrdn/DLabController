import os
import json
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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D

from dlab.boot import ROOT, get_config

# ----------------------------
# Waveplate Calibration Widget
# ----------------------------

# How many WPs you want to show
NUM_WAVEPLATES = 6

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']


def _calibration_root() -> Path:
    """Base folder for calibration; default 'ressources/calibration'."""
    cfg = get_config() or {}
    rel = (cfg.get("paths", {}) or {}).get("calibration", "ressources/calibration")
    return (ROOT / rel).resolve()


def _wp_default_yaml() -> Path:
    """Location of the YAML with default WP files."""
    return _calibration_root() / "wp_default.yaml"


def _resolve_path(p: str | Path) -> Path:
    """Resolve a path relative to the calibration root if not absolute."""
    p = Path(p)
    if p.is_absolute():
        return p
    return (_calibration_root() / p).resolve()


def _rel_to_root(p: Path) -> str:
    """Make a path relative to calibration root for nice YAML writes."""
    try:
        return p.resolve().relative_to(_calibration_root()).as_posix()
    except Exception:
        return p.as_posix()


def _load_xy_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust loader for 2-column data.
    - Skips lines starting with '#'
    - Accepts tab/space/semicolon/comma separators
    - Handles .txt/.twt produced by AutoWaveplateCalib
    """
    buf_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace(";", " ").replace(",", " ")
            buf_lines.append(s + "\n")
    if not buf_lines:
        return np.array([]), np.array([])
    from io import StringIO
    arr = np.loadtxt(StringIO("".join(buf_lines)))
    if arr.ndim == 1:
        if arr.size < 2:
            return np.array([]), np.array([])
        arr = arr.reshape(1, -1)
    return arr[:, 0], arr[:, 1]


class WaveplateCalibWidget(QWidget):
    """
    Loads default calibration files from `wp_default.yaml`, plots each WP, and
    exposes a single waveplate selector to replace its file. After loading or
    updating a calibration, calls `calibration_changed_callback(wp_index, (max_val, offset))`.
    """

    def __init__(self, log_callback=None, calibration_changed_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Waveplate Calibration")

        self.log_callback = log_callback
        self.calibration_changed_callback = calibration_changed_callback

        # {wp_index (int): absolute Path to file}
        self.default_calib: Dict[int, Path] = {}

        # {wp_index (int): (max_value, offset)}
        self.calibration_params: Dict[int, Tuple[float, float]] = {}

        # per-WP line edits for displaying current params
        self.wp_entries: Dict[str, Dict[str, QLineEdit]] = {}

        self._init_ui()
        self._load_defaults_from_yaml()
        self._load_all_calibrations()

    # ---------- UI ----------

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        # Left: controls
        options_group = QGroupBox("Calibration Options")
        options_layout = QVBoxLayout(options_group)

        for i in range(1, NUM_WAVEPLATES + 1):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"WP{i}:"), stretch=0)

            row.addWidget(QLabel("Max:"), stretch=0)
            max_edit = QLineEdit("0"); max_edit.setFixedWidth(70)
            row.addWidget(max_edit, stretch=0)

            row.addWidget(QLabel("Offset:"), stretch=0)
            off_edit = QLineEdit("0"); off_edit.setFixedWidth(70)
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

    # ---------- Calibration loading ----------

    def _load_all_calibrations(self) -> None:
        # Try to load configured files for 1..N
        for wp in range(1, NUM_WAVEPLATES + 1):
            p = self.default_calib.get(wp)
            if not p or not p.exists():
                self.log(f"Calibration file for WP{wp} not found.")
                continue
            self._open_calibration_file(wp, p)
        self._update_global_legend()
        self.canvas.draw()

    def _open_calibration_file(self, wp_index: int, filepath: Path) -> None:
        try:
            angles, powers = _load_xy_file(filepath)
            if angles.size == 0:
                raise ValueError("Empty calibration file")
            amp, phase = self._cos_fit(angles, powers)
            self.calibration_params[wp_index] = (2 * amp, phase)

            self.wp_entries[str(wp_index)]["max"].setText(f"{2 * amp:.2f}")
            self.wp_entries[str(wp_index)]["offset"].setText(f"{phase:.2f}")

            ax = self.axes[wp_index - 1]
            self._plot_waveplate(ax, angles, powers, COLORS[(wp_index - 1) % len(COLORS)], amp, phase)

            rel = _rel_to_root(filepath)
            self.log(f"WP{wp_index} loaded from ./{rel}")
            if self.calibration_changed_callback:
                self.calibration_changed_callback(wp_index, self.calibration_params[wp_index])
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

    # ---------- Plot / math ----------

    def _plot_waveplate(self, ax, angles, powers, color, amplitude, phase):
        ax.clear()
        ax.plot(angles, powers, marker='o', linestyle='None', color=color)
        xs = np.linspace(0, 360, 361)
        ax.plot(xs, self._cos_func(xs, amplitude, phase), color=color)
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Power (W)")

    def _update_global_legend(self):
        handles = [Line2D([], [], color=COLORS[i % len(COLORS)], marker='o', linestyle='None')
                   for i in range(NUM_WAVEPLATES)]
        labels = [f"WP{i + 1}" for i in range(NUM_WAVEPLATES)]
        self.fig.legend(handles, labels, loc='upper right')
        self.fig.tight_layout(rect=[0, 0, 0.85, 1])

    @staticmethod
    def _cos_func(x, amplitude, phase):
        return amplitude * np.cos(2 * np.pi / 90 * x - 2 * np.pi / 90 * phase) + amplitude

    def _cos_fit(self, x, y):
        guess = (np.max(y) / 2.0, 0.0)
        popt, _ = curve_fit(self._cos_func, x, y, p0=guess)
        return popt  # (amplitude, phase)

    # ---------- Logging ----------

    def log(self, message: str):
        msg = f"[WaveplateCalib] {message}"
        if self.log_callback:
            self.log_callback(msg)
        else:
            now = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] {msg}")
