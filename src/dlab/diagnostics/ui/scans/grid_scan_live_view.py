# src/dlab/diagnostics/ui/scans/grid_scan_live_view.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox,
    QPushButton, QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

try:
    from skimage.transform import rotate as _sk_rotate
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False
try:
    from scipy.ndimage import rotate as _sp_rotate
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from dlab.core.device_registry import REGISTRY

def _wp_index_from_stage_key(stage_key: str) -> int | None:
    try:
        if not stage_key.startswith("stage:"):
            return None
        i = int(stage_key.split(":")[1])
        from dlab.hardware.wrappers.waveplate_calib import NUM_WAVEPLATES
        return i if 1 <= i <= NUM_WAVEPLATES else None
    except Exception:
        return None

def _reg_key_powermode(wp_index: int) -> str:
    return f"waveplate:powermode:{wp_index}"

def _reg_key_calib(wp_index: int) -> str:
    return f"waveplate:calib:{wp_index}"


class AndorGridScanLiveView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Andor Grid Live View")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlag(Qt.Window, True)
        self.resize(1100, 860)
        self.axes: List[Tuple[str, List[float]]] = []
        self.axis_names: List[str] = []
        self.axis_lengths: List[int] = []
        self.axis_index: Dict[str, int] = {}
        self.axis_positions: List[List[float]] = []
        self.axis_units: List[str] = []
        self.cameras: List[str] = []
        self._first_frame_shape: Optional[Tuple[int, int]] = None
        self.panels: List[_Panel] = []
        self._build_ui()

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        cfg = QGroupBox("Context")
        c = QHBoxLayout(cfg)
        self.ctx_label = QLabel("Configure panels below, then start the scan.")
        c.addWidget(self.ctx_label)
        c.addStretch(1)
        self.reset_btn = QPushButton("Reset all panels")
        self.reset_btn.clicked.connect(self.reset_all)
        c.addWidget(self.reset_btn)
        main.addWidget(cfg)
        pre = QGroupBox("Preprocess (applied before metrics)")
        pl = QHBoxLayout(pre)
        self.pre_enable = QCheckBox("Enable"); self.pre_enable.setChecked(False)
        self.angle_sb = QDoubleSpinBox(); self.angle_sb.setRange(-360.0, 360.0); self.angle_sb.setDecimals(2); self.angle_sb.setValue(-95.0)
        self.angle_sb.setSingleStep(1.0)
        self.x0_sb = QSpinBox(); self.x1_sb = QSpinBox(); self.y0_sb = QSpinBox(); self.y1_sb = QSpinBox()
        for sb in (self.x0_sb, self.x1_sb, self.y0_sb, self.y1_sb):
            sb.setRange(0, 99999)
        self.x0_sb.setValue(165); self.x1_sb.setValue(490)
        self.y0_sb.setValue(210); self.y1_sb.setValue(340)
        pl.addWidget(self.pre_enable)
        pl.addWidget(QLabel("Angle (°)")); pl.addWidget(self.angle_sb)
        pl.addSpacing(16)
        pl.addWidget(QLabel("Crop x0")); pl.addWidget(self.x0_sb)
        pl.addWidget(QLabel("x1")); pl.addWidget(self.x1_sb)
        pl.addWidget(QLabel("y0")); pl.addWidget(self.y0_sb)
        pl.addWidget(QLabel("y1")); pl.addWidget(self.y1_sb)
        pl.addStretch(1)
        main.addWidget(pre)
        grid = QGridLayout()
        main.addLayout(grid, 1)
        for i in range(4):
            p = _Panel()
            self.panels.append(p)
            r, col = divmod(i, 2)
            grid.addWidget(p, r, col)
        foot = QHBoxLayout()
        self.note = QLabel("Tip: set Y source to a stage to see real values; 'time' shows scan order.")
        foot.addWidget(self.note)
        foot.addStretch(1)
        main.addLayout(foot)

    def preconfigure(self, axes: List[Tuple[str, List[float]]], andor_camera_keys: List[str]) -> None:
        self.axes = axes
        self.axis_names = [ax for ax, _ in axes]
        self.axis_lengths = [len(pos) for _, pos in axes]
        self.axis_index = {name: i for i, name in enumerate(self.axis_names)}
        self.axis_positions = [list(pos) for _, pos in axes]
        self.axis_units = [("°" if _wp_index_from_stage_key(name) is not None else "mm") for name, _ in axes]
        self.cameras = list(andor_camera_keys)
        for idx, p in enumerate(self.panels):
            p.set_choices(cameras=self.cameras, axis_names=self.axis_names)
            if idx == 0:
                p.metric_combo.setCurrentText("Total sum vs Y")
            elif idx == 1:
                p.metric_combo.setCurrentText("Sum by columns")
            elif idx == 2:
                p.metric_combo.setCurrentText("Sum by rows")
            else:
                p.metric_combo.setCurrentText("Total sum map (axes)")
            if self.axis_names:
                p.y_source_combo.setCurrentText(self.axis_names[0])
            else:
                p.y_source_combo.setCurrentText("time")
            if self.cameras:
                p.camera_combo.setCurrentText(self.cameras[0])

    def set_context(self, axes: List[Tuple[str, List[float]]], andor_camera_keys: List[str], preserve: bool = True) -> None:
        prev = []
        if preserve:
            for p in self.panels:
                prev.append((p.enabled_cb.isChecked(), p.metric_combo.currentText(),
                             p.camera_combo.currentText(), p.y_source_combo.currentText(),
                             getattr(p, "x_source_combo", QComboBox()).currentText() if hasattr(p, "x_source_combo") else ""))
        self.axes = axes
        self.axis_names = [ax for ax, _ in axes]
        self.axis_lengths = [len(pos) for _, pos in axes]
        self.axis_index = {name: i for i, name in enumerate(self.axis_names)}
        self.axis_positions = [list(pos) for _, pos in axes]
        self.axis_units = [("°" if _wp_index_from_stage_key(name) is not None else "mm") for name, _ in axes]
        self.cameras = list(andor_camera_keys)
        for i, p in enumerate(self.panels):
            p.set_choices(cameras=self.cameras, axis_names=self.axis_names)
            if preserve and i < len(prev):
                en, metric, cam, ysrc, xsrc = prev[i]
                p.enabled_cb.setChecked(en)
                if metric in p._metrics():
                    p.metric_combo.setCurrentText(metric)
                if cam in self.cameras:
                    p.camera_combo.setCurrentText(cam)
                if ysrc in ["time"] + self.axis_names:
                    p.y_source_combo.setCurrentText(ysrc)
                if xsrc in self.axis_names:
                    p.x_source_combo.setCurrentText(xsrc)
            p.realloc(y_len=self._len_for_source(p.y_source_combo.currentText()))
        self._update_ctx_label()

    def prepare_for_run(self) -> None:
        for p in self.panels:
            y_len = self._len_for_source(p.y_source_combo.currentText())
            p.realloc(y_len=y_len)

    def reset_all(self) -> None:
        for p in self.panels:
            p.clear_data()

    def _display_value_for_source(self, src: str, idxs: List[int]) -> tuple[float, str, str]:
        if src == "time" or src not in self.axis_index:
            return float(self._index_for_source("time", idxs)), "", "time"
        j = self.axis_index[src]
        if j >= len(self.axis_positions) or j >= len(idxs):
            return float(idxs[j] if j < len(idxs) else 0.0), "", src
        raw = float(self.axis_positions[j][idxs[j]])
        wp = _wp_index_from_stage_key(src)
        if wp is not None:
            pm = bool(REGISTRY.get(_reg_key_powermode(wp)) or False)
            if pm:
                return raw, "frac", f"{src} (power)"
            else:
                return raw, "°", f"{src} (angle)"
        return raw, "mm", src

    def on_andor_frame(self, axis_indices: List[int], frame_u16: np.ndarray, camera_key: str) -> None:
        img = frame_u16
        if self._first_frame_shape is None:
            self._first_frame_shape = img.shape
            H, W = self._first_frame_shape
            for sb, maxv in ((self.x0_sb, W-1), (self.x1_sb, W), (self.y0_sb, H-1), (self.y1_sb, H)):
                sb.setMaximum(max(0, int(maxv)))
            if self.x1_sb.value() <= self.x0_sb.value():
                self.x1_sb.setValue(min(W, self.x0_sb.value() + 1))
            if self.y1_sb.value() <= self.y0_sb.value():
                self.y1_sb.setValue(min(H, self.y0_sb.value() + 1))
        if self.pre_enable.isChecked():
            img = self._preprocess(img)
        for p in self.panels:
            if not p.enabled_cb.isChecked():
                continue
            if camera_key != p.camera_combo.currentText():
                continue
            metric = p.metric_combo.currentText()
            if metric == "Total sum vs Y":
                yi = self._index_for_source(p.y_source_combo.currentText(), axis_indices)
                yval, yunit, ylabel = self._display_value_for_source(p.y_source_combo.currentText(), axis_indices)
                val = float(np.sum(img, dtype=np.float64))
                p.update_line_value(yi, yval, val, y_label=f"{ylabel} [{yunit}]" if yunit else ylabel)
            elif metric == "Sum by columns":
                yi = self._index_for_source(p.y_source_combo.currentText(), axis_indices)
                yval, yunit, ylabel = self._display_value_for_source(p.y_source_combo.currentText(), axis_indices)
                prof = np.sum(img, axis=0, dtype=np.float64)
                p.update_heatmap_row_value(yi, yval, prof, x_len=prof.shape[0],
                                           y_label=f"{ylabel} [{yunit}]" if yunit else ylabel)
                p.ax.set_xlabel("pixel (column)")
            elif metric == "Sum by rows":
                yi = self._index_for_source(p.y_source_combo.currentText(), axis_indices)
                yval, yunit, ylabel = self._display_value_for_source(p.y_source_combo.currentText(), axis_indices)
                prof = np.sum(img, axis=1, dtype=np.float64)
                p.update_heatmap_row_value(yi, yval, prof, x_len=prof.shape[0],
                                           y_label=f"{ylabel} [{yunit}]" if yunit else ylabel)
                p.ax.set_xlabel("pixel (row)")
            else:
                yname = p.y_source_combo.currentText()
                xname = p.x_source_combo.currentText()
                if (xname not in self.axis_index) or (yname not in self.axis_index) or (xname == yname):
                    continue
                yi = self._index_for_source(yname, axis_indices)
                xi = self._index_for_source(xname, axis_indices)
                y_len = self.axis_lengths[self.axis_index[yname]]
                x_len = self.axis_lengths[self.axis_index[xname]]
                yval, yunit, ylabel = self._display_value_for_source(yname, axis_indices)
                xval, xunit, xlabel = self._display_value_for_source(xname, axis_indices)
                val = float(np.sum(img, dtype=np.float64))
                p.update_stage_map_value(yi, xi, x_len=x_len, y_len=y_len, value=val,
                                         x_val=xval, y_val=yval,
                                         x_label=f"{xlabel} [{xunit}]" if xunit else xlabel,
                                         y_label=f"{ylabel} [{yunit}]" if yunit else ylabel)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        out = image
        angle = float(self.angle_sb.value())
        if abs(angle) > 1e-9:
            if _HAS_SKIMAGE:
                out = _sk_rotate(out, angle, resize=True, order=3, mode="edge", preserve_range=True).astype(image.dtype)
            elif _HAS_SCIPY:
                out = _sp_rotate(out, angle, reshape=True, order=3, mode="nearest").astype(image.dtype)
        y0, y1 = int(self.y0_sb.value()), int(self.y1_sb.value())
        x0, x1 = int(self.x0_sb.value()), int(self.x1_sb.value())
        h, w = out.shape
        y0 = max(0, min(y0, h-1)); y1 = max(y0+1, min(y1, h))
        x0 = max(0, min(x0, w-1)); x1 = max(x0+1, min(x1, w))
        out = out[y0:y1, x0:x1]
        return out

    def _update_ctx_label(self) -> None:
        axes_txt = ", ".join(self.axis_names) if self.axis_names else "(no axes)"
        cams_txt = ", ".join(self.cameras) if self.cameras else "(no Andor cameras)"
        self.ctx_label.setText(f"Axes: {axes_txt}    |    Andor cameras: {cams_txt}")

    def _len_for_source(self, src: str) -> int:
        if src == "time":
            if not self.axis_lengths:
                return 0
            prod = 1
            for L in self.axis_lengths:
                prod *= max(1, L)
            return int(prod)
        if src in self.axis_index:
            return int(self.axis_lengths[self.axis_index[src]])
        return 0

    def _index_for_source(self, src: str, idxs: List[int]) -> int:
        if src == "time":
            s = 0
            for i, L in enumerate(self.axis_lengths):
                s = s * L + idxs[i]
            return int(s)
        j = self.axis_index.get(src)
        if j is None or j >= len(idxs):
            return 0
        return int(idxs[j])


class _Panel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        ctrl = QGroupBox()
        hc = QHBoxLayout(ctrl)
        self.enabled_cb = QCheckBox("Enable"); self.enabled_cb.setChecked(True)
        self.metric_combo = QComboBox(); self.metric_combo.addItems(self._metrics())
        self.camera_combo = QComboBox()
        self.y_source_combo = QComboBox()
        self.x_source_label = QLabel("X axis")
        self.x_source_combo = QComboBox()
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.clear_data)
        hc.addWidget(self.enabled_cb)
        hc.addWidget(QLabel("Metric")); hc.addWidget(self.metric_combo)
        hc.addWidget(QLabel("Camera")); hc.addWidget(self.camera_combo, 1)
        hc.addWidget(QLabel("Y source")); hc.addWidget(self.y_source_combo)
        hc.addWidget(self.x_source_label); hc.addWidget(self.x_source_combo)
        hc.addStretch(1)
        hc.addWidget(self.reset_btn)
        self.x_source_label.setVisible(False)
        self.x_source_combo.setVisible(False)
        self.fig = plt.figure(figsize=(4.8, 3.2))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        lay = QVBoxLayout(self)
        lay.addWidget(ctrl)
        lay.addWidget(self.canvas, 1)
        self._y_len: int = 0
        self._is_line: bool = True
        self._im = None
        self._line = None
        self._mat: Optional[np.ndarray] = None
        self._vec: Optional[np.ndarray] = None
        self._y_vals: Optional[np.ndarray] = None
        self._x_vals_for_map: Optional[np.ndarray] = None
        self._y_label_txt: str = "Y"
        self._x_label_txt: str = "X"
        self.metric_combo.currentTextChanged.connect(self._on_metric_changed)

    def set_choices(self, cameras: List[str], axis_names: List[str]) -> None:
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear(); self.camera_combo.addItems(cameras)
        self.camera_combo.blockSignals(False)
        self.y_source_combo.blockSignals(True)
        self.y_source_combo.clear(); self.y_source_combo.addItems(["time"] + axis_names)
        self.y_source_combo.blockSignals(False)
        self.x_source_combo.blockSignals(True)
        self.x_source_combo.clear(); self.x_source_combo.addItems(axis_names)
        self.x_source_combo.blockSignals(False)

    def realloc(self, y_len: int) -> None:
        self._y_len = int(max(0, y_len))
        metric = self.metric_combo.currentText()
        self._is_line = (metric == "Total sum vs Y")
        if self._is_line:
            self._vec = np.full((self._y_len,), np.nan, dtype=float)
            self._y_vals = np.full((self._y_len,), np.nan, dtype=float)
            self._mat = None
            self._ensure_line_artist()
        else:
            self._mat = np.full((self._y_len, 0), np.nan, dtype=float)
            self._y_vals = np.full((self._y_len,), np.nan, dtype=float)
            self._vec = None
            self._ensure_heatmap_artist()
        self._redraw()

    def clear_data(self) -> None:
        if self._is_line:
            if self._vec is not None:
                self._vec[:] = np.nan
            if self._y_vals is not None:
                self._y_vals[:] = np.nan
        else:
            if self._mat is not None and self._mat.size:
                self._mat[:] = np.nan
            if self._y_vals is not None:
                self._y_vals[:] = np.nan
            if self._x_vals_for_map is not None and self._x_vals_for_map.size:
                self._x_vals_for_map[:] = np.nan
        self._redraw()

    def update_line_value(self, yi: int, y_value: float, value: float, y_label: str):
        if not self._is_line:
            self._switch_to_line()
        if yi >= (self._vec.shape[0] if self._vec is not None else 0):
            new_len = yi + 1
            tmpv = np.full((new_len,), np.nan, dtype=float)
            tmpy = np.full((new_len,), np.nan, dtype=float)
            if self._vec is not None:
                tmpv[: self._vec.shape[0]] = self._vec
            if self._y_vals is not None:
                tmpy[: self._y_vals.shape[0]] = self._y_vals
            self._vec, self._y_vals = tmpv, tmpy
        self._vec[yi] = value
        self._y_vals[yi] = float(y_value)
        self._y_label_txt = y_label or "Y"
        self._redraw()

    def update_heatmap_row_value(self, yi: int, y_value: float, row: np.ndarray, x_len: int, y_label: str):
        if self._is_line:
            self._switch_to_heatmap()
        if self._mat is None or self._mat.shape[0] != self._y_len:
            self._mat = np.full((self._y_len, x_len), np.nan, dtype=float)
            self._y_vals = np.full((self._y_len,), np.nan, dtype=float)
        if self._mat.shape[1] != x_len:
            M = np.full((self._mat.shape[0], x_len), np.nan, dtype=float)
            c = min(self._mat.shape[1], x_len)
            if c > 0:
                M[:, :c] = self._mat[:, :c]
            self._mat = M
        if 0 <= yi < self._mat.shape[0]:
            self._mat[yi, :] = row
            self._y_vals[yi] = float(y_value)
        self._y_label_txt = y_label or "Y"
        self._redraw()

    def update_stage_map_value(self, yi: int, xi: int, x_len: int, y_len: int, value: float,
                               x_val: float, y_val: float,
                               x_label: str, y_label: str):
        if self._is_line:
            self._switch_to_heatmap()
        need_new = (self._mat is None or self._mat.shape[0] != y_len or self._mat.shape[1] != x_len)
        if need_new:
            M = np.full((y_len, x_len), np.nan, dtype=float)
            if self._mat is not None and self._mat.size:
                oy, ox = self._mat.shape
                cy, cx = min(oy, y_len), min(ox, x_len)
                M[:cy, :cx] = self._mat[:cy, :cx]
            self._mat = M
            self._y_vals = np.full((y_len,), np.nan, dtype=float)
            self._x_vals_for_map = np.full((x_len,), np.nan, dtype=float)
        if 0 <= yi < self._mat.shape[0] and 0 <= xi < self._mat.shape[1]:
            self._mat[yi, xi] = value
            self._y_vals[yi] = float(y_val)
            if self._x_vals_for_map is not None:
                self._x_vals_for_map[xi] = float(x_val)
        self._x_label_txt = x_label or "X"
        self._y_label_txt = y_label or "Y"
        self._redraw()

    def _metrics(self) -> List[str]:
        return ["Total sum vs Y", "Sum by columns", "Sum by rows", "Total sum map (axes)"]

    def _on_metric_changed(self, _m: str) -> None:
        m = self.metric_combo.currentText()
        is_stage_map = (m == "Total sum map (axes)")
        self.x_source_label.setVisible(is_stage_map)
        self.x_source_combo.setVisible(is_stage_map)
        self.realloc(self._y_len)

    def _switch_to_line(self) -> None:
        self._is_line = True
        self._vec = np.full((self._y_len,), np.nan, dtype=float)
        self._y_vals = np.full((self._y_len,), np.nan, dtype=float)
        self._mat = None
        self._ensure_line_artist()

    def _switch_to_heatmap(self) -> None:
        self._is_line = False
        self._mat = np.full((self._y_len, 0), np.nan, dtype=float)
        self._y_vals = np.full((self._y_len,), np.nan, dtype=float)
        self._vec = None
        self._ensure_heatmap_artist()

    def _ensure_line_artist(self) -> None:
        self.ax.cla()
        self._line = self.ax.plot(
            np.arange(max(1, self._y_len)),
            np.full((max(1, self._y_len),), np.nan, dtype=float)
        )[0]
        self.ax.set_xlabel("Y value")
        self.ax.set_ylabel("Total sum")
        self.ax.grid(True, alpha=0.3)
        self._im = None
        self.canvas.draw_idle()

    def _ensure_heatmap_artist(self) -> None:
        self.ax.cla()
        mat = np.zeros((max(1, self._y_len), 1), dtype=float)
        self._im = self.ax.imshow(mat, origin="lower", aspect="auto", cmap='turbo')
        self.ax.set_xlabel("pixel")
        self.ax.set_ylabel("Y value")
        self._line = None
        self.canvas.draw_idle()

    def _redraw(self) -> None:
        if self._is_line:
            if self._vec is None:
                return
            y = self._vec
            if self._line is None:
                self._ensure_line_artist()
            x = np.arange(y.shape[0]) if self._y_vals is None else self._y_vals
            self._line.set_xdata(x)
            self._line.set_ydata(y)
            self.ax.set_xlabel(self._y_label_txt)
            self.ax.set_ylabel("Total sum")
            self.ax.relim()
            self.ax.autoscale_view()
        else:
            if self._mat is None:
                return
            M = self._mat
            if self._im is None or self._im.get_array().shape != M.shape:
                self._ensure_heatmap_artist()
                self._im.set_data(M)
            else:
                self._im.set_data(M)
            finite = np.isfinite(M)
            if finite.any():
                vmin = float(np.nanmin(M)); vmax = float(np.nanmax(M))
                if vmin == vmax:
                    vmax = vmin + 1.0
                self._im.set_clim(vmin, vmax)
            self.ax.set_ylabel(self._y_label_txt)
            if self._y_vals is not None and self._y_vals.size:
                n = self._y_vals.size
                idxs = np.linspace(0, n-1, num=min(8, n), dtype=int)
                self.ax.set_yticks(idxs)
                self.ax.set_yticklabels([f"{self._y_vals[i]:.3g}" if np.isfinite(self._y_vals[i]) else "" for i in idxs])
            if self._x_vals_for_map is not None and self._x_vals_for_map.size:
                m = self._x_vals_for_map.size
                idxs = np.linspace(0, m-1, num=min(8, m), dtype=int)
                self.ax.set_xticks(idxs)
                self.ax.set_xticklabels([f"{self._x_vals_for_map[i]:.3g}" if np.isfinite(self._x_vals_for_map[i]) else "" for i in idxs])
                self.ax.set_xlabel(self._x_label_txt)
        self.canvas.draw_idle()


class AvaspecGridScanLiveView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Avaspec Grid Live View")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlag(Qt.Window, True)
        self.resize(1100, 860)
        self.axes: List[Tuple[str, List[float]]] = []
        self.axis_names: List[str] = []
        self.axis_lengths: List[int] = []
        self.axis_index: Dict[str, int] = {}
        self.axis_positions: List[List[float]] = []
        self._lin_counter: int = 0
        self._y_source: str = "time"
        self._y_len: int = 0
        self._x_wl: Optional[np.ndarray] = None
        self._sum_vec: Optional[np.ndarray] = None
        self._sum_yvals: Optional[np.ndarray] = None
        self._spec_mat: Optional[np.ndarray] = None
        self._spec_yvals: Optional[np.ndarray] = None
        self._build_ui()
        try:
            REGISTRY.register("ui:avaspec_live", self)
        except Exception:
            pass

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        ctx = QGroupBox("Context")
        cl = QHBoxLayout(ctx)
        self.y_combo = QComboBox()
        self.y_combo.currentTextChanged.connect(self._on_y_source_changed)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_all)
        cl.addWidget(QLabel("Y source"))
        cl.addWidget(self.y_combo)
        cl.addStretch(1)
        cl.addWidget(self.reset_btn)
        main.addWidget(ctx)
        top = QGroupBox("Total counts vs stage")
        tl = QVBoxLayout(top)
        self.fig_sum = plt.figure(figsize=(5.4, 3.2))
        self.ax_sum = self.fig_sum.add_subplot(111)
        self.ax_sum.set_xlabel("Stage value")
        self.ax_sum.set_ylabel("Total counts")
        self.ax_sum.grid(True, alpha=0.3)
        self.canvas_sum = FigureCanvas(self.fig_sum)
        tl.addWidget(self.canvas_sum)
        main.addWidget(top, 1)
        bot = QGroupBox("Spectrum (λ) vs stage")
        bl = QVBoxLayout(bot)
        self.fig_map = plt.figure(figsize=(5.4, 3.2))
        self.ax_map = self.fig_map.add_subplot(111)
        self.im_map = self.ax_map.imshow(np.zeros((1, 1)), origin="lower", aspect="auto", cmap='turbo')
        self.ax_map.set_xlabel("Wavelength (nm)")
        self.ax_map.set_ylabel("Stage value")
        self.canvas_map = FigureCanvas(self.fig_map)
        bl.addWidget(self.canvas_map)
        main.addWidget(bot, 2)

    def preconfigure(self, axes: List[Tuple[str, List[float]]]) -> None:
        self.axes = axes
        self.axis_names = [ax for ax, _ in axes]
        self.axis_lengths = [len(pos) for _, pos in axes]
        self.axis_index = {name: i for i, name in enumerate(self.axis_names)}
        self.axis_positions = [list(pos) for _, pos in axes]
        self.y_combo.blockSignals(True)
        self.y_combo.clear()
        self.y_combo.addItems(["time"] + self.axis_names)
        self.y_combo.blockSignals(False)
        self._y_source = self.axis_names[0] if self.axis_names else "time"
        self.y_combo.setCurrentText(self._y_source)
        self.prepare_for_run()

    def set_context(self, axes: List[Tuple[str, List[float]]], preserve: bool = True) -> None:
        prev_y = self._y_source if preserve else None
        self.preconfigure(axes)
        if preserve and prev_y in (["time"] + self.axis_names):
            self._y_source = prev_y
            self.y_combo.setCurrentText(prev_y)

    def prepare_for_run(self) -> None:
        self._lin_counter = 0
        self._x_wl = None
        self._y_len = self._compute_y_len(self._y_source)
        self._sum_vec = np.full((self._y_len,), np.nan, dtype=float)
        self._sum_yvals = np.full((self._y_len,), np.nan, dtype=float)
        self._spec_mat = None
        self._spec_yvals = np.full((self._y_len,), np.nan, dtype=float)
        self._redraw_all()

    def reset_all(self) -> None:
        self.prepare_for_run()

    def _compute_y_len(self, src: str) -> int:
        if src == "time" or src not in self.axis_index:
            prod = 1
            for L in self.axis_lengths:
                prod *= max(1, L)
            return int(prod)
        return int(self.axis_lengths[self.axis_index[src]])

    def _lin_to_indices(self, k: int) -> List[int]:
        idxs = []
        rem = int(k)
        for L in self.axis_lengths[::-1]:
            idxs.append(rem % L)
            rem //= L
        return list(reversed(idxs))

    def _y_value_from_indices(self, src: str, idxs: List[int]) -> float:
        if src == "time" or src not in self.axis_index:
            return float(self._index_for_time(idxs))
        j = self.axis_index[src]
        if j >= len(self.axis_positions) or j >= len(idxs):
            return float(idxs[j] if j < len(idxs) else 0.0)
        return float(self.axis_positions[j][idxs[j]])

    def _index_for_time(self, idxs: List[int]) -> int:
        s = 0
        for i, L in enumerate(self.axis_lengths):
            s = s * L + idxs[i]
        return int(s)

    def _on_y_source_changed(self, s: str) -> None:
        self._y_source = s
        self.prepare_for_run()

    def set_spectrum_from_scan(self, wl, counts):
        wl = np.asarray(wl, dtype=float).ravel()
        y = np.asarray(counts, dtype=float).ravel()
        if self._x_wl is None or self._x_wl.shape != wl.shape or not np.allclose(self._x_wl, wl):
            self._x_wl = wl.copy()
            if self._spec_mat is None or self._spec_mat.shape[1] != wl.size:
                self._spec_mat = np.full((self._y_len, wl.size), np.nan, dtype=float)
        idxs = self._lin_to_indices(self._lin_counter)
        yi = self._index_for_time(idxs) if (self._y_source == "time" or self._y_source not in self.axis_index) else idxs[self.axis_index[self._y_source]]
        if self._sum_vec is None or yi >= self._sum_vec.size:
            new_len = yi + 1
            sv = np.full((new_len,), np.nan, dtype=float)
            sy = np.full((new_len,), np.nan, dtype=float)
            if self._sum_vec is not None:
                sv[: self._sum_vec.size] = self._sum_vec
                sy[: self._sum_yvals.size] = self._sum_yvals
            self._sum_vec = sv
            self._sum_yvals = sy
        if self._spec_mat is None or yi >= self._spec_mat.shape[0]:
            rows = yi + 1
            cols = self._x_wl.size if self._x_wl is not None else y.size
            M = np.full((rows, cols), np.nan, dtype=float)
            if self._spec_mat is not None:
                rr, cc = self._spec_mat.shape
                M[:rr, :cc] = self._spec_mat
            self._spec_mat = M
            if self._spec_yvals is None or self._spec_yvals.size < rows:
                vy = np.full((rows,), np.nan, dtype=float)
                if self._spec_yvals is not None:
                    vy[: self._spec_yvals.size] = self._spec_yvals
                self._spec_yvals = vy
        self._sum_vec[yi] = float(np.nansum(y, dtype=np.float64))
        self._sum_yvals[yi] = self._y_value_from_indices(self._y_source, idxs)
        self._spec_mat[yi, : y.size] = y
        self._spec_yvals[yi] = self._sum_yvals[yi]
        self._lin_counter += 1
        self._redraw_all()

    def _redraw_all(self) -> None:
        self.ax_sum.cla()
        if self._sum_vec is not None and self._sum_yvals is not None:
            x = self._sum_yvals
            y = self._sum_vec
            self.ax_sum.plot(x, y)
            self.ax_sum.set_xlabel(self._y_source)
            self.ax_sum.set_ylabel("Total counts")
            self.ax_sum.grid(True, alpha=0.3)
        self.canvas_sum.draw_idle()
        self.ax_map.cla()
        if self._spec_mat is not None and self._x_wl is not None:
            self.im_map = self.ax_map.imshow(self._spec_mat, origin="lower", aspect="auto", cmap='turbo')
            self.ax_map.set_xlabel("Wavelength (nm)")
            self.ax_map.set_ylabel(self._y_source)
            if self._spec_yvals is not None and self._spec_yvals.size:
                n = self._spec_yvals.size
                idxs = np.linspace(0, n-1, num=min(8, n), dtype=int)
                self.ax_map.set_yticks(idxs)
                self.ax_map.set_yticklabels([f"{self._spec_yvals[i]:.3g}" if np.isfinite(self._spec_yvals[i]) else "" for i in idxs])
            if self._x_wl is not None and self._x_wl.size:
                m = self._x_wl.size
                idxs = np.linspace(0, m-1, num=min(8, m), dtype=int)
                self.ax_map.set_xticks(idxs)
                self.ax_map.set_xticklabels([f"{self._x_wl[i]:.3g}" if np.isfinite(self._x_wl[i]) else "" for i in idxs])
        self.canvas_map.draw_idle()
