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
import cmasher as cmr 
from scipy.ndimage import rotate as _sp_rotate


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
        self._im = self.ax.imshow(mat, origin="lower", aspect="auto", cmap='cmr.rainforest')
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
