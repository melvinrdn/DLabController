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

# optional rotators
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


class AndorGridScanLiveView(QWidget):
    """
    Compact 2x2 live dashboard for ANDOR grid scans + lightweight preprocessing (rotate/crop).

    Public API:
      - preconfigure(axes, andor_camera_keys)
      - set_context(axes, andor_camera_keys, preserve=True)
      - prepare_for_run()
      - on_andor_frame(indices, frame_u16, camera_key)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Andor Grid Live View")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlag(Qt.Window, True)
        self.resize(1100, 860)

        # Scan axes spec
        self.axes: List[Tuple[str, List[float]]] = []
        self.axis_names: List[str] = []
        self.axis_lengths: List[int] = []
        self.axis_index: Dict[str, int] = {}

        # Cameras (Andor only)
        self.cameras: List[str] = []

        # Preprocess state
        self._first_frame_shape: Optional[Tuple[int, int]] = None

        # Panels: fixed 2x2
        self.panels: List[_Panel] = []

        self._build_ui()

    # ----------- UI -----------
    def _build_ui(self) -> None:
        main = QVBoxLayout(self)

        # ---- Context bar
        cfg = QGroupBox("Context")
        c = QHBoxLayout(cfg)
        self.ctx_label = QLabel("Configure panels below, then start the scan.")
        c.addWidget(self.ctx_label)
        c.addStretch(1)
        self.reset_btn = QPushButton("Reset all panels")
        self.reset_btn.clicked.connect(self.reset_all)
        c.addWidget(self.reset_btn)
        main.addWidget(cfg)

        # ---- Preprocess controls (rotation + crop)
        pre = QGroupBox("Preprocess (applied before metrics)")
        pl = QHBoxLayout(pre)
        self.pre_enable = QCheckBox("Enable"); self.pre_enable.setChecked(False)

        self.angle_sb = QDoubleSpinBox(); self.angle_sb.setRange(-360.0, 360.0); self.angle_sb.setDecimals(2); self.angle_sb.setValue(-95.0)
        self.angle_sb.setSingleStep(1.0)

        # crop spinboxes; ranges will be set after first frame
        self.x0_sb = QSpinBox(); self.x1_sb = QSpinBox(); self.y0_sb = QSpinBox(); self.y1_sb = QSpinBox()
        for sb in (self.x0_sb, self.x1_sb, self.y0_sb, self.y1_sb):
            sb.setRange(0, 99999)
        # defaults like the example
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

        # ---- 2x2 grid of panels
        grid = QGridLayout()
        main.addLayout(grid, 1)

        for i in range(4):
            p = _Panel()
            self.panels.append(p)
            r, col = divmod(i, 2)
            grid.addWidget(p, r, col)

        # small footer
        foot = QHBoxLayout()
        self.note = QLabel("Tip: set Y source to 'time' for scan order along Y.")
        foot.addWidget(self.note)
        foot.addStretch(1)
        main.addLayout(foot)

    # ----------- Public API -----------
    def preconfigure(self, axes: List[Tuple[str, List[float]]], andor_camera_keys: List[str]) -> None:
        self.axes = axes
        self.axis_names = [ax for ax, _ in axes]
        self.axis_lengths = [len(pos) for _, pos in axes]
        self.axis_index = {name: i for i, name in enumerate(self.axis_names)}
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

    def on_andor_frame(self, axis_indices: List[int], frame_u16: np.ndarray, camera_key: str) -> None:
        """Called by GridScanWorker for each Andor frame (uint16)."""
        img = frame_u16

        # remember first frame shape to clamp crop ranges
        if self._first_frame_shape is None:
            self._first_frame_shape = img.shape  # (H, W)
            H, W = self._first_frame_shape
            # set sensible limits
            for sb, maxv in ((self.x0_sb, W-1), (self.x1_sb, W), (self.y0_sb, H-1), (self.y1_sb, H)):
                sb.setMaximum(max(0, int(maxv)))
            # ensure x1>x0, y1>y0
            if self.x1_sb.value() <= self.x0_sb.value():
                self.x1_sb.setValue(min(W, self.x0_sb.value() + 1))
            if self.y1_sb.value() <= self.y0_sb.value():
                self.y1_sb.setValue(min(H, self.y0_sb.value() + 1))

        # --- preprocess (rotate + crop) if enabled
        if self.pre_enable.isChecked():
            img = self._preprocess(img)

        # Dispatch to active panels (using preprocessed image)
        for p in self.panels:
            if not p.enabled_cb.isChecked():
                continue
            if camera_key != p.camera_combo.currentText():
                continue

            metric = p.metric_combo.currentText()

            if metric == "Total sum vs Y":
                yi = self._index_for_source(p.y_source_combo.currentText(), axis_indices)
                val = float(np.sum(img, dtype=np.float64))
                p.update_line(yi, val)

            elif metric == "Sum by columns":
                yi = self._index_for_source(p.y_source_combo.currentText(), axis_indices)
                prof = np.sum(img, axis=0, dtype=np.float64)
                p.update_heatmap_row(yi, prof, x_len=prof.shape[0])
                p.ax.set_xlabel("pixel (column)")
                p.ax.set_ylabel(f"{p.y_source_combo.currentText()} index")

            elif metric == "Sum by rows":
                yi = self._index_for_source(p.y_source_combo.currentText(), axis_indices)
                prof = np.sum(img, axis=1, dtype=np.float64)
                p.update_heatmap_row(yi, prof, x_len=prof.shape[0])
                p.ax.set_xlabel("pixel (row)")
                p.ax.set_ylabel(f"{p.y_source_combo.currentText()} index")

            else:  # "Total sum map (axes)"
                yname = p.y_source_combo.currentText()
                xname = p.x_source_combo.currentText()
                if (xname not in self.axis_index) or (yname not in self.axis_index) or (xname == yname):
                    continue
                yi = self._index_for_source(yname, axis_indices)
                xi = self._index_for_source(xname, axis_indices)
                y_len = self.axis_lengths[self.axis_index[yname]]
                x_len = self.axis_lengths[self.axis_index[xname]]
                val = float(np.sum(img, dtype=np.float64))
                p.update_stage_map(yi, xi, x_len=x_len, y_len=y_len, value=val)
                p.ax.set_xlabel(f"{xname} index")
                p.ax.set_ylabel(f"{yname} index")

    # ----------- Preprocess helpers -----------
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Rotate then crop; keeps dtype and range."""
        out = image
        # rotate (resize=True to keep all content)
        angle = float(self.angle_sb.value())
        if abs(angle) > 1e-9:
            if _HAS_SKIMAGE:
                out = _sk_rotate(out, angle, resize=True, order=3, mode="edge", preserve_range=True).astype(image.dtype)
            elif _HAS_SCIPY:
                # scipy rotates CCW degrees; reshape=True ~ resize
                out = _sp_rotate(out, angle, reshape=True, order=3, mode="nearest").astype(image.dtype)
            else:
                # no deps: skip rotation
                pass

        # crop (y, x)
        y0, y1 = int(self.y0_sb.value()), int(self.y1_sb.value())
        x0, x1 = int(self.x0_sb.value()), int(self.x1_sb.value())
        h, w = out.shape
        y0 = max(0, min(y0, h-1)); y1 = max(y0+1, min(y1, h))
        x0 = max(0, min(x0, w-1)); x1 = max(x0+1, min(x1, w))
        out = out[y0:y1, x0:x1]
        return out

    # ----------- Internals -----------
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
        """Map 'time' or a stage axis name to an index for this sample."""
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
    """
    One dashboard cell. Supports:
      - Total sum vs Y (1D line)
      - Sum by columns (2D heatmap)
      - Sum by rows (2D heatmap)
      - Total sum map (axes) (2D heatmap over two stage axes)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)

        # Controls
        ctrl = QGroupBox()
        hc = QHBoxLayout(ctrl)
        self.enabled_cb = QCheckBox("Enable"); self.enabled_cb.setChecked(True)
        self.metric_combo = QComboBox(); self.metric_combo.addItems(self._metrics())
        self.camera_combo = QComboBox()
        self.y_source_combo = QComboBox()  # "time" + axis names
        self.x_source_label = QLabel("X axis")
        self.x_source_combo = QComboBox()  # axis names only (for 2D stage map)
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

        # Figure
        self.fig = plt.figure(figsize=(4.8, 3.2))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        lay = QVBoxLayout(self)
        lay.addWidget(ctrl)
        lay.addWidget(self.canvas, 1)

        # Data containers
        self._y_len: int = 0
        self._is_line: bool = True
        self._im = None
        self._line = None
        self._mat: Optional[np.ndarray] = None  # (Y, X) for heatmaps
        self._vec: Optional[np.ndarray] = None  # (Y,) for lines

        # React to metric changes
        self.metric_combo.currentTextChanged.connect(self._on_metric_changed)

    # ---- public-ish API used by parent ----
    def set_choices(self, cameras: List[str], axis_names: List[str]) -> None:
        # cameras
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear(); self.camera_combo.addItems(cameras)
        self.camera_combo.blockSignals(False)

        # Y source supports "time" or any axis
        self.y_source_combo.blockSignals(True)
        self.y_source_combo.clear(); self.y_source_combo.addItems(["time"] + axis_names)
        self.y_source_combo.blockSignals(False)

        # X axis (for 2D stage map) = axis names only
        self.x_source_combo.blockSignals(True)
        self.x_source_combo.clear(); self.x_source_combo.addItems(axis_names)
        self.x_source_combo.blockSignals(False)

    def realloc(self, y_len: int) -> None:
        self._y_len = int(max(0, y_len))
        metric = self.metric_combo.currentText()
        self._is_line = (metric == "Total sum vs Y")
        # allocate blank containers
        if self._is_line:
            self._vec = np.full((self._y_len,), np.nan, dtype=float)
            self._mat = None
            self._ensure_line_artist()
        else:
            self._mat = np.full((self._y_len, 0), np.nan, dtype=float)  # will set width on first row
            self._vec = None
            self._ensure_heatmap_artist()

        self._redraw()

    def clear_data(self) -> None:
        if self._is_line:
            if self._vec is not None:
                self._vec[:] = np.nan
        else:
            if self._mat is not None and self._mat.size:
                self._mat[:] = np.nan
        self._redraw()

    def update_line(self, yi: int, value: float) -> None:
        if not self._is_line:
            self._switch_to_line()
        if yi >= (self._vec.shape[0] if self._vec is not None else 0):
            new_len = yi + 1
            tmp = np.full((new_len,), np.nan, dtype=float)
            if self._vec is not None:
                tmp[: self._vec.shape[0]] = self._vec
            self._vec = tmp
        self._vec[yi] = value
        self._redraw()

    def update_heatmap_row(self, yi: int, row: np.ndarray, x_len: int) -> None:
        if self._is_line:
            self._switch_to_heatmap()
        if self._mat is None or self._mat.shape[0] != self._y_len:
            self._mat = np.full((self._y_len, x_len), np.nan, dtype=float)
        if self._mat.shape[1] != x_len:
            M = np.full((self._mat.shape[0], x_len), np.nan, dtype=float)
            c = min(self._mat.shape[1], x_len)
            if c > 0:
                M[:, :c] = self._mat[:, :c]
            self._mat = M
        if 0 <= yi < self._mat.shape[0]:
            self._mat[yi, :] = row
        self._redraw()

    def update_stage_map(self, yi: int, xi: int, x_len: int, y_len: int, value: float) -> None:
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
        if 0 <= yi < self._mat.shape[0] and 0 <= xi < self._mat.shape[1]:
            self._mat[yi, xi] = value
        self._redraw()

    # ---- internals ----
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
        self._mat = None
        self._ensure_line_artist()

    def _switch_to_heatmap(self) -> None:
        self._is_line = False
        self._mat = np.full((self._y_len, 0), np.nan, dtype=float)
        self._vec = None
        self._ensure_heatmap_artist()

    def _ensure_line_artist(self) -> None:
        self.ax.cla()
        self._line = self.ax.plot(
            np.arange(max(1, self._y_len)),
            np.full((max(1, self._y_len),), np.nan, dtype=float)
        )[0]
        self.ax.set_xlabel("Y index")
        self.ax.set_ylabel("Total sum")
        self.ax.grid(True, alpha=0.3)
        self._im = None
        self.canvas.draw_idle()

    def _ensure_heatmap_artist(self) -> None:
        self.ax.cla()
        mat = np.zeros((max(1, self._y_len), 1), dtype=float)
        self._im = self.ax.imshow(mat, origin="lower", aspect="auto")
        self.ax.set_xlabel("pixel")
        self.ax.set_ylabel("Y index")
        self._line = None
        self.canvas.draw_idle()

    def _redraw(self) -> None:
        if self._is_line:
            if self._vec is None:
                return
            y = self._vec
            if self._line is None:
                self._ensure_line_artist()
            x = np.arange(y.shape[0])
            self._line.set_xdata(x)
            self._line.set_ydata(y)
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
        self.canvas.draw_idle()
