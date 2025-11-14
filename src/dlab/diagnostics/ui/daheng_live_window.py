from __future__ import annotations
import sys, os, time, datetime, numpy as np, yaml, logging, threading
from PIL import Image, PngImagePlugin
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QMessageBox, QSplitter, QCheckBox, QGroupBox, QSpinBox, QShortcut, QComboBox
)
from PyQt5.QtGui import QIntValidator, QKeySequence
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
from dlab.hardware.wrappers.daheng_controller import (
    DahengController, DahengControllerError,
    DEFAULT_EXPOSURE_US, MIN_EXPOSURE_US, MAX_EXPOSURE_US,
    DEFAULT_GAIN, MIN_GAIN, MAX_GAIN,
)
from dlab.boot import get_config
from dlab.core.device_registry import REGISTRY
import cmasher as cmr

PIXEL_SIZE_M = 3.45e-6
MIN_INTERVAL_US = 500_000
logging.getLogger("matplotlib").setLevel(logging.WARNING)
DATE_FORMAT = "%H:%M:%S"

def _data_root():
    cfg = get_config() or {}
    return str((cfg.get("paths", {}) or {}).get("data_dir", r"C:/data"))

def _config_path():
    return os.path.abspath(os.path.join("config", "config.yaml"))

def _read_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _write_yaml(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def _resolve_cmap(key):
    if isinstance(key, str) and key.startswith("cmr."):
        name = key.split(".", 1)[1]
        return getattr(cmr, name)
    return plt.get_cmap(key)

class LiveCaptureThread(QThread):
    image_signal = pyqtSignal(np.ndarray)
    def __init__(self, cam, exposure_us, gain, interval_us, cap_lock=None):
        super().__init__()
        self.cam = cam
        self.exposure_us = exposure_us
        self.gain = gain
        self.interval_s = interval_us / 1e6
        self._running = True
        self._lock = threading.Lock()
        self._cap_lock = cap_lock
    def update_parameters(self, exposure_us, gain, interval_us):
        with self._lock:
            self.exposure_us = exposure_us
            self.gain = gain
            self.interval_s = interval_us / 1e6
    def run(self):
        while self._running:
            try:
                with self._lock:
                    exp = self.exposure_us
                    g = self.gain
                    wait_s = self.interval_s
                lock = self._cap_lock or threading.Lock()
                with lock:
                    frame = self.cam.capture_single(exp, g)
                self.image_signal.emit(frame)
                time.sleep(wait_s)
            except Exception:
                break
    def stop(self):
        self._running = False

class DahengLiveWindow(QWidget):
    closed = pyqtSignal()
    gui_update_image = pyqtSignal(object)
    gui_log = pyqtSignal(str)
    def __init__(self, camera_name="Daheng", fixed_index=1):
        super().__init__()
        self.live_running = False
        self.capture_lock = threading.Lock()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.camera_name = camera_name
        self.fixed_index = fixed_index
        self.setWindowTitle(f"DahengLiveWindow - {camera_name}")
        self.cam = None
        self.thread = None
        self.image_artist = None
        self.cbar = None
        self.last_frame = None
        self.fix_cbar = False
        self.fixed_vmax = None
        self.cmap_key = "cmr.rainforest"
        self.cmap = _resolve_cmap(self.cmap_key)
        self.roi_px = None
        self.roi_artist = None
        self.rect_selector = None
        self.use_roi_cb = QCheckBox("Use ROI")
        self.use_roi_cb.setChecked(False)
        self.preview_roi_cb = QCheckBox("Preview crop")
        self.preview_roi_cb.setChecked(False)
        self.center_w_spin = QSpinBox()
        self.center_h_spin = QSpinBox()
        for sp, dv in ((self.center_w_spin, 200), (self.center_h_spin, 200)):
            sp.setRange(4, 4096); sp.setSingleStep(10); sp.setValue(dv)

        # Crosshair 1
        self.crosshair_visible = False
        self.crosshair_locked = False
        self.crosshair_pos_mm = None
        self.crosshair_hline = None
        self.crosshair_vline = None

        # Crosshair 2
        self.crosshair2_visible = False
        self.crosshair2_locked = False
        self.crosshair2_pos_mm = None
        self.crosshair2_hline = None
        self.crosshair2_vline = None

        self._mpl_cid_click = None
        self._mpl_cid_motion = None
        self.initUI()
        self.gui_update_image.connect(self.update_image, Qt.QueuedConnection)
        self.gui_log.connect(self.log, Qt.QueuedConnection)

    def initUI(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter()
        param_panel = QWidget()
        param_layout = QVBoxLayout(param_panel)

        idx_layout = QHBoxLayout()
        idx_layout.addWidget(QLabel("Camera Index:"))
        idx_layout.addWidget(QLabel(str(self.fixed_index)))
        param_layout.addLayout(idx_layout)

        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure (µs):"))
        self.exposure_edit = QLineEdit(str(DEFAULT_EXPOSURE_US))
        self.exposure_edit.setValidator(QIntValidator(MIN_EXPOSURE_US, MAX_EXPOSURE_US, self))
        self.exposure_edit.textChanged.connect(self.update_params)
        exp_layout.addWidget(self.exposure_edit)
        param_layout.addLayout(exp_layout)

        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Update Interval (µs):"))
        init_interval = max(DEFAULT_EXPOSURE_US, MIN_INTERVAL_US)
        self.interval_edit = QLineEdit(str(init_interval))
        self.interval_edit.setEnabled(False)
        int_layout.addWidget(self.interval_edit)
        param_layout.addLayout(int_layout)

        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain:"))
        self.gain_edit = QLineEdit(str(DEFAULT_GAIN))
        self.gain_edit.setValidator(QIntValidator(MIN_GAIN, MAX_GAIN, self))
        self.gain_edit.textChanged.connect(self.update_params)
        gain_layout.addWidget(self.gain_edit)
        param_layout.addLayout(gain_layout)

        cmt_layout = QHBoxLayout()
        cmt_layout.addWidget(QLabel("Comment:"))
        self.comment_edit = QLineEdit()
        cmt_layout.addWidget(self.comment_edit)
        param_layout.addLayout(cmt_layout)

        nsave_layout = QHBoxLayout()
        nsave_layout.addWidget(QLabel("Frames to Save:"))
        self.frames_to_save_edit = QLineEdit("1")
        self.frames_to_save_edit.setValidator(QIntValidator(1, 1000, self))
        nsave_layout.addWidget(self.frames_to_save_edit)
        param_layout.addLayout(nsave_layout)

        self.background_cb = QCheckBox("Background")
        param_layout.addWidget(self.background_cb)

        self.fix_cb = QCheckBox("Fix Colorbar Max")
        self.fix_cb.toggled.connect(self.on_fix_cbar)
        param_layout.addWidget(self.fix_cb)
        self.fix_value_edit = QLineEdit("10000")
        self.fix_value_edit.setValidator(QIntValidator(0, 1_000_000_000, self))
        self.fix_value_edit.setEnabled(False)
        self.fix_value_edit.textChanged.connect(self.on_fix_value_changed)
        self.fix_cb.toggled.connect(self.fix_value_edit.setEnabled)
        param_layout.addWidget(self.fix_value_edit)

        cmap_group = QGroupBox("Colormap")
        cmap_layout = QHBoxLayout(cmap_group)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["cmr.rainforest", "cmr.neutral", "cmr.sunburst", "turbo"])
        self.cmap_combo.setCurrentText(self.cmap_key)
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)
        cmap_layout.addWidget(QLabel("Map:"))
        cmap_layout.addWidget(self.cmap_combo)
        param_layout.addWidget(cmap_group)

        roi_group = QGroupBox("ROI")
        roi_layout = QHBoxLayout(roi_group)
        set_btn = QPushButton("Set ROI")
        clear_btn = QPushButton("Clear ROI")
        center_btn = QPushButton("Center on max")
        set_btn.clicked.connect(self.start_roi_selection)
        clear_btn.clicked.connect(self.clear_roi)
        center_btn.clicked.connect(self.center_on_max)
        roi_layout.addWidget(self.use_roi_cb)
        roi_layout.addWidget(self.preview_roi_cb)
        roi_layout.addWidget(QLabel("W(px):")); roi_layout.addWidget(self.center_w_spin)
        roi_layout.addWidget(QLabel("H(px):")); roi_layout.addWidget(self.center_h_spin)
        roi_layout.addWidget(center_btn)
        roi_layout.addWidget(set_btn)
        roi_layout.addWidget(clear_btn)
        param_layout.addWidget(roi_group)

        xhair_group = QGroupBox("Crosshair")
        xh_layout = QVBoxLayout(xhair_group)

        row1 = QHBoxLayout()
        self.btn_xhair_toggle = QPushButton("X1 Toggle (Shift+C)")
        self.btn_xhair_toggle.clicked.connect(self.toggle_crosshair)
        self.btn_xhair_lock = QPushButton("X1 Lock/Unlock")
        self.btn_xhair_lock.clicked.connect(self.toggle_lock_manual)
        self.btn_xhair_save = QPushButton("X1 Save Pos")
        self.btn_xhair_goto = QPushButton("X1 Go To Saved")
        self.btn_xhair_save.clicked.connect(self.save_crosshair_position)
        self.btn_xhair_goto.clicked.connect(self.goto_saved_crosshair_position)
        row1.addWidget(self.btn_xhair_toggle)
        row1.addWidget(self.btn_xhair_lock)
        row1.addWidget(self.btn_xhair_save)
        row1.addWidget(self.btn_xhair_goto)
        xh_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_xhair2_toggle = QPushButton("X2 Toggle (Ctrl+D)")
        self.btn_xhair2_toggle.clicked.connect(self.toggle_crosshair2)
        self.btn_xhair2_lock = QPushButton("X2 Lock/Unlock")
        self.btn_xhair2_lock.clicked.connect(self.toggle_lock_manual2)
        self.btn_xhair2_save = QPushButton("X2 Save Pos")
        self.btn_xhair2_goto = QPushButton("X2 Go To Saved")
        self.btn_xhair2_save.clicked.connect(self.save_crosshair2_position)
        self.btn_xhair2_goto.clicked.connect(self.goto_saved_crosshair2_position)
        row2.addWidget(self.btn_xhair2_toggle)
        row2.addWidget(self.btn_xhair2_lock)
        row2.addWidget(self.btn_xhair2_save)
        row2.addWidget(self.btn_xhair2_goto)
        xh_layout.addLayout(row2)

        param_layout.addWidget(xhair_group)

        btn_layout = QVBoxLayout()
        def make_btn(text, slot, enabled=True):
            b = QPushButton(text); b.clicked.connect(slot); b.setEnabled(enabled); btn_layout.addWidget(b); return b
        self.activate_camera_btn = make_btn("Activate Camera", self.activate_camera)
        self.deactivate_camera_btn = make_btn("Deactivate Camera", self.deactivate_camera, False)
        self.start_capture_btn = make_btn("Start Live Capture", self.start_capture, False)
        self.stop_capture_btn = make_btn("Stop Live Capture", self.stop_capture, False)
        self.save_btn = make_btn("Save Frame(s)", self.save_frames)
        param_layout.addLayout(btn_layout)

        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        param_layout.addWidget(self.log_text)
        splitter.addWidget(param_panel)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax.set_title(f"DahengLiveWindow - {self.camera_name}")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.figure.subplots_adjust(right=0.85)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        main_layout.addWidget(splitter)

        self.shortcut_toggle_cross = QShortcut(QKeySequence("Shift+C"), self)
        self.shortcut_toggle_cross.activated.connect(self.toggle_crosshair)

        self.shortcut_toggle_cross2 = QShortcut(QKeySequence("Ctrl+D"), self)
        self.shortcut_toggle_cross2.activated.connect(self.toggle_crosshair2)

        self._mpl_cid_click = self.canvas.mpl_connect("button_press_event", self._on_mpl_click)
        self._mpl_cid_motion = self.canvas.mpl_connect("motion_notify_event", self._on_mpl_motion)
        self.resize(1080, 720)

    def on_cmap_changed(self, key):
        self.cmap_key = key
        self.cmap = _resolve_cmap(self.cmap_key)
        if self.image_artist is not None:
            self.image_artist.set_cmap(self.cmap)
            if self.cbar:
                self.cbar.update_normal(self.image_artist)
            self.canvas.draw_idle()
        self.log(f"Colormap set to {key}")

    def log(self, message):
        now = datetime.datetime.now().strftime(DATE_FORMAT)
        self.log_text.append(f"[{now}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        logging.getLogger("dlab.ui.DahengLiveWindow").info(message)

    def _update_interval_field(self, exp_us):
        interval_us = exp_us if exp_us >= MIN_INTERVAL_US else MIN_INTERVAL_US
        self.interval_edit.setText(str(interval_us))
        return interval_us

    def _px_per_mm(self):
        return 1.0 / (PIXEL_SIZE_M * 1e3)

    def start_roi_selection(self):
        if self.last_frame is None:
            self.log("No image yet — start live or capture one to set ROI.")
            return
        if self.rect_selector is not None:
            try: self.rect_selector.disconnect_events()
            except Exception: pass
            self.rect_selector = None
        def _on_select(eclick, erelease):
            if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
                return
            px_per_mm = self._px_per_mm()
            x0_mm, y0_mm = eclick.xdata, eclick.ydata
            x1_mm, y1_mm = erelease.xdata, erelease.ydata
            x0 = int(np.floor(min(x0_mm, x1_mm) * px_per_mm))
            x1 = int(np.ceil (max(x0_mm, x1_mm) * px_per_mm))
            y0 = int(np.floor(min(y0_mm, y1_mm) * px_per_mm))
            y1 = int(np.ceil (max(y0_mm, y1_mm) * px_per_mm))
            h, w = self.last_frame.shape
            x0 = max(0, min(w-1, x0)); x1 = max(1, min(w, x1))
            y0 = max(0, min(h-1, y0)); y1 = max(1, min(h, y1))
            if x1 <= x0 or y1 <= y0:
                self.log("Invalid ROI")
                return
            self.roi_px = (x0, y0, x1, y1)
            self._draw_roi_overlay()
            self.log(f"ROI set: x={x0}:{x1}, y={y0}:{y1} (px)")
            if self.rect_selector:
                self.rect_selector.set_visible(False)
                self.canvas.draw_idle()
        self.rect_selector = RectangleSelector(self.ax, _on_select, useblit=True, button=[1], minspanx=2, minspany=2, spancoords='pixels', interactive=True)

    def _draw_roi_overlay(self):
        try:
            if self.roi_artist is not None:
                self.roi_artist.remove()
                self.roi_artist = None
        except Exception:
            pass
        if self.roi_px is None:
            self.canvas.draw_idle()
            return
        x0, y0, x1, y1 = self.roi_px
        mm_per_px = PIXEL_SIZE_M * 1e3
        x0_mm, x1_mm = x0 * mm_per_px, x1 * mm_per_px
        y0_mm, y1_mm = y0 * mm_per_px, y1 * mm_per_px
        from matplotlib.patches import Rectangle
        self.roi_artist = self.ax.add_patch(Rectangle((x0_mm, y0_mm), (x1_mm - x0_mm), (y1_mm - y0_mm), fill=False, linewidth=1.5, linestyle="--"))
        self.canvas.draw_idle()

    def clear_roi(self):
        self.roi_px = None
        self._draw_roi_overlay()
        self.log("ROI cleared")

    def center_on_max(self):
        if self.last_frame is None or self.last_frame.size == 0:
            self.log("No image to center on.")
            return
        h, w = self.last_frame.shape
        idx = int(np.argmax(self.last_frame))
        y_max, x_max = divmod(idx, w)
        rw = int(self.center_w_spin.value())
        rh = int(self.center_h_spin.value())
        rw = max(4, min(w, rw))
        rh = max(4, min(h, rh))
        half_w = rw // 2
        half_h = rh // 2
        x0 = max(0, x_max - half_w)
        y0 = max(0, y_max - half_h)
        x1 = min(w, x0 + rw)
        y1 = min(h, y0 + rh)
        x0 = max(0, x1 - rw)
        y0 = max(0, y1 - rh)
        if x1 <= x0 or y1 <= y0:
            self.log("Center ROI failed due to size.")
            return
        self.roi_px = (int(x0), int(y0), int(x1), int(y1))
        self._draw_roi_overlay()
        self.use_roi_cb.setChecked(True)
        self.preview_roi_cb.setChecked(True)
        self.log(f"ROI centered on max at ({x_max},{y_max}), size=({rw}x{rh})")

    def _ensure_crosshair_artists(self):
        if self.crosshair_hline is None or self.crosshair_vline is None:
            self.crosshair_hline = self.ax.axhline(0, linestyle="--", linewidth=1.2, color="r")
            self.crosshair_vline = self.ax.axvline(0, linestyle="--", linewidth=1.2, color="r")
            self.crosshair_hline.set_visible(self.crosshair_visible)
            self.crosshair_vline.set_visible(self.crosshair_visible)

    def _ensure_crosshair2_artists(self):
        if self.crosshair2_hline is None or self.crosshair2_vline is None:
            self.crosshair2_hline = self.ax.axhline(0, linestyle="-.", linewidth=1.2, color="green")
            self.crosshair2_vline = self.ax.axvline(0, linestyle="-.", linewidth=1.2, color="green")
            self.crosshair2_hline.set_visible(self.crosshair2_visible)
            self.crosshair2_vline.set_visible(self.crosshair2_visible)

    def toggle_crosshair(self):
        if not self.crosshair_visible and self.crosshair_pos_mm is None:
            if self.last_frame is not None:
                h, w = self.last_frame.shape
                mm_per_px = PIXEL_SIZE_M * 1e3
                cx = (w * mm_per_px) / 2.0
                cy = (h * mm_per_px) / 2.0
                self.crosshair_pos_mm = (cx, cy)
            else:
                self.crosshair_pos_mm = (0.0, 0.0)
        self.crosshair_visible = not self.crosshair_visible
        self._ensure_crosshair_artists()
        self._refresh_crosshair()
        self.log(f"Crosshair 1 {'shown' if self.crosshair_visible else 'hidden'}")

    def toggle_crosshair2(self):
        if not self.crosshair2_visible and self.crosshair2_pos_mm is None:
            if self.last_frame is not None:
                h, w = self.last_frame.shape
                mm_per_px = PIXEL_SIZE_M * 1e3
                cx = (w * mm_per_px) / 2.0
                cy = (h * mm_per_px) / 2.0
                self.crosshair2_pos_mm = (cx, cy)
            else:
                self.crosshair2_pos_mm = (0.0, 0.0)
        self.crosshair2_visible = not self.crosshair2_visible
        self._ensure_crosshair2_artists()
        self._refresh_crosshair2()
        self.log(f"Crosshair 2 {'shown' if self.crosshair2_visible else 'hidden'}")

    def toggle_lock_manual(self):
        if not self.crosshair_visible:
            return
        self.crosshair_locked = not self.crosshair_locked
        self._refresh_crosshair()

    def toggle_lock_manual2(self):
        if not self.crosshair2_visible:
            return
        self.crosshair2_locked = not self.crosshair2_locked
        self._refresh_crosshair2()

    def _refresh_crosshair(self):
        self._ensure_crosshair_artists()
        vis = bool(self.crosshair_visible)
        self.crosshair_hline.set_visible(vis)
        self.crosshair_vline.set_visible(vis)
        if vis and self.crosshair_pos_mm is not None:
            x_mm, y_mm = self.crosshair_pos_mm
            self.crosshair_hline.set_ydata([y_mm, y_mm])
            self.crosshair_vline.set_xdata([x_mm, x_mm])
        self.canvas.draw_idle()

    def _refresh_crosshair2(self):
        self._ensure_crosshair2_artists()
        vis = bool(self.crosshair2_visible)
        self.crosshair2_hline.set_visible(vis)
        self.crosshair2_vline.set_visible(vis)
        if vis and self.crosshair2_pos_mm is not None:
            x_mm, y_mm = self.crosshair2_pos_mm
            self.crosshair2_hline.set_ydata([y_mm, y_mm])
            self.crosshair2_vline.set_xdata([x_mm, x_mm])
        self.canvas.draw_idle()

    def _on_mpl_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 3 and self.crosshair_visible:
            self.crosshair_locked = not self.crosshair_locked
            if self.crosshair_locked:
                self.crosshair_pos_mm = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair()
        elif event.button == 2 and self.crosshair2_visible:
            self.crosshair2_locked = not self.crosshair2_locked
            if self.crosshair2_locked:
                self.crosshair2_pos_mm = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair2()

    def _on_mpl_motion(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if self.crosshair_visible and not self.crosshair_locked:
            self.crosshair_pos_mm = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair()
        if self.crosshair2_visible and not self.crosshair2_locked:
            self.crosshair2_pos_mm = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair2()

    def save_crosshair_position(self):
        if not self.crosshair_visible or self.crosshair_pos_mm is None:
            QMessageBox.warning(self, "Crosshair", "Crosshair 1 must be visible to save its position.")
            return
        path = _config_path()
        data = _read_yaml(path)
        cam_key = f"daheng_{self.fixed_index}"
        node = data.get("crosshair", {})
        node[cam_key] = {"x_mm": float(self.crosshair_pos_mm[0]), "y_mm": float(self.crosshair_pos_mm[1])}
        data["crosshair"] = node
        try:
            _write_yaml(path, data)
            self.log(f"Crosshair 1 position saved to {path} under key crosshair.{cam_key}")
        except Exception as e:
            QMessageBox.critical(self, "Crosshair", f"Failed to save crosshair 1: {e}")

    def save_crosshair2_position(self):
        if not self.crosshair2_visible or self.crosshair2_pos_mm is None:
            QMessageBox.warning(self, "Crosshair 2", "Crosshair 2 must be visible to save its position.")
            return
        path = _config_path()
        data = _read_yaml(path)
        cam_key = f"daheng_{self.fixed_index}"
        node = data.get("crosshair2", {})
        node[cam_key] = {"x_mm": float(self.crosshair2_pos_mm[0]), "y_mm": float(self.crosshair2_pos_mm[1])}
        data["crosshair2"] = node
        try:
            _write_yaml(path, data)
            self.log(f"Crosshair 2 position saved to {path} under key crosshair2.{cam_key}")
        except Exception as e:
            QMessageBox.critical(self, "Crosshair 2", f"Failed to save crosshair 2: {e}")

    def goto_saved_crosshair_position(self):
        path = _config_path()
        data = _read_yaml(path)
        cam_key = f"daheng_{self.fixed_index}"
        pos = ((data.get("crosshair") or {}).get(cam_key)) if isinstance(data, dict) else None
        if not pos:
            QMessageBox.information(self, "Crosshair", "No saved position found for crosshair 1 on this camera.")
            return
        try:
            x_mm = float(pos["x_mm"]); y_mm = float(pos["y_mm"])
        except Exception:
            QMessageBox.critical(self, "Crosshair", "Saved position for crosshair 1 is invalid.")
            return
        self.crosshair_pos_mm = (x_mm, y_mm)
        if not self.crosshair_visible:
            self.crosshair_visible = True
        self._refresh_crosshair()
        self.log(f"Crosshair 1 moved to saved position ({x_mm:.3f} mm, {y_mm:.3f} mm)")

    def goto_saved_crosshair2_position(self):
        path = _config_path()
        data = _read_yaml(path)
        cam_key = f"daheng_{self.fixed_index}"
        pos = ((data.get("crosshair2") or {}).get(cam_key)) if isinstance(data, dict) else None
        if not pos:
            QMessageBox.information(self, "Crosshair 2", "No saved position found for crosshair 2 on this camera.")
            return
        try:
            x_mm = float(pos["x_mm"]); y_mm = float(pos["y_mm"])
        except Exception:
            QMessageBox.critical(self, "Crosshair 2", "Saved position for crosshair 2 is invalid.")
            return
        self.crosshair2_pos_mm = (x_mm, y_mm)
        if not self.crosshair2_visible:
            self.crosshair2_visible = True
        self._refresh_crosshair2()
        self.log(f"Crosshair 2 moved to saved position ({x_mm:.3f} mm, {y_mm:.3f} mm)")

    def update_params(self):
        try:
            exp_us = int(self.exposure_edit.text())
            gain = int(self.gain_edit.text())
            interval_us = self._update_interval_field(exp_us)
            if self.thread:
                self.thread.update_parameters(exp_us, gain, interval_us)
            elif self.cam:
                self.cam.set_exposure(exp_us)
                self.cam.set_gain(gain)
        except ValueError:
            pass

    def activate_camera(self):
        if self.cam:
            try: self.cam.deactivate()
            except Exception: pass
            self.cam = None
        try:
            self.cam = DahengController(self.fixed_index)
            self.cam.activate()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate: {e}")
            self.log(f"Activation error: {e}")
            return
        self.log(f"Camera {self.fixed_index} activated")
        key_name  = f"camera:daheng:{self.camera_name.lower()}"
        key_index = f"camera:daheng:index:{self.fixed_index}"
        for k in (key_name, key_index):
            prev = REGISTRY.get(k)
            if prev and prev is not self.cam:
                self.log(f"Registry key '{k}' already in use. Replacing.")
            REGISTRY.register(k, self)
        self.activate_camera_btn.setEnabled(False)
        self.deactivate_camera_btn.setEnabled(True)
        self.start_capture_btn.setEnabled(True)

    def deactivate_camera(self):
        if self.cam:
            try: self.cam.deactivate()
            except Exception: pass
            self.log(f"Camera {self.fixed_index} deactivated")
        for k in (f"camera:daheng:{self.camera_name.lower()}",
                  f"camera:daheng:index:{self.fixed_index}"):
            REGISTRY.unregister(k)
        self.cam = None
        self.activate_camera_btn.setEnabled(True)
        self.deactivate_camera_btn.setEnabled(False)
        self.start_capture_btn.setEnabled(False)
        self.stop_capture_btn.setEnabled(False)

    def start_capture(self):
        if not self.cam:
            QMessageBox.critical(self, "Error", "Camera not activated.")
            return
        try:
            exp_us = int(self.exposure_edit.text())
            gain = int(self.gain_edit.text())
            interval_us = self._update_interval_field(exp_us)
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameters.")
            return
        self.thread = LiveCaptureThread(self.cam, exp_us, gain, interval_us, cap_lock=self.capture_lock)
        self.thread.image_signal.connect(self.update_image)
        self.thread.start()
        self.live_running = True
        self.log("Live capture started")
        self.start_capture_btn.setEnabled(False)
        self.stop_capture_btn.setEnabled(True)

    def stop_capture(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None
            self.live_running = False
            self.log("Live capture stopped")
        self.start_capture_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(False)

    def update_image(self, image):
        if not isinstance(image, np.ndarray):
            self.log("Invalid image received")
            return
        self.last_frame = image
        disp = image
        extent_mm = None
        if self.preview_roi_cb.isChecked() and self.roi_px is not None and self.use_roi_cb.isChecked():
            x0, y0, x1, y1 = self.roi_px
            h0, w0 = disp.shape
            x0 = max(0, min(w0-1, x0)); x1 = max(1, min(w0, x1))
            y0 = max(0, min(h0-1, y0)); y1 = max(1, min(h0, y1))
            disp = disp[y0:y1, x0:x1]
            mm_per_px = PIXEL_SIZE_M * 1e3
            extent_mm = [x0*mm_per_px, x1*mm_per_px, y0*mm_per_px, y1*mm_per_px]
        vmin, vmax = float(disp.min()), float(disp.max())
        h, w = disp.shape
        if extent_mm is None:
            x_mm = w * (PIXEL_SIZE_M * 1e3)
            y_mm = h * (PIXEL_SIZE_M * 1e3)
            extent = [0, x_mm, 0, y_mm]
        else:
            extent = extent_mm
        if self.image_artist is None:
            self.ax.clear()
            self.image_artist = self.ax.imshow(disp, cmap=self.cmap, vmin=vmin, vmax=vmax, extent=extent, origin="lower", aspect="equal")
            self.cbar = self.figure.colorbar(self.image_artist, ax=self.ax, fraction=0.046, pad=0.04)
        else:
            self.image_artist.set_data(disp)
            if self.fix_cbar and self.fixed_vmax is not None:
                self.image_artist.set_clim(vmin, self.fixed_vmax)
            else:
                self.image_artist.set_clim(vmin, vmax)
            self.image_artist.set_extent(extent)
            self.image_artist.set_cmap(self.cmap)
            if self.cbar:
                self.cbar.update_normal(self.image_artist)
        if not (self.preview_roi_cb.isChecked() and self.use_roi_cb.isChecked()):
            self._draw_roi_overlay()
        if self.crosshair_visible:
            self._refresh_crosshair()
        if self.crosshair2_visible:
            self._refresh_crosshair2()
        self.canvas.draw_idle()

    def on_fix_cbar(self, checked):
        self.fix_cbar = checked
        if checked:
            try:
                self.fixed_vmax = float(self.fix_value_edit.text())
                self.log(f"Colorbar max set to {self.fixed_vmax:.1f}")
            except ValueError:
                self.fixed_vmax = None
                self.fix_cb.setChecked(False)
                self.log("Invalid colorbar max")
        else:
            self.fixed_vmax = None
            self.log("Colorbar auto scale")

    def on_fix_value_changed(self, text):
        if not self.fix_cb.isChecked() or not self.image_artist:
            return
        try:
            self.fixed_vmax = float(text)
            vmin, _ = self.image_artist.get_clim()
            self.image_artist.set_clim(vmin, self.fixed_vmax)
            if self.cbar:
                self.cbar.update_normal(self.image_artist)
            self.canvas.draw_idle()
            self.log(f"Colorbar max updated to {self.fixed_vmax:.1f}")
        except ValueError:
            pass

    def _capture_one(self, exp_us, gain):
        if not self.cam:
            raise DahengControllerError("Camera not activated.")
        return self.cam.capture_single(exp_us, gain)

    def save_frames(self):
        try:
            exp_us = int(self.exposure_edit.text())
            gain = int(self.gain_edit.text())
            n_frames = int(self.frames_to_save_edit.text())
            if n_frames <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid 'Frames to Save' / exposure / gain.")
            return
        now = datetime.datetime.now()
        base_dir = _data_root()
        folder = os.path.join(base_dir, f"{now:%Y-%m-%d}", self.camera_name)
        os.makedirs(folder, exist_ok=True)
        ts_prefix = now.strftime("%Y%m%d_%H%M%S")
        stem = f"{self.camera_name}_{'Background' if self.background_cb.isChecked() else 'Image'}"
        comment = self.comment_edit.text()
        saved = []
        for i in range(1, n_frames + 1):
            try:
                frame = self._capture_one(exp_us, gain) if self.cam else self.last_frame
                if frame is None:
                    raise RuntimeError("No frame captured.")
                if self.use_roi_cb.isChecked() and self.roi_px is not None:
                    x0, y0, x1, y1 = self.roi_px
                    h0, w0 = frame.shape
                    x0 = max(0, min(w0-1, x0)); x1 = max(1, min(w0, x1))
                    y0 = max(0, min(h0-1, y0)); y1 = max(1, min(h0, y1))
                    frame = frame[y0:y1, x0:x1]
                f8 = np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))
                fn = f"{stem}_{ts_prefix}_{i}.png"
                path = os.path.join(folder, fn)
                meta = PngImagePlugin.PngInfo()
                meta.add_text("Exposure_us", str(exp_us))
                meta.add_text("Gain", str(gain))
                meta.add_text("Comment", comment)
                Image.fromarray(f8, mode="L").save(path, format="PNG", pnginfo=meta)
                saved.append(fn)
                self.log(f"Saved {path}")
            except Exception as e:
                self.log(f"Error saving frame {i}: {e}")
                QMessageBox.critical(self, "Error", f"Error saving frame {i}: {e}")
                break
        if not saved:
            return
        log_name = f"{self.camera_name}_log_{now:%Y-%m-%d}.log"
        log_path = os.path.join(folder, log_name)
        header = "ImageFile\tExposure_us\tGain\tComment\n"
        try:
            if not os.path.exists(log_path):
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(header)
            with open(log_path, "a", encoding="utf-8") as f:
                for fn in saved:
                    f.write(f"{fn}\t{exp_us}\t{gain}\t{comment}\n")
            self.log(f"Logged {len(saved)} file(s) to {log_path}")
        except Exception as e:
            self.log(f"Error writing log: {e}")
            QMessageBox.critical(self, "Error", f"Error writing log: {e}")

    def grab_frame_for_scan(self, averages=1, adaptive=None, dead_pixel_cleanup=False, background=False, *, force_roi=False):
        if not self.cam:
            raise DahengControllerError("Camera not activated.")
        was_live = bool(self.live_running)
        if was_live:
            try: self.stop_capture()
            except Exception: pass
        try: exp_us = int(self.exposure_edit.text())
        except ValueError: exp_us = DEFAULT_EXPOSURE_US
        try: device_gain = int(self.gain_edit.text())
        except ValueError: device_gain = DEFAULT_GAIN
        def _cap_once(cur_exp_us):
            self.cam.set_exposure(cur_exp_us); self.cam.set_gain(device_gain)
            return self.cam.capture_single(cur_exp_us, device_gain)
        n = max(1, int(averages))
        acc = None
        for _ in range(n):
            f = np.asarray(_cap_once(exp_us), dtype=np.float32)
            if (force_roi or self.use_roi_cb.isChecked()) and self.roi_px is not None:
                x0, y0, x1, y1 = self.roi_px
                h0, w0 = f.shape
                x0 = max(0, min(w0 - 1, x0)); x1 = max(1, min(w0, x1))
                y0 = max(0, min(h0 - 1, y0)); y1 = max(1, min(h0, y1))
                f = f[y0:y1, x0:x1]
            acc = f if acc is None else (acc + f)
        avg = acc / n
        if dead_pixel_cleanup:
            avg[avg >= 65535.0] = 0.0
            avg[avg < 0.0] = 0.0
            p9999 = np.percentile(avg, 99.99)
            if p9999 > 65535.0:
                avg[avg > p9999] = 0.0
        frame_u8 = np.clip(avg, 0, 255).astype(np.uint8)
        self.gui_update_image.emit(frame_u8)
        meta = {
            "CameraName": f"DahengCam_{self.fixed_index}",
            "CameraIndex": self.fixed_index,
            "Exposure_us": exp_us,
            "Gain": "",
            "Background": "1" if background else "0",
            "ROI_px": "" if self.roi_px is None else f"{self.roi_px[0]},{self.roi_px[1]},{self.roi_px[2]},{self.roi_px[3]}",
            "ROI_Used": "1" if (force_roi or (self.use_roi_cb.isChecked() and self.roi_px is not None)) else "0",
        }
        return frame_u8, meta

    def closeEvent(self, event):
        if self.thread:
            self.stop_capture()
        if self.cam:
            try: self.cam.deactivate()
            except Exception: pass
        for k in (f"camera:daheng:{self.camera_name.lower()}", f"camera:daheng:index:{self.fixed_index}"):
            REGISTRY.unregister(k)
        try:
            if self._mpl_cid_click is not None:
                self.canvas.mpl_disconnect(self._mpl_cid_click)
            if self._mpl_cid_motion is not None:
                self.canvas.mpl_disconnect(self._mpl_cid_motion)
        except Exception:
            pass
        self.closed.emit()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DahengLiveWindow(camera_name="Nomarski", fixed_index=1)
    gui.show()
    sys.exit(app.exec_())
