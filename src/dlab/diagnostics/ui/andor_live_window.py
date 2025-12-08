from __future__ import annotations
import sys, time, datetime, os, numpy as np, matplotlib.pyplot as plt, yaml, logging, threading
from pathlib import Path
from PIL import Image, PngImagePlugin
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QMessageBox, QSplitter, QCheckBox, QSpinBox, QGroupBox,
    QDoubleSpinBox, QShortcut, QComboBox
)
from PyQt5.QtGui import QIntValidator, QKeySequence
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.ndimage import rotate as sp_rotate

from dlab.hardware.wrappers.andor_controller import (
    AndorController, AndorControllerError,
    DEFAULT_EXPOSURE_US, MIN_EXPOSURE_US, MAX_EXPOSURE_US
)
from dlab.boot import get_config
from dlab.core.device_registry import REGISTRY
from dlab.hardware.wrappers.andor_registry_adapter import AndorRegistryCamera
import cmasher as cmr

logging.getLogger("matplotlib").setLevel(logging.WARNING)
DATE_FORMAT = "%H:%M:%S"

CAMERA_NAME = "AndorCam_1"
REGISTRY_KEY = "camera:andor:andorcam_1"

DEFAULT_PREPROCESS = {
    "enabled": False,
    "angle": -95.0,
    "x0": 165,
    "x1": 490,
    "y0": 210,
    "y1": 340
}

PLOT_HEIGHT_RATIOS = [3, 1]
SPLITTER_RATIOS = [1, 4]
FPS_UPDATE_INTERVAL_MS = 500

def _data_root():
    cfg = get_config() or {}
    return str((cfg.get("paths", {}) or {}).get("data_dir", r"C:/data"))

def _config_path():
    return os.path.join("config", "config.yaml")

def _resolve_cmap(key):
    if isinstance(key, str) and key.startswith("cmr."):
        name = key.split(".", 1)[1]
        return getattr(cmr, name)
    return plt.get_cmap(key)

class LiveCaptureThread(QThread):
    image_signal = pyqtSignal(np.ndarray)
    fps_signal = pyqtSignal(float)
    
    def __init__(self, camera_controller, exposure, update_interval_ms):
        super().__init__()
        self.camera_controller = camera_controller
        self.exposure = exposure
        self.update_interval_ms = update_interval_ms
        self.interval_sec = update_interval_ms / 1000.0
        self._running = True
        self._param_lock = threading.Lock()
        self._frame_times = []
        self._fps = 0.0
        
    def update_parameters(self, exposure, update_interval_ms):
        with self._param_lock:
            self.exposure = exposure
            self.update_interval_ms = update_interval_ms
            self.interval_sec = update_interval_ms / 1000.0
            
    def _capture_one(self, exp_us):
        if hasattr(self.camera_controller, "capture_single"):
            return self.camera_controller.capture_single(exp_us)
        return self.camera_controller.take_image(exp_us, 1)
        
    def _update_fps(self):
        now = time.time()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t < 2.0]
        if len(self._frame_times) >= 2:
            self._fps = len(self._frame_times) / (self._frame_times[-1] - self._frame_times[0])
        else:
            self._fps = 0.0
        self.fps_signal.emit(self._fps)
        
    def run(self):
        while self._running:
            try:
                with self._param_lock:
                    exp = self.exposure
                    sleep_interval = self.interval_sec
                image = self._capture_one(exp)
                self.image_signal.emit(image)
                self._update_fps()
                time.sleep(sleep_interval)
            except AndorControllerError:
                break
            except Exception:
                break
                
    def stop(self):
        self._running = False

class AndorLiveWindow(QWidget):
    closed = pyqtSignal()
    external_image_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Andor Camera Feed")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.cam = None
        self.capture_thread = None
        self.registered_key = None
        self.image_artist = None
        self.cbar = None
        self.fixed_cbar_max = None
        self.last_frame = None
        self._frame_lock = threading.Lock()
        self.external_image_signal.connect(self.update_image)
        self.crosshair_visible = False
        self.crosshair_locked = False
        self.crosshair_pos = None
        self.ch_h = None
        self.ch_v = None
        self.line_mode_active = False
        self.line_start = None
        self.line_artists = []
        self.cmap_key = "cmr.rainforest"
        self.cmap = _resolve_cmap(self.cmap_key)
        self.current_fps = 0.0
        self.initUI()
        self.load_preprocess_from_config()
        
    def initUI(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter()

        param_panel = QWidget()
        param_layout = QVBoxLayout(param_panel)

        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Exposure (µs):"))
        self.exposure_edit = QLineEdit(f"{DEFAULT_EXPOSURE_US}")
        self.exposure_edit.setValidator(QIntValidator(MIN_EXPOSURE_US, MAX_EXPOSURE_US, self))
        self.exposure_edit.textChanged.connect(self.update_capture_parameters)
        exp_layout.addWidget(self.exposure_edit)
        param_layout.addLayout(exp_layout)

        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("Update Interval (ms):"))
        self.interval_edit = QLineEdit("100")
        self.interval_edit.setValidator(QIntValidator(100, 10000, self))
        self.interval_edit.textChanged.connect(self.update_capture_parameters)
        int_layout.addWidget(self.interval_edit)
        param_layout.addLayout(int_layout)
        
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_label = QLabel("0.0")
        fps_layout.addWidget(self.fps_label)
        fps_layout.addStretch()
        param_layout.addLayout(fps_layout)

        mcp_layout = QHBoxLayout()
        mcp_layout.addWidget(QLabel("MCP Voltage:"))
        self.mcp_voltage_edit = QLineEdit("Not specified")
        mcp_layout.addWidget(self.mcp_voltage_edit)
        param_layout.addLayout(mcp_layout)

        comment_layout = QHBoxLayout()
        comment_layout.addWidget(QLabel("Comment:"))
        self.comment_edit = QLineEdit()
        comment_layout.addWidget(self.comment_edit)
        param_layout.addLayout(comment_layout)

        nsave_layout = QHBoxLayout()
        nsave_layout.addWidget(QLabel("Frames to Save:"))
        self.frames_to_save_edit = QLineEdit("1")
        self.frames_to_save_edit.setValidator(QIntValidator(1, 1000, self))
        nsave_layout.addWidget(self.frames_to_save_edit)
        param_layout.addLayout(nsave_layout)

        self.fix_cbar_checkbox = QCheckBox("Fix Colorbar Max")
        param_layout.addWidget(self.fix_cbar_checkbox)
        self.fix_value_edit = QLineEdit("10000")
        self.fix_value_edit.setValidator(QIntValidator(0, 1_000_000_000, self))
        self.fix_value_edit.setEnabled(False)
        param_layout.addWidget(self.fix_value_edit)
        self.fix_cbar_checkbox.toggled.connect(self.fix_value_edit.setEnabled)
        self.fix_cbar_checkbox.toggled.connect(self.on_fix_cbar)
        self.fix_value_edit.textChanged.connect(self.on_fix_value_changed)

        self.background_checkbox = QCheckBox("Background")
        self.background_checkbox.setChecked(False)
        param_layout.addWidget(self.background_checkbox)

        cmap_group = QGroupBox("Colormap")
        cmap_layout = QHBoxLayout(cmap_group)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["cmr.rainforest", "cmr.neutral", "cmr.sunburst", "cmr.freeze", "turbo", "viridis", "plasma"])
        self.cmap_combo.setCurrentText(self.cmap_key)
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)
        cmap_layout.addWidget(QLabel("Map:"))
        cmap_layout.addWidget(self.cmap_combo)
        param_layout.addWidget(cmap_group)

        ch_grp = QGroupBox("Crosshair")
        ch_l = QVBoxLayout(ch_grp)
        ch_row1 = QHBoxLayout()
        self.btn_ch_toggle = QPushButton("Toggle (Shift+C)")
        self.btn_ch_toggle.clicked.connect(self.toggle_crosshair)
        self.btn_ch_lock = QPushButton("Lock/Unlock")
        self.btn_ch_lock.clicked.connect(self.toggle_lock_manual)
        ch_row1.addWidget(self.btn_ch_toggle)
        ch_row1.addWidget(self.btn_ch_lock)
        ch_l.addLayout(ch_row1)
        ch_row2 = QHBoxLayout()
        self.btn_ch_save = QPushButton("Save Position")
        self.btn_ch_save.clicked.connect(self.save_crosshair_to_config)
        self.btn_ch_goto = QPushButton("Load Position")
        self.btn_ch_goto.clicked.connect(self.goto_saved_crosshair)
        ch_row2.addWidget(self.btn_ch_save)
        ch_row2.addWidget(self.btn_ch_goto)
        ch_l.addLayout(ch_row2)
        param_layout.addWidget(ch_grp)

        ln_grp = QGroupBox("Lines")
        ln_l = QHBoxLayout(ln_grp)
        self.btn_line_start = QPushButton("Start Line")
        self.btn_line_start.clicked.connect(self.start_line_mode)
        self.btn_line_clear = QPushButton("Clear Lines")
        self.btn_line_clear.clicked.connect(self.clear_lines)
        ln_l.addWidget(self.btn_line_start)
        ln_l.addWidget(self.btn_line_clear)
        param_layout.addWidget(ln_grp)

        btn_layout = QVBoxLayout()
        self.activate_button = QPushButton("Activate Camera")
        self.activate_button.clicked.connect(self.activate_camera)
        btn_layout.addWidget(self.activate_button)
        self.deactivate_button = QPushButton("Deactivate Camera")
        self.deactivate_button.clicked.connect(self.deactivate_camera)
        self.deactivate_button.setEnabled(False)
        btn_layout.addWidget(self.deactivate_button)
        self.start_button = QPushButton("Start Live Capture")
        self.start_button.clicked.connect(self.start_capture)
        self.start_button.setEnabled(False)
        btn_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop Live Capture")
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False)
        btn_layout.addWidget(self.stop_button)
        self.save_button = QPushButton("Save Frame(s) (Ctrl+S)")
        self.save_button.clicked.connect(self.save_frames)
        btn_layout.addWidget(self.save_button)
        param_layout.addLayout(btn_layout)

        pre_grp = QGroupBox("Display Preprocess (rotate + crop)")
        pre_lay = QVBoxLayout(pre_grp)
        self.pre_enable_cb = QCheckBox("Enable")
        self.pre_enable_cb.setChecked(DEFAULT_PREPROCESS["enabled"])
        pre_lay.addWidget(self.pre_enable_cb)
        row_angle = QHBoxLayout()
        row_angle.addWidget(QLabel("Angle (°):"))
        self.pre_angle_sb = QDoubleSpinBox()
        self.pre_angle_sb.setRange(-360.0, 360.0)
        self.pre_angle_sb.setDecimals(2)
        self.pre_angle_sb.setSingleStep(1.0)
        self.pre_angle_sb.setValue(DEFAULT_PREPROCESS["angle"])
        row_angle.addWidget(self.pre_angle_sb)
        pre_lay.addLayout(row_angle)
        row_x = QHBoxLayout()
        self.pre_x0_sb = QSpinBox(); self.pre_x1_sb = QSpinBox()
        self.pre_x0_sb.setRange(0, 99999); self.pre_x1_sb.setRange(1, 99999)
        self.pre_x0_sb.setValue(DEFAULT_PREPROCESS["x0"])
        self.pre_x1_sb.setValue(DEFAULT_PREPROCESS["x1"])
        row_x.addWidget(QLabel("Crop x0:")); row_x.addWidget(self.pre_x0_sb)
        row_x.addWidget(QLabel("x1:")); row_x.addWidget(self.pre_x1_sb)
        pre_lay.addLayout(row_x)
        row_y = QHBoxLayout()
        self.pre_y0_sb = QSpinBox(); self.pre_y1_sb = QSpinBox()
        self.pre_y0_sb.setRange(0, 99999); self.pre_y1_sb.setRange(1, 99999)
        self.pre_y0_sb.setValue(DEFAULT_PREPROCESS["y0"])
        self.pre_y1_sb.setValue(DEFAULT_PREPROCESS["y1"])
        row_y.addWidget(QLabel("Crop y0:")); row_y.addWidget(self.pre_y0_sb)
        row_y.addWidget(QLabel("y1:")); row_y.addWidget(self.pre_y1_sb)
        pre_lay.addLayout(row_y)
        
        pre_btn_row = QHBoxLayout()
        self.btn_pre_save = QPushButton("Save Settings")
        self.btn_pre_save.clicked.connect(self.save_preprocess_to_config)
        self.btn_pre_load = QPushButton("Load Settings")
        self.btn_pre_load.clicked.connect(self.load_preprocess_from_config)
        self.btn_pre_reset = QPushButton("Reset to Default")
        self.btn_pre_reset.clicked.connect(self.reset_preprocess_to_default)
        pre_btn_row.addWidget(self.btn_pre_save)
        pre_btn_row.addWidget(self.btn_pre_load)
        pre_btn_row.addWidget(self.btn_pre_reset)
        pre_lay.addLayout(pre_btn_row)
        
        param_layout.addWidget(pre_grp)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        param_layout.addWidget(self.log_text)

        splitter.addWidget(param_panel)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.figure, (self.ax_img, self.ax_profile) = plt.subplots(
            2, 1, 
            gridspec_kw={"height_ratios": PLOT_HEIGHT_RATIOS}, 
            sharex=True
        )
        self.ax_img.set_title("Andor Camera Image")
        self.ax_img.set_xlabel("X (px)")
        self.ax_img.set_ylabel("Y (px)")
        self.ax_profile.set_title("Integrated Profile")
        self.ax_profile.grid(True, alpha=0.3)
        self.figure.subplots_adjust(right=0.85, hspace=0.15)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_panel)

        splitter.setStretchFactor(0, SPLITTER_RATIOS[0])
        splitter.setStretchFactor(1, SPLITTER_RATIOS[1])
        main_layout.addWidget(splitter)
        self.resize(1200, 800)
        
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self._shortcut_cross = QShortcut(QKeySequence("Shift+C"), self)
        self._shortcut_cross.activated.connect(self.toggle_crosshair)
        self._shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self._shortcut_save.activated.connect(self.save_frames)
        
    def log(self, message):
        current_time = datetime.datetime.now().strftime(DATE_FORMAT)
        self.log_text.append(f"[{current_time}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        logging.getLogger("dlab.ui.AndorLiveWindow").info(message)
        
    def _validate_preprocess_settings(self):
        x0, x1 = int(self.pre_x0_sb.value()), int(self.pre_x1_sb.value())
        y0, y1 = int(self.pre_y0_sb.value()), int(self.pre_y1_sb.value())
        if x0 >= x1:
            raise ValueError(f"x0 ({x0}) must be < x1 ({x1})")
        if y0 >= y1:
            raise ValueError(f"y0 ({y0}) must be < y1 ({y1})")
        
    def _profile_axis_for_display(self):
        if not self.pre_enable_cb.isChecked():
            return 1
        ang = float(self.pre_angle_sb.value()) % 180.0
        ang = min(ang, 180.0 - ang)
        return 0 if ang > 45.0 else 1
        
    def _display_preprocess(self, image):
        out = image
        if self.pre_enable_cb.isChecked():
            angle = float(self.pre_angle_sb.value())
            if abs(angle) > 1e-9:
                try:
                    out = sp_rotate(out, angle, reshape=True, order=3, mode="nearest").astype(image.dtype)
                except Exception as e:
                    self.log(f"Rotation failed: {e}")
            y0, y1 = int(self.pre_y0_sb.value()), int(self.pre_y1_sb.value())
            x0, x1 = int(self.pre_x0_sb.value()), int(self.pre_x1_sb.value())
            h, w = out.shape[:2]
            y0 = max(0, min(y0, max(0, h - 1)))
            y1 = max(y0 + 1, min(y1, h))
            x0 = max(0, min(x0, max(0, w - 1)))
            x1 = max(x0 + 1, min(x1, w))
            out = out[y0:y1, x0:x1]
        return out
        
    def save_preprocess_to_config(self):
        try:
            self._validate_preprocess_settings()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Settings", str(e))
            return
            
        path = _config_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {}
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                data = {}
            
            andor = data.get("andor", {}) if isinstance(data.get("andor"), dict) else {}
            andor["preprocess"] = {
                "enabled": bool(self.pre_enable_cb.isChecked()),
                "angle": float(self.pre_angle_sb.value()),
                "x0": int(self.pre_x0_sb.value()),
                "x1": int(self.pre_x1_sb.value()),
                "y0": int(self.pre_y0_sb.value()),
                "y1": int(self.pre_y1_sb.value())
            }
            data["andor"] = andor
            
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            self.log(f"Preprocess settings saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save preprocess settings: {e}")
            self.log(f"Error saving preprocess settings: {e}")
    
    def load_preprocess_from_config(self):
        path = _config_path()
        try:
            if not os.path.exists(path):
                self.log("No config file found, using defaults")
                return
            
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            preprocess = (((data.get("andor") or {}).get("preprocess")) or {})
            if not preprocess:
                self.log("No saved preprocess settings in config")
                return
            
            self.pre_enable_cb.setChecked(bool(preprocess.get("enabled", DEFAULT_PREPROCESS["enabled"])))
            self.pre_angle_sb.setValue(float(preprocess.get("angle", DEFAULT_PREPROCESS["angle"])))
            self.pre_x0_sb.setValue(int(preprocess.get("x0", DEFAULT_PREPROCESS["x0"])))
            self.pre_x1_sb.setValue(int(preprocess.get("x1", DEFAULT_PREPROCESS["x1"])))
            self.pre_y0_sb.setValue(int(preprocess.get("y0", DEFAULT_PREPROCESS["y0"])))
            self.pre_y1_sb.setValue(int(preprocess.get("y1", DEFAULT_PREPROCESS["y1"])))
            
            self.log(f"Preprocess settings loaded from {path}")
        except Exception as e:
            self.log(f"Warning: Failed to load preprocess settings: {e}")
    
    def reset_preprocess_to_default(self):
        self.pre_enable_cb.setChecked(DEFAULT_PREPROCESS["enabled"])
        self.pre_angle_sb.setValue(DEFAULT_PREPROCESS["angle"])
        self.pre_x0_sb.setValue(DEFAULT_PREPROCESS["x0"])
        self.pre_x1_sb.setValue(DEFAULT_PREPROCESS["x1"])
        self.pre_y0_sb.setValue(DEFAULT_PREPROCESS["y0"])
        self.pre_y1_sb.setValue(DEFAULT_PREPROCESS["y1"])
        self.log("Preprocess settings reset to defaults")
    
    def update_capture_parameters(self):
        try:
            exposure = int(self.exposure_edit.text())
            update_interval = int(self.interval_edit.text())
            if self.capture_thread:
                self.capture_thread.update_parameters(exposure, update_interval)
            elif self.cam:
                if hasattr(self.cam, "set_exposure"):
                    self.cam.set_exposure(exposure)
        except ValueError:
            pass
            
    def on_cmap_changed(self, key):
        self.cmap_key = key
        self.cmap = _resolve_cmap(self.cmap_key)
        if self.image_artist is not None:
            self.image_artist.set_cmap(self.cmap)
            if self.cbar:
                self.cbar.update_normal(self.image_artist)
            self.canvas.draw_idle()
        self.log(f"Colormap set to {key}")
        
    def activate_camera(self):
        try:
            self.cam = AndorController(device_index=0)
            self.cam.activate()
            self.log("Camera 0 activated.")
            self.adapter = AndorRegistryCamera(self.cam, name=CAMERA_NAME, live_window=self)
            REGISTRY.register(REGISTRY_KEY, self.adapter)
            self.registered_key = REGISTRY_KEY
            self.log(f"Registered {REGISTRY_KEY} in DeviceRegistry.")
            self.activate_button.setEnabled(False)
            self.deactivate_button.setEnabled(True)
            self.start_button.setEnabled(True)
        except AndorControllerError as ce:
            QMessageBox.critical(self, "Error", f"Failed to activate camera: {ce}")
            self.log(f"Error activating camera: {ce}")
            
    def deactivate_camera(self):
        try:
            if self.registered_key:
                try:
                    REGISTRY.unregister(self.registered_key)
                    self.log(f"Unregistered {self.registered_key} from DeviceRegistry.")
                except Exception as e:
                    self.log(f"Warning: Failed to unregister {self.registered_key}: {e}")
                self.registered_key = None
            if self.cam:
                self.cam.deactivate()
                self.log("Camera deactivated.")
            self.cam = None
            self.activate_button.setEnabled(True)
            self.deactivate_button.setEnabled(False)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
        except AndorControllerError as ce:
            QMessageBox.critical(self, "Error", f"Failed to deactivate camera: {ce}")
            self.log(f"Error deactivating camera: {ce}")
            
    def start_capture(self):
        if self.cam is None:
            QMessageBox.critical(self, "Error", "Camera not activated.")
            return
        try:
            exposure = int(self.exposure_edit.text())
            update_interval = int(self.interval_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameter values.")
            return
        self.capture_thread = LiveCaptureThread(self.cam, exposure, update_interval)
        self.capture_thread.image_signal.connect(self.update_image)
        self.capture_thread.fps_signal.connect(self.update_fps)
        self.capture_thread.start()
        self.log("Live capture started.")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
    def stop_capture(self):
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread.wait()
            self.log("Live capture stopped.")
            self.capture_thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.fps_label.setText("0.0")
        
    def update_fps(self, fps):
        self.current_fps = fps
        self.fps_label.setText(f"{fps:.1f}")
        
    def on_fix_cbar(self, checked):
        if checked:
            try:
                self.fixed_cbar_max = float(self.fix_value_edit.text())
                self.log(f"Colorbar max set to {self.fixed_cbar_max:.1f}")
            except ValueError:
                self.log("Invalid colorbar max value")
                self.fix_cbar_checkbox.setChecked(False)
        else:
            self.fixed_cbar_max = None
            self.log("Colorbar auto scale")
            
    def on_fix_value_changed(self, text):
        if not self.fix_cbar_checkbox.isChecked() or not self.image_artist:
            return
        try:
            self.fixed_cbar_max = float(text)
            vmin, _ = self.image_artist.get_clim()
            self.image_artist.set_clim(vmin, self.fixed_cbar_max)
            if self.cbar:
                self.cbar.update_normal(self.image_artist)
            self.canvas.draw_idle()
        except ValueError:
            pass
            
    def _refresh_crosshair(self):
        if self.ch_h is not None:
            try: 
                self.ch_h.remove()
            except Exception: 
                pass
            self.ch_h = None
        if self.ch_v is not None:
            try: 
                self.ch_v.remove()
            except Exception: 
                pass
            self.ch_v = None
        if not self.crosshair_visible:
            return
        if self.crosshair_pos is None:
            y0, y1 = self.ax_img.get_ylim()
            x0, x1 = self.ax_img.get_xlim()
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            self.crosshair_pos = (cx, cy)
        x, y = self.crosshair_pos
        color = 'red' if self.crosshair_locked else 'yellow'
        self.ch_h = self.ax_img.axhline(y, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
        self.ch_v = self.ax_img.axvline(x, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
        
    def start_line_mode(self):
        self.line_mode_active = True
        self.line_start = None
        self.log("Line: click to set start point.")
        
    def clear_lines(self):
        for ln in list(self.line_artists):
            try: 
                ln.remove()
            except Exception: 
                pass
        self.line_artists.clear()
        self.line_start = None
        self.line_mode_active = False
        self.canvas.draw_idle()
        self.log("Lines cleared.")
        
    def _on_mouse_move(self, event):
        if self.crosshair_visible and not self.crosshair_locked:
            if event.xdata is None or event.ydata is None or event.inaxes != self.ax_img:
                return
            self.crosshair_pos = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair()
            self.canvas.draw_idle()
            
    def _on_mouse_press(self, event):
        if event.inaxes == self.ax_img:
            if self.line_mode_active and event.button == 1:
                if event.xdata is None or event.ydata is None:
                    return
                if self.line_start is None:
                    self.line_start = (float(event.xdata), float(event.ydata))
                    self.log(f"Line start anchored at ({self.line_start[0]:.1f}, {self.line_start[1]:.1f}).")
                else:
                    x0, y0 = self.line_start
                    x1, y1 = float(event.xdata), float(event.ydata)
                    ln, = self.ax_img.plot([x0, x1], [y0, y1], linewidth=2.0, color="red", alpha=0.8)
                    self.line_artists.append(ln)
                    self.line_start = None
                    self.line_mode_active = False
                    self.canvas.draw_idle()
                    self.log(f"Line added from ({x0:.1f}, {y0:.1f}) to ({x1:.1f}, {y1:.1f}).")
                    return
            if event.button == 3 and self.crosshair_visible:
                self.crosshair_locked = not self.crosshair_locked
                if self.crosshair_locked and (event.xdata is not None and event.ydata is not None):
                    self.crosshair_pos = (float(event.xdata), float(event.ydata))
                self._refresh_crosshair()
                self.canvas.draw_idle()
                
    def toggle_crosshair(self):
        self.crosshair_visible = not self.crosshair_visible
        if not self.crosshair_visible:
            self.crosshair_locked = False
        self._refresh_crosshair()
        self.canvas.draw_idle()
        
    def toggle_lock_manual(self):
        if not self.crosshair_visible:
            return
        self.crosshair_locked = not self.crosshair_locked
        self._refresh_crosshair()
        self.canvas.draw_idle()
        
    def save_crosshair_to_config(self):
        if not self.crosshair_visible or self.crosshair_pos is None:
            QMessageBox.warning(self, "Crosshair", "Crosshair must be visible to save.")
            return
        path = _config_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {}
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                data = {}
            andor = data.get("andor", {}) if isinstance(data.get("andor"), dict) else {}
            andor["crosshair"] = {"x": float(self.crosshair_pos[0]), "y": float(self.crosshair_pos[1])}
            data["andor"] = andor
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            self.log(f"Crosshair saved to {path}: ({self.crosshair_pos[0]:.1f}, {self.crosshair_pos[1]:.1f})")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save crosshair: {e}")
            
    def goto_saved_crosshair(self):
        path = _config_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
        ch = (((data.get("andor") or {}).get("crosshair")) or {})
        if not ch or "x" not in ch or "y" not in ch:
            QMessageBox.information(self, "Crosshair", "No saved crosshair position in config.")
            return
        self.crosshair_visible = True
        self.crosshair_locked = True
        self.crosshair_pos = (float(ch["x"]), float(ch["y"]))
        self._refresh_crosshair()
        self.canvas.draw_idle()
        self.log(f"Loaded crosshair: ({self.crosshair_pos[0]:.1f}, {self.crosshair_pos[1]:.1f})")
        
    def update_image(self, image):
        with self._frame_lock:
            self.last_frame = image
        disp = self._display_preprocess(image)
        sum_val = float(np.sum(disp))
        max_val = float(np.max(disp))
        min_val = float(np.min(disp))
        mean_val = float(np.mean(disp))
        title_text = f"Sum: {sum_val:.0f} | Max: {max_val:.0f} | Mean: {mean_val:.1f}"
        
        if self.image_artist is None:
            self.ax_img.clear()
            self.image_artist = self.ax_img.imshow(disp, cmap=self.cmap, interpolation='None')
            self.ax_img.set_title(title_text)
            self.ax_img.set_xlabel("X (px)")
            self.ax_img.set_ylabel("Y (px)")
            self.cbar = self.figure.colorbar(self.image_artist, ax=self.ax_img, fraction=0.046, pad=0.04)
            self.cbar.ax.set_ylabel("Intensity", rotation=270, labelpad=15)
        else:
            xlim = self.ax_img.get_xlim()
            ylim = self.ax_img.get_ylim()
            self.image_artist.set_data(disp)
            self.ax_img.set_xlim(xlim)
            self.ax_img.set_ylim(ylim)
            self.ax_img.set_title(title_text)
            if self.fix_cbar_checkbox.isChecked() and self.fixed_cbar_max is not None:
                self.image_artist.set_clim(min_val, self.fixed_cbar_max)
            else:
                self.fixed_cbar_max = None
                self.image_artist.set_clim(min_val, max_val)
            self.image_artist.set_cmap(self.cmap)
            self.cbar.update_normal(self.image_artist)
            
        axis = self._profile_axis_for_display()
        profile = np.sum(disp, axis=axis)
        self.ax_profile.clear()
        self.ax_profile.grid(True, alpha=0.3)
        if axis == 1:
            x = np.arange(disp.shape[0])
            self.ax_profile.plot(x, profile, linewidth=1.5)
            self.ax_profile.fill_between(x, profile, alpha=0.3)
            self.ax_profile.set_xlabel("Row (px)")
            self.ax_profile.set_ylabel("Integrated Intensity")
            self.ax_profile.set_xlim(0, disp.shape[0] - 1)
        else:
            x = np.arange(disp.shape[1])
            self.ax_profile.plot(x, profile, linewidth=1.5)
            self.ax_profile.fill_between(x, profile, alpha=0.3)
            self.ax_profile.set_xlabel("Column (px)")
            self.ax_profile.set_ylabel("Integrated Intensity")
            self.ax_profile.set_xlim(0, disp.shape[1] - 1)
        self.ax_profile.set_ylim(bottom=0)
        
        self._refresh_crosshair()
        self.canvas.draw_idle()
        
    def _capture_one(self, exp_us):
        if self.cam is None:
            raise AndorControllerError("Camera not activated.")
        if hasattr(self.cam, "capture_single"):
            return self.cam.capture_single(exp_us)
        return self.cam.take_image(exp_us, 1)
        
    def _get_save_directory(self):
        now = datetime.datetime.now()
        base_dir = _data_root()
        dir_path = Path(base_dir) / now.strftime("%Y-%m-%d") / CAMERA_NAME
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path, now
        
    def _generate_filename(self, timestamp, frame_index, is_background):
        stem = f"{CAMERA_NAME}_Background" if is_background else f"{CAMERA_NAME}_Image"
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        return f"{stem}_{ts_str}_{frame_index}.png"
        
    def _save_single_frame(self, frame, filepath, exposure_us, mcp_voltage, comment):
        frame_uint16 = np.clip(frame, 0, 65535).astype(np.uint16, copy=False)
        img = Image.fromarray(frame_uint16, mode="I;16")
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Exposure", str(exposure_us))
        metadata.add_text("MCP Voltage", mcp_voltage)
        metadata.add_text("Comment", comment)
        img.save(filepath, format="PNG", pnginfo=metadata)
        
    def _write_log_entry(self, log_path, filename, exposure_us, mcp_voltage, comment):
        header = "File Name\tExposure (µs)\tMCP Voltage\tComment\n"
        if not log_path.exists():
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(header)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{filename}\t{exposure_us}\t{mcp_voltage}\t{comment}\n")
            
    def save_frames(self):
        with self._frame_lock:
            if self.cam is None and self.last_frame is None:
                QMessageBox.warning(self, "Warning", "No frame available to save.")
                return
                
        try:
            exposure_us = int(self.exposure_edit.text())
            n_frames = int(self.frames_to_save_edit.text())
            if n_frames <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid 'Frames to Save' or 'Exposure' value.")
            return
            
        dir_path, base_timestamp = self._get_save_directory()
        mcp_voltage = self.mcp_voltage_edit.text()
        comment = self.comment_edit.text()
        is_background = self.background_checkbox.isChecked()
        
        saved_files = []
        for i in range(1, n_frames + 1):
            try:
                timestamp = datetime.datetime.now()
                with self._frame_lock:
                    frame = self._capture_one(exposure_us) if self.cam else self.last_frame
                if frame is None:
                    raise RuntimeError("No frame captured.")
                    
                filename = self._generate_filename(timestamp, i, is_background)
                filepath = dir_path / filename
                self._save_single_frame(frame, filepath, exposure_us, mcp_voltage, comment)
                saved_files.append(filename)
                self.log(f"Saved {filename}")
            except Exception as e:
                self.log(f"Error saving frame {i}: {e}")
                QMessageBox.critical(self, "Error", f"Error saving frame {i}: {e}")
                break
                
        if not saved_files:
            return
            
        log_filename = f"{CAMERA_NAME}_log_{base_timestamp.strftime('%Y-%m-%d')}.log"
        log_path = dir_path / log_filename
        try:
            for filename in saved_files:
                self._write_log_entry(log_path, filename, exposure_us, mcp_voltage, comment)
            self.log(f"Logged {len(saved_files)} file(s) to {log_path}")
        except Exception as e:
            self.log(f"Error writing log file: {e}")
            QMessageBox.critical(self, "Error", f"Error writing log file: {e}")
            
    def closeEvent(self, event):
        if self.capture_thread:
            self.stop_capture()
        if self.cam:
            self.deactivate_camera()
        self.closed.emit()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AndorLiveWindow()
    gui.show()
    sys.exit(app.exec_())