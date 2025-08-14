from __future__ import annotations
import json
import datetime
from pathlib import Path
import numpy as np
import logging
import yaml  

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from dlab.boot import ROOT, get_config
from dlab.hardware.wrappers.phase_settings import PhaseSettings
from dlab.hardware.wrappers.slm_controller import SLMController

logger = logging.getLogger("dlab.ui.SlmWindow")

# ---------- Helpers chemins / config ----------

def _ressources_root() -> Path:
    cfg = get_config() or {}
    rel = (cfg.get("paths", {}) or {}).get("ressources", "ressources")
    return (ROOT / rel).resolve()

def _defaults_yaml_path() -> Path:
    """Chemin du fichier d'override (modifiable par l'app)."""
    cfg = get_config() or {}
    rel = (cfg.get("slm", {}) or {}).get("defaults_file", "ressources/saved_settings/slm_defaults.yaml")
    return (ROOT / rel).resolve()

def _default_from_config(color: str) -> Path:
    """Chemin par défaut défini dans config.yaml (fallback)."""
    cfg = get_config() or {}
    key = "red_default" if color == "red" else "green_default"
    rel = (cfg.get("slm", {}) or {}).get(key)
    if not rel:
        # fallback si la clé n'existe pas
        rel = f"ressources/saved_settings/SLM_{color}/{color}_default_settings.txt"
    return (ROOT / rel).resolve()

def _saved_dir(color: str) -> Path:
    return _ressources_root() / "saved_settings" / f"SLM_{color}"

def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True, allow_unicode=True)


class SlmWindow(QtWidgets.QMainWindow):
    closed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SlmWindow")
        self.setMinimumSize(700, 900)
        self.setAttribute(Qt.WA_DeleteOnClose)

        # Contrôleurs SLM
        self.SLM_red = SLMController("red")
        self.SLM_green = SLMController("green")

        self.slm_red_status = "closed"
        self.slm_green_status = "closed"

        self.initUI()

        self.update_log("Loading the default parameters...")
        for color in ["red", "green"]:
            self.load_default_parameters(color)

    def initUI(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        self.main_layout = QtWidgets.QVBoxLayout(central)

        self.createMenuBar()

        self.slm_tabs = QtWidgets.QTabWidget()
        panel_red = self.create_slm_panel("red")
        panel_green = self.create_slm_panel("green")
        self.slm_tabs.addTab(panel_red, "Red SLM")
        self.slm_tabs.addTab(panel_green, "Green SLM")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.slm_tabs)

        self.logText = QtWidgets.QTextEdit()
        self.logText.setReadOnly(True)
        self.logText.setFixedHeight(120)
        splitter.addWidget(self.logText)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.main_layout.addWidget(splitter)
        self.statusBar().showMessage("Red SLM: closed | Green SLM: closed")

    def update_status_bar(self):
        msg = f"Red SLM: {self.slm_red_status} | Green SLM: {self.slm_green_status}"
        self.statusBar().showMessage(msg)
        if hasattr(self, "status_label_red"):
            self.status_label_red.setText(f"Status: {self.slm_red_status}")
            self.status_label_red.setStyleSheet(
                "background-color: lightgreen;" if "displaying" in self.slm_red_status else "background-color: lightgray;"
            )
        if hasattr(self, "status_label_green"):
            self.status_label_green.setText(f"Status: {self.slm_green_status}")
            self.status_label_green.setStyleSheet(
                "background-color: lightgreen;" if "displaying" in self.slm_green_status else "background-color: lightgray;"
            )

    def createMenuBar(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")
        exitAction = QtWidgets.QAction("Exit", self)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

    def create_slm_panel(self, color: str):
        panel = QtWidgets.QGroupBox(f"{color.capitalize()} SLM Interface")
        layout = QtWidgets.QVBoxLayout(panel)

        top_group = QtWidgets.QGroupBox(f"{color.capitalize()} SLM - Phase Display")
        top_layout = QtWidgets.QVBoxLayout(top_group)

        btn_save = QtWidgets.QPushButton(f"Save {color} settings")
        btn_load = QtWidgets.QPushButton(f"Load {color} settings")
        btn_save.clicked.connect(lambda: self.save_settings(color))
        btn_load.clicked.connect(lambda: self.load_settings(color))
        hlayout_save = QtWidgets.QHBoxLayout()
        hlayout_save.addWidget(btn_load)
        hlayout_save.addWidget(btn_save)
        top_layout.addLayout(hlayout_save)

        # Display selection
        h_layout_display = QtWidgets.QHBoxLayout()
        label_display = QtWidgets.QLabel("Display number:")
        screens = QtWidgets.QApplication.instance().screens()
        num_screens = len(screens) if screens else 1
        spin_display = QtWidgets.QSpinBox()
        spin_display.setRange(1, num_screens)
        spin_display.setValue(2 if color == "green" and num_screens >= 2 else 1)
        setattr(self, f"spin_{color}", spin_display)
        h_layout_display.addWidget(label_display)
        h_layout_display.addWidget(spin_display)
        top_layout.addLayout(h_layout_display)

        # Status indicator
        status_label = QtWidgets.QLabel("Status: closed")
        status_label.setAlignment(QtCore.Qt.AlignCenter)
        status_label.setStyleSheet("background-color: lightgray;")
        setattr(self, f"status_label_{color}", status_label)
        top_layout.addWidget(status_label)

        # Matplotlib preview (in radians)
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        slm = getattr(self, f"SLM_{color}")
        h, w = slm.slm_size
        extent = (-w / 2, w / 2, -h / 2, h / 2)
        phase_image = ax.imshow(np.zeros(slm.slm_size), cmap="hsv", vmin=0, vmax=2 * np.pi, extent=extent)
        cbar = fig.colorbar(phase_image, ax=ax, orientation="horizontal", fraction=0.07, pad=0.03)
        cbar.set_ticks([0, np.pi, 2 * np.pi])
        cbar.set_ticklabels(["0", "π", "2π"])
        ax.set_xticks([])
        ax.set_yticks([])
        canvas = FigureCanvas(fig)
        setattr(self, f"fig_{color}", fig)
        setattr(self, f"ax_{color}", ax)
        setattr(self, f"phase_image_{color}", phase_image)
        setattr(self, f"canvas_{color}", canvas)
        top_layout.addWidget(canvas)

        # Phase types checkboxes + params
        check_group = QtWidgets.QGroupBox("Phases enabled")
        check_layout = QtWidgets.QVBoxLayout(check_group)
        checkboxes = []
        for typ in PhaseSettings.types:
            cb = QtWidgets.QCheckBox(typ)
            cb.setChecked(False)
            check_layout.addWidget(cb)
            checkboxes.append(cb)
        tab_widget = QtWidgets.QTabWidget()
        phase_refs = []
        for typ in PhaseSettings.types:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)
            phase_ref = PhaseSettings.new_type(tab, typ)
            tab_layout.addWidget(phase_ref)
            tab_widget.addTab(tab, typ)
            phase_refs.append(phase_ref)
        setattr(self, f"checkboxes_{color}", checkboxes)
        setattr(self, f"phase_refs_{color}", phase_refs)
        setattr(self, f"tab_widget_{color}", tab_widget)
        top_tab_layout = QtWidgets.QHBoxLayout()
        top_tab_layout.addWidget(check_group)
        top_tab_layout.addWidget(tab_widget)
        top_layout.addLayout(top_tab_layout)

        # Action buttons
        bottom_layout = QtWidgets.QHBoxLayout()
        btn_preview = QtWidgets.QPushButton(f"Preview {color}")
        btn_publish = QtWidgets.QPushButton(f"Publish {color}")
        btn_close = QtWidgets.QPushButton(f"Close {color}")
        btn_preview.clicked.connect(lambda: self.get_phase(color))
        btn_publish.clicked.connect(lambda: self.open_publish_win(color))
        btn_close.clicked.connect(lambda: self.close_publish_win(color))
        bottom_layout.addWidget(btn_preview)
        bottom_layout.addWidget(btn_publish)
        bottom_layout.addWidget(btn_close)
        layout.addLayout(bottom_layout)

        layout.addWidget(top_group)
        return panel

    # ---------- Helpers ----------
    @staticmethod
    def _levels_to_radians(levels: np.ndarray, bit_depth: int) -> np.ndarray:
        # map [0..bit_depth] → [0..2π]
        return (levels.astype(np.float64) * (2.0 * np.pi / bit_depth))

    # ---------- Logic ----------
    def get_phase(self, color: str):
        """Compute phases and update the preview; return phase types used for publishing."""
        self.update_log(f"Preview requested for {color} SLM.")
        slm: SLMController = getattr(self, f"SLM_{color}")
        phase_refs = getattr(self, f"phase_refs_{color}")
        checkboxes = getattr(self, f"checkboxes_{color}")

        total_levels = np.zeros(slm.slm_size, dtype=np.uint16)
        preview_levels = np.zeros(slm.slm_size, dtype=np.uint16)
        publish_types, preview_types = [], []

        for cb, phase_ref in zip(checkboxes, phase_refs):
            if cb.isChecked():
                # IMPORTANT: assume phase_ref.phase() returns *device levels* (0..bit_depth).
                levels = phase_ref.phase()
                total_levels = (total_levels + levels) % (slm.bit_depth + 1)
                publish_types.append(phase_ref.name_())
                if "background" not in phase_ref.name_().lower():
                    preview_levels = (preview_levels + levels) % (slm.bit_depth + 1)
                    preview_types.append(phase_ref.name_())

        slm.phase = total_levels
        display_phase = self._levels_to_radians(preview_levels, slm.bit_depth)

        phase_image = getattr(self, f"phase_image_{color}")
        phase_image.set_data(display_phase)
        canvas = getattr(self, f"canvas_{color}")
        canvas.draw()
        self.update_log(f"Preview updated for {color} SLM. Types: {', '.join(preview_types)}")
        return publish_types

    def open_publish_win(self, color: str):
        self.update_log(f"Publish requested for {color} SLM.")
        slm: SLMController = getattr(self, f"SLM_{color}")
        spin = getattr(self, f"spin_{color}")
        screen_num = spin.value()

        # screen lock check
        if color == "red":
            if self.slm_green_status != "closed" and f"Screen {screen_num}" in self.slm_green_status:
                QtWidgets.QMessageBox.warning(self, "Error", f"Screen {screen_num} is already in use by Green SLM.")
                self.update_log(f"Error: Cannot publish red SLM on screen {screen_num} because it is already in use by Green SLM.")
                return
        else:
            if self.slm_red_status != "closed" and f"Screen {screen_num}" in self.slm_red_status:
                QtWidgets.QMessageBox.warning(self, "Error", f"Screen {screen_num} is already in use by Red SLM.")
                self.update_log(f"Error: Cannot publish green SLM on screen {screen_num} because it is already in use by Red SLM.")
                return

        publish_types = self.get_phase(color)
        if slm.phase is None or np.all(slm.phase == 0):
            QtWidgets.QMessageBox.warning(self, "Error", f"No background image provided for {color} SLM. Please provide a background image.")
            self.update_log(f"Error: No background image provided for {color} SLM. Please provide a background image.")
            return

        slm.publish(slm.phase, screen_num)
        self.update_log(f"Published {color} SLM phase on screen {screen_num}. Types: {', '.join(publish_types)}")
        if color == "red":
            self.slm_red_status = f"displaying (Screen {screen_num})"
        else:
            self.slm_green_status = f"displaying (Screen {screen_num})"
        self.update_status_bar()

    def close_publish_win(self, color: str):
        self.update_log(f"Close requested for {color} SLM.")
        slm: SLMController = getattr(self, f"SLM_{color}")
        slm.close()
        slm.phase = np.zeros(slm.slm_size, dtype=np.uint16)
        self.update_log(f"Closed {color} SLM connection.")
        if color == "red":
            self.slm_red_status = "closed"
        else:
            self.slm_green_status = "closed"
        self.update_status_bar()

    def save_settings(self, color: str):
        dlg = QtWidgets.QFileDialog(self)
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setNameFilter("Text Files (*.txt);;All Files (*)")
        dlg.setDirectory(str(_saved_dir(color)))
        if not dlg.exec_():
            return
        filepath = Path(dlg.selectedFiles()[0])

        settings = {}
        phase_refs = getattr(self, f"phase_refs_{color}")
        checkboxes = getattr(self, f"checkboxes_{color}")
        spin = getattr(self, f"spin_{color}")
        for phase_ref, cb in zip(phase_refs, checkboxes):
            settings[phase_ref.name_()] = {"Enabled": cb.isChecked(), "Params": phase_ref.save_()}
        settings["screen_pos"] = spin.value()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(settings, f)
        self.update_default_path(filepath, color)

    def load_settings(self, color: str, filepath: Path | None = None):
        dlg = QtWidgets.QFileDialog(self)
        dlg.setNameFilter("Text Files (*.txt);;All Files (*)")
        dlg.setDirectory(str(_saved_dir(color)))
        if filepath is None:
            if not dlg.exec_():
                return
            filepath = Path(dlg.selectedFiles()[0])

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            phase_refs = getattr(self, f"phase_refs_{color}")
            checkboxes = getattr(self, f"checkboxes_{color}")
            spin = getattr(self, f"spin_{color}")

            for key, phase_ref, cb in zip(data.keys(), phase_refs, checkboxes):
                if key != "screen_pos" and key in data:
                    phase_data = data[key]
                    phase_ref.load_(phase_data["Params"])
                    cb.setChecked(phase_data["Enabled"])
            if "screen_pos" in data:
                spin.setValue(data["screen_pos"])
            self.update_log(f"{color.capitalize()} settings loaded successfully")
        except Exception as e:
            self.update_log(f"Error loading settings for {color}: {e}")

    def update_default_path(self, path: Path, color: str):
        """
        Écrit l'override dans slm_defaults.yaml (relatif à ressources/ si possible).
        """
        base = _ressources_root()
        try:
            rel_path = path.resolve().relative_to(base).as_posix()
        except ValueError:
            rel_path = str(path.resolve())  # hors de ressources/

        overrides_path = _defaults_yaml_path()
        data = _read_yaml(overrides_path)
        data[f"{color}_default_path"] = rel_path
        try:
            _write_yaml(overrides_path, data)
            self.update_log(f"Default {color} settings path updated to {rel_path}")
        except Exception as e:
            self.update_log(f"Error updating default settings path for {color}: {e}")

    def load_default_parameters(self, color: str):
        """
        Charge d'abord l'override YAML, sinon fallback sur config.yaml.
        """
        try:
            overrides_path = _defaults_yaml_path()
            overrides = _read_yaml(overrides_path)
            key = f"{color}_default_path"

            if key in overrides and overrides[key]:
                rel_or_abs = overrides[key]
                base = _ressources_root()
                filepath = (base / rel_or_abs).resolve() if not Path(rel_or_abs).is_absolute() else Path(rel_or_abs)
                self.update_log(f"Loading default {color} settings from override: {rel_or_abs}")
            else:
                filepath = _default_from_config(color)
                self.update_log(f"Loading default {color} settings from config.yaml")

            self.load_settings(color, filepath)
        except Exception as e:
            self.update_log(f"Error loading default {color} settings: {e}")

    def update_log(self, message: str):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.logText.append(f"[{current_time}] {message}")
        logger.info(message)

    def closeEvent(self, event):
        try:
            self.SLM_red.close()
            self.SLM_green.close()
        except Exception as e:
            self.update_log(f"Error during shutdown: {e}")

        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = SlmWindow()
    window.show()
    sys.exit(app.exec_())
