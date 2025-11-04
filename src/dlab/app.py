from __future__ import annotations
from pathlib import Path
import datetime
import sys
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QGroupBox, QTextEdit, QSpinBox
)

from dlab.boot import ROOT, bootstrap, get_config
from dlab.hardware.wrappers.pressure_sensor import PressureMonitorWidget
from dlab.diagnostics.utils import GUI_LOG_DATE_FORMAT

CFG = bootstrap(ROOT / "config" / "config.yaml")


class DlabControllerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.andor_window = None
        self.avaspec_window = None
        self.powermeter_window = None
        self.daheng_windows: dict[str, object] = {}
        self.stage_control_window = None
        self.slm_window = None
        self.scan_window = None
        self.smaract_proc: subprocess.Popen | None = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("DlabControllerWindow")
        self.resize(300, 300)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QVBoxLayout()
        windows_group = QGroupBox("Windows")
        view_layout = QVBoxLayout()

        self.slm_button = QPushButton("Open SLM Window")
        self.slm_button.clicked.connect(self.open_slm_window)
        view_layout.addWidget(self.slm_button)

        self.andor_button = QPushButton("Open Andor Window")
        self.andor_button.clicked.connect(self.open_andor_window)
        view_layout.addWidget(self.andor_button)

        self.avaspec_button = QPushButton("Open Avaspec Window")
        self.avaspec_button.clicked.connect(self.open_avaspec_window)
        view_layout.addWidget(self.avaspec_button)
        
        self.powermeter_button = QPushButton("Open Powermeter Window")
        self.powermeter_button.clicked.connect(self.open_powermeter_window)
        view_layout.addWidget(self.powermeter_button)
        
        self.thorlabs_button = QPushButton("Open Stage Control Window")
        self.thorlabs_button.clicked.connect(self.open_stage_control_window)
        view_layout.addWidget(self.thorlabs_button)

        self.smaract_button = QPushButton("Open SmarAct Control")
        self.smaract_button.clicked.connect(self.open_smaract_control)
        view_layout.addWidget(self.smaract_button)
        
        self.scan_button = QPushButton("Open Scan Window")
        self.scan_button.clicked.connect(self.open_scan_window)
        view_layout.addWidget(self.scan_button)

        self.camera_controls = {}
        default_indices = {"DahengCam_1": 1, "DahengCam_2": 2, "DahengCam_3": 3}
        for label in ["DahengCam_1", "DahengCam_2", "DahengCam_3"]:
            box = QGroupBox(f"{label}")
            layout = QHBoxLayout()

            spinbox = QSpinBox()
            spinbox.setRange(1, 5)
            spinbox.setValue(default_indices[label])
            layout.addWidget(QLabel("Index:"))
            layout.addWidget(spinbox)

            button = QPushButton("Open")
            layout.addWidget(button)
            button.clicked.connect(lambda _, name=label, sb=spinbox: self.open_daheng_window(name, sb.value()))

            box.setLayout(layout)
            view_layout.addWidget(box)
            self.camera_controls[label] = spinbox

        windows_group.setLayout(view_layout)
        left_panel.addWidget(windows_group)

        dashboard_url = "http://localhost:3000/d/ad6bbh8/pressure-dashboard?orgId=1&from=now-30m&to=now&timezone=browser&refresh=auto"
        self.path_label = QLabel(f'<a href="{dashboard_url}">Open Pressure Dashboard</a>')
        self.path_label.setOpenExternalLinks(True)
        left_panel.addWidget(self.path_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_panel.addWidget(self.log_text)

        left_panel.addStretch(1)
        main_layout.addLayout(left_panel)

        self.setup_pressure_log()

    def append_log(self, message: str):
        now = datetime.datetime.now().strftime(GUI_LOG_DATE_FORMAT)
        self.log_text.append(f"[{now}] {message}")

    def setup_pressure_log(self):
        self.pressure_monitor = PressureMonitorWidget(self)
        self.pressure_monitor.log_signal.connect(self.append_log)

    def open_andor_window(self):
        from dlab.diagnostics.ui.andor_live_window import AndorLiveWindow
        if self.andor_window is None:
            self.andor_window = AndorLiveWindow()
            self.andor_window.closed.connect(self.on_andor_window_closed)
            self.andor_window.show()
            self.andor_window.raise_()
            self.andor_window.activateWindow()
            self.append_log("Andor window opened.")
        else:
            self.append_log("Andor window is already open.")

    def on_andor_window_closed(self):
        self.andor_window = None
        self.append_log("Andor window closed.")

    def open_avaspec_window(self):
        from dlab.diagnostics.ui.avaspec_live_window import AvaspecLiveWindow
        if self.avaspec_window is None:
            self.avaspec_window = AvaspecLiveWindow()
            self.avaspec_window.closed.connect(self.on_avaspec_window_closed)
            self.avaspec_window.show()
            self.avaspec_window.raise_()
            self.avaspec_window.activateWindow()
            self.append_log("Avaspec window opened.")
        else:
            self.append_log("Avaspec window is already open.")

    def on_avaspec_window_closed(self):
        self.avaspec_window = None
        self.append_log("Avaspec window closed.")

    def open_slm_window(self):
        from dlab.diagnostics.ui.slm_window import SlmWindow
        if self.slm_window is None:
            self.slm_window = SlmWindow()
            self.slm_window.closed.connect(self.on_slm_window_closed)
            self.slm_window.show()
            self.slm_window.raise_()
            self.slm_window.activateWindow()
            self.append_log("SLM window opened.")
        else:
            self.append_log("SLM window is already open.")

    def on_slm_window_closed(self):
        self.slm_window = None
        self.append_log("SLM window closed.")

    def open_stage_control_window(self):
        from dlab.diagnostics.ui.stage_control_window import StageControlWindow
        if self.stage_control_window is None:
            self.stage_control_window = StageControlWindow()
        self.stage_control_window.show()
        self.stage_control_window.raise_()
        self.stage_control_window.activateWindow()
        self.append_log("Stage Control window opened.")

    def open_smaract_control(self):
        script = ROOT / "src" / "dlab" / "diagnostics" / "ui" / "run_smaract_gui.py"
        if not script.exists():
            self.append_log(f"SmarAct launcher not found: {script}")
            return
        if self.smaract_proc is not None and self.smaract_proc.poll() is None:
            self.append_log("SmarAct Control already running.")
            return
        try:
            self.smaract_proc = subprocess.Popen([sys.executable, str(script)], cwd=str(ROOT))
            self.append_log("SmarAct Control launched.")
        except Exception as e:
            self.append_log(f"Failed to launch SmarAct Control: {e}")

    def open_daheng_window(self, camera_name: str, fixed_index: int):
        from dlab.diagnostics.ui.daheng_live_window import DahengLiveWindow
        if camera_name not in self.daheng_windows:
            win = DahengLiveWindow(camera_name=camera_name, fixed_index=fixed_index)
            win.closed.connect(lambda name=camera_name: self.on_daheng_window_closed(name))
            self.daheng_windows[camera_name] = win
            win.show()
            win.raise_()
            win.activateWindow()
            self.append_log(f"Daheng window opened for '{camera_name}'.")
        else:
            self.append_log(f"Daheng window for '{camera_name}' is already open.")

    def on_daheng_window_closed(self, camera_name: str):
        if camera_name in self.daheng_windows:
            del self.daheng_windows[camera_name]
        self.append_log(f"Daheng window closed for '{camera_name}'.")

    def open_scan_window(self):
        from dlab.diagnostics.ui.scans.scan_window import ScanWindow
        if self.scan_window is None:
            self.scan_window = ScanWindow()
            self.scan_window.closed.connect(self.on_scan_window_closed)
        self.scan_window.show()
        self.scan_window.raise_()
        self.scan_window.activateWindow()
        self.append_log("Scan window opened.")

    def on_scan_window_closed(self):
        self.scan_window = None
        self.append_log("Scan window closed.")

    def open_powermeter_window(self):
        from dlab.diagnostics.ui.powermeter_live_window import PowermeterLiveWindow
        if self.powermeter_window is None:
            self.powermeter_window = PowermeterLiveWindow()
            self.powermeter_window.closed.connect(self.on_powermeter_window_closed)
        self.powermeter_window.show()
        self.powermeter_window.raise_()
        self.powermeter_window.activateWindow()
        self.append_log("Powermeter window opened.")

    def on_powermeter_window_closed(self):
        self.powermeter_window = None
        self.append_log("Powermeter window closed.")


def main():
    app = QApplication(sys.argv)
    window = DlabControllerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
