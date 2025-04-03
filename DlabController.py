import sys
import time
import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QGroupBox, QTextEdit
)
from diagnostics.view.AndorLive import AndorLive
from diagnostics.view.DahengLive import DahengLive
from diagnostics.view.StageControl import StageControl
from diagnostics.view.SLMView import SLMView
from hardware.wrappers.PressureSensor import GRAFANA_PATH, PressureMonitorWidget

class DlabController(QMainWindow):
    """
    Central control application that provides group boxes for opening various GUIs
    and for running scans. The main window displays only the clickable dashboard URL
    and a log text box. The pressure polling runs in the background without showing its UI.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Central Control App")
        self.resize(800, 500)

        self.andor_live = None
        self.daheng_live = None
        self.stage_control = None
        self.slm_view = None

        # Instantiate the PressureMonitorWidget to start pressure polling in the background.
        self.pressure_monitor = PressureMonitorWidget(self)
        self.pressure_monitor.log_signal.connect(self.append_log)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QVBoxLayout()

        # "View" group box.
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout()
        self.slm_button = QPushButton("Open SLM View")
        self.slm_button.clicked.connect(self.open_slm_view)
        view_layout.addWidget(self.slm_button)
        self.andor_button = QPushButton("Open Andor Live")
        self.andor_button.clicked.connect(self.open_andor_live)
        view_layout.addWidget(self.andor_button)
        self.daheng_button = QPushButton("Open Daheng Live")
        self.daheng_button.clicked.connect(self.open_daheng_live)
        view_layout.addWidget(self.daheng_button)
        self.thorlabs_button = QPushButton("Open Thorlabs Control")
        self.thorlabs_button.clicked.connect(self.open_thorlabs_view)
        view_layout.addWidget(self.thorlabs_button)
        view_group.setLayout(view_layout)
        left_panel.addWidget(view_group)

        # "Scan" group box.
        scan_group = QGroupBox("Scan")
        scan_layout = QVBoxLayout()
        self.scan_button = QPushButton("Run Scan")
        self.scan_button.clicked.connect(self.run_scan)
        scan_layout.addWidget(self.scan_button)
        scan_group.setLayout(scan_layout)
        left_panel.addWidget(scan_group)

        # Clickable Dashboard URL.
        self.path_label = QLabel(
            f'Dashboard URL: <a href="{GRAFANA_PATH}">{GRAFANA_PATH}</a>'
        )
        self.path_label.setOpenExternalLinks(True)
        left_panel.addWidget(self.path_label)

        # Log text box.
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_panel.addWidget(self.log_text)

        left_panel.addStretch(1)
        main_layout.addLayout(left_panel)

    def append_log(self, message):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{current_time}] {message}")

    def open_andor_live(self):
        if self.andor_live is None:
            self.andor_live = AndorLive()
        self.andor_live.show()
        self.andor_live.raise_()
        self.andor_live.activateWindow()
        self.append_log("AndorLive GUI opened.")

    def open_slm_view(self):
        if self.slm_view is None:
            self.slm_view = SLMView()
        self.slm_view.show()
        self.slm_view.raise_()
        self.slm_view.activateWindow()
        self.append_log("SLMView GUI opened.")

    def open_daheng_live(self):
        if self.daheng_live is None:
            self.daheng_live = DahengLive()
        self.daheng_live.show()
        self.daheng_live.raise_()
        self.daheng_live.activateWindow()
        self.append_log("DahengLive GUI opened.")

    def open_thorlabs_view(self):
        if self.stage_control is None:
            self.stage_control = StageControl()
        self.stage_control.show()
        self.stage_control.raise_()
        self.stage_control.activateWindow()
        self.append_log("Thorlabs Control GUI opened.")

    def run_scan(self):
        self.append_log("Starting scan...")
        if self.daheng_live is None:
            self.open_daheng_live()
        if self.andor_live is None:
            self.open_andor_live()
        if self.daheng_live.camera_controller is None:
            self.append_log("Daheng camera not activated.")
            return
        try:
            exposure_d = int(self.daheng_live.exposure_edit.text())
            gain = int(self.daheng_live.gain_edit.text())
            avgs_d = int(self.daheng_live.avgs_edit.text())
            daheng_img = self.daheng_live.camera_controller.take_image(exposure_d, gain, avgs_d)
            self.append_log("Daheng image captured.")
        except Exception as e:
            self.append_log(f"Error capturing Daheng image: {e}")
            return

        time.sleep(1)

        if self.andor_live.camera_controller is None:
            self.append_log("Andor camera not activated.")
            return
        try:
            exposure_a = int(self.andor_live.exposure_edit.text())
            avgs_a = int(self.andor_live.avgs_edit.text())
            andor_img = self.andor_live.camera_controller.take_image(exposure_a, avgs_a)
            self.append_log("Andor image captured.")
        except Exception as e:
            self.append_log(f"Error capturing Andor image: {e}")
            return

        self.display_scan_images(daheng_img, andor_img)

    def display_scan_images(self, daheng_img, andor_img):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(daheng_img, cmap='viridis')
        ax1.set_title("Daheng Image")
        ax2.imshow(andor_img, cmap='viridis')
        ax2.set_title("Andor Image")
        plt.tight_layout()
        plt.show()
        self.append_log("Scan images displayed.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DlabController()
    window.show()
    sys.exit(app.exec_())