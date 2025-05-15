from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QGroupBox, QTextEdit
)
from hardware.wrappers.PressureSensor import GRAFANA_PATH, PressureMonitorWidget
from diagnostics.utils import LOG_DATE_FORMAT
import datetime


class DlabController(QMainWindow):
    """Main window for the Dlab Controller application."""
    def __init__(self):
        super().__init__()

        # Initialize instance variables.
        self.andor_live = None # Will hold the AndorLive instance
        self.daheng_live = {}  # Empty dictionary for the Daheng cameras.
        self.stage_control = None # Will hold the Thorlabs StageControl instance
        self.slm_view = None # Will hold the SLMView instance
        self.scan_panel = None  # Will hold the Scan Panel instance

        self.setup_ui() # Set up the UI
        
    def setup_ui(self):
        """
        Sets up the UI components.
        """
        # Set up the main window.
        self.setWindowTitle("DlabController")
        self.resize(800, 500)
        
        # Set up the main layout.
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        left_panel = QVBoxLayout()
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout()   
        
        # SLM button
        self.slm_button = QPushButton("Open SLM View")
        self.slm_button.clicked.connect(self.open_slm_view)
        view_layout.addWidget(self.slm_button)
        
        # Andor button
        self.andor_button = QPushButton("Open Andor Live")
        self.andor_button.clicked.connect(self.open_andor_live)
        view_layout.addWidget(self.andor_button)

        # Three separate Daheng buttons.
        self.daheng_nozzle_button = QPushButton("Open Daheng Live – Nomarski")
        self.daheng_nozzle_button.clicked.connect(lambda: self.open_daheng_live("Nomarski", 1))
        view_layout.addWidget(self.daheng_nozzle_button)

        self.daheng_focus_button = QPushButton("Open Daheng Live – Nozzle")
        self.daheng_focus_button.clicked.connect(lambda: self.open_daheng_live("Focus", 2))
        view_layout.addWidget(self.daheng_focus_button)

        self.daheng_focus_button = QPushButton("Open Daheng Live – Focus")
        self.daheng_focus_button.clicked.connect(lambda: self.open_daheng_live("Focus", 3))
        view_layout.addWidget(self.daheng_focus_button)

        # Thorlabs button.
        self.thorlabs_button = QPushButton("Open Thorlabs Control")
        self.thorlabs_button.clicked.connect(self.open_thorlabs_view)
        view_layout.addWidget(self.thorlabs_button)
        view_group.setLayout(view_layout)
        left_panel.addWidget(view_group)

        # Scan group box.
        scan_group = QGroupBox("Scan")
        scan_layout = QVBoxLayout()
        self.scan_panel_button = QPushButton("Open Scan Panel")
        self.scan_panel_button.clicked.connect(self.open_scan_panel)
        scan_layout.addWidget(self.scan_panel_button)
        scan_group.setLayout(scan_layout)
        left_panel.addWidget(scan_group)

        # Clickable Dashboard URL for Grafana.
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
            
        # Set up the pressure sensor.
        self.setup_pressure_log()   
            
    def append_log(self, message):
        """
        Appends a message to the log text box with a timestamp.
        """
        now = datetime.datetime.now().strftime(LOG_DATE_FORMAT)
        self.log_text.append(f"[{now}] {message}")
        
    def setup_pressure_log(self):
        """
        Sets up the pressure sensor and starts polling in a separate thread.
        """
        self.pressure_monitor = PressureMonitorWidget(self)
        self.pressure_monitor.log_signal.connect(self.append_log)

    def open_andor_live(self):
        """Opens the AndorLive window."""
        from diagnostics.view.AndorLive import AndorLive
        if self.andor_live is None:
            self.andor_live = AndorLive()
            self.andor_live.closed.connect(lambda: self.on_andor_closed())
        self.andor_live.show()
        self.andor_live.raise_()
        self.andor_live.activateWindow()
        self.append_log("AndorLive opened.")
    
    def on_andor_closed(self):
        """Handles the closing of the AndorLive window."""
        self.append_log("AndorLive closed.")
        self.andor_live = None

    def open_slm_view(self):
        """Opens the SLMView window."""
        from diagnostics.view.SLMView import SLMView
        if self.slm_view is None:
            self.slm_view = SLMView()
        self.slm_view.show()
        self.slm_view.raise_()
        self.slm_view.activateWindow()
        self.append_log("SLMView opened.")

    def open_thorlabs_view(self):
        """Opens the Thorlabs Control window."""
        from diagnostics.view.StageControl import StageControl
        if self.stage_control is None:
            self.stage_control = StageControl()
        self.stage_control.show()
        self.stage_control.raise_()
        self.stage_control.activateWindow()
        self.append_log("Thorlabs Control opened.")
        
    def open_daheng_live(self, camera_name, fixed_index):
        """
        Opens a DahengLive window with a fixed camera index and a unique camera name.
        """
        from diagnostics.view.DahengLive import DahengLive
        if camera_name not in self.daheng_live:
            self.daheng_live[camera_name] = DahengLive(camera_name=camera_name, fixed_index=fixed_index)
        self.daheng_live[camera_name].show()
        self.daheng_live[camera_name].raise_()
        self.daheng_live[camera_name].activateWindow()
        self.append_log(f"DahengLive opened for {camera_name} camera.")

    def open_scan_panel(self):
        """Opens the Scan Panel window."""
        from diagnostics.view.ScanPanel import ScanPanel
        if self.andor_live is None:
            self.open_andor_live()
        # Ensure all three Daheng cameras are available.
        if "Nozzle" not in self.daheng_live:
            self.open_daheng_live("Nozzle", 1)
        if "Focus" not in self.daheng_live:
            self.open_daheng_live("Focus", 2)
        if self.stage_control is None:
            self.open_thorlabs_view()

        # Pass all three Daheng camera instances to the ScanPanel.
        self.scan_panel = ScanPanel(
            self.andor_live,
            self.daheng_live["Focus"],
            self.daheng_live["Nozzle"], #Its reversed, to fix later
            self.stage_control.thorlabs_view
        )
        self.scan_panel.show()
        self.append_log("Scan Panel opened.")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = DlabController()
    window.show()
    sys.exit(app.exec_())
