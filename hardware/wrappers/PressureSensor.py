import sys
import time
import datetime
import os
import numpy as np

from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QGroupBox, QTextEdit
)
from prometheus_client import start_http_server, Gauge
from hardware.wrappers.PfeifferVacuum import MaxiGauge, MaxiGaugeError

# Prometheus gauges
pressure_gauge_1 = Gauge('sensor_pressure_1', 'Grating and MCP')
pressure_gauge_2 = Gauge('sensor_pressure_2', 'Generation')
pressure_gauge_3 = Gauge('sensor_pressure_3', 'Gas Catcher')

GRAFANA_PATH = 'https://melvinrdn.grafana.net/public-dashboards/e5426f0a3aad4370ab8bb64b3c76a8cc'
COM_PORT = 'COM9'
UPDATE_INTERVAL = 5  # seconds

mg = None

def initialize_maxigauge():
    global mg
    try:
        mg = MaxiGauge(COM_PORT)
    except Exception as e:
        mg = None

class PressureWorker(QObject):
    """
    Worker that continuously polls the MaxiGauge for pressure values.
    It emits signals when new data is available or an error occurs.
    """
    pressure_updated = pyqtSignal(float, float, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            global mg
            if mg is None:
                self.error_occurred.emit("Pressure: Not initialized")
                time.sleep(UPDATE_INTERVAL)
                continue
            try:
                pressures = mg.pressures()
                p1 = pressures[0].pressure if pressures[0].status in [0, 1, 2] else float('nan')
                p2 = pressures[1].pressure if pressures[1].status in [0, 1, 2] else float('nan')
                p3 = pressures[2].pressure if pressures[2].status in [0, 1, 2] else float('nan')
                pressure_gauge_1.set(p1)
                pressure_gauge_2.set(p2)
                pressure_gauge_3.set(p3)
                # (Not used in the GUI; we only log errors.)
                self.pressure_updated.emit(p1, p2, p3)
            except MaxiGaugeError:
                self.error_occurred.emit("Pressure: Error")
            time.sleep(UPDATE_INTERVAL)

class PressureMonitorWidget(QObject):
    """
    A non-visual QObject that starts the pressure polling in the background
    and emits log messages (e.g., when starting the Prometheus server).
    """
    log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        initialize_maxigauge()
        # Delay the Prometheus startup to ensure any connections to log_signal are in place.
        QTimer.singleShot(0, self.start_prometheus)
        self.worker = PressureWorker()
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.error_occurred.connect(lambda msg: self.log_signal.emit(msg))
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def start_prometheus(self):
        if mg is not None:
            try:
                start_http_server(8000)
                self.log_signal.emit("Prometheus metrics server running on http://localhost:8000")
            except Exception as e:
                self.log_signal.emit(f"Failed to start Prometheus server: {e}")
        else:
            self.log_signal.emit("Failed to initialize MaxiGauge.")

    def stop(self):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()