import subprocess
import time
from typing import Optional

from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QObject
from prometheus_client import start_http_server, Gauge
from hardware.wrappers.PfeifferVacuum import MaxiGauge, MaxiGaugeError

COM_PORT: str = 'COM9'
UPDATE_INTERVAL: float = 5.0
PROMETHEUS_EXEC: str = r'hardware/prometheus/prometheus.exe'
PROMETHEUS_CONFIG: str = r'hardware/prometheus/prometheus.yml'
PROMETHEUS_PORT: int = 8000
GRAFANA_PATH: str = 'http://localhost:3000/goto/FsiuGkbHg?orgId=1'

def initialize_maxigauge(port: str) -> Optional[MaxiGauge]:
    """
    Try to open the MaxiGauge on `port`.

    Parameters
    ----------
    port : str
        Serial port to connect to.

    Returns
    -------
    MaxiGauge or None
        Connected device or None on failure.
    """
    try:
        return MaxiGauge(port)
    except Exception:
        return None


class PressureWorker(QObject):
    """
    Polls the MaxiGauge and updates Prometheus gauges.

    Signals
    -------
    pressure_updated : float, float, float
        Latest (p1, p2, p3) readings.
    error_occurred : str
        Error messages.
    """
    pressure_updated = pyqtSignal(float, float, float)
    error_occurred  = pyqtSignal(str)

    def __init__(self, mg: Optional[MaxiGauge], parent: QObject = None) -> None:
        super().__init__(parent)
        self._running = True
        self._mg = mg
        self._g1 = Gauge('sensor_pressure_1', 'Grating and MCP')
        self._g2 = Gauge('sensor_pressure_2', 'Generation')
        self._g3 = Gauge('sensor_pressure_3', 'Gas Catcher')

    def stop(self) -> None:
        """
        Stop the polling loop.
        """
        self._running = False

    def run(self) -> None:
        """
        Main loop: read pressures, set gauges, emit signals.
        """
        if self._mg is None:
            self.error_occurred.emit("Pressure: Not initialized")
            return

        while self._running:
            try:
                p = self._mg.pressures()
                p1 = p[0].pressure if p[0].status in (0, 1, 2) else float('nan')
                p2 = p[1].pressure if p[1].status in (0, 1, 2) else float('nan')
                p3 = p[2].pressure if p[2].status in (0, 1, 2) else float('nan')
                self._g1.set(p1)
                self._g2.set(p2)
                self._g3.set(p3)
                self.pressure_updated.emit(p1, p2, p3)
            except MaxiGaugeError as e:
                self.error_occurred.emit(f"Pressure: Error ({e})")
            time.sleep(UPDATE_INTERVAL)


class PressureMonitorWidget(QObject):
    """
    Manages Prometheus HTTP server, external process, and polling.

    Signals
    -------
    log_signal : str
        Status and error messages.
    """
    log_signal = pyqtSignal(str)

    def __init__(self, parent: QObject = None) -> None:
        super().__init__(parent)
        mg = initialize_maxigauge(COM_PORT)

        QTimer.singleShot(0, self.start_prometheus)

        self.worker = PressureWorker(mg)
        self.worker.error_occurred.connect(self.log_signal)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def start_prometheus(self) -> None:
        """
        Start metrics HTTP server and launch prometheus.exe.
        """
        if self.worker._mg is None:
            self.log_signal.emit("Skipping Prometheus: MaxiGauge not initialized.")
            return

        try:
            start_http_server(PROMETHEUS_PORT)
            self.log_signal.emit(f"Metrics at http://localhost:{PROMETHEUS_PORT}")
        except Exception as e:
            self.log_signal.emit(f"Failed to start HTTP server: {e}")

        self.log_signal.emit(f"You can now launch prometheus.exe")

    def __del__(self):
        """
        Cleanup thread and Prometheus process.
        """
        if hasattr(self, 'worker'):
            self.worker.stop()
        if hasattr(self, 'thread'):
            self.thread.quit()
            self.thread.wait()
