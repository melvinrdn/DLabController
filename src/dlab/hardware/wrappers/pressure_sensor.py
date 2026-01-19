from __future__ import annotations

import time

from PyQt5.QtCore import QThread, pyqtSignal, QObject, QCoreApplication
from prometheus_client import start_http_server, Gauge, CollectorRegistry

from dlab.utils.config_utils import cfg_get
from dlab.hardware.wrappers.pfeiffer_vacuum import MaxiGauge, MaxiGaugeError
from dlab.utils.log_panel import LogPanel


def _initialize_maxigauge(port: str) -> MaxiGauge | None:
    """Attempt to connect to MaxiGauge on given port."""
    try:
        return MaxiGauge(port)
    except Exception:
        return None


class PressureWorker(QObject):
    """Background worker that polls pressure readings."""

    pressure_updated = pyqtSignal(float, float, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, mg: MaxiGauge | None, interval_s: float, gauge: Gauge, parent=None):
        super().__init__(parent)
        self._running = True
        self._mg = mg
        self._interval_s = interval_s
        self._gauge = gauge

    @property
    def has_device(self) -> bool:
        return self._mg is not None

    def stop(self):
        self._running = False

    def _read_triplet(self) -> tuple[float, float, float]:
        p = self._mg.pressures()

        def val(i):
            return p[i].pressure if p[i].status in (0, 1, 2) else float("nan")

        return val(0), val(1), val(2)

    def run(self):
        if self._mg is None:
            self.error_occurred.emit("Not initialized")
            return

        while self._running:
            try:
                p1, p2, p3 = self._read_triplet()
                self._gauge.labels(sensor="1").set(p1)
                self._gauge.labels(sensor="2").set(p2)
                self._gauge.labels(sensor="3").set(p3)
                self.pressure_updated.emit(p1, p2, p3)
            except MaxiGaugeError as e:
                self.error_occurred.emit(f"MaxiGauge error ({e})")
                self._reconnect()
            except Exception as e:
                self.error_occurred.emit(f"Unexpected error ({e})")
                time.sleep(1.0)
                continue

            time.sleep(self._interval_s)

        self._close()

    def _reconnect(self):
        """Attempt to reconnect to MaxiGauge."""
        try:
            if hasattr(self._mg, "close"):
                self._mg.close()
        except Exception:
            pass

        time.sleep(2.0)
        port = cfg_get("pressure_sensor.port", "COM3")
        self._mg = _initialize_maxigauge(port)

        if self._mg is None:
            time.sleep(1.0)

    def _close(self):
        """Close MaxiGauge connection."""
        try:
            if self._mg and hasattr(self._mg, "close"):
                self._mg.close()
        except Exception:
            pass


class PressureMonitorWidget(QObject):
    """Pressure monitor with Prometheus metrics export."""

    log_signal = pyqtSignal(str)

    def __init__(self, parent=None, log_panel: LogPanel | None = None):
        super().__init__(parent)
        self._log = log_panel

        # Config
        port = cfg_get("pressure_sensor.port", "COM3")
        interval_s = float(cfg_get("pressure_sensor.pressure_update_interval_s", 5.0))
        prom_port = int(cfg_get("pressure_sensor.prometheus_port", 8000))
        self._grafana_url = cfg_get("pressure_sensor.grafana_url", "http://localhost:3000")

        # Prometheus
        self._registry = CollectorRegistry()
        self._gauge = Gauge(
            "sensor_pressure",
            "Pressure reading by sensor (mbar)",
            ["sensor"],
            registry=self._registry,
        )
        self._start_metrics_server(prom_port)

        # Worker thread
        self._mg = _initialize_maxigauge(port)
        self._worker = PressureWorker(self._mg, interval_s, self._gauge)
        self._worker.error_occurred.connect(self._on_error)

        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    @property
    def grafana_url(self) -> str:
        return self._grafana_url

    def _start_metrics_server(self, port: int):
        """Start Prometheus HTTP metrics server."""
        try:
            start_http_server(port, addr="0.0.0.0", registry=self._registry)
            self._log_message(f"Metrics at http://127.0.0.1:{port}/metrics")
            self._log_message(f"Grafana dashboard: {self._grafana_url}")
            self._log_message(f"user:admin password:admin")
        except Exception as e:
            self._log_message(f"Failed to start metrics server: {e}")

    def _on_error(self, message: str):
        """Handle worker error signals."""
        self._log_message(message)
        self.log_signal.emit(message)

    def _log_message(self, message: str):
        """Log to panel if available."""
        if self._log:
            self._log.log(message, source="Pressure")

    def shutdown(self):
        """Stop worker and thread."""
        if getattr(self, "_worker", None):
            self._worker.stop()
        if getattr(self, "_thread", None):
            self._thread.quit()
            self._thread.wait(3000)

    def __del__(self):
        self.shutdown()