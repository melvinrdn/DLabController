from __future__ import annotations
import time
import logging
from typing import Optional

from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QObject, QCoreApplication
from prometheus_client import start_http_server, Gauge, CollectorRegistry

from dlab.boot import get_config
from dlab.hardware.wrappers.pfeiffer_vacuum import MaxiGauge, MaxiGaugeError

log = logging.getLogger(__name__)


def _cfg(path: str, default=None):
    """Accès simple à la config YAML via dlab.boot.get_config()."""
    cfg = get_config() or {}
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def initialize_maxigauge(port: str) -> Optional[MaxiGauge]:
    """Essaie d'ouvrir le MaxiGauge sur `port`."""
    try:
        mg = MaxiGauge(port)
        log.info("MaxiGauge connected on %s", port)
        return mg
    except Exception as e:
        log.exception("MaxiGauge init failed on %s: %s", port, e)
        return None


class PressureWorker(QObject):
    """Thread worker: lit le MaxiGauge et met à jour les métriques Prometheus."""
    pressure_updated = pyqtSignal(float, float, float)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        mg: Optional[MaxiGauge],
        interval_s: float,
        registry: CollectorRegistry,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._running = True
        self._mg = mg
        self._interval_s = interval_s
        # Un seul Gauge avec label 'sensor' → évite collisions si recréé
        self._gauge = Gauge(
            "sensor_pressure",
            "Pressure reading by sensor",
            ["sensor"],
            registry=registry,
        )

    @property
    def has_device(self) -> bool:
        return self._mg is not None

    def stop(self) -> None:
        self._running = False

    def _read_triplet(self) -> tuple[float, float, float]:
        assert self._mg is not None
        p = self._mg.pressures()
        def val(i: int) -> float:
            return p[i].pressure if p[i].status in (0, 1, 2) else float("nan")
        return val(0), val(1), val(2)

    def run(self) -> None:
        if self._mg is None:
            self.error_occurred.emit("Pressure: Not initialized")
            return

        while self._running:
            try:
                p1, p2, p3 = self._read_triplet()
                self._gauge.labels(sensor="1").set(p1)
                self._gauge.labels(sensor="2").set(p2)
                self._gauge.labels(sensor="3").set(p3)
                self.pressure_updated.emit(p1, p2, p3)
            except MaxiGaugeError as e:
                msg = f"Pressure: MaxiGauge error ({e})"
                log.warning(msg)
                self.error_occurred.emit(msg)
                # tentative de reconnexion simple
                try:
                    if hasattr(self._mg, "close"):
                        self._mg.close()
                except Exception:
                    pass
                time.sleep(2.0)
                port = _cfg("ports.pfeiffer", "COM9")
                self._mg = initialize_maxigauge(port)
            except Exception as e:
                msg = f"Pressure: Unexpected error ({e})"
                log.exception(msg)
                self.error_occurred.emit(msg)
            finally:
                time.sleep(self._interval_s)

        # fermeture propre
        try:
            if self._mg and hasattr(self._mg, "close"):
                self._mg.close()
        except Exception:
            pass


class PressureMonitorWidget(QObject):
    """Gère le serveur HTTP Prometheus et le thread de polling."""
    log_signal = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        if not _cfg("devices.pressure", True):
            self.log_signal.emit("Pressure device disabled by config.")
            self.worker = None
            self.thread = None
            return

        port = _cfg("ports.pfeiffer", "COM9")
        interval_s = float(_cfg("runtime.pressure_update_interval_s", 5.0))
        self._prom_port = int(_cfg("metrics.prometheus_port", 8000))
        self._grafana_url = _cfg("metrics.grafana_url", "http://localhost:3000")

        # Registry dédié pour éviter collisions si recréé
        self._registry = CollectorRegistry()

        self._mg = initialize_maxigauge(port)
        self.worker = PressureWorker(self._mg, interval_s, registry=self._registry)
        self.worker.error_occurred.connect(self.log_signal)

        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

        # démarre HTTP après init
        QTimer.singleShot(0, self._start_http_if_ready)

        # arrêt propre à la fermeture app
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    @property
    def grafana_url(self) -> str:
        return self._grafana_url

    def _start_http_if_ready(self) -> None:
        if not self.worker or not self.worker.has_device:
            self.log_signal.emit("Skipping Prometheus: MaxiGauge not initialized.")
            return
        try:
            start_http_server(self._prom_port, registry=self._registry)
            self.log_signal.emit(f"Metrics at http://localhost:{self._prom_port}")
            self.log_signal.emit(f"Grafana dashboard: {self._grafana_url}")
        except Exception as e:
            self.log_signal.emit(f"Failed to start metrics server: {e}")

    def shutdown(self) -> None:
        """Stop thread + close device."""
        try:
            if self.worker:
                self.worker.stop()
        except Exception:
            pass
        try:
            if self.thread:
                self.thread.quit()
                self.thread.wait(3000)
        except Exception:
            pass

    def __del__(self):
        self.shutdown()
