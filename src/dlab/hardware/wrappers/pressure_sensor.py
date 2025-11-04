import time
import logging
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QObject, QCoreApplication
from prometheus_client import start_http_server, Gauge, CollectorRegistry
from dlab.boot import get_config
from dlab.hardware.wrappers.pfeiffer_vacuum import MaxiGauge, MaxiGaugeError

log = logging.getLogger(__name__)

def _cfg(path, default=None):
    cfg = get_config() or {}
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def initialize_maxigauge(port):
    try:
        mg = MaxiGauge(port)
        log.info("MaxiGauge connected on %s", port)
        return mg
    except Exception as e:
        log.exception("MaxiGauge init failed on %s: %s", port, e)
        return None

class PressureWorker(QObject):
    pressure_updated = pyqtSignal(float, float, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, mg, interval_s, gauge, parent=None):
        super().__init__(parent)
        self._running = True
        self._mg = mg
        self._interval_s = interval_s
        self._gauge = gauge

    @property
    def has_device(self):
        return self._mg is not None

    def stop(self):
        self._running = False

    def _read_triplet(self):
        p = self._mg.pressures()
        def val(i):
            return p[i].pressure if p[i].status in (0, 1, 2) else float("nan")
        return val(0), val(1), val(2)

    def run(self):
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
                self.error_occurred.emit(f"Pressure: MaxiGauge error ({e})")
                try:
                    if hasattr(self._mg, "close"):
                        self._mg.close()
                except Exception:
                    pass
                time.sleep(2.0)
                self._mg = initialize_maxigauge(_cfg("pressure.port", "COM3"))
                if self._mg is None:
                    time.sleep(1.0)
                    continue
            except Exception as e:
                self.error_occurred.emit(f"Pressure: Unexpected error ({e})")
                time.sleep(1.0)
                continue
            time.sleep(self._interval_s)
        try:
            if self._mg and hasattr(self._mg, "close"):
                self._mg.close()
        except Exception:
            pass

class PressureMonitorWidget(QObject):
    log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        port = _cfg("pressure.port", "COM3")
        interval_s = float(_cfg("runtime.pressure_update_interval_s", 5.0))
        self._prom_port = int(_cfg("metrics.prometheus_port", 8000))
        self._grafana_url = _cfg("metrics.grafana_url", "http://localhost:3000")
        self._registry = CollectorRegistry()
        self._gauge = Gauge("sensor_pressure", "Pressure reading by sensor (mbar)", ["sensor"], registry=self._registry)
        try:
            start_http_server(self._prom_port, addr="0.0.0.0", registry=self._registry)
            self.log_signal.emit(f"Metrics at http://127.0.0.1:{self._prom_port}/metrics")
            self.log_signal.emit(f"Grafana dashboard: {self._grafana_url}")
        except Exception as e:
            self.log_signal.emit(f"Failed to start metrics server: {e}")
        self._mg = initialize_maxigauge(port)
        self.worker = PressureWorker(self._mg, interval_s, self._gauge)
        self.worker.error_occurred.connect(self.log_signal)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    @property
    def grafana_url(self):
        return self._grafana_url

    def shutdown(self):
        try:
            if getattr(self, "worker", None):
                self.worker.stop()
        except Exception:
            pass
        try:
            if getattr(self, "thread", None):
                self.thread.quit()
                self.thread.wait(3000)
        except Exception:
            pass

    def __del__(self):
        self.shutdown()
