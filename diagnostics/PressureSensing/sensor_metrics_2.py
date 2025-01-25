from prometheus_client import start_http_server, Gauge
import time
from PfeifferVacuum import MaxiGauge, MaxiGaugeError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pressure_gauge_1 = Gauge('sensor_pressure_1', 'Gas catcher')
pressure_gauge_2 = Gauge('sensor_pressure_2', 'Generation')
pressure_gauge_3 = Gauge('sensor_pressure_3', 'Grating and MCP')

COM_PORT = 'COM9'
UPDATE_INTERVAL = 5  # seconds

mg = None


def initialize_maxigauge():
    global mg
    try:
        mg = MaxiGauge(COM_PORT)
        logging.info("MaxiGauge initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize MaxiGauge: {e}")
        mg = None


def update_metrics():
    if mg is None:
        logging.warning("MaxiGauge not initialized. Cannot update pressure data.")
        return

    while True:
        try:
            pressures = mg.pressures()
            pressure_gauge_1.set(pressures[0].pressure if pressures[0].status in [0, 1, 2] else float('nan'))
            pressure_gauge_2.set(pressures[1].pressure if pressures[1].status in [0, 1, 2] else float('nan'))
            pressure_gauge_3.set(pressures[2].pressure if pressures[2].status in [0, 1, 2] else float('nan'))
        except MaxiGaugeError as e:
            logging.error(f"Error reading from MaxiGauge: {e}")
            pressure_gauge_1.set(float('nan'))
            pressure_gauge_2.set(float('nan'))
            pressure_gauge_3.set(float('nan'))
            continue

        time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    initialize_maxigauge()

    if mg is not None:
        start_http_server(8000)
        logging.info("Prometheus metrics server running on http://localhost:8000")
        update_metrics()
    else:
        logging.error("Exiting: MaxiGauge failed to initialize.")
