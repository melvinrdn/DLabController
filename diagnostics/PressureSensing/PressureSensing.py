import threading
import time
import datetime
import logging
from PfeifferVacuum import MaxiGauge, MaxiGaugeError
from influxdb_client import InfluxDBClient, Point, WriteOptions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

COM_PORT = 'COM9'
INFLUXDB_TOKEN = "TCFR-UwJzE8b52u2wRK3LXrDMfKomqCQRLtWqGolrDyV8s0JDMYPPdd_SCad745t_dnnzK47oYK6DO9S_WL0rA=="
INFLUXDB_URL = "http://localhost:8086"
ORG = "DLab"
BUCKET = "pressure_data"
UPDATE_INTERVAL = 5  # seconds
ERROR_RETRY_INTERVAL = 30  # seconds

mg = None

def initialize_maxigauge():
    global mg
    try:
        mg = MaxiGauge(COM_PORT)
        logging.info("MaxiGauge initialized successfully.")
    except serial.SerialException as e:
        logging.error(f"Failed to open COM port: {e}")
        mg = None


def update_pressure():
    if mg is None:
        logging.warning("MaxiGauge not initialized. Cannot update pressure data.")
        return

    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=ORG)
    write_api = client.write_api(write_options=WriteOptions(batch_size=1))

    try:
        while True:
            try:
                pressures = mg.pressures()
                sensor_1, sensor_2, sensor_3 = pressures[0], pressures[1], pressures[2]
                current_time = datetime.datetime.utcnow().isoformat()

                point = (
                    Point("pressure_readings")
                    .time(current_time)
                    .field("pressure_1", sensor_1.pressure if sensor_1.status in [0, 1, 2] else None)
                    .field("pressure_2", sensor_2.pressure if sensor_2.status in [0, 1, 2] else None)
                    .field("pressure_3", sensor_3.pressure if sensor_3.status in [0, 1, 2] else None)
                )
                write_api.write(bucket=BUCKET, org=ORG, record=point)
                logging.info("Pressure data updated successfully.")
                time.sleep(UPDATE_INTERVAL)

            except MaxiGaugeError as e:
                logging.error(f"Error reading from MaxiGauge: {e}")
                current_time = datetime.datetime.utcnow().isoformat()
                point = (
                    Point("pressure_readings")
                    .time(current_time)
                    .field("pressure_1", None)
                    .field("pressure_2", None)
                    .field("pressure_3", None)
                )
                write_api.write(bucket=BUCKET, org=ORG, record=point)
                time.sleep(ERROR_RETRY_INTERVAL)
    finally:
        client.close()
        logging.info("InfluxDB client closed.")


if __name__ == "__main__":
    initialize_maxigauge()

    if mg is not None:
        pressure_thread = threading.Thread(target=update_pressure)
        pressure_thread.daemon = True
        pressure_thread.start()

    while True:
        time.sleep(1)
