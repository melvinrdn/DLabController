import threading
import time
import datetime
from PfeifferVacuum import MaxiGauge, MaxiGaugeError
import serial
from influxdb_client import InfluxDBClient, Point, WriteOptions

# Pressure data storage
data_lock = threading.Lock()
mg = None


def initialize_maxigauge():
    global mg
    try:
        mg = MaxiGauge('COM9')  # Replace with your actual COM port
        print("MaxiGauge initialized successfully.")
    except serial.SerialException as e:
        print(f"Failed to open COM port: {e}")
        mg = None


def update_pressure():
    if mg is None:
        print("MaxiGauge not initialized. Cannot update pressure data.")
        return

    token = "TCFR-UwJzE8b52u2wRK3LXrDMfKomqCQRLtWqGolrDyV8s0JDMYPPdd_SCad745t_dnnzK47oYK6DO9S_WL0rA=="
    org = "DLab"
    bucket = "pressure_data"

    client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)
    write_api = client.write_api(write_options=WriteOptions(batch_size=1))

    while True:
        try:
            pressures = mg.pressures()
            sensor_1, sensor_2, sensor_3 = pressures[0], pressures[1], pressures[2]

            current_time = datetime.datetime.utcnow().isoformat()

            with data_lock:
                point = (
                    Point("pressure_readings")
                    .time(current_time)
                    .field("pressure_1", sensor_1.pressure if sensor_1.status in [0, 1, 2] else None)
                    .field("pressure_2", sensor_2.pressure if sensor_2.status in [0, 1, 2] else None)
                    .field("pressure_3", sensor_3.pressure if sensor_3.status in [0, 1, 2] else None)
                )
                write_api.write(bucket=bucket, org=org, record=point)

            time.sleep(5)  # Update every minute

        except MaxiGaugeError as e:
            print(f"Error reading from MaxiGauge: {e}")
            with data_lock:
                current_time = datetime.datetime.utcnow().isoformat()
                point = (
                    Point("pressure_readings")
                    .time(current_time)
                    .field("pressure_1", None)
                    .field("pressure_2", None)
                    .field("pressure_3", None)
                )
                write_api.write(bucket=bucket, org=org, record=point)
            time.sleep(5)


if __name__ == "__main__":
    initialize_maxigauge()

    if mg is not None:
        pressure_thread = threading.Thread(target=update_pressure)
        pressure_thread.daemon = True
        pressure_thread.start()

    while True:
        time.sleep(1)
