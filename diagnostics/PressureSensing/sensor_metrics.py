from prometheus_client import start_http_server, Gauge
import time
import random  # Replace with your actual sensor logic

# Create a Gauge metric for the pressure
pressure_gauge = Gauge('sensor_pressure', 'Pressure value from the sensor')

def get_pressure_value():
    # Replace this function with actual code to read from your sensor
    return random.uniform(0, 100)  # Simulated pressure value

def update_metrics():
    while True:
        pressure_value = get_pressure_value()
        pressure_gauge.set(pressure_value)  # Update the metric
        time.sleep(1)  # Update every second (adjust as needed)

if __name__ == "__main__":
    # Start the Prometheus HTTP server on port 8000
    start_http_server(8000)
    print("Prometheus metrics server running on http://localhost:8000")
    update_metrics()
