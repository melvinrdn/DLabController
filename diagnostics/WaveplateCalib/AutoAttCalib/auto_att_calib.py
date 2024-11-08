import pyvisa
import time
import logging
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import thorlabs_apt as apt
from ThorlabsPM100 import ThorlabsPM100

# Setup logging with color formatting
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

class ColorFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.WARNING:
            record.msg = f"{YELLOW}{record.msg}{RESET}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{RED}{record.msg}{RESET}"
        else:
            record.msg = f"{GREEN}{record.msg}{RESET}"
        return super().format(record)

# Setup logging with color formatting
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])


class AutoAttCalib:
    PLOT_UPDATE_INTERVAL = 0.1  # Interval for plot update in seconds

    def __init__(self, motor_id: object, powermeter_id, stabilization_time=1):
        """
        Initialize the AutoCalib instance with motor ID, power meter ID, and stabilization time.

        Args:
            motor_id (str): The ID of the motor.
            powermeter_id (str): The resource ID of the power meter.
            stabilization_time (float): Waiting time (in seconds) for motor stabilization after moving to a position.
        """
        self.motor_id = motor_id
        self.powermeter_id = powermeter_id
        self.stabilization_time = stabilization_time
        self.motor = None
        self.power_meter = None
        self.instrument = None
        self.fig, self.ax, self.line = None, None, None

    def init_motor(self):
        """Initialize and home the motor."""
        try:
            self.motor = apt.Motor(int(self.motor_id))
            self.motor.move_home(blocking=True)
            logging.info(f"Motor with ID {self.motor_id} initialized successfully.")
        except Exception as e:
            logging.warning(f"Failed to initialize motor with ID {self.motor_id}: {e}")

    def init_powermeter(self):
        """Initialize the power meter."""
        try:
            rm = pyvisa.ResourceManager()
            self.instrument = rm.open_resource(self.powermeter_id)
            self.power_meter = ThorlabsPM100(inst=self.instrument)
            logging.info("Power meter initialized successfully.")
        except Exception as e:
            logging.warning(f"Failed to initialize power meter: {e}")

    def setup_plot(self):
        """Set up the live plot for power measurement."""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.fig.tight_layout()
        self.ax.set_title("Live Power Measurement")
        self.ax.set_xlabel("Angle (degrees)")
        self.ax.set_ylabel("Power (W)")
        self.line, = self.ax.plot([], [], 'bo')

    def update_plot(self, angle_data, power_data):
        """Update plot with new data."""
        self.line.set_xdata(angle_data)
        self.line.set_ydata(power_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.tight_layout()
        plt.draw()
        plt.pause(self.PLOT_UPDATE_INTERVAL)

    def measure_power_at_angles(self, angle_list, waveplate_name, waveplate_path):
        """Measure power at specified angles with live plotting."""
        if not all([self.motor, self.power_meter]):
            logging.warning("Please ensure both motor and power meter are initialized before proceeding.")
            return

        os.makedirs(waveplate_path, exist_ok=True)
        output_file = os.path.join(waveplate_path, f"calib_{waveplate_name}_{datetime.now().strftime('%Y_%m_%d')}.txt")

        angle_data, power_data = [], []
        self.setup_plot()

        try:
            with open(output_file, "w") as f:
                for angle in angle_list:
                    logging.info(f"Moving motor to {angle} degrees.")
                    self.motor.move_to(angle, blocking=True)
                    time.sleep(self.stabilization_time)

                    power_value = self.power_meter.read
                    logging.info(f"Angle: {angle} degrees, Power: {power_value:.6f} W")
                    f.write(f"{angle}\t{power_value:.6f}\n")

                    angle_data.append(angle)
                    power_data.append(power_value)
                    self.update_plot(angle_data, power_data)

            logging.info("Power measurement completed.")
            logging.info(f"Data saved to {output_file}")

        except Exception as e:
            logging.warning(f"An error occurred during power measurement: {e}")

        finally:
            plt.ioff()
            plt.show()
            self.close_hardware()

    def close_hardware(self):
        """Close the connection to the motor and power meter."""
        if self.instrument:
            self.instrument.close()
            logging.info("Power meter connection closed.")
        if self.motor:
            self.motor.disable()
            logging.info("Motor connection closed.")

    def run(self, angle_list, waveplate_name, waveplate_path):
        """Run the full calibration sequence."""
        self.init_motor()
        self.init_powermeter()
        self.measure_power_at_angles(angle_list, waveplate_name, waveplate_path)


