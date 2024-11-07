import numpy as np
import thorlabs_apt as apt
import pyvisa
from ThorlabsPM100 import ThorlabsPM100
import time
import logging
from datetime import datetime

from diagnostics.diagnostics_helpers import ColorFormatter
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])

class WaveplateController:

    def __init__(self):
        self.motor = None

    def init_motor(self, motor_id):
        try:
            self.motor = apt.Motor(int(motor_id))
            logging.info(f"Motor with ID {motor_id} initialized successfully.")
        except Exception as e:
            logging.warning(f"Failed to initialize motor with ID {motor_id}: {e}")
            self.motor = None

    def measure_power_at_angles(self, angle_list, waveplate_name, waveplate_path, resource_name):
        if self.motor is None:
            logging.warning("Motor is not initialized. Please initialize the motor before measuring.")
            return None

        date_str = datetime.now().strftime("%Y_%m_%d")
        output_file = f"{waveplate_path}/calib_{waveplate_name}_{date_str}.txt"


        try:
            rm = pyvisa.ResourceManager()
            instrument = rm.open_resource(resource_name)
            power_meter = ThorlabsPM100(inst=instrument)

            with open(output_file, "w") as f:
                for angle in angle_list:
                    logging.info(f"Moving motor to {angle} degrees.")
                    self.motor.move_to(angle, blocking=True)
                    time.sleep(1)

                    power_value = power_meter.read
                    logging.info(f"Angle: {angle} degrees, Power: {power_value:.6f} W")

                    f.write(f"{angle}\t{power_value:.6f}\n")

            instrument.close()
            logging.info("Power measurement completed and connection closed.")
            logging.info(f"Data saved to {output_file}")

        except Exception as e:
            logging.warning(f"An error occurred during power measurement: {e}")

# Motor and Powermeter
motor_id = '83837725'
resource_name = 'USB0::0x1313::0x8078::P0045634::INSTR'

# Waveplate choice
waveplate_name = 'wp_1'
waveplate_path = '../../ressources/calibration/' + waveplate_name
steps = 3
angles_to_measure = np.linspace(0,30,steps)

# Run
waveplate_controller = WaveplateController()
waveplate_controller.init_motor(motor_id)
waveplate_controller.measure_power_at_angles(angles_to_measure, waveplate_name, waveplate_path, resource_name)
