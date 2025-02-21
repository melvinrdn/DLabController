import pyvisa
from ThorlabsPM100 import ThorlabsPM100

class PowermeterController:
    def __init__(self, powermeter_id: str):
        """
        Initialize the power meter controller with the given VISA resource string.

        Parameters:
            powermeter_id (str): The VISA resource string (e.g., 'USB0::0x1313::0x8078::P0045634::INSTR')
        """
        self.powermeter_id = powermeter_id
        self.instrument = None
        self.power_meter = None

    def activate(self) -> None:
        """
        Activate (initialize) the power meter.
        """
        rm = pyvisa.ResourceManager()
        self.instrument = rm.open_resource(self.powermeter_id)
        self.power_meter = ThorlabsPM100(inst=self.instrument)
        self.set_auto_range(1)

    def get_config(self) -> dict:
        """
        Retrieve the current configuration from the power meter.
        """
        config = {
            "measurement_configuration": self.power_meter.getconfigure,
            "wavelength": self.power_meter.sense.correction.wavelength,
            "averaging_count": self.power_meter.sense.average.count,
            "auto_range": self.power_meter.sense.power.dc.range.auto,
            "lpass_bandwidth": self.power_meter.input.pdiode.filter.lpass.state,
        }
        return config

    def read_power(self) -> float:
        """
        Retrieve a power measurement.
        """
        return self.power_meter.read

    def fetch_power(self) -> float:
        """
        Retrieve last measurement data.
        """
        return self.power_meter.fetch

    def set_auto_range(self, auto_range: int) -> None:
        """
        Set the power range to auto mode.
        """
        if auto_range == 1:
            self.power_meter.sense.power.dc.range.auto = "ON"
        elif auto_range == 0:
            self.power_meter.sense.power.dc.range.auto = "OFF"

    def set_avg(self, no_avg: int) -> None:
        """
        Sets the averaging rate (1 sample takes approx. 3ms)
        """
        self.power_meter.sense.average.count = no_avg

    def set_wavelength(self, wavelength: float) -> None:
        """
        Set the operation wavelength in nm.
        """
        self.power_meter.sense.correction.wavelength = wavelength

    def set_bandwidth(self, bandwidth: str) -> None:
        """
        Set the bandwidth (high or low).
        """
        mode = bandwidth.lower()
        if mode == "high":
            self.power_meter.input.pdiode.filter.lpass.state = 0
        elif mode == "low":
            self.power_meter.input.pdiode.filter.lpass.state = 1

    def deactivate(self) -> None:
        """
        Deactivate the power meter by closing the instrument connection.
        """
        if self.instrument is not None:
            self.instrument.close()
            self.instrument = None
            self.power_meter = None

## Example usage
"""
powermeter_id = 'USB0::0x1313::0x8078::P0045634::INSTR'
Powermeter =  PowermeterController(powermeter_id)
Powermeter.activate()
print(Powermeter.get_config())
Powermeter.deactivate()
"""