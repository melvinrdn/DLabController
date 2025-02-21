import time
import hardware.drivers.avaspec_driver._avs_py as avs
from typing import Any, List, Tuple

class AvaspecController:
    def __init__(self, spec_handle: Any) -> None:
        """
        Initialize the spectrometer controller.
        """
        self.original_spec_handle = spec_handle
        self.avs = avs
        self.spec_handle = None
        self.wavelength = None
        self.num_pixels = None

    def activate(self) -> None:
        """
        Activate the spectrometer and retrieve its wavelength array.
        """
        self.avs.AVS_Init()
        self.spec_handle = self.avs.AVS_Activate(self.original_spec_handle)
        self.wavelength = self.avs.AVS_GetLambda(self.spec_handle)
        self.num_pixels = len(self.wavelength)

    def set_measurement_parameters(self, int_time: float, no_avg: int) -> None:
        """
        Set the measurement parameters for the spectrometer.
        """
        self.avs.set_measure_params(self.spec_handle, int_time, no_avg)

    def measure_spectrum(self, int_time: float, no_avg: int) -> Tuple[float, List[float]]:
        """
        Perform a measurement and return the timestamp and spectrum data.
        """
        self.avs.AVS_StopMeasure(self.spec_handle)
        time.sleep(0.5)
        self.set_measurement_parameters(int_time, no_avg)
        self.avs.AVS_Measure(self.spec_handle)
        timestamp, data = self.avs.get_spectrum(self.spec_handle)
        return timestamp, data

    def deactivate(self) -> None:
        """
        Deactivate the spectrometer.
        """
        if self.spec_handle:
            self.avs.AVS_Deactivate(self.spec_handle)
            self.spec_handle = None

    @classmethod
    def list_spectrometers(cls):
        """
        Initialize and return a list of available spectrometers.
        """
        try:
            avs.AVS_Init()
            spectrometer_list = avs.AVS_GetList()
            return spectrometer_list
        except Exception as e:
            return None


## Example usage
"""
spec_list = AvaspecController.list_spectrometers()
selected_spec_handle = spec_list[0]
controller = AvaspecController(selected_spec_handle)
controller.activate()
integration_time = 100
number_of_averages = 1
timestamp, spectrum_data = controller.measure_spectrum(integration_time, number_of_averages)
print("Spectrum data:", spectrum_data)
controller.deactivate()
"""