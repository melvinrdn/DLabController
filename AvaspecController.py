import time
import hardware.avaspec_driver._avs_py as avs

class AvaspecController:
    def __init__(self, spec_handle):
        """
        Initialize the spectrometer controller by activating the spectrometer.
        """
        self.avs = avs
        self.spec_handle = self.avs.AVS_Activate(spec_handle)
        self.wavelength = self.avs.AVS_GetLambda(self.spec_handle)
        self.num_pixels = len(self.wavelength)

    def set_measurement_parameters(self, int_time, no_avg):
        """Set the measurement parameters for the spectrometer."""
        self.avs.set_measure_params(self.spec_handle, int_time, no_avg)

    def measure_spectrum(self, int_time, no_avg):
        """
        Perform a measurement and return the timestamp and spectrum data.
        """
        self.avs.AVS_StopMeasure(self.spec_handle)
        time.sleep(0.5)
        self.set_measurement_parameters(int_time, no_avg)
        self.avs.AVS_Measure(self.spec_handle)
        timestamp, data = self.avs.get_spectrum(self.spec_handle)
        return timestamp, data

    def deactivate(self):
        """Deactivate the spectrometer."""
        self.avs.AVS_Deactivate(self.spec_handle)

    @classmethod
    def list_spectrometers(cls):
        """
        Initialize and return a list of available spectrometers.
        """
        avs.AVS_Init()
        return avs.AVS_GetList()