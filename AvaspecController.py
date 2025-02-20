# AvaspecController.py
import time
import hardware.avaspec_driver._avs_py as avs

class AvaspecController:
    def __init__(self, spec_handle):
        """
        Initialize the spectrometer controller.
        (Activation is done manually by calling activate().)
        """
        self.original_spec_handle = spec_handle
        self.avs = avs
        self.spec_handle = None
        self.wavelength = None
        self.num_pixels = None

    def activate(self):
        """
        Activate the spectrometer and retrieve its wavelength array.
        """
        # Reinitialize the library to ensure the device can be activated.
        self.avs.AVS_Init()
        self.spec_handle = self.avs.AVS_Activate(self.original_spec_handle)
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
