import hardware.drivers.SLM_driver._slm_py as slm_driver

## SLM-300 Santec
slm_size = (1200, 1920)
chip_width = 15.36e-3
chip_height = 9.6e-3
pixel_size = 8e-6
bit_depth = 1023

class SLMController:
    """
    A class to represent a Spatial Light Modulator (SLM) and interact with it.
    """
    def __init__(self, color, slm_size=slm_size, chip_width=chip_width, chip_height=chip_height, pixel_size=pixel_size, bit_depth=bit_depth):
        self.color = color

        self.slm_size = slm_size
        self.chip_width = chip_width
        self.chip_height = chip_height
        self.pixel_size = pixel_size
        self.bit_depth = bit_depth

        self.background_phase = None
        self.phase = None
        self.screen_num = None

    def _convert_phase(self, phase):
        """
        Convert the phase values to be within the allowed range (rom 0 to 1023 by default).
        """
        return phase % (self.bit_depth + 1)

    def publish(self, phase, screen_num):
        """
        Publishes the phase to the specified SLM screen.
        """
        self.phase = self._convert_phase(phase)
        self.screen_num = screen_num

        slm_driver.SLM_Disp_Open(self.screen_num)
        slm_driver.SLM_Disp_Data(self.screen_num, self.phase, self.slm_size[1], self.slm_size[0])


    def close(self):
        """
        Closes the connection to the SLM screen.
        """
        if self.screen_num is not None:
            slm_driver.SLM_Disp_Close(self.screen_num)
            self.screen_num = None

