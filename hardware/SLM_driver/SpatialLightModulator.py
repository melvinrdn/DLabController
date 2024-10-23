import numpy as np
import hardware.SLM_driver._slm_py as slm_driver

## SLM-300 Santec
slm_size = (1200, 1920)
chip_width = 15.36e-3
chip_height = 9.6e-3
pixel_size = 8e-6
bit_depth = 1023

class SpatialLightModulator:
    """
    A class to represent a Spatial Light Modulator (SLM) and interact with it.

    Parameters
    ----------
    color : str
        The color of the SLM, used for identification.
    slm_size : tuple[int, int], optional
        The resolution of the SLM as (height, width). Default is (1200, 1920).
    chip_width : float, optional
        The width of the SLM chip in meters. Default is 15.36e-3.
    chip_height : float, optional
        The height of the SLM chip in meters. Default is 9.6e-3.
    pixel_size : float, optional
        The size of each pixel on the SLM chip in meters. Default is 8e-6.
    bit_depth : int, optional
        The bit depth of the SLM, defining the range of phase values. Default is 1023.
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

        Parameters
        ----------
        phase : np.ndarray
            The phase data to validate.

        Returns
        -------
        np.ndarray
            The adjusted phase data.
        """
        return phase % (self.bit_depth + 1)

    def publish(self, phase, screen_num):
        """
        Publishes the phase to the specified SLM screen.

        Parameters
        ----------
        phase : np.ndarray
            A 2D numpy array representing the phase data to be displayed.
        screen_num : int
            The number of the screen to display the phase on.

        Raises
        ------
        ValueError
            If the phase size does not match the SLM size.
        Exception
            If an error occurs during communication with the SLM hardware.
        """
        if phase.shape != self.slm_size:
            raise ValueError(f'Phase shape {phase.shape} does not match the SLM size {self.slm_size}.')

        self.phase = self._convert_phase(phase)
        self.screen_num = screen_num

        try:
            slm_driver.SLM_Disp_Open(self.screen_num)
            slm_driver.SLM_Disp_Data(self.screen_num, self.phase, self.slm_size[1], self.slm_size[0])
            print(f'Published to the {self.color} SLM.')
        except Exception as e:
            print(f'Error publishing to the {self.color} SLM: {e}.')
        finally:
            print('---------')

    def close(self):
        """
        Closes the connection to the SLM screen.

        Raises
        ------
        Exception
            If an error occurs during the closing of the SLM screen.
        """
        if self.screen_num is not None:
            try:
                slm_driver.SLM_Disp_Close(self.screen_num)
                print(f'Connection to the {self.color} SLM has been closed.')
            except Exception as e:
                print(f'Error closing the {self.color} SLM: {e}')
            finally:
                self.screen_num = None
        else:
            print(f'No open connection to close for {self.color} SLM.')
