import logging
import threading
from hardware.drivers import gxipy_driver as gx
import numpy as np

# Suppress Matplotlib debug messages.
logging.getLogger('matplotlib').setLevel(logging.WARNING)

DEFAULT_EXPOSURE_US = 1000
DEFAULT_GAIN = 0
DEFAULT_AVERAGES = 1

MIN_EXPOSURE_US = 20
MAX_EXPOSURE_US = 100000
MIN_GAIN = 0
MAX_GAIN = 24

class DahengControllerError(Exception):
    """
    Exception raised for errors encountered during camera operations.
    """
    pass


class DahengController:
    """
    A controller class for interfacing with a Daheng camera.

    This class handles camera initialization, configuration, image acquisition with frame averaging,
    and resource cleanup. It is designed to be thread-safe and logs key operations with messages
    prefixed by the camera index.
    """

    def __init__(self, index: int):
        """
        Initializes the DahengController instance without connecting to the device.

        Parameters:
            index (int): The index of the camera to be managed.
        """
        self.index: int = index
        self._cam = None
        self._imshape = None
        self.device_manager = gx.DeviceManager()
        self._lock = threading.Lock()  # Ensures thread-safe operations
        self.logger = logging.getLogger(f"DahengController_{self.index}")
        self.logger.propagate = False
        self.current_exposure = None
        self.current_gain = None
        self.current_avgs = None

    def enable_debug(self, debug_on: bool = True) -> None:
        """
        Enables or disables debug mode for the camera controller.

        Parameters:
            debug_on (bool): If True, sets logger level to DEBUG; otherwise, INFO.
        """
        level = logging.DEBUG if debug_on else logging.INFO
        self.logger.setLevel(level)

    def activate(self) -> None:
        """
        Activates and configures the camera with default settings.

        Raises:
            CameraError: If the camera fails to activate.
        """
        try:
            self.device_manager.update_device_list()
            self._cam = self.device_manager.open_device_by_index(self.index)
            self._cam.ExposureTime.set(DEFAULT_EXPOSURE_US)
            self._cam.Gain.set(DEFAULT_GAIN)
            self._cam.TriggerMode.set(gx.GxSwitchEntry.ON)
            self._cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
            self._cam.stream_on()
            self._cam.TriggerSoftware.send_command()
            im = self._cam.data_stream[0].get_image()
            np_im = im.get_numpy_array()
            self._imshape = np.shape(np_im)
            self._cam.stream_off()
            self.current_exposure = DEFAULT_EXPOSURE_US
            self.current_gain = DEFAULT_GAIN
            self.current_avgs = DEFAULT_AVERAGES
        except Exception as e:
            self._cam = None
            self._imshape = None
            raise DahengControllerError(f"Camera {self.index}: Failed to activate camera: {e}") from e

    def set_exposure(self, exposure: int) -> None:
        """
        Sets the camera's exposure time if it differs from the current setting.

        Parameters:
            exposure (int): Desired exposure time in microiseconds.

        Raises:
            ValueError: If exposure is not valid.
            CameraError: If the camera is not activated.
        """
        if not isinstance(exposure, int):
            raise ValueError("Exposure time must be an integer.")
        if exposure <= 0:
            raise ValueError("Exposure time must be a positive integer.")
        if not self._cam:
            raise DahengControllerError("Camera is not activated. Call activate() first.")
        if self.current_exposure != exposure:
            self._cam.ExposureTime.set(exposure)
            self.current_exposure = exposure
            self.logger.debug(f"Camera {self.index}: Exposure set to {exposure} ms")

    def set_gain(self, gain: int) -> None:
        """
        Sets the camera's gain if it differs from the current setting.

        Parameters:
            gain (int): Desired gain value.

        Raises:
            ValueError: If gain is not valid.
            CameraError: If the camera is not activated.
        """
        if not isinstance(gain, int):
            raise ValueError("Gain must be an integer.")
        if gain < 0:
            raise ValueError("Gain must be a non-negative integer.")
        if not self._cam:
            raise DahengControllerError("Camera is not activated. Call activate() first.")
        if self.current_gain != gain:
            self._cam.Gain.set(gain)
            self.current_gain = gain
            self.logger.debug(f"Camera {self.index}: Gain set to {gain}")

    def set_average(self, avgs: int) -> None:
        """
        Updates the average count if it differs from the current value.

        Parameters:
            avgs (int): Desired number of frames to average.

        Raises:
            ValueError: If avgs is not a positive integer.
        """
        if not isinstance(avgs, int):
            raise ValueError("Averaging count must be an integer.")
        if avgs <= 0:
            raise ValueError("Averaging count must be a positive integer.")
        if self.current_avgs != avgs:
            self.current_avgs = avgs
            self.logger.debug(f"Camera {self.index}: Averaging count set to {avgs}")

    def take_image(self, exposure: int, gain: int, avgs: int) -> np.ndarray:
        """
        Captures and returns an averaged image based on the specified parameters.

        Parameters:
            exposure (int): Exposure time in microseconds.
            gain (int): Gain value.
            avgs (int): Number of frames to average.

        Returns:
            np.ndarray: Averaged image array.

        Raises:
            ValueError: If parameters are invalid.
            CameraError: If the camera is not activated.
        """
        if not isinstance(exposure, int) or not isinstance(gain, int) or not isinstance(avgs, int):
            raise ValueError("Exposure, gain, and averaging count must be integers.")
        if exposure <= 0:
            raise ValueError("Exposure must be a positive integer.")
        if gain < 0:
            raise ValueError("Gain must be a non-negative integer.")
        if avgs <= 0:
            raise ValueError("Averaging count must be a positive integer.")
        if self._cam is None or self._imshape is None:
            raise DahengControllerError("Camera is not activated. Call activate() first.")

        with self._lock:
            # Update settings only if necessary.
            self.set_exposure(exposure)
            self.set_gain(gain)
            self.set_average(avgs)

            res = np.zeros(self._imshape, dtype=np.float64)
            self._cam.stream_on()
            try:
                for _ in range(avgs):
                    self._cam.TriggerSoftware.send_command()
                    im = self._cam.data_stream[0].get_image()
                    np_im = im.get_numpy_array()
                    res += np_im
            finally:
                self._cam.stream_off()
            self.logger.debug(f"Camera {self.index}: Captured image with exposure={exposure}, gain={gain}, avgs={avgs}")
            return res / avgs

    def deactivate(self) -> None:
        """
        Deactivates the camera and releases resources.
        """
        if self._cam:
            self._cam.close_device()
            self._cam = None

    def __enter__(self) -> "DahengController":
        """
        Activates the camera upon entering a context.
        """
        self.activate()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Deactivates the camera upon exiting a context.
        """
        self.deactivate()

    @staticmethod
    def get_available_indices() -> list:
        """
        Retrieves available camera indices.

        Returns:
            list: A list of available camera indices.
        """
        device_manager = gx.DeviceManager()
        dev_num, _ = device_manager.update_device_list()
        return list(range(1, dev_num + 1))