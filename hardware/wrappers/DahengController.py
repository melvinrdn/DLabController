from hardware.drivers import gxipy_driver as gx
import numpy as np
import matplotlib.pyplot as plt
import logging
import threading

# Suppress Matplotlib debug messages.
logging.getLogger('matplotlib').setLevel(logging.WARNING)

DEFAULT_EXPOSURE_MS = 1000
DEFAULT_GAIN = 0

class CameraError(Exception):
    """
    Exception raised for errors encountered during camera operations.

    Attributes:
        message (str): Explanation of the error.
    """
    pass

class DahengController:
    """
    A controller class for interfacing with a Daheng camera.

    This class handles camera initialization, configuration, image acquisition with frame averaging,
    and resource cleanup. It is designed to be thread-safe and logs key operations with messages
    prefixed by the camera index.

    Attributes:
        index (int): Identifier for the camera.
        _cam: The internal camera device object.
        _imshape: Shape of the captured image array (height, width).
        device_manager: Object for managing available camera devices.
        _lock: A threading lock to synchronize access to camera operations.
        logger: Logger instance for logging messages.
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
        self.logger = logging.getLogger(__name__)

    def activate(self) -> None:
        """
        Activates and configures the camera with default settings.

        This method opens the camera device, sets default exposure and gain values, configures
        software triggering, and captures an initial image to determine the image dimensions.

        Raises:
            CameraError: If the camera fails to activate. The error message includes available
                         camera indices if activation fails.
        """
        available_indices = DahengController.get_available_indices()
        try:
            self.logger.debug(f"Available indices: {available_indices}")
            self.logger.debug(f"Camera {self.index}: Opening...")
            self.device_manager.update_device_list()
            self._cam = self.device_manager.open_device_by_index(self.index)
            self._cam.ExposureTime.set(DEFAULT_EXPOSURE_MS)
            self._cam.Gain.set(DEFAULT_GAIN)
            self._cam.TriggerMode.set(gx.GxSwitchEntry.ON)
            self._cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
            self._cam.stream_on()
            self._cam.TriggerSoftware.send_command()
            im = self._cam.data_stream[0].get_image()
            np_im = im.get_numpy_array()
            self._imshape = np.shape(np_im)
            self._cam.stream_off()
            self.logger.info(f"Camera {self.index}: activated successfully with shape {self._imshape}")
        except Exception as e:
            self._cam = None
            self._imshape = None
            raise CameraError(
                f"Camera {self.index}: Failed to activate camera: {e}. "
            ) from e

    def set_exposure(self, exposure: int) -> None:
        """
        Sets the camera's exposure time.

        Parameters:
            exposure (int): Desired exposure time in milliseconds. Must be a positive integer.

        Raises:
            ValueError: If the exposure is not an integer or is less than or equal to zero.
            CameraError: If the camera is not activated.
        """
        if not isinstance(exposure, int):
            raise ValueError("Exposure time must be an integer.")
        if exposure <= 0:
            raise ValueError("Exposure time must be a positive integer.")
        if not self._cam:
            raise CameraError("Camera is not activated. Call activate() first.")
        self._cam.ExposureTime.set(exposure)
        self.logger.debug(f"Camera {self.index}: Exposure set to {exposure} ms")

    def set_gain(self, gain: int) -> None:
        """
        Sets the camera's gain.

        Parameters:
            gain (int): Desired gain value. Must be a non-negative integer.

        Raises:
            ValueError: If the gain is not an integer or is negative.
            CameraError: If the camera is not activated.
        """
        if not isinstance(gain, int):
            raise ValueError("Gain must be an integer.")
        if gain < 0:
            raise ValueError("Gain must be a non-negative integer.")
        if not self._cam:
            raise CameraError("Camera is not activated. Call activate() first.")
        self._cam.Gain.set(gain)
        self.logger.debug(f"Camera {self.index}: Gain set to {gain}")

    def take_image(self, exposure: int, gain: int, avgs: int) -> np.ndarray:
        """
        Captures and returns an averaged image based on the specified parameters.

        This method sets the camera's exposure and gain, then triggers the camera to capture
        a series of frames. The returned image is the pixel-wise average of the captured frames.

        Parameters:
            exposure (int): Exposure time in milliseconds for image capture. Must be positive integer.
            gain (int): Gain value for image capture. Must be non-negative.
            avgs (int): Number of frames to average. Must be a positive integer.

        Returns:
            np.ndarray: The averaged image array as a floating-point numpy array.

        Raises:
            ValueError: If any parameter is not an integer or is out of the allowed range.
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
            raise CameraError("Camera is not activated. Call activate() first.")

        with self._lock:  # Ensure thread-safe operations
            self.set_exposure(exposure)
            self.set_gain(gain)

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
        Deactivates the camera by closing the device connection and releasing resources.

        This method should be called when camera operations are complete to prevent resource leaks.
        """
        if self._cam:
            self._cam.close_device()
            self._cam = None
            self.logger.info(f"Camera {self.index}: deactivated.")

    def __enter__(self) -> "DahengController":
        """
        Activates the camera when entering a context (a with statement for instance).

        Returns:
            DahengController: The current camera controller instance.
        """
        self.activate()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Deactivates the camera when exiting a context, ensuring proper cleanup.
        """
        self.deactivate()

    @staticmethod
    def get_available_indices() -> list:
        """
        Retrieves a list of available camera indices based on the current device list.

        Returns:
            list: A list of available camera indices (1-indexed).
        """
        device_manager = gx.DeviceManager()
        dev_num, _ = device_manager.update_device_list()
        return list(range(1, dev_num + 1))

# Execution example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    exposure = 1000
    gain = 0
    avgs = 5
    idx = 1

    try:
        with DahengController(idx) as camera:
            averaged_image = camera.take_image(exposure, gain, avgs)
            plt.imshow(averaged_image, cmap='turbo')
            plt.title(f"Averaged Image ({avgs} Frames)")
            plt.axis('off')
            plt.show()
    except CameraError as e:
        logging.error(f"Camera {idx}: {e}")
