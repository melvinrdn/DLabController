import numpy as np
import logging
import threading
from pylablib.devices import Andor

# Default settings (in microseconds for exposure)
DEFAULT_EXPOSURE_US = 100000  # e.g., 100000 µs = 100 ms
DEFAULT_AVERAGES = 1

# Exposure range (in µs)
MIN_EXPOSURE_US = 1000
MAX_EXPOSURE_US = 1000000

class AndorControllerError(Exception):
    """
    Exception raised for errors encountered during Andor camera operations.
    """
    pass

class AndorController:
    """
    A controller class for interfacing with an Andor camera.

    Provides methods to activate the camera, set the exposure time,
    capture an averaged image, and deactivate the camera.
    Exposure is specified in microseconds.
    """
    def __init__(self, device_index=0):
        """
        Initializes the AndorController instance without connecting to the camera.

        Parameters:
            device_index (int): The index of the Andor camera to use (default: 0).
        """
        self.device_index = device_index
        self.cam = None
        self.image_shape = None
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"AndorController_{self.device_index}")
        self.logger.propagate = False
        self.current_exposure = None

    def enable_debug(self, debug_on=True):
        """
        Enables or disables debug mode for the Andor controller.

        Parameters:
            debug_on (bool): If True, sets logger level to DEBUG; otherwise, INFO.
        """
        level = logging.DEBUG if debug_on else logging.INFO
        self.logger.setLevel(level)

    def activate(self):
        """
        Activates and connects to the Andor camera.

        Configures the camera with the default exposure time, captures an initial image
        to determine the image shape, and logs the activation status.

        Raises:
            AndorControllerError: If the camera fails to activate.
        """
        try:
            # Create an instance of the Andor camera.
            self.cam = Andor.AndorSDK2Camera()  # Adjust parameters as needed.
            # Set default exposure. API expects exposure in seconds.
            self.cam.set_exposure(DEFAULT_EXPOSURE_US / 1e6)
            # Optional: Setup shutter if required by your camera.
            self.cam.setup_shutter('open')
            # Start acquisition to capture an initial frame.
            self.cam.start_acquisition()
            self.cam.wait_for_frame(timeout=20)
            frame = self.cam.read_oldest_image()
            self.cam.stop_acquisition()
            self.image_shape = np.shape(frame)
            self.current_exposure = DEFAULT_EXPOSURE_US
            self.logger.info(f"Camera {self.device_index}: activated successfully with shape {self.image_shape}")
        except Exception as e:
            self.cam = None
            self.image_shape = None
            raise AndorControllerError(f"Camera {self.device_index}: Failed to activate camera: {e}") from e

    def set_exposure(self, exposure):
        """
        Sets the camera's exposure time if it differs from the current setting.

        Parameters:
            exposure (int): Desired exposure time in microseconds.

        Raises:
            ValueError: If exposure is not a positive integer.
            AndorControllerError: If the camera is not activated.
        """
        if not isinstance(exposure, int) or exposure <= 0:
            raise ValueError("Exposure must be a positive integer in microseconds.")
        if self.cam is None:
            raise AndorControllerError("Camera not activated. Call activate() first.")
        if self.current_exposure != exposure:
            try:
                self.cam.set_exposure(exposure / 1e6)
                self.current_exposure = exposure
                self.logger.debug(f"Camera {self.device_index}: Exposure set to {exposure} µs")
            except Exception as e:
                raise AndorControllerError(f"Failed to set exposure: {e}") from e

    def take_image(self, exposure, avgs):
        """
        Captures and returns an averaged image based on the specified parameters.

        Parameters:
            exposure (int): Exposure time in microseconds.
            avgs (int): Number of frames to average.

        Returns:
            np.ndarray: The averaged image.

        Raises:
            ValueError: If parameters are invalid.
            AndorControllerError: If the camera is not activated or capture fails.
        """
        if not isinstance(exposure, int) or exposure <= 0:
            raise ValueError("Exposure must be a positive integer in microseconds.")
        if not isinstance(avgs, int) or avgs <= 0:
            raise ValueError("Averaging count must be a positive integer.")
        if self.cam is None or self.image_shape is None:
            raise AndorControllerError("Camera not activated. Call activate() first.")
        with self._lock:
            self.set_exposure(exposure)
            try:
                self.cam.start_acquisition()
                image_acc = np.zeros(self.image_shape, dtype=np.float64)
                for i in range(avgs):
                    self.cam.wait_for_frame(timeout=20)
                    frame = self.cam.read_oldest_image()
                    image_acc += frame.astype(np.float64)
                self.cam.stop_acquisition()
                avg_image = image_acc / avgs
                self.logger.debug(f"Camera {self.device_index}: Captured image with exposure={exposure} µs, avgs={avgs}")
                return avg_image
            except Exception as e:
                raise AndorControllerError(f"Failed to capture image: {e}") from e

    def deactivate(self):
        """
        Deactivates the camera and releases resources.
        """
        if self.cam:
            try:
                self.cam.close()
                self.logger.info(f"Camera {self.device_index}: deactivated.")
            except Exception as e:
                self.logger.error(f"Error deactivating camera: {e}")
            finally:
                self.cam = None

    def __enter__(self):
        """
        Activates the camera upon entering a context.
        """
        self.activate()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Deactivates the camera upon exiting a context.
        """
        self.deactivate()

