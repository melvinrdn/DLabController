from __future__ import annotations
import logging
import threading
from typing import Optional

import numpy as np
from pylablib.devices import Andor

# Defaults (microseconds)
DEFAULT_EXPOSURE_US = 100_000  # 100 ms
DEFAULT_AVERAGES = 1

# Exposure limits (microseconds) — tune to your camera
MIN_EXPOSURE_US = 1_000
MAX_EXPOSURE_US = 1_000_000

class AndorControllerError(Exception):
    """Raised for Andor camera operation errors."""
    pass


class AndorController:
    """
    Controller for an Andor SDK2 camera via pylablib.

    Exposure is specified in microseconds. Methods are thread-safe for capture.
    """

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.cam: Optional[Andor.AndorSDK2Camera] = None
        self.image_shape: Optional[tuple[int, ...]] = None
        self._lock = threading.Lock()
        self._log = logging.getLogger(__name__)
        self.current_exposure: Optional[int] = None

    # --------------- lifecycle ---------------

    def is_active(self) -> bool:
        return self.cam is not None

    def activate(self) -> None:
        """
        Connect to the camera, configure default exposure, and cache image shape.
        """
        if self.cam is not None:
            return  # already active

        try:
            cam = Andor.AndorSDK2Camera()  # selects first camera by default
            # If you need a specific camera/index, adapt here (enumeration depends on pylablib version)
            exp_us = self._clamp_exposure(DEFAULT_EXPOSURE_US)
            cam.set_exposure(exp_us / 1e6)  # seconds
            # Optional: shutter setup depending on your rig
            try:
                cam.setup_shutter("open")
            except Exception:
                # Not all cameras/firmwares support this; ignore if unsupported
                pass

            # Prime one frame to get shape
            cam.start_acquisition()
            try:
                cam.wait_for_frame(timeout=20)
                frame = cam.read_oldest_image()
            finally:
                cam.stop_acquisition()

            self.cam = cam
            self.image_shape = np.shape(frame)
            self.current_exposure = exp_us
            self._log.info("Andor[%s] activated; image shape=%s; exposure=%dus",
                           self.device_index, self.image_shape, self.current_exposure)
        except Exception as e:
            # Ensure we leave no half-open handle around
            try:
                cam.close()  # type: ignore[name-defined]
            except Exception:
                pass
            self.cam = None
            self.image_shape = None
            raise AndorControllerError(f"activate failed: {e}") from e

    def deactivate(self) -> None:
        """Close the camera handle."""
        if self.cam:
            try:
                self.cam.close()
                self._log.info("Andor[%s] deactivated", self.device_index)
            except Exception as e:
                self._log.error("Andor[%s] deactivate error: %s", self.device_index, e)
            finally:
                self.cam = None
                self.image_shape = None
                self.current_exposure = None

    def __enter__(self) -> "AndorController":
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.deactivate()

    # --------------- configuration ---------------

    def _clamp_exposure(self, exposure_us: int) -> int:
        if exposure_us < MIN_EXPOSURE_US or exposure_us > MAX_EXPOSURE_US:
            clamped = max(MIN_EXPOSURE_US, min(MAX_EXPOSURE_US, exposure_us))
            self._log.warning(
                "Andor[%s] exposure %dus out of range [%d..%d]; clamped to %dus",
                self.device_index, exposure_us, MIN_EXPOSURE_US, MAX_EXPOSURE_US, clamped
            )
            return clamped
        return exposure_us

    def set_exposure(self, exposure_us: int) -> None:
        """
        Set exposure in microseconds.
        """
        if not isinstance(exposure_us, int) or exposure_us <= 0:
            raise ValueError("exposure_us must be a positive integer (µs)")
        if self.cam is None:
            raise AndorControllerError("camera not active; call activate() first")

        exposure_us = self._clamp_exposure(exposure_us)
        if self.current_exposure == exposure_us:
            return

        try:
            self.cam.set_exposure(exposure_us / 1e6)
            self.current_exposure = exposure_us
            self._log.debug("Andor[%s] exposure set to %dus", self.device_index, exposure_us)
        except Exception as e:
            raise AndorControllerError(f"set_exposure failed: {e}") from e

    # --------------- capture ---------------

    def get_image_shape(self) -> tuple[int, ...]:
        if self.image_shape is None:
            raise AndorControllerError("image shape unknown; call activate() first")
        return self.image_shape

    def capture_single(self, exposure_us: int | None = None, timeout_s: float = 20.0) -> np.ndarray:
        """
        Capture a single frame. If exposure_us is given, it will be applied first.
        """
        if self.cam is None or self.image_shape is None:
            raise AndorControllerError("camera not active; call activate() first")

        with self._lock:
            if exposure_us is not None:
                self.set_exposure(int(exposure_us))

            self.cam.start_acquisition()
            try:
                self.cam.wait_for_frame(timeout=timeout_s)
                frame = self.cam.read_oldest_image()
            finally:
                # Always stop acquisition even if wait/read raised
                self.cam.stop_acquisition()

        # Ensure float64 for downstream averaging/processing
        return frame.astype(np.float64, copy=False)

    def take_image(self, exposure: int, avgs: int) -> np.ndarray:
        """
        Capture and return an averaged image.

        Parameters
        ----------
        exposure : int
            Exposure time in microseconds.
        avgs : int
            Number of frames to average (>= 1).
        """
        if not isinstance(exposure, int) or exposure <= 0:
            raise ValueError("exposure must be a positive integer (µs)")
        if not isinstance(avgs, int) or avgs <= 0:
            raise ValueError("avgs must be a positive integer")
        if self.cam is None or self.image_shape is None:
            raise AndorControllerError("camera not active; call activate() first")

        with self._lock:
            self.set_exposure(exposure)
            acc = np.zeros(self.image_shape, dtype=np.float64)

            self.cam.start_acquisition()
            try:
                for i in range(avgs):
                    self.cam.wait_for_frame(timeout=20.0)
                    frame = self.cam.read_oldest_image()
                    acc += frame.astype(np.float64, copy=False)
            finally:
                self.cam.stop_acquisition()

        avg = acc / avgs
        self._log.debug("Andor[%s] captured avg image: exposure=%dus, avgs=%d",
                        self.device_index, self.current_exposure, avgs)
        return avg
