from __future__ import annotations
import logging
import threading
from typing import Optional, Tuple, List

import numpy as np
from dlab.hardware.drivers import gxipy_driver as gx

# Exposure and gain defaults/ranges (microseconds for exposure)
DEFAULT_EXPOSURE_US = 1_000
DEFAULT_GAIN = 0
MIN_EXPOSURE_US = 20
MAX_EXPOSURE_US = 100_000
MIN_GAIN = 0
MAX_GAIN = 24


class DahengControllerError(Exception):
    """Raised for Daheng camera operation errors."""
    pass


class DahengController:
    """
    Controller for a Daheng camera via gxipy wrapper.

    Thread-safe for capture. Exposure in microseconds. Gain is unitless (device-specific).
    """

    def __init__(self, index: int) -> None:
        self.index = index
        self._cam = None
        self._imshape: Optional[Tuple[int, ...]] = None
        self._lock = threading.Lock()
        self._mgr = gx.DeviceManager()
        self._log = logging.getLogger(__name__)
        self.current_exposure: Optional[int] = None
        self.current_gain: Optional[int] = None

    # --------------- lifecycle ---------------

    def activate(self) -> None:
        """Open device, set defaults, and cache image shape."""
        try:
            self._mgr.update_device_list()
            self._cam = self._mgr.open_device_by_index(self.index)
            # Defaults
            self._cam.ExposureTime.set(self._clamp_exposure(DEFAULT_EXPOSURE_US))
            self._cam.Gain.set(self._clamp_gain(DEFAULT_GAIN))
            # Software trigger mode
            self._cam.TriggerMode.set(gx.GxSwitchEntry.ON)
            self._cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
            self._cam.PixelFormat.set(gx.GxPixelFormatEntry.MONO8)
            # Prime one frame to know the shape
            self._cam.stream_on()
            try:
                self._cam.TriggerSoftware.send_command()
                img = self._cam.data_stream[0].get_image()
                np_im = img.get_numpy_array()
                if np_im is None:
                    raise DahengControllerError("Failed to get image on activation (None array).")
                self._imshape = tuple(np.shape(np_im))
            finally:
                self._cam.stream_off()

            self.current_exposure = int(self._cam.ExposureTime.get())
            self.current_gain = int(self._cam.Gain.get())

            self._log.info("Daheng[%s] activated; shape=%s; exposure=%dus; gain=%d",
                           self.index, self._imshape, self.current_exposure, self.current_gain)
        except Exception as e:
            self._safe_close()
            raise DahengControllerError(f"activate failed: {e}") from e

    def deactivate(self) -> None:
        """Close device."""
        self._safe_close()
        self._log.info("Daheng[%s] deactivated", self.index)

    def _safe_close(self) -> None:
        try:
            if self._cam is not None:
                try:
                    # Make sure stream is off before closing
                    self._cam.stream_off()
                except Exception:
                    pass
                self._cam.close_device()
        finally:
            self._cam = None
            self._imshape = None
            self.current_exposure = None
            self.current_gain = None

    # --------------- configuration ---------------

    def _clamp_exposure(self, us: int) -> int:
        if us < MIN_EXPOSURE_US or us > MAX_EXPOSURE_US:
            clamped = max(MIN_EXPOSURE_US, min(MAX_EXPOSURE_US, us))
            self._log.warning("Daheng[%s] exposure %dus out of range [%d..%d]; clamped to %dus",
                              self.index, us, MIN_EXPOSURE_US, MAX_EXPOSURE_US, clamped)
            return clamped
        return us

    def _clamp_gain(self, g: int) -> int:
        if g < MIN_GAIN or g > MAX_GAIN:
            clamped = max(MIN_GAIN, min(MAX_GAIN, g))
            self._log.warning("Daheng[%s] gain %d out of range [%d..%d]; clamped to %d",
                              self.index, g, MIN_GAIN, MAX_GAIN, clamped)
            return clamped
        return g

    def set_exposure(self, exposure_us: int) -> None:
        """Set exposure in microseconds."""
        if not isinstance(exposure_us, int) or exposure_us <= 0:
            raise ValueError("exposure_us must be a positive integer (µs)")
        if self._cam is None:
            raise DahengControllerError("camera not active; call activate() first")

        exposure_us = self._clamp_exposure(exposure_us)
        if self.current_exposure == exposure_us:
            return
        self._cam.ExposureTime.set(exposure_us)
        self.current_exposure = exposure_us
        self._log.debug("Daheng[%s] exposure set to %dus", self.index, exposure_us)

    def set_gain(self, gain: int) -> None:
        """Set device gain."""
        if not isinstance(gain, int):
            raise ValueError("gain must be integer")
        if self._cam is None:
            raise DahengControllerError("camera not active; call activate() first")

        gain = self._clamp_gain(gain)
        if self.current_gain == gain:
            return
        self._cam.Gain.set(gain)
        self.current_gain = gain
        self._log.debug("Daheng[%s] gain set to %d", self.index, gain)

    def get_image_shape(self) -> Tuple[int, ...]:
        if self._imshape is None:
            raise DahengControllerError("image shape unknown; call activate() first")
        return self._imshape

    # --------------- capture ---------------

    def capture_single(self, exposure_us: int, gain: Optional[int] = None) -> np.ndarray:
        """
        Capture a single frame (float64). If gain is given, update it first.
        """
        if self._cam is None or self._imshape is None:
            raise DahengControllerError("camera not active; call activate() first")

        with self._lock:
            self.set_exposure(int(exposure_us))
            if gain is not None:
                self.set_gain(int(gain))

            self._cam.stream_on()
            try:
                self._cam.TriggerSoftware.send_command()
                img = self._cam.data_stream[0].get_image()
                arr = img.get_numpy_array()
                if arr is None:
                    raise DahengControllerError("capture_single: image array is None")
                return arr.astype(np.float64, copy=False)
            finally:
                self._cam.stream_off()

    def capture_sequence(self, n: int, exposure_us: int, gain: Optional[int] = None) -> List[np.ndarray]:
        """
        Capture N frames as a list (each float64). Faster than calling capture_single N times.
        """
        if self._cam is None or self._imshape is None:
            raise DahengControllerError("camera not active; call activate() first")
        if n <= 0:
            raise ValueError("n must be >= 1")

        frames: List[np.ndarray] = []
        with self._lock:
            self.set_exposure(int(exposure_us))
            if gain is not None:
                self.set_gain(int(gain))

            self._cam.stream_on()
            try:
                for _ in range(n):
                    self._cam.TriggerSoftware.send_command()
                    img = self._cam.data_stream[0].get_image()
                    arr = img.get_numpy_array()
                    if arr is None:
                        raise DahengControllerError("capture_sequence: image array is None")
                    frames.append(arr.astype(np.float64, copy=False))
            finally:
                self._cam.stream_off()

        return frames

    # --------------- legacy compatibility ---------------

    def take_image(self, exposure: int, gain: int, avgs: int) -> np.ndarray:
        """
        Legacy API: capture with 'avgs' frames averaged. Prefer capture_single / capture_sequence.
        """
        if avgs <= 0:
            raise ValueError("avgs must be >= 1")
        frames = self.capture_sequence(avgs, exposure, gain)
        acc = np.zeros_like(frames[0], dtype=np.float64)
        for f in frames:
            acc += f
        return acc / avgs

    # --------------- misc ---------------

    @staticmethod
    def get_available_indices() -> list[int]:
        """Return available 1-based camera indices."""
        mgr = gx.DeviceManager()
        dev_num, _ = mgr.update_device_list()
        return list(range(1, dev_num + 1))
