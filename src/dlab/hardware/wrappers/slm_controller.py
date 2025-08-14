from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import dlab.hardware.drivers.SLM_driver._slm_py as slm_driver

# SLM-300 Santec
DEFAULT_SLM_SIZE: Tuple[int, int] = (1200, 1920)  # (rows, cols) = (height, width)
DEFAULT_CHIP_W = 15.36e-3
DEFAULT_CHIP_H = 9.6e-3
DEFAULT_PIXEL_SIZE = 8e-6
DEFAULT_BIT_DEPTH = 1023  # 10 bits
DEFAULT_BEAM_RADIUS_ON_SLM = 3.5e-3  

class SLMController:
    """
    Minimal controller for a Spatial Light Modulator.
    You provide phase maps in device units [0..bit_depth].
    """
    def __init__(
        self,
        color: str,
        slm_size: Tuple[int, int] = DEFAULT_SLM_SIZE,
        chip_width: float = DEFAULT_CHIP_W,
        chip_height: float = DEFAULT_CHIP_H,
        pixel_size: float = DEFAULT_PIXEL_SIZE,
        bit_depth: int = DEFAULT_BIT_DEPTH,
    ):
        self.color = color
        self.slm_size = slm_size
        self.chip_width = chip_width
        self.chip_height = chip_height
        self.pixel_size = pixel_size
        self.bit_depth = bit_depth

        self.background_phase: Optional[np.ndarray] = None
        self.phase: Optional[np.ndarray] = None
        self.screen_num: Optional[int] = None

    def _convert_phase(self, phase: np.ndarray) -> np.ndarray:
        """
        Ensure integer phase in [0..bit_depth], contiguous C array.
        If float given, we modulo and cast to uint16.
        """
        arr = np.asarray(phase)
        arr = np.mod(arr, self.bit_depth + 1)
        # uint16 is enough for 10-bit; adjust if driver needs int32:
        arr = arr.astype(np.uint16, copy=False)
        return np.ascontiguousarray(arr)

    def publish(self, phase: np.ndarray, screen_num: int) -> None:
        """
        Publish the phase to the specified SLM screen.
        """
        self.phase = self._convert_phase(phase)
        self.screen_num = screen_num

        slm_driver.SLM_Disp_Open(self.screen_num)
        h, w = self.slm_size  # (rows, cols)
        slm_driver.SLM_Disp_Data(self.screen_num, self.phase, w, h)

    def close(self) -> None:
        """
        Explicit close if you kept a screen open (safe no-op otherwise).
        """
        if self.screen_num is not None:
            try:
                slm_driver.SLM_Disp_Close(self.screen_num)
            except Exception:
                pass
            self.screen_num = None
