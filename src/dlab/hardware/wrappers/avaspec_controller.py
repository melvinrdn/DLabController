from __future__ import annotations
from typing import Any, List, Tuple
import time
import numpy as np
import dlab.hardware.drivers.avaspec_driver._avs_py as avs

class AvaspecError(Exception):
    pass

class AvaspecController:
    def __init__(self, spec_handle: Any) -> None:
        self._orig_handle = spec_handle
        self._h = None
        self.wavelength: np.ndarray | None = None
        self.num_pixels: int | None = None
        self.name: str = "Avaspec" 

    def activate(self) -> None:
        """Initialize library, activate device, fetch wavelength axis."""
        try:
            avs.AVS_Init()
            self._h = avs.AVS_Activate(self._orig_handle)
            self.wavelength = np.asarray(avs.AVS_GetLambda(self._h), dtype=float)
            self.num_pixels = int(self.wavelength.size)
            if self.num_pixels <= 0:
                raise AvaspecError("Empty wavelength array")
        except Exception as e:
            self._h = None
            raise AvaspecError(f"Activate failed: {e}") from e

    def set_measurement_parameters(self, int_time_ms: float, no_avg: int) -> None:
        """Push measurement parameters to device."""
        if self._h is None:
            raise AvaspecError("Not activated")
        avs.set_measure_params(self._h, float(int_time_ms), int(no_avg))
        
    def set_integration_time_ms(self, int_time_ms: float) -> None:
        self.set_measurement_parameters(float(int_time_ms), 1)

    def get_wavelengths(self) -> np.ndarray:
        if self.wavelength is None:
            raise AvaspecError("Not activated")
        return np.asarray(self.wavelength, dtype=float)

    def grab_spectrum_for_scan(self, int_ms: float, averages: int) -> tuple[np.ndarray, dict]:
        if averages <= 0:
            averages = 1
        buf = []
        last_ts = 0.0
        for _ in range(int(averages)):
            ts, data = self.measure_spectrum(float(int_ms), 1)
            last_ts = ts
            buf.append(np.asarray(data, dtype=float))
        counts = np.mean(np.stack(buf, axis=0), axis=0)
        meta = {
            "Integration_ms": float(int_ms),
            "Averages": int(averages),
            "Timestamp": float(last_ts),
            "DeviceName": self.name,
        }
        return counts, meta

    def measure_spectrum(self, int_time_ms: float, no_avg: int) -> Tuple[float, np.ndarray]:
        """Perform one acquisition and return (unix_ts, counts array)."""
        if self._h is None:
            raise AvaspecError("Not activated")
        # ensure a clean state
        avs.AVS_StopMeasure(self._h)
        # small guard; many devices need a tiny pause
        time.sleep(0.05)
        self.set_measurement_parameters(int_time_ms, no_avg)
        avs.AVS_Measure(self._h)
        ts, data = avs.get_spectrum(self._h)
        arr = np.asarray(data, dtype=float)
        if self.num_pixels and arr.size != self.num_pixels:
            # pad/trim to expected length to avoid downstream shape issues
            n = self.num_pixels
            if arr.size > n:
                arr = arr[:n]
            else:
                tmp = np.zeros(n, dtype=float)
                tmp[:arr.size] = arr
                arr = tmp
        return float(ts), arr

    def deactivate(self) -> None:
        """Deactivate device."""
        if self._h:
            try:
                avs.AVS_Deactivate(self._h)
            finally:
                self._h = None

    @classmethod
    def list_spectrometers(cls) -> list[Any]:
        """Return available spectrometer handles (possibly empty)."""
        try:
            avs.AVS_Init()
            return avs.AVS_GetList() or []
        except Exception:
            return []
