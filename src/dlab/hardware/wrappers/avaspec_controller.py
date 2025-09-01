# dlab/hardware/wrappers/avaspec_controller.py
from __future__ import annotations
from typing import Any, Tuple, Optional
import time
import numpy as np
from pathlib import Path
import dlab.hardware.drivers.avaspec_driver._avs_py as avs
import dlab.hardware.drivers.avaspec_driver._avs_win as avs_win
from dlab.boot import ROOT, get_config

def _cal_path_from_config() -> Optional[Path]:
    cfg = get_config() or {}
    rel = (cfg.get("avaspec", {}) or {}).get("calibration_file")
    if not rel:
        return None
    p = Path(rel)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p

class AvaspecError(Exception):
    pass

class AvaspecController:
    def __init__(self, spec_handle: Any) -> None:
        self._orig_handle = spec_handle
        self._h = None
        self.wavelength: np.ndarray | None = None
        self.num_pixels: int | None = None
        self.name: str = "Avaspec"
        self._bg_wavelength: np.ndarray | None = None
        self._bg_counts: np.ndarray | None = None
        self._cal_path: Optional[Path] = None
        self._cal_wavelength: np.ndarray | None = None
        self._cal_values: np.ndarray | None = None
        self._suppress_until: float = 0.0
    def activate(self) -> None:
        try:
            avs.AVS_Init()
            self._h = avs.AVS_Activate(self._orig_handle)
            avs_win.AVS_UseHighResAdc(self._h, True)
            self.wavelength = np.asarray(avs.AVS_GetLambda(self._h), dtype=float)
            self.num_pixels = int(self.wavelength.size)
            if self.num_pixels <= 0:
                raise AvaspecError("Empty wavelength array")
            self.load_calibration_from_config(silent=True)
        except Exception as e:
            self._h = None
            raise AvaspecError(f"Activate failed: {e}") from e
    def deactivate(self) -> None:
        if self._h:
            try:
                avs.AVS_Deactivate(self._h)
            finally:
                self._h = None
    @classmethod
    def list_spectrometers(cls) -> list[Any]:
        try:
            avs.AVS_Init()
            return avs.AVS_GetList() or []
        except Exception:
            return []
    def set_measurement_parameters(self, int_time_ms: float, no_avg: int) -> None:
        if self._h is None:
            raise AvaspecError("Not activated")
        avs.set_measure_params(self._h, float(int_time_ms), int(no_avg))
    def set_integration_time_ms(self, int_time_ms: float) -> None:
        self.set_measurement_parameters(float(int_time_ms), 1)
    def get_wavelengths(self) -> np.ndarray:
        if self.wavelength is None:
            raise AvaspecError("Not activated")
        return np.asarray(self.wavelength, dtype=float)
    def set_background(self, wl: np.ndarray, counts: np.ndarray) -> None:
        self._bg_wavelength = np.asarray(wl, dtype=float)
        self._bg_counts = np.asarray(counts, dtype=float)
    def reset_background(self) -> None:
        self._bg_wavelength = None
        self._bg_counts = None
    def _apply_background(self, wl: np.ndarray, counts: np.ndarray) -> np.ndarray:
        if self._bg_wavelength is None or self._bg_counts is None:
            return counts
        try:
            if (self._bg_counts.size == counts.size
                and np.allclose(self._bg_wavelength, wl, rtol=0, atol=0)):
                bg = self._bg_counts
            else:
                bg = np.interp(wl, self._bg_wavelength, self._bg_counts, left=np.nan, right=np.nan)
                if np.isnan(bg[0]):
                    i0 = np.flatnonzero(~np.isnan(bg))
                    if i0.size: bg[:i0[0]] = bg[i0[0]]
                if np.isnan(bg[-1]):
                    i1 = np.flatnonzero(~np.isnan(bg))
                    if i1.size: bg[i1[-1]:] = bg[i1[-1]]
            return counts - bg
        except Exception:
            return counts
    def has_calibration(self) -> bool:
        if time.monotonic() < self._suppress_until:
            return False
        return self._cal_wavelength is not None and self._cal_values is not None
    def load_calibration_from_config(self, silent: bool=False) -> Optional[Path]:
        path = _cal_path_from_config()
        if not path or not path.exists():
            self._cal_path = None
            self._cal_wavelength = None
            self._cal_values = None
            return None
        wl, val = self._read_calibration_dat(path)
        wl = np.asarray(wl, dtype=float).ravel()
        val = np.asarray(val, dtype=float).ravel()
        if wl.size == 0 or wl.size != val.size:
            raise AvaspecError("Invalid calibration file: empty or mismatched lengths.")
        eps = np.finfo(float).tiny
        val = np.where(np.isfinite(val) & (np.abs(val) > 0), val, eps)
        self._cal_path = path
        self._cal_wavelength = wl
        self._cal_values = val
        return path
    def _read_calibration_dat(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.loadtxt(path, dtype=float, delimiter=',')
        if arr.ndim == 1 and arr.size >= 2:
            arr = arr.reshape(-1, 2)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise AvaspecError("Calibration .dat must have 2 columns: wavelength, value.")
        return arr[:, 0], arr[:, 1]
    def _apply_calibration(self, wl: np.ndarray, counts: np.ndarray) -> np.ndarray:
        if not self.has_calibration():
            return counts
        try:
            cal = np.interp(wl, self._cal_wavelength, self._cal_values, left=np.nan, right=np.nan)
            if np.isnan(cal[0]):
                i0 = np.flatnonzero(~np.isnan(cal))
                if i0.size: cal[:i0[0]] = cal[i0[0]]
            if np.isnan(cal[-1]):
                i1 = np.flatnonzero(~np.isnan(cal))
                if i1.size: cal[i1[-1]:] = cal[i1[-1]]
            eps = np.finfo(float).tiny
            cal = np.where(np.isfinite(cal) & (np.abs(cal) > 0), cal, eps)
            return counts / cal
        except Exception:
            return counts
    def apply_processing(self, wl: np.ndarray, counts: np.ndarray, use_calibration: bool) -> np.ndarray:
        y = self._apply_background(wl, counts)
        if use_calibration:
            if not self.has_calibration():
                self.load_calibration_from_config(silent=True)
            y = self._apply_calibration(wl, y)
        return y
    def _suppress_processing_for(self, seconds: float) -> None:
        self._suppress_until = max(self._suppress_until, time.monotonic() + float(seconds))
    def grab_spectrum_for_scan(self, int_ms: float, averages: int) -> Tuple[np.ndarray, dict]:
        if averages <= 0:
            averages = 1
        buf = []
        last_ts = 0.0
        for _ in range(int(averages)):
            ts, data = self.measure_spectrum(float(int_ms), 1)
            last_ts = ts
            buf.append(np.asarray(data, dtype=float))
        counts_raw = np.mean(np.stack(buf, axis=0), axis=0)
        wl = self.get_wavelengths()
        y = np.asarray(counts_raw, dtype=float)
        meta = {
            "Integration_ms": float(int_ms),
            "Averages": int(averages),
            "Timestamp": float(last_ts),
            "DeviceName": self.name,
            "BackgroundApplied": False,
            "CalibrationApplied": False,
            "CalibrationFile": "",
        }
        self._suppress_processing_for(2.0)
        return y, meta
    def measure_spectrum(self, int_time_ms: float, no_avg: int) -> Tuple[float, np.ndarray]:
        if self._h is None:
            raise AvaspecError("Not activated")
        avs.AVS_StopMeasure(self._h)
        time.sleep(0.02)
        self.set_measurement_parameters(int_time_ms, no_avg)
        avs.AVS_Measure(self._h)
        time.sleep(max(0.0, 0.001 * max(1, no_avg) + 0.001))
        ts, data = avs.get_spectrum(self._h)
        arr = np.asarray(data, dtype=float)
        if self.num_pixels and arr.size != self.num_pixels:
            n = self.num_pixels
            if arr.size > n:
                arr = arr[:n]
            else:
                tmp = np.zeros(n, dtype=float)
                tmp[:arr.size] = arr
                arr = tmp
        return float(ts), arr
