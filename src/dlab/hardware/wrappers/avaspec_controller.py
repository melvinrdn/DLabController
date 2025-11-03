from __future__ import annotations
import time, threading
import numpy as np
from pathlib import Path
import dlab.hardware.drivers.avaspec_driver._avs_py as avs
import dlab.hardware.drivers.avaspec_driver._avs_win as avs_win
from dlab.boot import ROOT, get_config

class AvaspecError(Exception):
    pass

def _cal_path_from_config():
    cfg = get_config() or {}
    rel = (cfg.get("avaspec", {}) or {}).get("calibration_file")
    if not rel:
        return None
    p = Path(rel)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p if p.exists() else None

def _is_pending(exc):
    s = str(exc)
    return "ERR_OPERATION_PENDING" in s or "-5" in s

class AvaspecController:
    def __init__(self, spec_handle):
        self._orig_handle = spec_handle
        self._h = None
        self._wl = None
        self._npx = None
        self._bg_wl = None
        self._bg_counts = None
        self._cal_wl = None
        self._cal_vals = None
        self._cal_enabled = False
        self._int_ms = 100.0
        self._avg = 1
        self._io_lock = threading.Lock()

    def activate(self):
        try:
            avs.AVS_Init()
            self._h = avs.AVS_Activate(self._orig_handle)
            if self._h is None:
                raise AvaspecError("AVS_Activate returned None")
            try:
                avs_win.AVS_UseHighResAdc(self._h, True)
            except Exception:
                pass
            wl = np.asarray(avs.AVS_GetLambda(self._h), dtype=float)
            if wl.size == 0:
                raise AvaspecError("Empty wavelength array")
            self._wl = wl
            self._npx = int(wl.size)
        except Exception as e:
            self._h = None
            raise AvaspecError(f"Activate failed: {e}") from e

    def deactivate(self):
        with self._io_lock:
            try:
                if self._h is not None:
                    try:
                        avs.AVS_StopMeasure(self._h)
                    except Exception:
                        pass
                    avs.AVS_Deactivate(self._h)
            finally:
                self._h = None

    @classmethod
    def list_spectrometers(cls):
        try:
            avs.AVS_Init()
            return avs.AVS_GetList() or []
        except Exception:
            return []

    def set_params(self, int_time_ms, averages):
        if self._h is None:
            raise AvaspecError("Not activated")
        int_time_ms = float(max(1.0, float(int_time_ms)))
        averages = int(max(1, int(averages)))
        with self._io_lock:
            tries = 0
            while True:
                try:
                    try:
                        avs.AVS_StopMeasure(self._h)
                    except Exception:
                        pass
                    avs.set_measure_params(self._h, int_time_ms, averages)
                    self._int_ms = int_time_ms
                    self._avg = averages
                    return
                except Exception as e:
                    if _is_pending(e) and tries < 100:
                        tries += 1
                        time.sleep(0.01)
                        continue
                    raise

    def set_background(self, wl_nm, counts):
        self._bg_wl = np.asarray(wl_nm, float).ravel()
        self._bg_counts = np.asarray(counts, float).ravel()

    def clear_background(self):
        self._bg_wl = None
        self._bg_counts = None

    def enable_calibration(self, enabled):
        self._cal_enabled = bool(enabled)
        if enabled and (self._cal_wl is None or self._cal_vals is None):
            p = _cal_path_from_config()
            if p:
                self._load_calibration_file(p)

    def _load_calibration_file(self, path):
        arr = np.loadtxt(str(path), dtype=float, delimiter=",")
        if arr.ndim == 1 and arr.size >= 2:
            arr = arr.reshape(-1, 2)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise AvaspecError("Calibration .dat must have 2 columns: wavelength,value")
        wl = np.asarray(arr[:, 0], float)
        vals = np.asarray(arr[:, 1], float)
        eps = np.finfo(float).tiny
        vals = np.where(np.isfinite(vals) & (np.abs(vals) > 0), vals, eps)
        self._cal_wl = wl
        self._cal_vals = vals

    def _apply_background(self, wl_nm, counts):
        if self._bg_wl is None or self._bg_counts is None:
            return counts
        bg = np.interp(wl_nm, self._bg_wl, self._bg_counts, left=np.nan, right=np.nan)
        if np.isnan(bg[0]):
            j = np.flatnonzero(~np.isnan(bg))
            if j.size:
                bg[:j[0]] = bg[j[0]]
        if np.isnan(bg[-1]):
            j = np.flatnonzero(~np.isnan(bg))
            if j.size:
                bg[j[-1]:] = bg[j[-1]]
        return counts - bg

    def _apply_calibration(self, wl_nm, counts):
        if not self._cal_enabled or self._cal_wl is None or self._cal_vals is None:
            return counts
        cal = np.interp(wl_nm, self._cal_wl, self._cal_vals, left=np.nan, right=np.nan)
        if np.isnan(cal[0]):
            j = np.flatnonzero(~np.isnan(cal))
            if j.size:
                cal[:j[0]] = cal[j[0]]
        if np.isnan(cal[-1]):
            j = np.flatnonzero(~np.isnan(cal))
            if j.size:
                cal[j[-1]:] = cal[j[-1]]
        eps = np.finfo(float).tiny
        cal = np.where(np.isfinite(cal) & (np.abs(cal) > 0), cal, eps)
        return counts / cal

    def measure_once(self):
        if self._h is None:
            raise AvaspecError("Not activated")
        with self._io_lock:
            tries = 0
            while True:
                try:
                    try:
                        avs.AVS_StopMeasure(self._h)
                    except Exception:
                        pass
                    avs.set_measure_params(self._h, self._int_ms, self._avg)
                    avs.AVS_Measure(self._h)
                    wait_s = max(0.0, 0.001 * max(1, self._avg) + 0.001)
                    time.sleep(wait_s)
                    ts, data = avs.get_spectrum(self._h)
                    counts = np.asarray(data, float).ravel()
                    if self._npx and counts.size != self._npx:
                        n = self._npx
                        out = np.zeros(n, float)
                        out[:min(n, counts.size)] = counts[:min(n, counts.size)]
                        counts = out
                    wl = np.asarray(self._wl, float).ravel()
                    return float(ts), wl, counts
                except Exception as e:
                    if _is_pending(e) and tries < 100:
                        tries += 1
                        time.sleep(0.01)
                        continue
                    raise

    def process_counts(self, wl_nm, counts):
        y = self._apply_background(wl_nm, counts)
        y = self._apply_calibration(wl_nm, y)
        return y
