import ctypes
import time
import threading
from ctypes import (
    c_uint, c_char_p, c_void_p, c_int, c_uint32, c_size_t,
    byref, create_string_buffer
)

SA_OK = 0
# According to MCS docs, status is a bitfield; use bitwise check instead of equality
SA_TARGET_STATUS = 1 << 2  # 0b0100

class MCSLib:
    """ctypes binding with explicit signatures to avoid default int coercions."""
    def __init__(self, dll_path: str):
        lib = ctypes.windll.LoadLibrary(dll_path)

        # Discovery & open/close
        lib.SA_FindSystems.argtypes = [c_char_p, ctypes.c_char_p, ctypes.POINTER(c_uint)]
        lib.SA_FindSystems.restype = c_int

        lib.SA_OpenSystem.argtypes = [ctypes.POINTER(c_uint), c_char_p, c_char_p]
        lib.SA_OpenSystem.restype = c_int

        lib.SA_CloseSystem.argtypes = [c_uint]
        lib.SA_CloseSystem.restype = c_int

        # Info
        lib.SA_GetNumberOfChannels.argtypes = [c_uint, ctypes.POINTER(c_uint)]
        lib.SA_GetNumberOfChannels.restype = c_int

        # Status
        lib.SA_GetStatus_S.argtypes = [c_uint, c_int, ctypes.POINTER(c_uint)]
        lib.SA_GetStatus_S.restype = c_int

        # Motion
        lib.SA_StepMove_S.argtypes = [c_uint, c_int, c_int, c_int, c_int]
        lib.SA_StepMove_S.restype = c_int

        self.lib = lib


class SmarActMCS:
    """
    Python wrapper for SmarAct MCS Control using MCSControl.dll
    - Safe ctypes signatures
    - Bitfield status handling
    - Timeout & thread-safety
    - Context-manager support
    """
    def __init__(self, dll_path: str):
        self._lib = MCSLib(dll_path).lib
        self._system_index = c_uint(0)
        self._lock = threading.Lock()
        self._is_open = False

    # ---------- helpers ----------
    def _check(self, code: int, func: str):
        if code != SA_OK:
            raise RuntimeError(f"{func} failed with code {code}")

    @staticmethod
    def _parse_locators(buf: bytes) -> list[str]:
        s = buf.decode(errors="ignore").strip()
        if not s:
            return []
        # SA_FindSystems may return multiple locators separated by newlines or semicolons
        parts = []
        for sep in ("\n", ";"):
            if sep in s:
                parts = [p.strip() for p in s.split(sep)]
                break
        if not parts:
            parts = [s]
        return [p for p in parts if p]

    # ---------- lifecycle ----------
    def open(self, locator: str | None = None, options: str = "sync"):
        """
        Discover and open the first system unless a specific 'locator' is provided.
        """
        with self._lock:
            if self._is_open:
                return

            if locator is None:
                buf = create_string_buffer(1024)
                size = c_uint(len(buf))
                self._check(self._lib.SA_FindSystems(None, buf, byref(size)), "SA_FindSystems")
                locators = self._parse_locators(buf.value)
                if not locators:
                    raise RuntimeError("No MCS system found.")
                locator = locators[0]

            self._check(
                self._lib.SA_OpenSystem(byref(self._system_index),
                                        c_char_p(locator.encode()),
                                        c_char_p(options.encode())),
                "SA_OpenSystem",
            )
            self._is_open = True

    def close(self):
        with self._lock:
            if not self._is_open:
                return
            self._check(self._lib.SA_CloseSystem(self._system_index), "SA_CloseSystem")
            self._is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------- info ----------
    def get_num_channels(self) -> int:
        with self._lock:
            count = c_uint(0)
            self._check(self._lib.SA_GetNumberOfChannels(self._system_index, byref(count)),
                        "SA_GetNumberOfChannels")
            return int(count.value)

    def get_status(self, channel: int) -> int:
        with self._lock:
            status = c_uint(0)
            self._check(self._lib.SA_GetStatus_S(self._system_index, int(channel), byref(status)),
                        "SA_GetStatus_S")
            return int(status.value)

    # ---------- motion ----------
    def step_move(self,
                  channel: int,
                  steps: int,
                  amplitude: int = 4095,
                  frequency: int = 2000,
                  timeout: float = 10.0,
                  poll_period: float = 0.02):
        """
        Perform a step move and wait until the channel is idle.

        Parameters
        ----------
        channel : int
            Channel index (0-based per MCS convention).
        steps : int
            Number of steps (pulses), can be negative.
        amplitude : int
            Step amplitude (1..4095 typically).
        frequency : int
            Step frequency in Hz.
        timeout : float
            Max time to wait for completion in seconds.
        poll_period : float
            Polling interval for status in seconds.
        """
        with self._lock:
            self._check(
                self._lib.SA_StepMove_S(self._system_index, int(channel),
                                        int(steps), int(amplitude), int(frequency)),
                "SA_StepMove_S",
            )

        # Wait without holding the lock to not block other read ops
        t0 = time.perf_counter()
        while True:
            status = self.get_status(channel)
            # Bitfield check: active target movement bit cleared => done
            if (status & SA_TARGET_STATUS) == 0:
                break
            if time.perf_counter() - t0 > timeout:
                raise TimeoutError(f"StepMove on channel {channel} timed out after {timeout}s "
                                   f"(status=0x{status:08X}).")
            time.sleep(poll_period)


if __name__ == "__main__":
    dll = "./src/dlab/hardware/drivers/smaract_driver/MCSControl.dll"
    with SmarActMCS(dll) as mcs:
        print("Opening MCS system...")
        print("Number of channels:", mcs.get_num_channels())
        channel = 1
        steps = -2000
        amplitude = 4095
        frequency = 2000
        print(f"Moving channel {channel} {steps} steps with amplitude {amplitude} and frequency {frequency}")
        mcs.step_move(channel, steps, amplitude, frequency)
        print("Move completed.")
    print("MCS system closed.")
