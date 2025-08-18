# src/dlab/hardware/wrappers/zaber_controller.py
from __future__ import annotations
from typing import Optional

from zaber_motion import Library, Units
from zaber_motion.binary import Connection, Device


class ZaberNotActivatedError(RuntimeError):
    """Raised when an operation requires an activated stage."""


class ZaberBinaryController:
    """
    Thin wrapper for Zaber Binary devices.
    Public API mirrors ThorlabsController: activate/home/move_to/get_position/identify/disable.
    """

    def __init__(self, port: str, baud_rate: int = 9600, units: Units = Units.LENGTH_MILLIMETRES):
        self.port = port
        self.baud_rate = baud_rate
        self.units = units

        self.conn: Optional[Connection] = None
        self.device: Optional[Device] = None

    def _ensure(self) -> Device:
        if self.device is None:
            raise ZaberNotActivatedError("Stage not activated. Call activate() first.")
        return self.device

    def activate(self, homing: bool = True) -> None:
        """Open serial, detect first device, optionally home."""
        Library.enable_device_db_store()
        self.conn = Connection.open_serial_port(self.port, baud_rate=self.baud_rate)
        devices = self.conn.detect_devices()
        if not devices:
            self.disable()
            raise ZaberNotActivatedError(f"No Zaber devices found on {self.port}@{self.baud_rate}")
        self.device = devices[0]
        if homing:
            self.home()

    def home(self, blocking: bool = True) -> None:
        """Home the stage."""
        dev = self._ensure()
        dev.home()
        if blocking:
            dev.wait_until_idle()

    def move_to(self, position: float, blocking: bool = True) -> None:
        """Move to an absolute position in units (default mm)."""
        dev = self._ensure()
        dev.move_absolute(float(position), self.units)
        if blocking:
            dev.wait_until_idle()

    def get_position(self) -> Optional[float]:
        """Return the current position in configured units."""
        if self.device is None:
            return None
        return float(self.device.get_position(self.units))

    def identify(self) -> None:
        """Flash LEDs or otherwise identify the device."""
        self._ensure().identify()

    def disable(self) -> None:
        """Stop motion and close the connection."""
        try:
            if self.device is not None:
                self.device.stop()
        finally:
            if self.conn is not None:
                self.conn.close()
        self.device = None
        self.conn = None