import serial
import time


class NV40:
    ERRORS = {
        'err,1': 'Unknown command',
        'err,2': 'Too many characters in the command',
        'err,3': 'Too many characters in the parameter',
        'err,4': 'Too many parameters',
        'err,5': 'Wrong character in parameter',
        'err,6': 'Wrong separator',
        'err,7': 'Position out of range',
    }

    V_MIN = 0.0
    V_MAX = 140.0
    RANGE_UM = 100.0

    def __init__(self, port, timeout=0.2, closed_loop=False):
        self.ser = serial.Serial(
            port=port,
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_NONE,
            timeout=timeout,
        )
        time.sleep(0.1)

        self.set_remote_control(True)

        if closed_loop:
            raise RuntimeError(
                "Closed-loop mode was requested but this NV40 controller "
                "does not support it. Use closed_loop=False."
            )

        self.__execute("ol")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __execute(self, cmd: str) -> str:
        self.ser.reset_input_buffer()
        self.ser.write((cmd + "\r").encode())
        time.sleep(0.1)

        ans = self.ser.read(100).decode(errors="ignore").strip()

        if ans in self.ERRORS:
            raise ValueError(self.ERRORS[ans])

        return ans

    def set_remote_control(self, enable: bool = True) -> str:
        return self.__execute("i1" if enable else "i0")

    def set_closed_loop(self, enable: bool = True) -> str:
        if enable:
            raise RuntimeError(
                "Closed-loop mode is not supported on this controller. "
                "This call would result in undefined behavior."
            )
        return self.__execute("ol")

    def set_position(self, value: float) -> str:
        value = max(self.V_MIN, min(self.V_MAX, value))
        return self.__execute(f"wr,{value:.3f}")

    def get_position(self) -> float:
        ans = self.__execute("rd")
        return float(ans.split(",")[1])

    @classmethod
    def get_voltage_limits(cls):
        return cls.V_MIN, cls.V_MAX


if __name__ == "__main__":
    dev = NV40("COM6")

    import numpy as np

    for v in np.arange(10, 11, 0.1):
        dev.set_position(v)
        print(v, dev.get_position())

    dev.close()
