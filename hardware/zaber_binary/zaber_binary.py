import serial
import zaber_motion.binary
from zaber_motion.binary import Connection


class ZaberStage:

    def __init__(self, comport):
        self.comport = comport
        self.binit = False
        self.baud_rate = 9600
        self.device_address = 0
        self.port = comport
        self.protocol = None
        self.device = None
        self.range = None
        self.con = None

        try:
            connection = Connection.open_serial_port(self.port)
            self.con = connection
            device_list = self.con.detect_devices()
            print("Found {} devices".format(len(device_list)))

            if not device_list:
                print("No devices found.")
                return

            self.device = device_list[0]
        except:
            print("Oh no, init failed")

    def set_position(self, pos):
        self.device.move_absolute(pos, "mm")

    def get_position(self):
        current_position = self.device.get_position("mm")
        return current_position

    def close(self):
        if self.con is not None:
            self.con.close()
            self.con = None
            self.device = None

    #def __del__(self):
    #    if self.device is not None:
    #        self.port.close()
    #        self.device = None
