import serial
import time


class NV40:
    def __init__(self, comport, closed_loop = False):
        self.comport = comport
        self.s = None
        self.binit = False
        self.bLock = False
        self.s = serial.Serial(self.comport, timeout=1)
        self.s.baudrate = 9600
        self.s.write_timeout = 1
        self.s.close()
        self.s.open()
        self.closedLoop(closed_loop)
        print('from jena_piezo_V3: PiezoJena stage initialized.')
        self.binit = True
        self.remoteControl(True)


    def closedLoop(self, bClosed):
        cmd = b'cl\r' if bClosed else b'ol\r'
        self.s.write(cmd)

    def remoteControl(self, bRemote):
        time.sleep(0.1)
        cmd = b'i1\r' if bRemote else b'i0\r'
        self.s.write(cmd)
        time.sleep(0.1)

    def set_position(self, pos):
        if self.bLock:
            print('from jena_piezo_V3: Too many setposition commands, ignoring one.')
            return
        if self.binit:
            self.bLock = True
            cmd = f'wr,{pos:.2f}\r'
            self.s.write(cmd.encode())
            cnt = 0
            while True:
                time.sleep(0.05)
                pos_current = self.get_position()
                if pos_current > pos - 0.01 or pos_current < pos + 0.01:
                    break
                cnt += 1
                if cnt > 20:
                    self.bLock = False
                    raise ValueError('from jena_piezo_V3: Target position not reached.')
            self.bLock = False
        else:
            raise ValueError('from jena_piezo_V3: Class not initialized using the init() method.')

    def get_position(self):
        if self.binit:
            cnt = 0
            while True:
                self.s.write(b'rd\r')
                response = self.s.readline().decode().strip()
                if response:
                    break
                time.sleep(0.1)
                cnt += 1
                if cnt > 5:
                    raise ValueError('from jena_piezo_V3: Reading failed!')

            if response.startswith('err'):
                raise ValueError(f'from jena_piezo_V3: There was an error reading: {response}')

            pos = float(response[3:])
            return pos

        else:
            raise ValueError('from jena_piezo_V3: Class not initialized using the init() method.')

    def __del__(self):
        if self.s and self.s.is_open:
            self.s.write(b'i0\r')
            self.s.close()
        self.binit = False

# Example usage:
# piezo = PiezoJenaNV40('COM1')  # Replace 'COM1' with your actual serial port
# piezo.init()
# piezo.setPosition(10.0)
# print(piezo.getPosition())
# del piezo
