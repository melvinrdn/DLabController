#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-

# Author: Philipp Klaus, philipp.l.klaus AT web.de

# This file is part of PfeifferVacuum.py.
#
# PfeifferVacuum.py is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PfeifferVacuum.py is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PfeifferVacuum.py. If not, see <http://www.gnu.org/licenses/>.

import serial
import time
import signal
import math
from threading import Thread, Event


class MaxiGauge:
    def __init__(self, serialPort, baud=9600, debug=False):
        self.debug = debug
        try:
            self.connection = serial.Serial(serialPort, baudrate=baud, timeout=0.2)
        except serial.serialutil.SerialException as se:
            raise MaxiGaugeError(se)
        self.logfilename = 'measurement-data.txt'

    def checkDevice(self):
        message = "The Display Contrast is currently set to %d (out of 20).\n" % self.displayContrast()
        message += "Keys since MaxiGauge was switched on: %s (out of 1,2,3,4,5).\n" % ", ".join(map(str, self.pressedKeys()))
        return message

    def pressedKeys(self):
        keys = int(self.send('TKB', 1)[0])
        pressedKeys = []
        for i in [4, 3, 2, 1, 0]:  # It's got 5 keys
            if keys // 2 ** i == 1:
                pressedKeys.append(i + 1)
                keys = keys % 2 ** i
        pressedKeys.reverse()
        return pressedKeys

    def displayContrast(self, newContrast=-1):
        if newContrast == -1:
            return int(self.send('DCC', 1)[0])
        else:
            return int(self.send('DCC,%d' % (newContrast,), 1)[0])

    def pressures(self):
        return [self.pressure(i + 1) for i in range(6)]

    def pressure(self, sensor):
        if sensor < 1 or sensor > 6:
            raise MaxiGaugeError(f"Sensor can only be between 1 and 6. You chose {sensor}")
        reading = self.send(f'PR{sensor}', 1)  # Reading will have the form x,x.xxxEsx <CR><LF> (see p.88)
        try:
            r = reading[0].split(',')
            status = int(r[0])
            pressure = float(r[-1])
        except:
            raise MaxiGaugeError(f"Problem interpreting the returned line:\n{reading}")
        return PressureReading(sensor, status, pressure)

    def signal_handler(self, sig, frame):
        self.stopping_continuous_update.set()
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def start_continuous_pressure_updates(self, update_time, log_every=0):
        self.stopping_continuous_update = Event()
        signal.signal(signal.SIGINT, self.signal_handler)
        self.update_time = update_time
        self.log_every = log_every
        self.update_counter = 1
        self.t = Thread(target=self.continuous_pressure_updates)
        self.t.daemon = True
        self.t.start()

    def continuous_pressure_updates(self):
        cache = []
        while not self.stopping_continuous_update.is_set():
            startTime = time.time()
            self.update_counter += 1
            self.cached_pressures = self.pressures()
            cache.append([time.time()] + [sensor.pressure if sensor.status in [0, 1, 2] else float('nan') for sensor in self.cached_pressures])
            if self.log_every > 0 and (self.update_counter % self.log_every == 0):
                logtime = cache[self.log_every // 2][0]
                cache = list(zip(*cache))  # Transpose cache
                cache = cache[1:]  # Remove the first element
                avgs = [sum(vals) / self.log_every for vals in cache]
                self.log_to_file(logtime=logtime, logvalues=avgs)
                cache = []
            time.sleep(0.1)  # We want a minimum pause of 0.1 s
            while not self.stopping_continuous_update.is_set() and (self.update_time - (time.time() - startTime) > .2):
                time.sleep(.2)
            time.sleep(max([0., self.update_time - (time.time() - startTime)]))
        if self.log_every > 0 and (self.update_counter % self.log_every == 0):
            self.flush_logfile()

    def log_to_file(self, logtime=None, logvalues=None):
        if not hasattr(self, 'logfile'):
            self.logfile = open(self.logfilename, 'a')
        if not logtime:
            logtime = time.time()
        if not logvalues:
            logvalues = [sensor.pressure if sensor.status in [0, 1, 2] else float('nan') for sensor in self.cached_pressures]
        line = "%d, " % logtime + ', '.join(["%.3E" % val if not math.isnan(val) else '' for val in logvalues])
        self.logfile.write(line + '\n')

    def flush_logfile(self):
        try:
            self.logfile.flush()
        except:
            pass

    def debugMessage(self, message):
        if self.debug:
            print(repr(message))

    def send(self, mnemonic, numEnquiries=0):
        self.connection.flushInput()
        self.write(mnemonic + LINE_TERMINATION)
        self.getACQorNAK()
        response = []
        for i in range(numEnquiries):
            self.enquire()
            response.append(self.read())
        return response

    def write(self, what):
        self.debugMessage(what)
        self.connection.write(what.encode())  # Encode string for serial communication

    def enquire(self):
        self.write(C['ENQ'])

    def read(self):
        data = ""
        while True:
            x = self.connection.read().decode()  # Decode bytes to string
            self.debugMessage(x)
            data += x
            if len(data) > 1 and data[-2:] == LINE_TERMINATION:
                break
        return data[:-len(LINE_TERMINATION)]

    def getACQorNAK(self):
        returncode = self.connection.readline().decode()
        self.debugMessage(returncode)
        if len(returncode) > 2 and returncode[-3] == C['NAK']:
            self.enquire()
            returnedError = self.read()
            error = str(returnedError).split(',', 1)
            errmsg = {'System Error': ERR_CODES[0][int(error[0])], 'Gauge Error': ERR_CODES[1][int(error[1])]}
            raise MaxiGaugeNAK(errmsg)

    def disconnect(self):
        try:
            self.stopping_continuous_update.set()
        except:
            pass

    def __del__(self):
        self.disconnect()
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()


class PressureReading:
    def __init__(self, id, status, pressure):
        if int(id) not in range(1, 7):
            raise MaxiGaugeError('Pressure Gauge ID must be between 1-6')
        self.id = int(id)
        if int(status) not in PRESSURE_READING_STATUS.keys():
            raise MaxiGaugeError(f'The Pressure Status must be in the range {list(PRESSURE_READING_STATUS.keys())}')
        self.status = int(status)
        self.pressure = float(pressure)

    def statusMsg(self):
        return PRESSURE_READING_STATUS[self.status]

    def __repr__(self):
        return f"Gauge #{self.id}: Status {self.status} ({self.statusMsg()}), Pressure: {self.pressure} mbar\n"


### ------ Exceptions ------

class MaxiGaugeError(Exception):
    pass


class MaxiGaugeNAK(MaxiGaugeError):
    pass


### ------- Control Symbols -----------

C = {
    'ETX': "\x03",  # End of Text (Ctrl-C)   Reset the interface
    'CR': "\x0D",   # Carriage Return        Go to the beginning of line
    'LF': "\x0A",   # Line Feed              Advance by one line
    'ENQ': "\x05",  # Enquiry                Request for data transmission
    'ACQ': "\x06",  # Acknowledge            Positive report signal
    'NAK': "\x15",  # Negative Acknowledge   Negative report signal
    'ESC': "\x1b",  # Escape
}

LINE_TERMINATION = C['CR'] + C['LF']

### ------- Error codes -----------

ERR_CODES = [
    {
        0: 'No error',
        1: 'Watchdog has responded',
        2: 'Task fail error',
        4: 'IDCX idle error',
        8: 'Stack overflow error',
        16: 'EPROM error',
        32: 'RAM error',
        64: 'EEPROM error',
        128: 'Key error',
        4096: 'Syntax error',
        8192: 'Inadmissible parameter',
        16384: 'No hardware',
        32768: 'Fatal error'
    },
    {
        0: 'No error',
        1: 'Sensor 1: Measurement error',
        2: 'Sensor 2: Measurement error',
        4: 'Sensor 3: Measurement error',
        8: 'Sensor 4: Measurement error',
        16: 'Sensor 5: Measurement error',
        32: 'Sensor 6: Measurement error',
        512: 'Sensor 1: Identification error',
        1024: 'Sensor 2: Identification error',
        2048: 'Sensor 3: Identification error',
        4096: 'Sensor 4: Identification error',
        8192: 'Sensor 5: Identification error',
        16384: 'Sensor 6: Identification error',
    }
]

### ------ Pressure reading status ------

PRESSURE_READING_STATUS = {
    0: 'Measurement data okay',
    1: 'Underrange',
    2: 'Overrange',
    3: 'Sensor error',
    4: 'Sensor off',
    5: 'No sensor',
    6: 'Identification error'
}
