import os
import numpy as np
from auto_att_calib import AutoAttCalib
# TODO add wavelength dependency with .sense.correction.wavelength

# Settings
motor_id = 83838295
powermeter_id = 'USB0::0x1313::0x8078::P0045634::INSTR'
stabilization_time = 1
waveplate_name = 'wp_2'
waveplate_path = os.path.join('../../../ressources/calibration', waveplate_name)
angles_to_measure = np.linspace(0, 180, 181)

# Run calibration
autoattcalib = AutoAttCalib(motor_id, powermeter_id, stabilization_time=stabilization_time)
autoattcalib.run(angles_to_measure, waveplate_name, waveplate_path)
