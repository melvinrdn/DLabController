import time
import numpy as np

from PyQt5.QtWidgets import QApplication

from dlab.hardware.wrappers.slm_controller import SLMController
from dlab.hardware.wrappers.phase_settings import PhaseSettings


def get_phase_ref(type_name: str):
    dummy_parent = None
    return PhaseSettings.new_type(dummy_parent, type_name)

def main():
    app = QApplication([])
    TYPE = "TwoFociStochastic"
    phase_ref = get_phase_ref(TYPE)
    slm = SLMController("red")
    for A in np.linspace(0, 1.0, 11):
        phase_ref.le_amp.setText(f"{A:.3f}")
        levels = phase_ref.phase()
        slm.publish(levels, screen_num=1)
        time.sleep(1)
    slm.close()

if __name__ == "__main__":
    main()
