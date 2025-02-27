import thorlabs_apt as apt
from typing import Optional

class ThorlabsController:
    def __init__(self, motor_id: int) -> None:
        """
        Initialize the Thorlabs motor controller with the given motor ID.
        """
        self.motor_id: int = motor_id
        self.motor: Optional[apt.Motor] = None

    def activate(self,homing=True) -> None:
        """
        Activate (initialize) the motor.
        """
        self.motor = apt.Motor(self.motor_id)
        if homing:
            self.home()

    def home(self) -> None:
        """
        Send the motor to its home position.
        """
        if self.motor:
            self.motor.move_home(blocking=True)

    def move_to(self, position: float, blocking: bool = True) -> None:
        """
        Move the motor to a specified position.
        """
        if self.motor:
            self.motor.move_to(position, blocking=blocking)

    def get_position(self) -> Optional[float]:
        """
        Return the current motor position.
        """
        return self.motor.position if self.motor else None


    def identify(self) -> None:
        """
        Flashes the 'Active' LED at the motor to identify it.
        """
        self.motor.identify()

    def disable(self) -> None:
        """
        Disable the motor.
        """
        if self.motor:
            self.motor.disable()
            self.motor = None

## Example Usage
"""
motor_id: int = 83837725
controller = ThorlabsController(motor_id)
controller.activate()
print("Motor activated. Current position:", controller.get_position())
controller.identify()
print("Motor identified.")
controller.disable()
"""

