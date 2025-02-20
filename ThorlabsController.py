# ThorlabsController.py
import thorlabs_apt as apt

class ThorlabsController:
    def __init__(self, motor_id):
        """
        Initialize the Thorlabs motor controller with the given motor ID.
        (Activation is done manually by calling activate().)
        """
        self.motor_id = motor_id
        self.motor = None

    def activate(self):
        """
        Activate (initialize) the motor.
        """
        self.motor = apt.Motor(self.motor_id)
        self.home()

    def home(self):
        """Send the motor to its home position."""
        if self.motor:
            self.motor.move_home(blocking=True)

    def move_to(self, position, blocking=True):
        """Move the motor to a specified position."""
        if self.motor:
            self.motor.move_to(position, blocking=blocking)

    def get_position(self):
        """Return the current motor position."""
        return self.motor.position if self.motor else None

    def disable(self):
        """Disable (close) the motor."""
        if self.motor:
            self.motor.disable()
            self.motor = None
