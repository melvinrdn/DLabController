import thorlabs_apt as apt

class ThorlabsController:
    def __init__(self, motor_id):
        """
        Initialize the Thorlabs motor controller with the given motor ID
        """
        self.motor = apt.Motor(motor_id)
    def home(self):
        """Send the motor to its home position."""
        self.motor.move_home(blocking=True)

    def move_to(self, position, blocking=True):
        """Move the motor to a specified position."""
        self.motor.move_to(position, blocking=blocking)

    def get_position(self):
        """Return the current motor position."""
        return self.motor.position

    def disable(self):
        """Disable (close) the motor."""
        self.motor.disable()