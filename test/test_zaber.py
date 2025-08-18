# test/test_zaber.py
from dlab.hardware.wrappers.zaber_controller import ZaberBinaryController

PORT = "COM4"   # change if needed
BAUD = 9600

def main():
    stage = ZaberBinaryController(PORT, BAUD)
    try:
        print("[TEST] Activating stage...")
        stage.activate(homing=True)
        print(f"[OK] Stage homed. Current pos = {stage.get_position():.3f} mm")

        print("[TEST] Moving to 5.0 mm...")
        stage.move_to(5.0)
        print(f"[OK] Position = {stage.get_position():.3f} mm")

        print("[TEST] Identify device...")
        stage.identify()
        print(f"[OK] ")

    finally:
        print("[TEST] Disabling stage...")
        stage.disable()
        print("[OK] Done.")

if __name__ == "__main__":
    main()
