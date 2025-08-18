# test/test_zaber.py
from dlab.hardware.wrappers.zaber_controller import ZaberBinaryController
from dlab.boot import get_config


def main():
    cfg = get_config()
    zcfg = cfg.get("zaber", {})

    stage = ZaberBinaryController(
        port=zcfg.get("port", "COM4"),
        baud_rate=zcfg.get("baud"),
        range_min=zcfg.get("range", {}).get("min"),
        range_max=zcfg.get("range", {}).get("max"),
)

    try:
        print("[TEST] Activating stage...")
        stage.activate(homing=True)
        print(f"[OK] Stage homed. Current pos = {stage.get_position():.3f} mm")

        print("[TEST] Moving to 50.0 mm...")
        stage.move_to(50.0)
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
