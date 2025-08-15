from __future__ import annotations
from typing import Tuple, Dict, Optional
import numpy as np

class AndorRegistryCamera:
    """
    DeviceRegistry adapter:
      - name
      - set_exposure_us(int)
      - grab_frame_for_scan(...) -> (frame_u16, meta)
      - optionally pushes frames to a live UI via Qt signal
    """
    def __init__(self, controller, name: str = "AndorCam_1", live_window: Optional[object] = None):
        self.controller = controller
        self.name = name
        self.live_window = live_window  # <-- add this

    def set_exposure_us(self, exposure_us: int) -> None:
        # AndorController semble accepter les µs directement:
        if hasattr(self.controller, "set_exposure"):
            self.controller.set_exposure(int(exposure_us))
        elif hasattr(self.controller, "setExposure"):
            self.controller.setExposure(int(exposure_us))

    def _capture_one(self, exposure_us: int) -> np.ndarray:
        if hasattr(self.controller, "capture_single"):
            img = self.controller.capture_single(int(exposure_us))
        else:
            img = self.controller.take_image(int(exposure_us), 1)
        return np.asarray(img, dtype=np.float64)

    def grab_frame_for_scan(
        self,
        averages: int = 1,
        background: bool = False,
        dead_pixel_cleanup: bool = True,
        exposure_us: int | None = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, int | str]]:
        if exposure_us is not None:
            self.set_exposure_us(int(exposure_us))

        n = max(1, int(averages))
        frames = [self._capture_one(int(exposure_us) if exposure_us is not None else 0) for _ in range(n)]
        avg = np.mean(np.stack(frames, axis=0), axis=0)
        
        out = np.clip(avg, 0, 65535).astype(np.uint16, copy=False)
        meta = {
            "CameraName": self.name,
            "Exposure_us": int(exposure_us) if exposure_us is not None else 0,
            "Background": "1" if background else "0",
        }

        # NEW: push the grabbed frame to the AndorLive UI
        try:
            if self.live_window is not None and hasattr(self.live_window, "external_image_signal"):
                self.live_window.external_image_signal.emit(out)
        except Exception:
            pass

        return out, meta
