# src/dlab/diagnostics/ui/grating_compressor.py
from __future__ import annotations
import datetime
from typing import Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QGroupBox, QTextEdit, QMessageBox
)
from PyQt5.QtGui import QDoubleValidator

from dlab.boot import get_config
from dlab.hardware.wrappers.zaber_controller import ZaberBinaryController
from dlab.core.device_registry import REGISTRY


class GratingCompressorWindow(QWidget):
    """
    Dedicated UI to control a single Zaber (Binary) translation stage in millimetres.
    No power mode, no waveplate logic. Safe clamping using range from config.
    """
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Grating Compressor")
        self.stage: Optional[ZaberBinaryController] = None

        # --- config ---
        cfg = get_config() or {}
        zcfg = (cfg.get("zaber", {}) or {})
        self.port = str(zcfg.get("port", "COM4"))
        self.baud = int(zcfg.get("baud", 9600))
        rng = zcfg.get("range", {}) or {}
        self.range_min = float(rng.get("min", 0.0))
        self.range_max = float(rng.get("max", 50.0))

        self._build_ui()
        self._poll = QTimer(self)
        self._poll.setInterval(200)  # ms
        self._poll.timeout.connect(self._update_position)

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        main = QVBoxLayout(self)

        # Connection panel
        conn_box = QGroupBox("Connection")
        cl = QHBoxLayout(conn_box)

        self.port_edit = QLineEdit(self.port); self.port_edit.setFixedWidth(120)
        self.baud_edit = QLineEdit(str(self.baud)); self.baud_edit.setFixedWidth(80)
        self.baud_edit.setValidator(QDoubleValidator(0, 1e9, 0, self))

        self.btn_activate = QPushButton("Activate")
        self.btn_deactivate = QPushButton("Deactivate"); self.btn_deactivate.setEnabled(False)

        self.btn_activate.clicked.connect(self._on_activate)
        self.btn_deactivate.clicked.connect(self._on_deactivate)

        cl.addWidget(QLabel("Port:")); cl.addWidget(self.port_edit)
        cl.addWidget(QLabel("Baud:")); cl.addWidget(self.baud_edit)
        cl.addStretch(1)
        cl.addWidget(self.btn_activate); cl.addWidget(self.btn_deactivate)
        main.addWidget(conn_box)

        # Motion panel
        mot_box = QGroupBox("Motion (mm)")
        ml = QVBoxLayout(mot_box)

        row1 = QHBoxLayout()
        self.target_abs = QLineEdit(); self.target_abs.setPlaceholderText("Absolute position (mm)")
        self.target_abs.setValidator(QDoubleValidator(-1e6, 1e6, 3, self))
        self.btn_move_abs = QPushButton("Move To")
        self.btn_move_abs.setEnabled(False)
        self.btn_move_abs.clicked.connect(self._on_move_abs)
        row1.addWidget(self.target_abs, 1); row1.addWidget(self.btn_move_abs)
        ml.addLayout(row1)

        row2 = QHBoxLayout()
        self.step_rel = QLineEdit("0.001"); self.step_rel.setFixedWidth(100)
        self.btn_rel_minus = QPushButton("− Step")
        self.btn_rel_plus  = QPushButton("+ Step")
        for b in (self.btn_rel_minus, self.btn_rel_plus):
            b.setEnabled(False)
        self.btn_rel_minus.clicked.connect(lambda: self._on_move_rel(sign=-1))
        self.btn_rel_plus.clicked.connect(lambda: self._on_move_rel(sign=+1))
        row2.addWidget(QLabel("Step (mm):")); row2.addWidget(self.step_rel)
        row2.addStretch(1)
        row2.addWidget(self.btn_rel_minus); row2.addWidget(self.btn_rel_plus)
        ml.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_home = QPushButton("Home"); self.btn_home.setEnabled(False)
        self.btn_ident = QPushButton("Identify"); self.btn_ident.setEnabled(False)
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.btn_home.clicked.connect(self._on_home)
        self.btn_ident.clicked.connect(self._on_ident)
        self.btn_stop.clicked.connect(self._on_stop)
        row3.addWidget(self.btn_home); row3.addWidget(self.btn_ident); row3.addWidget(self.btn_stop)
        ml.addLayout(row3)

        row4 = QHBoxLayout()
        self.cur_pos = QLineEdit(); self.cur_pos.setReadOnly(True); self.cur_pos.setPlaceholderText("—")
        self.cur_pos.setFixedWidth(140)
        row4.addWidget(QLabel("Current:")); row4.addWidget(self.cur_pos)
        row4.addStretch(1)
        row4.addWidget(QLabel(f"Limits: {self.range_min:g} … {self.range_max:g} mm"))
        ml.addLayout(row4)

        main.addWidget(mot_box)

        # Log panel
        log_box = QGroupBox("Log")
        ll = QVBoxLayout(log_box)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        ll.addWidget(self.log)
        main.addWidget(log_box)

        main.addStretch(1)

    # ------------- helpers -------------
    def _log(self, msg: str) -> None:
        t = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{t}] {msg}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _update_position(self) -> None:
        if not self.stage:
            self._poll.stop(); return
        try:
            p = self.stage.get_position()
            if p is not None:
                self.cur_pos.setText(f"{p:.3f}")
        except Exception as e:
            self._log(f"Position read failed: {e}")
            self._poll.stop()

    def _ui_enable(self, on: bool) -> None:
        self.btn_deactivate.setEnabled(on)
        self.btn_move_abs.setEnabled(on)
        self.btn_rel_minus.setEnabled(on)
        self.btn_rel_plus.setEnabled(on)
        self.btn_home.setEnabled(on)
        self.btn_ident.setEnabled(on)
        self.btn_stop.setEnabled(on)

    # ------------- slots -------------
    def _on_activate(self) -> None:
        if self.stage:
            return
        port = self.port_edit.text().strip() or self.port
        try:
            baud = int(float(self.baud_edit.text().strip()))
        except Exception:
            baud = self.baud

        try:
            stg = ZaberBinaryController(
                port=port,
                baud_rate=baud,
                range_min=self.range_min,
                range_max=self.range_max,
            )
            stg.activate(homing=True)
            self.stage = stg
            self._log(f"Activated on {port} @ {baud}. Homed.")
            # registry
            key = "stage:zaber:grating_compressor"
            try:
                prev = REGISTRY.get(key)
                if prev is not None and prev is not self.stage:
                    REGISTRY.unregister(key)
            except Exception:
                pass
            REGISTRY.register(key, self.stage)

            self._ui_enable(True)
            self.btn_activate.setEnabled(False)
            self.port_edit.setEnabled(False)
            self.baud_edit.setEnabled(False)
            self._poll.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Activation failed: {e}")
            self._log(f"Activation failed: {e}")
            self.stage = None

    def _on_deactivate(self) -> None:
        try:
            if self.stage:
                try:
                    self.stage.disable()
                finally:
                    # unregister
                    try:
                        REGISTRY.unregister("stage:zaber:grating_compressor")
                    except Exception:
                        pass
            self._log("Deactivated.")
        finally:
            self.stage = None
            self._ui_enable(False)
            self.btn_activate.setEnabled(True)
            self.port_edit.setEnabled(True)
            self.baud_edit.setEnabled(True)
            self._poll.stop()

    def _on_home(self) -> None:
        if not self.stage:
            QMessageBox.warning(self, "Error", "Stage not activated."); return
        try:
            self.stage.home(blocking=False)
            self._log("Homing…")
            self._poll.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Home failed: {e}")
            self._log(f"Home failed: {e}")

    def _on_ident(self) -> None:
        if not self.stage:
            QMessageBox.warning(self, "Error", "Stage not activated."); return
        try:
            self.stage.identify()
            self._log("Identify.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Identify failed: {e}")
            self._log(f"Identify failed: {e}")

    def _on_stop(self) -> None:
        if not self.stage:
            return
        try:
            # best effort stop
            self.stage._ensure().stop()  # direct device stop
            self._log("Stop requested.")
        except Exception as e:
            self._log(f"Stop failed: {e}")

    def _on_move_abs(self) -> None:
        if not self.stage:
            QMessageBox.warning(self, "Error", "Stage not activated."); return
        t = self.target_abs.text().strip()
        if not t:
            QMessageBox.warning(self, "Error", "Enter a target position (mm)."); return
        try:
            val = float(t)
            self.stage.move_to(val, blocking=False)  # clamp inside
            self._log(f"Move to {val:.3f} mm …")
            self._poll.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Move failed: {e}")
            self._log(f"Move failed: {e}")

    def _on_move_rel(self, sign: int) -> None:
        if not self.stage:
            QMessageBox.warning(self, "Error", "Stage not activated."); return
        try:
            step = float(self.step_rel.text().strip())
        except Exception:
            QMessageBox.warning(self, "Error", "Invalid step."); return
        try:
            cur = self.stage.get_position() or 0.0
            tgt = cur + sign * step
            # clamp
            if tgt < self.range_min: tgt = self.range_min
            if tgt > self.range_max: tgt = self.range_max
            self.stage.move_to(tgt, blocking=False)
            self._log(f"Move to {tgt:.3f} mm …")
            self._poll.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Relative move failed: {e}")
            self._log(f"Relative move failed: {e}")

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = GratingCompressorWindow()
    window.show()
    sys.exit(app.exec_())