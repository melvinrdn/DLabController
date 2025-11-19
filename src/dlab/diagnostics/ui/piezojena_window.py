from __future__ import annotations
import datetime
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QGroupBox, QTextEdit, QMessageBox
)
from PyQt5.QtGui import QDoubleValidator

from dlab.boot import get_config
from dlab.hardware.wrappers.piezojena_controller import NV40
from dlab.core.device_registry import REGISTRY


class PiezoJenaStageWindow(QWidget):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PiezoJena NV40 Controller")

        self.stage: Optional[NV40] = None

        cfg = get_config() or {}
        pcfg = (cfg.get("piezojena", {}) or {})

        self.port = str(pcfg.get("port", "COM6"))

        self.range_min, self.range_max = NV40.get_voltage_limits()

        self._build_ui()

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)

        conn_box = QGroupBox("Connection")
        cl = QHBoxLayout(conn_box)

        self.port_edit = QLineEdit(self.port)
        self.port_edit.setFixedWidth(100)

        self.btn_activate = QPushButton("Activate")
        self.btn_deactivate = QPushButton("Deactivate")
        self.btn_deactivate.setEnabled(False)

        self.btn_activate.clicked.connect(self._on_activate)
        self.btn_deactivate.clicked.connect(self._on_deactivate)

        cl.addWidget(QLabel("Port:"))
        cl.addWidget(self.port_edit)
        cl.addStretch(1)
        cl.addWidget(self.btn_activate)
        cl.addWidget(self.btn_deactivate)

        main.addWidget(conn_box)

        mot_box = QGroupBox("Motion (Volts)")
        ml = QVBoxLayout(mot_box)

        row1 = QHBoxLayout()
        self.target_abs = QLineEdit()
        self.target_abs.setPlaceholderText("Absolute V")
        self.target_abs.setValidator(QDoubleValidator(-1e3, 1e3, 3, self))
        self.btn_move_abs = QPushButton("Move To")
        self.btn_move_abs.setEnabled(False)
        self.btn_move_abs.clicked.connect(self._on_move_abs)
        row1.addWidget(self.target_abs, 1)
        row1.addWidget(self.btn_move_abs)
        ml.addLayout(row1)

        row2 = QHBoxLayout()
        self.step_rel = QLineEdit("0.1")
        self.step_rel.setFixedWidth(100)
        self.step_rel.setValidator(QDoubleValidator(-1e3, 1e3, 3, self))

        self.btn_rel_minus = QPushButton("− Step")
        self.btn_rel_plus = QPushButton("+ Step")
        for b in (self.btn_rel_minus, self.btn_rel_plus):
            b.setEnabled(False)

        self.btn_rel_minus.clicked.connect(lambda: self._on_move_rel(sign=-1))
        self.btn_rel_plus.clicked.connect(lambda: self._on_move_rel(sign=+1))

        row2.addWidget(QLabel("Step (V):"))
        row2.addWidget(self.step_rel)
        row2.addStretch(1)
        row2.addWidget(self.btn_rel_minus)
        row2.addWidget(self.btn_rel_plus)
        ml.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_force_remote = QPushButton("Re-enable Remote")
        self.btn_force_remote.setEnabled(False)
        self.btn_force_remote.clicked.connect(self._on_force_remote)
        row3.addWidget(self.btn_force_remote)
        row3.addStretch(1)
        ml.addLayout(row3)

        row4 = QHBoxLayout()
        self.cur_pos = QLineEdit()
        self.cur_pos.setReadOnly(True)
        self.cur_pos.setPlaceholderText("—")
        self.cur_pos.setFixedWidth(140)
        row4.addWidget(QLabel("Current:"))
        row4.addWidget(self.cur_pos)
        row4.addStretch(1)
        row4.addWidget(QLabel(f"Limits: {self.range_min:g} … {self.range_max:g} V"))
        ml.addLayout(row4)

        main.addWidget(mot_box)

        log_box = QGroupBox("Log")
        ll = QVBoxLayout(log_box)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        ll.addWidget(self.log)
        main.addWidget(log_box)

        main.addStretch(1)
        
    def _log(self, msg: str) -> None:
        t = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{t}] {msg}")
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum()
        )

    def _ui_enable(self, on: bool) -> None:
        self.btn_deactivate.setEnabled(on)
        self.btn_move_abs.setEnabled(on)
        self.btn_rel_minus.setEnabled(on)
        self.btn_rel_plus.setEnabled(on)
        self.btn_force_remote.setEnabled(on)

    def _on_activate(self) -> None:
        if self.stage:
            return

        port = self.port_edit.text().strip() or self.port

        try:
            stg = NV40(port=port, closed_loop=False)
            self.stage = stg
            self._log(f"Activated on {port}.")

            key = "stage:piezojena:nv40"
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

            try:
                p = self.stage.get_position()
                self.cur_pos.setText(f"{p:.3f}")
            except Exception:
                self.cur_pos.setText("—")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Activation failed: {e}")
            self._log(f"Activation failed: {e}")
            self.stage = None

    def _on_deactivate(self) -> None:
        try:
            if self.stage:
                try:
                    self.stage.set_remote_control(False)
                finally:
                    try:
                        REGISTRY.unregister("stage:piezojena:nv40")
                    except Exception:
                        pass
            self._log("Deactivated.")
        finally:
            self.stage = None
            self._ui_enable(False)
            self.btn_activate.setEnabled(True)
            self.port_edit.setEnabled(True)
            self.cur_pos.setText("—")

    def _on_force_remote(self) -> None:
        if self.stage:
            try:
                self.stage.set_remote_control(True)
                self._log("Remote mode re-enabled.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Remote enable failed: {e}")
                self._log(f"Remote enable failed: {e}")

    def _safe_refresh_position(self) -> None:
        if not self.stage:
            return
        try:
            p = self.stage.get_position()
            self.cur_pos.setText(f"{p:.3f}")
        except Exception as e:
            self._log(f"Position read failed: {e}")
            self.cur_pos.setText("—")

    def _on_move_abs(self) -> None:
        if not self.stage:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return

        t = self.target_abs.text().strip()
        if not t:
            QMessageBox.warning(self, "Error", "Enter a voltage.")
            return

        try:
            val = float(t)
        except Exception:
            QMessageBox.warning(self, "Error", "Invalid number.")
            return

        val = max(self.range_min, min(self.range_max, val))

        try:
            self.stage.set_position(val)
            self._log(f"Move to {val:.3f} V …")
            self._safe_refresh_position()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Move failed: {e}")
            self._log(f"Move failed: {e}")

    def _on_move_rel(self, sign: int) -> None:
        if not self.stage:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return

        try:
            step = float(self.step_rel.text().strip())
        except Exception:
            QMessageBox.warning(self, "Error", "Invalid step.")
            return

        try:
            cur = self.stage.get_position() or 0.0
            tgt = cur + sign * step

            if tgt < self.range_min:
                tgt = self.range_min
            if tgt > self.range_max:
                tgt = self.range_max

            self.stage.set_position(tgt)
            self._log(f"Move to {tgt:.3f} V …")
            self._safe_refresh_position()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Relative move failed: {e}")
            self._log(f"Relative move failed: {e}")


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = PiezoJenaStageWindow()
    window.show()
    sys.exit(app.exec_())
