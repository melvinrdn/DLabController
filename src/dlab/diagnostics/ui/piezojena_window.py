from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QMessageBox,
)
from PyQt5.QtGui import QDoubleValidator

from dlab.boot import get_config
from dlab.hardware.wrappers.piezojena_controller import NV40
from dlab.core.device_registry import REGISTRY
from dlab.utils.log_panel import LogPanel

REGISTRY_KEY = "stage:piezojena:nv40"


class PiezoJenaStageWindow(QWidget):
    """Control window for PiezoJena NV40 stage."""

    def __init__(
        self, log_panel: LogPanel | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("PiezoJena Stage")
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._log = log_panel
        self._stage: Optional[NV40] = None

        # Load configuration
        cfg = get_config() or {}
        pcfg = cfg.get("piezojena", {}) or {}
        self._port = str(pcfg.get("port", "COM6"))
        self._range_min, self._range_max = NV40.get_voltage_limits()

        self._init_ui()

    def _init_ui(self) -> None:
        main = QVBoxLayout(self)

        # Connection group
        conn_box = QGroupBox("Connection")
        cl = QHBoxLayout(conn_box)

        self._port_edit = QLineEdit(self._port)
        self._port_edit.setFixedWidth(120)

        self._activate_btn = QPushButton("Activate")
        self._deactivate_btn = QPushButton("Deactivate")
        self._deactivate_btn.setEnabled(False)

        self._activate_btn.clicked.connect(self._on_activate)
        self._deactivate_btn.clicked.connect(self._on_deactivate)

        cl.addWidget(QLabel("Port:"))
        cl.addWidget(self._port_edit)
        cl.addStretch(1)
        cl.addWidget(self._activate_btn)
        cl.addWidget(self._deactivate_btn)
        main.addWidget(conn_box)

        # Motion group
        mot_box = QGroupBox("Motion (Volts)")
        ml = QVBoxLayout(mot_box)

        # Absolute move row
        row1 = QHBoxLayout()
        self._target_abs_edit = QLineEdit()
        self._target_abs_edit.setPlaceholderText("Absolute V")
        self._target_abs_edit.setValidator(QDoubleValidator(-1e3, 1e3, 3, self))
        self._move_abs_btn = QPushButton("Move To")
        self._move_abs_btn.setEnabled(False)
        self._move_abs_btn.clicked.connect(self._on_move_abs)
        row1.addWidget(self._target_abs_edit, 1)
        row1.addWidget(self._move_abs_btn)
        ml.addLayout(row1)

        # Relative move row
        row2 = QHBoxLayout()
        self._step_rel_edit = QLineEdit("0.1")
        self._step_rel_edit.setFixedWidth(100)
        self._step_rel_edit.setValidator(QDoubleValidator(-1e3, 1e3, 3, self))
        self._rel_minus_btn = QPushButton("− Step")
        self._rel_plus_btn = QPushButton("+ Step")
        self._rel_minus_btn.setEnabled(False)
        self._rel_plus_btn.setEnabled(False)
        self._rel_minus_btn.clicked.connect(lambda: self._on_move_rel(sign=-1))
        self._rel_plus_btn.clicked.connect(lambda: self._on_move_rel(sign=+1))
        row2.addWidget(QLabel("Step (V):"))
        row2.addWidget(self._step_rel_edit)
        row2.addStretch(1)
        row2.addWidget(self._rel_minus_btn)
        row2.addWidget(self._rel_plus_btn)
        ml.addLayout(row2)

        # Control buttons row
        row3 = QHBoxLayout()
        self._force_remote_btn = QPushButton("Re-enable Remote")
        self._force_remote_btn.setEnabled(False)
        self._force_remote_btn.clicked.connect(self._on_force_remote)
        row3.addWidget(self._force_remote_btn)
        row3.addStretch(1)
        ml.addLayout(row3)

        # Position display row
        row4 = QHBoxLayout()
        self._cur_pos_edit = QLineEdit()
        self._cur_pos_edit.setReadOnly(True)
        self._cur_pos_edit.setPlaceholderText("—")
        self._cur_pos_edit.setFixedWidth(140)
        row4.addWidget(QLabel("Current:"))
        row4.addWidget(self._cur_pos_edit)
        row4.addStretch(1)
        row4.addWidget(QLabel(f"Limits: {self._range_min:g} … {self._range_max:g} V"))
        ml.addLayout(row4)

        main.addWidget(mot_box)
        main.addStretch(1)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_message(self, msg: str) -> None:
        if self._log:
            self._log.log(msg, source="PiezoJena")

    # -------------------------------------------------------------------------
    # UI state management
    # -------------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._deactivate_btn.setEnabled(enabled)
        self._move_abs_btn.setEnabled(enabled)
        self._rel_minus_btn.setEnabled(enabled)
        self._rel_plus_btn.setEnabled(enabled)
        self._force_remote_btn.setEnabled(enabled)

    # -------------------------------------------------------------------------
    # Stage control
    # -------------------------------------------------------------------------

    def _on_activate(self) -> None:
        if self._stage:
            return

        port = self._port_edit.text().strip() or self._port

        try:
            stg = NV40(port=port, closed_loop=False)
            self._stage = stg
            self._log_message(f"Activated on {port}.")

            # Register in device registry
            try:
                prev = REGISTRY.get(REGISTRY_KEY)
                if prev is not None and prev is not self._stage:
                    REGISTRY.unregister(REGISTRY_KEY)
            except Exception:
                pass
            REGISTRY.register(REGISTRY_KEY, self._stage)

            self._set_controls_enabled(True)
            self._activate_btn.setEnabled(False)
            self._port_edit.setEnabled(False)

            # Read initial position
            try:
                p = self._stage.get_position()
                self._cur_pos_edit.setText(f"{p:.3f}")
            except Exception:
                self._cur_pos_edit.setText("—")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Activation failed: {e}")
            self._log_message(f"Activation failed: {e}")
            self._stage = None

    def _on_deactivate(self) -> None:
        try:
            if self._stage:
                try:
                    self._stage.set_remote_control(False)
                finally:
                    try:
                        REGISTRY.unregister(REGISTRY_KEY)
                    except Exception:
                        pass
            self._log_message("Deactivated.")
        finally:
            self._stage = None
            self._set_controls_enabled(False)
            self._activate_btn.setEnabled(True)
            self._port_edit.setEnabled(True)
            self._cur_pos_edit.setText("—")

    def _on_force_remote(self) -> None:
        if not self._stage:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return
        try:
            self._stage.set_remote_control(True)
            self._log_message("Remote mode re-enabled.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Remote enable failed: {e}")
            self._log_message(f"Remote enable failed: {e}")

    def _on_move_abs(self) -> None:
        if not self._stage:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return

        t = self._target_abs_edit.text().strip()
        if not t:
            QMessageBox.warning(self, "Error", "Enter a voltage.")
            return

        try:
            val = float(t)
        except Exception:
            QMessageBox.warning(self, "Error", "Invalid number.")
            return

        val = max(self._range_min, min(self._range_max, val))

        try:
            self._stage.set_position(val)
            self._cur_pos_edit.setText(f"{val:.3f}")
            self._log_message(f"Move to {val:.3f} V.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Move failed: {e}")
            self._log_message(f"Move failed: {e}")

    def _on_move_rel(self, sign: int) -> None:
        if not self._stage:
            QMessageBox.warning(self, "Error", "Stage not activated.")
            return

        try:
            step = float(self._step_rel_edit.text().strip())
        except Exception:
            QMessageBox.warning(self, "Error", "Invalid step.")
            return

        try:
            cur = (
                float(self._cur_pos_edit.text())
                if self._cur_pos_edit.text() not in ("", "—")
                else 0.0
            )
            tgt = cur + sign * step
            tgt = max(self._range_min, min(self._range_max, tgt))

            self._stage.set_position(tgt)
            self._cur_pos_edit.setText(f"{tgt:.3f}")
            self._log_message(f"Move to {tgt:.3f} V.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Relative move failed: {e}")
            self._log_message(f"Relative move failed: {e}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def closeEvent(self, event):
        if self._stage:
            self._on_deactivate()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = PiezoJenaStageWindow()
    window.show()
    sys.exit(app.exec_())