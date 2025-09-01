# src/dlab/diagnostics/view/PowermeterLive.py
from __future__ import annotations
import datetime, time
from typing import List
import numpy as np
import pyvisa

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QTextEdit, QMessageBox, QSplitter, QCheckBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from dlab.core.device_registry import REGISTRY
from dlab.hardware.wrappers.powermeter_controller import PowermeterController, PowermeterControllerError


class LivePowerThread(QThread):
    power_signal = pyqtSignal(float, float)
    error_signal = pyqtSignal(str)

    def __init__(self, ctrl: PowermeterController, period_s: float):
        super().__init__()
        self.ctrl = ctrl
        self.period = float(max(0.02, period_s))
        self._running = True

    def update_period(self, period_s: float) -> None:
        self.period = float(max(0.02, period_s))

    def run(self) -> None:
        while self._running:
            try:
                val = float(self.ctrl.read_power())
                ts = time.time()
                self.power_signal.emit(ts, val)
                time.sleep(self.period)
            except Exception as e:
                self.error_signal.emit(str(e))
                break

    def stop(self) -> None:
        self._running = False


class PowermeterLiveWindow(QWidget):
    closed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Powermeter Live")
        self.ctrl: PowermeterController | None = None
        self.thread: LivePowerThread | None = None
        self.registry_key = None
        self._t: List[float] = []
        self._y: List[float] = []
        self._build_ui()
        try:
            REGISTRY.register("ui:powermeter_live", self)
        except Exception:
            pass

    def _build_ui(self):
        main = QHBoxLayout(self)
        splitter = QSplitter()

        left = QWidget(); left_l = QVBoxLayout(left)

        self.pm_combo = QComboBox()
        btn_search = QPushButton("Search Powermeters")
        btn_search.clicked.connect(self.search_powermeters)
        left_l.addWidget(QLabel("Select Powermeter:"))
        left_l.addWidget(self.pm_combo)
        left_l.addWidget(btn_search)

        btn_act = QPushButton("Activate")
        btn_deact = QPushButton("Deactivate"); btn_deact.setEnabled(False)
        btn_act.clicked.connect(self.activate_hardware)
        btn_deact.clicked.connect(self.deactivate_hardware)
        self.btn_act, self.btn_deact = btn_act, btn_deact
        left_l.addWidget(btn_act); left_l.addWidget(btn_deact)

        self.wl_combo = QComboBox(); self.wl_combo.addItems(["1030","515","343"])
        self.avg_edit = QLineEdit("1")
        self.cb_auto = QCheckBox("Auto range"); self.cb_auto.setChecked(True)
        self.bw_combo = QComboBox(); self.bw_combo.addItems(["high","low"])
        self.period_edit = QLineEdit("0.1")
        self.win_edit = QLineEdit("30")
        left_l.addWidget(QLabel("Wavelength (nm):")); left_l.addWidget(self.wl_combo)
        left_l.addWidget(QLabel("Averaging count:")); left_l.addWidget(self.avg_edit)
        left_l.addWidget(self.cb_auto)
        left_l.addWidget(QLabel("Bandwidth:")); left_l.addWidget(self.bw_combo)
        left_l.addWidget(QLabel("Sampling period (s):")); left_l.addWidget(self.period_edit)
        left_l.addWidget(QLabel("Time window (s):")); left_l.addWidget(self.win_edit)

        btn_apply = QPushButton("Apply Settings")
        btn_apply.clicked.connect(self.apply_settings)
        left_l.addWidget(btn_apply)

        btn_start = QPushButton("Start Live"); btn_start.clicked.connect(self.start_live)
        btn_stop  = QPushButton("Stop Live");  btn_stop.setEnabled(False); btn_stop.clicked.connect(self.stop_live)
        self.btn_start, self.btn_stop = btn_start, btn_stop
        left_l.addWidget(btn_start); left_l.addWidget(btn_stop)

        self.lbl_now = QLabel("Power: — W")
        left_l.addWidget(self.lbl_now)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        left_l.addWidget(self.log)

        splitter.addWidget(left)

        right = QWidget(); right_l = QVBoxLayout(right)
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Time (s)"); self.ax.set_ylabel("Power (W)"); self.ax.grid(True)
        self.canvas = FigureCanvas(self.fig)
        right_l.addWidget(NavigationToolbar(self.canvas, self))
        right_l.addWidget(self.canvas)
        splitter.addWidget(right)

        main.addWidget(splitter)
        self.resize(1000, 640)

    def _ui_log(self, msg: str):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{t}] {msg}")

    def _plot_color(self) -> str:
        w = self.current_wavelength_nm()
        if w == 343:
            return "violet"
        if w == 515:
            return "green"
        return "red"

    def current_wavelength_nm(self) -> int:
        try:
            return int(self.wl_combo.currentText())
        except Exception:
            return 1030

    def search_powermeters(self):
        self.pm_combo.clear()
        try:
            rm = pyvisa.ResourceManager()
            res = list(rm.list_resources())
        except Exception:
            res = []
        if not res:
            QMessageBox.critical(self, "Error", "No VISA resources found.")
            self._ui_log("No VISA resources found.")
            return
        self.pm_combo.addItems(res)
        self._ui_log(f"Found {len(res)} resource(s).")

    def activate_hardware(self):
        idx = self.pm_combo.currentIndex()
        if idx < 0:
            QMessageBox.critical(self, "Error", "No powermeter selected.")
            return
        res = self.pm_combo.currentText().strip()
        if not res:
            QMessageBox.critical(self, "Error", "Invalid resource.")
            return
        try:
            self.ctrl = PowermeterController(res)
            self.ctrl.activate()
            self.apply_settings()
            key = f"powermeter:thorlabs:pm_{idx+1}"
            try:
                for k, v in REGISTRY.items(prefix="powermeter:thorlabs:"):
                    if k == key or v is self.ctrl:
                        REGISTRY.unregister(k)
            except Exception:
                pass
            REGISTRY.register(key, self.ctrl)
            self.registry_key = key
            self._ui_log(f"Activated and registered '{key}'.")
            self.btn_act.setEnabled(False)
            self.btn_deact.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate: {e}")
            self._ui_log(f"Activate failed: {e}")

    def deactivate_hardware(self):
        try:
            if self.thread:
                self.thread.stop(); self.thread.wait()
                self.thread = None
            if self.ctrl:
                self.ctrl.deactivate()
                self._ui_log("Powermeter deactivated.")
        finally:
            try:
                if self.registry_key:
                    REGISTRY.unregister(self.registry_key)
                    self._ui_log(f"Unregistered '{self.registry_key}'.")
            except Exception:
                pass
            self.registry_key = None
            self.ctrl = None
            self.btn_act.setEnabled(True)
            self.btn_deact.setEnabled(False)
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def apply_settings(self):
        if not self.ctrl:
            QMessageBox.warning(self, "Warning", "Powermeter not activated.")
            return
        try:
            wl = float(self.current_wavelength_nm())
            av = int(float(self.avg_edit.text()))
            ar = bool(self.cb_auto.isChecked())
            bw = self.bw_combo.currentText().strip().lower()
            self.ctrl.set_wavelength(wl)
            self.ctrl.set_avg(av)
            self.ctrl.set_auto_range(ar)
            self.ctrl.set_bandwidth(bw)
            self._ui_log("Settings applied.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {e}")
            self._ui_log(f"Apply failed: {e}")

    def start_live(self):
        if not self.ctrl:
            QMessageBox.critical(self, "Error", "Powermeter not activated.")
            return
        try:
            period = float(self.period_edit.text())
        except Exception:
            period = 0.1
        self._t.clear(); self._y.clear()
        self.thread = LivePowerThread(self.ctrl, period)
        self.thread.power_signal.connect(self.update_power)
        self.thread.error_signal.connect(lambda e: self._ui_log(f"Error: {e}"))
        self.thread.start()
        self._ui_log("Live started.")
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)

    def stop_live(self):
        if self.thread:
            self.thread.stop(); self.thread.wait()
            self.thread = None
            self._ui_log("Live stopped.")
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)

    def update_power(self, ts: float, val: float):
        self._t.append(ts)
        self._y.append(val)
        try:
            win = float(self.win_edit.text())
        except Exception:
            win = 30.0
        if win <= 0:
            win = 30.0
        t_now = self._t[-1]
        t0_keep = t_now - win
        i0 = 0
        for i in range(len(self._t)):
            if self._t[i] >= t0_keep:
                i0 = i
                break
        x = np.asarray(self._t[i0:]) - float(self._t[i0] if len(self._t[i0:]) else t_now)
        y = np.asarray(self._y[i0:])
        self.ax.cla()
        self.ax.plot(x, y, color=self._plot_color())
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Power (W)")
        self.ax.grid(True)
        self.canvas.draw_idle()
        self.lbl_now.setText(f"Power: {val:.6g} W")

    def closeEvent(self, event):
        try:
            self.stop_live()
        finally:
            try:
                if self.ctrl:
                    self.ctrl.deactivate()
            finally:
                try:
                    if self.registry_key:
                        REGISTRY.unregister(self.registry_key)
                except Exception:
                    pass
                try:
                    REGISTRY.unregister("ui:powermeter_live")
                except Exception:
                    pass
                self.registry_key = None
                self.ctrl = None
        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    w = PowermeterLiveWindow()
    w.show()
    sys.exit(app.exec_())
