from __future__ import annotations
import datetime, time, os
from typing import List
import numpy as np
import pyvisa

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QTextEdit, QMessageBox, QSplitter, QCheckBox, QGroupBox
)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from dlab.core.device_registry import REGISTRY
from dlab.hardware.wrappers.powermeter_controller import PowermeterController


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
        main.setContentsMargins(6,6,6,6)
        main.setSpacing(6)
        splitter = QSplitter()

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(6,6,6,6)
        left_l.setSpacing(6)
        left.setMaximumWidth(230)

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
        self.period_edit = QLineEdit("0.1")
        self.win_edit = QLineEdit("10")
        for w in (self.wl_combo, self.avg_edit, self.period_edit, self.win_edit, self.pm_combo):
            w.setMaximumWidth(110)
        left_l.addWidget(QLabel("Wavelength (nm):")); left_l.addWidget(self.wl_combo)
        left_l.addWidget(QLabel("Averaging count:")); left_l.addWidget(self.avg_edit)
        left_l.addWidget(QLabel("Sampling period (s):")); left_l.addWidget(self.period_edit)
        left_l.addWidget(QLabel("Time window (s):")); left_l.addWidget(self.win_edit)

        range_grp = QGroupBox("Y range")
        rg_l = QVBoxLayout(range_grp); rg_l.setContentsMargins(6,6,6,6); rg_l.setSpacing(4)
        self.cb_ylim = QCheckBox("Choose range"); self.cb_ylim.toggled.connect(self._toggle_ylim_inputs)
        row_min = QHBoxLayout(); row_max = QHBoxLayout()
        self.ymin_edit = QLineEdit(""); self.ymin_edit.setMaximumWidth(90)
        self.ymax_edit = QLineEdit(""); self.ymax_edit.setMaximumWidth(90)
        self.ymin_edit.setEnabled(False); self.ymax_edit.setEnabled(False)
        row_min.addWidget(QLabel("Power min:")); row_min.addWidget(self.ymin_edit, 1)
        row_max.addWidget(QLabel("Power max:")); row_max.addWidget(self.ymax_edit, 1)
        rg_l.addWidget(self.cb_ylim)
        rg_l.addLayout(row_min)
        rg_l.addLayout(row_max)
        left_l.addWidget(range_grp)

        left_l.addWidget(QLabel("Comment:"))
        self.comment_edit = QLineEdit("")
        self.comment_edit.setMaximumWidth(200)
        left_l.addWidget(self.comment_edit)

        btn_apply = QPushButton("Apply Settings")
        btn_apply.clicked.connect(self.apply_settings)
        left_l.addWidget(btn_apply)

        btn_start = QPushButton("Start Live"); btn_start.clicked.connect(self.start_live)
        btn_stop  = QPushButton("Stop Live");  btn_stop.setEnabled(False); btn_stop.clicked.connect(self.stop_live)
        self.btn_start, self.btn_stop = btn_start, btn_stop
        left_l.addWidget(btn_start); left_l.addWidget(btn_stop)

        self.log = QTextEdit(); self.log.setReadOnly(True)
        left_l.addWidget(self.log)

        splitter.addWidget(left)

        right = QWidget(); right_l = QVBoxLayout(right)
        right_l.setContentsMargins(6,6,6,6)
        right_l.setSpacing(6)

        header = QWidget(); header_l = QHBoxLayout(header)
        header_l.setContentsMargins(0,0,0,0); header_l.setSpacing(12)
        self.lbl_big_now = QLabel("—")
        self.lbl_big_max = QLabel("—")
        self.lbl_big_mean = QLabel("—")
        self.lbl_big_now.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_big_max.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_big_mean.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_big_now.setStyleSheet("font-size: 36pt; font-weight: 800;")
        self.lbl_big_max.setStyleSheet("font-size: 24pt; font-weight: 700;")
        self.lbl_big_mean.setStyleSheet("font-size: 24pt; font-weight: 700;")
        lab_now = QLabel("Power:"); lab_now.setStyleSheet("font-size: 12pt;")
        lab_max = QLabel("Max:");   lab_max.setStyleSheet("font-size: 12pt;")
        lab_mean = QLabel("Mean:"); lab_mean.setStyleSheet("font-size: 12pt;")
        self.btn_save_window = QPushButton("Save Window")
        self.btn_save_window.clicked.connect(self.save_current_window)
        header_l.addWidget(lab_now)
        header_l.addWidget(self.lbl_big_now, 1)
        header_l.addSpacing(8)
        header_l.addWidget(lab_max)
        header_l.addWidget(self.lbl_big_max)
        header_l.addSpacing(8)
        header_l.addWidget(lab_mean)
        header_l.addWidget(self.lbl_big_mean)
        header_l.addStretch(1)
        header_l.addWidget(self.btn_save_window)
        right_l.addWidget(header)

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Time (s)"); self.ax.set_ylabel("Power (W)"); self.ax.grid(True)
        self.canvas = FigureCanvas(self.fig)
        right_l.addWidget(NavigationToolbar(self.canvas, self))
        right_l.addWidget(self.canvas)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 780])

        main.addWidget(splitter)
        self.resize(1100, 680)

    def _toggle_ylim_inputs(self, checked: bool):
        self.ymin_edit.setEnabled(checked)
        self.ymax_edit.setEnabled(checked)

    @pyqtSlot(float, float)
    def set_power_from_scan(self, ts: float, val: float) -> None:
        self.update_power(ts, val)

    @pyqtSlot(object)
    def refresh_from_device(self, dev) -> None:
        try:
            val = float(dev.fetch_power())
            self.update_power(time.time(), val)
        except Exception as e:
            self._ui_log(f"Live refresh error: {e}")

    def _ui_log(self, msg: str):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{t}] {msg}")

    def _plot_color(self) -> str:
        w = self.current_wavelength_nm()
        if w == 343:
            return "#9400D3"
        if w == 515:
            return "#00A000"
        return "#CC0000"

    def _fmt_power(self, v: float) -> str:
        a = abs(v)
        if a < 1e-6:
            return f"{v*1e9:.3g} nW"
        if a < 1e-3:
            return f"{v*1e6:.3g} µW"
        if a < 1:
            return f"{v*1e3:.3g} mW"
        return f"{v:.3g} W"

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
        if self.thread is not None:
            QMessageBox.information(self, "Live running", "Stop Live before deactivating the device.")
            return
        try:
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
            self.ctrl.set_wavelength(wl)
            self.ctrl.set_avg(av)
            try:
                self.ctrl.set_auto_range(True)
            except Exception:
                pass
            try:
                self.ctrl.set_bandwidth("high")
            except Exception:
                pass
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
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_deact.setEnabled(False)

    def stop_live(self):
        if self.thread:
            self.thread.stop(); self.thread.wait()
            self.thread = None
            self._ui_log("Live stopped.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_deact.setEnabled(True)

    def _window_arrays(self):
        if not self._t:
            return np.array([]), np.array([])
        try:
            win = float(self.win_edit.text())
        except Exception:
            win = 10.0
        if win <= 0:
            win = 10.0
        t_now = self._t[-1]
        t0_keep = t_now - win
        i0 = 0
        for i in range(len(self._t)):
            if self._t[i] >= t0_keep:
                i0 = i
                break
        x = np.asarray(self._t[i0:]) - float(self._t[i0] if len(self._t[i0:]) else t_now)
        y = np.asarray(self._y[i0:])
        return x, y

    def update_power(self, ts: float, val: float):
        self._t.append(ts)
        self._y.append(val)
        x, y = self._window_arrays()
        self.ax.cla()
        color = self._plot_color()
        self.ax.plot(x, y, color=color, linewidth=2.0)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Power (W)")
        self.ax.grid(True)
        self.ax.set_title(f"Powermeter Live — {self.current_wavelength_nm()} nm")
        if self.cb_ylim.isChecked():
            try:
                ymin = float(self.ymin_edit.text())
                ymax = float(self.ymax_edit.text())
                if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                    self.ax.set_ylim(ymin, ymax)
            except Exception:
                pass
        self.canvas.draw_idle()
        p_now = self._fmt_power(val)
        p_max = self._fmt_power(np.max(y) if y.size else val)
        p_mean = self._fmt_power(float(np.mean(y)) if y.size else val)
        self.lbl_big_now.setText(p_now)
        self.lbl_big_max.setText(p_max)
        self.lbl_big_mean.setText(p_mean)
        self.lbl_big_now.setStyleSheet(f"font-size: 36pt; font-weight: 800; color: {color};")
        self.lbl_big_max.setStyleSheet(f"font-size: 24pt; font-weight: 700; color: {color};")
        self.lbl_big_mean.setStyleSheet(f"font-size: 24pt; font-weight: 700; color: {color};")

    def save_current_window(self):
        x, y = self._window_arrays()
        if y.size == 0:
            QMessageBox.warning(self, "Save Window", "No data to save.")
            return
        now = datetime.datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        base_dir = os.path.join("C:/data", date_dir, "powermeter")
        os.makedirs(base_dir, exist_ok=True)
        fname = f"powermeter_log_{now.strftime('%Y-%m-%d_%H_%M_%S')}.txt"
        path = os.path.join(base_dir, fname)
        try:
            with open(path, "w", encoding="utf-8") as f:
                comment = self.comment_edit.text().strip()
                if comment:
                    f.write(comment + "\n")
                f.write(f"Wavelength: {self.current_wavelength_nm()} nm\n")
                f.write("t_s\tpower_W\n")
                for xi, yi in zip(x, y):
                    f.write(f"{xi:.6f}\t{yi:.9g}\n")
            self._ui_log(f"Saved window to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Window", f"Failed to save: {e}")

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
