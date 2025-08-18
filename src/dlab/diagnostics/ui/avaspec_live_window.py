# src/dlab/diagnostics/view/AvaspecLive.py
from __future__ import annotations
import os, datetime, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QTextEdit, QMessageBox, QSplitter, QCheckBox, QApplication
)
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from dlab.boot import ROOT, get_config
from dlab.hardware.wrappers.avaspec_controller import AvaspecController, AvaspecError
from dlab.core.device_registry import REGISTRY

def _data_root() -> str:
    cfg = get_config() or {}
    return str((ROOT / (cfg.get("paths", {}).get("data_root", "C:/data"))).resolve())

def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def _append_avaspec_log(folder: str, spec_name: str, fn: str, int_ms: float, averages: int, comment: str) -> None:
    log_path = os.path.join(folder, f"{spec_name}_log_{datetime.datetime.now():%Y-%m-%d}.log")
    exists = os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write("File Name\tIntegration_ms\tAverages\tComment\n")
        f.write(f"{fn}\t{int_ms}\t{averages}\t{comment}\n")


class LiveMeasurementThread(QThread):
    spectrum_signal = pyqtSignal(float, object)
    error_signal = pyqtSignal(str)

    def __init__(self, ctrl, int_time_ms: float, no_avg: int):
        super().__init__()
        self.ctrl = ctrl
        self.int_time = float(int_time_ms)
        self.no_avg = int(no_avg)
        self._running = True

    def update_params(self, int_time_ms: float, no_avg: int) -> None:
        self.int_time = float(int_time_ms)
        self.no_avg = int(no_avg)

    def run(self) -> None:
        while self._running:
            try:
                ts, data = self.ctrl.measure_spectrum(self.int_time, self.no_avg)
                self.spectrum_signal.emit(ts, data)
                # pace by integration time (ms -> s), cap min interval
                time.sleep(max(0.02, self.int_time / 1000.0))
            except Exception as e:
                self.error_signal.emit(str(e))
                break

    def stop(self) -> None:
        self._running = False

class AvaspecLiveWindow(QWidget):
    closed = pyqtSignal()   # <-- add this
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Avaspec Live")
        self.ctrl = None
        self.thread: LiveMeasurementThread | None = None
        self.handles = []
        self.line = None
        self.fit_line = None
        self.last_data = None
        self.registry_key = None

        self._build_ui()
        
        try:
            REGISTRY.register("ui:avaspec_live", self)
        except Exception:
            pass


    def _build_ui(self):
        main = QHBoxLayout(self)
        splitter = QSplitter()

        # left panel
        left = QWidget(); left_l = QVBoxLayout(left)

        # spectrometer select
        self.spec_combo = QComboBox()
        btn_search = QPushButton("Search Spectrometers")
        btn_search.clicked.connect(self.search_spectrometers)
        left_l.addWidget(QLabel("Select Spectrometer:"))
        left_l.addWidget(self.spec_combo)
        left_l.addWidget(btn_search)

        # activate/deactivate
        btn_act = QPushButton("Activate")
        btn_deact = QPushButton("Deactivate"); btn_deact.setEnabled(False)
        btn_act.clicked.connect(self.activate_hardware)
        btn_deact.clicked.connect(self.deactivate_hardware)
        self.btn_act, self.btn_deact = btn_act, btn_deact
        left_l.addWidget(btn_act); left_l.addWidget(btn_deact)

        # measurement params
        self.int_edit = QLineEdit("100")   # ms
        self.avg_edit = QLineEdit("1")     # device-side averages (keep it here)
        left_l.addWidget(QLabel("Integration Time (ms):")); left_l.addWidget(self.int_edit)
        left_l.addWidget(QLabel("Number of Averages:"));     left_l.addWidget(self.avg_edit)

        # options
        self.cb_autoscale = QCheckBox("Autoscale"); self.cb_autoscale.setChecked(True)
        self.cb_log = QCheckBox("Log scale")
        self.cb_fit = QCheckBox("Gaussian fit")
        left_l.addWidget(self.cb_autoscale); left_l.addWidget(self.cb_log); left_l.addWidget(self.cb_fit)

        # live control
        btn_start = QPushButton("Start Live"); btn_start.clicked.connect(self.start_live)
        btn_stop  = QPushButton("Stop Live");  btn_stop.setEnabled(False); btn_stop.clicked.connect(self.stop_live)
        self.btn_start, self.btn_stop = btn_start, btn_stop
        left_l.addWidget(btn_start); left_l.addWidget(btn_stop)

        # save current spectrum
        self.comment_edit = QLineEdit("")
        self.btn_save = QPushButton("Save Spectrum"); self.btn_save.clicked.connect(self.save_spectrum)
        left_l.addWidget(QLabel("Comment:")); left_l.addWidget(self.comment_edit); left_l.addWidget(self.btn_save)

        # log
        self.log = QTextEdit(); self.log.setReadOnly(True)
        left_l.addWidget(self.log)

        splitter.addWidget(left)

        # right panel (plot)
        right = QWidget(); right_l = QVBoxLayout(right)
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Wavelength (nm)"); self.ax.set_ylabel("Counts"); self.ax.grid(True)
        self.canvas = FigureCanvas(self.fig)
        right_l.addWidget(NavigationToolbar(self.canvas, self))
        right_l.addWidget(self.canvas)
        splitter.addWidget(right)

        main.addWidget(splitter)
        self.resize(1080, 720)

    def _ui_log(self, msg: str):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{t}] {msg}")

    # ---- hardware mgmt ----
    def search_spectrometers(self):
        self.spec_combo.clear()
        lst = AvaspecController.list_spectrometers()
        if not lst:
            QMessageBox.critical(self, "Error", "No spectrometer found.")
            self._ui_log("No spectrometer found.")
            return
        self.handles = lst
        self.spec_combo.addItems([f"Spectrometer {i+1}" for i in range(len(lst))])
        self._ui_log(f"Found {len(lst)} spectrometer(s).")

    def activate_hardware(self):
        idx = self.spec_combo.currentIndex()
        if idx < 0 or idx >= len(self.handles):
            QMessageBox.critical(self, "Error", "No spectrometer selected.")
            return
        try:
            self.ctrl = AvaspecController(self.handles[idx])
            self.ctrl.activate()
            self._ui_log("Spectrometer activated.")

            from dlab.core.device_registry import REGISTRY
            key = f"spectrometer:avaspec:spec_{idx+1}"

            # évite doublons (même key ou même objet)
            try:
                for k, v in REGISTRY.items(prefix="spectrometer:avaspec:"):
                    if k == key or v is self.ctrl:
                        REGISTRY.unregister(k)
            except Exception:
                pass

            REGISTRY.register(key, self.ctrl)
            self.registry_key = key
            self._ui_log(f"Registered '{key}' in device registry.")

            self.btn_act.setEnabled(False)
            self.btn_deact.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to activate: {e}")
            self._ui_log(f"Activate failed: {e}")


    def deactivate_hardware(self):
        try:
            if self.ctrl:
                self.ctrl.deactivate()
                self._ui_log("Spectrometer deactivated.")
        finally:
            # unregister propre
            try:
                if self.registry_key:
                    from dlab.core.device_registry import REGISTRY
                    REGISTRY.unregister(self.registry_key)
                    self._ui_log(f"Unregistered '{self.registry_key}' from device registry.")
            except Exception:
                pass
            self.registry_key = None
            self.ctrl = None
            self.btn_act.setEnabled(True)
            self.btn_deact.setEnabled(False)


    # ---- live ----
    def start_live(self):
        if not self.ctrl:
            QMessageBox.critical(self, "Error", "Spectrometer not activated.")
            return
        try:
            it = float(self.int_edit.text())
            av = int(self.avg_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid integration/averages.")
            return
        self.thread = LiveMeasurementThread(self.ctrl, it, av)
        self.thread.spectrum_signal.connect(self.update_spectrum)
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

    # ---- plotting ----
    def update_spectrum(self, ts: float, data: np.ndarray):
        self.last_data = np.asarray(data, dtype=float)
        wl = np.asarray(self.ctrl.wavelength, dtype=float)

        if self.line is None:
            (self.line,) = self.ax.plot(wl, self.last_data, color="C0", label="Data")
        else:
            self.line.set_ydata(self.last_data)

        # optional gaussian fit
        if self.cb_fit.isChecked():
            def g(x, A, mu, sigma, d): return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + d
            try:
                A0 = float(self.last_data.max())
                mu0 = float(wl[self.last_data.argmax()])
                sigma0, d0 = 2.0, float(self.last_data.min())
                popt, _ = curve_fit(g, wl, self.last_data, p0=[A0, mu0, sigma0, d0], maxfev=10000)
                A_fit, mu_fit, sigma_fit, d_fit = popt
                fwhm = 2.355 * abs(sigma_fit)
                # transform-limited pulse estimate (fs) for Gaussian spectrum
                c = 299792458.0
                mu_m = mu_fit * 1e-9
                fwhm_m = fwhm * 1e-9
                tau_fs = 0.441 * (mu_m**2) / (c * fwhm_m) * 1e15
                yfit = g(wl, *popt)
                if self.fit_line is None:
                    (self.fit_line,) = self.ax.plot(wl, yfit, "r--", label="Gaussian Fit")
                else:
                    self.fit_line.set_ydata(yfit)
                self.ax.legend()
                self.ax.set_title(f"Peak≈{A_fit:.0f}, FWHM={fwhm:.2f} nm, TL pulse≈{tau_fs:.0f} fs")
            except Exception:
                self.ax.set_title("Gaussian fit failed")
        else:
            if self.fit_line is not None:
                self.fit_line.remove()
                self.fit_line = None
                self.ax.set_title("")

        # scale
        if self.cb_log.isChecked():
            self.ax.set_yscale("log")
            # avoid log(0)
            ymin = max(1e-12, float(np.nanmin(self.last_data[np.isfinite(self.last_data)])))
            self.ax.set_ylim(bottom=ymin)
        else:
            self.ax.set_yscale("linear")

        if self.cb_autoscale.isChecked():
            self.ax.relim(); self.ax.autoscale_view()

        self.canvas.draw_idle()

    # ---- save ----
    def save_spectrum(self):
        if self.last_data is None or self.ctrl is None:
            QMessageBox.warning(self, "Warning", "No spectrum to save.")
            return

        now = datetime.datetime.now()
        # Folder name forced to lowercase 'avaspec' as requested
        folder = os.path.join(_data_root(), f"{now:%Y-%m-%d}", "avaspec")
        _ensure_dir(folder)

        safe_ts = now.strftime("%Y-%m-%d_%H-%M-%S")
        base = "Avaspec"
        fn = os.path.join(folder, f"{base}_Spectrum_{safe_ts}.txt")

        comment = self.comment_edit.text() or ""
        try:
            wl = np.asarray(self.ctrl.wavelength, dtype=float)
            with open(fn, "w", encoding="utf-8") as f:
                if comment:
                    f.write(f"# Comment: {comment}\n")
                f.write(f"# Timestamp: {now:%Y-%m-%d %H:%M:%S}\n")
                f.write(f"# IntegrationTime_ms: {self.int_edit.text()}\n")
                f.write(f"# Averages: {self.avg_edit.text()}\n")
                f.write("Wavelength_nm;Counts\n")
                for x, y in zip(wl, self.last_data):
                    f.write(f"{float(x):.6f};{float(y):.6f}\n")

            # append avaspec log
            try:
                int_ms = float(self.int_edit.text())
            except Exception:
                int_ms = float("nan")
            try:
                averages = int(self.avg_edit.text())
            except Exception:
                averages = 1

            _append_avaspec_log(folder, base, os.path.basename(fn), int_ms, averages, comment)

            self._ui_log(f"Spectrum saved to {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")
            self._ui_log(f"Save failed: {e}")
            
            
    def set_spectrum_from_scan(self, wl, counts):
        try:
            # si le live tourne, ne pas écraser
            if self.thread is not None:
                return

            import numpy as np
            wl = np.asarray(wl, dtype=float)
            counts = np.asarray(counts, dtype=float)
            self.last_data = counts  # pour le bouton 'Save Spectrum'

            # initialisation/MAJ des courbes
            if self.line is None:
                (self.line,) = self.ax.plot(wl, counts, label="Data")
            else:
                # si taille identique : MAJ y ; sinon on replot
                if self.line.get_xdata() is not None and len(self.line.get_xdata()) == len(wl):
                    self.line.set_ydata(counts)
                else:
                    self.ax.cla()
                    self.ax.set_xlabel("Wavelength (nm)")
                    self.ax.set_ylabel("Counts")
                    self.ax.grid(True)
                    (self.line,) = self.ax.plot(wl, counts, label="Data")
                    # reset éventuel de la courbe de fit
                    if self.fit_line is not None:
                        self.fit_line.remove()
                        self.fit_line = None
                        self.ax.set_title("")

            # autoscale/log comme dans update_spectrum()
            if self.cb_log.isChecked():
                self.ax.set_yscale("log")
                import numpy as np
                ymin = max(1e-12, float(np.nanmin(counts[np.isfinite(counts)])))
                self.ax.set_ylim(bottom=ymin)
            else:
                self.ax.set_yscale("linear")

            if self.cb_autoscale.isChecked():
                self.ax.relim()
                self.ax.autoscale_view()

            self.canvas.draw_idle()
        except Exception as e:
            self._ui_log(f"Scan-plot update failed: {e}")


    # ---- lifecycle ----
    def closeEvent(self, event):
        try:
            self.stop_live()
        finally:
            try:
                if self.ctrl:
                    self.ctrl.deactivate()
            finally:
                # idem: clean registry si besoin
                try:
                    if self.registry_key:
                        from dlab.core.device_registry import REGISTRY
                        REGISTRY.unregister(self.registry_key)
                        self._ui_log(f"Unregistered '{self.registry_key}' from device registry.")
                except Exception:
                    pass
                
                try:
                    REGISTRY.unregister("ui:avaspec_live")
                except Exception:
                    pass
                self.registry_key = None
                self.ctrl = None
        self.closed.emit()
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = AvaspecLiveWindow()
    w.show()
    sys.exit(app.exec_())
