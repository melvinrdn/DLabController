from __future__ import annotations
import os
import json
import numpy as np

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.image as mpimg

from dlab.hardware.wrappers.slm_controller import DEFAULT_BEAM_RADIUS_ON_SLM, DEFAULT_SLM_SIZE, DEFAULT_CHIP_W, DEFAULT_CHIP_H, DEFAULT_PIXEL_SIZE, DEFAULT_BIT_DEPTH
from prysm import coordinates, polynomials

from dlab.boot import get_config

def _cfg(path, default=None):
    cfg = get_config() or {}
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

w_L = float(_cfg("slm.beam_radius_on_slm", DEFAULT_BEAM_RADIUS_ON_SLM))

slm_size   = DEFAULT_SLM_SIZE
chip_width = DEFAULT_CHIP_W
chip_height= DEFAULT_CHIP_H
pixel_size = DEFAULT_PIXEL_SIZE
bit_depth  = DEFAULT_BIT_DEPTH

phase_types = ['Background', 'Lens', 'Zernike', 'Flat', 'Binary', 'Vortex', 'PhaseJumps', 'Grating', 'Square','Checkerboard', 'Tilt', 'Axicon']

class BaseTypeWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def _read_file(self, filepath):
        if not filepath:
            return
        try:
            if filepath.endswith('.csv'):
                try:
                    self.img = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=np.arange(1920) + 1)
                except Exception:
                    self.img = np.loadtxt(filepath, delimiter=',')
            else:
                self.img = mpimg.imread(filepath)
                if self.img.ndim == 3:
                    self.img = self.img.sum(axis=2)
        except Exception as e:
            print('Error reading file "{}": {}'.format(filepath, e))

    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open File", "",
                                                  "CSV Files (*.csv);;Image Files (*.bmp);;All Files (*)")
        if filepath:
            self._read_file(filepath)
            if hasattr(self, 'lbl_file'):
                self.lbl_file.setText(filepath)

    def name_(self):
        return self.name

    def close_(self):
        self.close()

class TypeFlat(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = 'Flat'
        layout = QVBoxLayout(self)
        group = QGroupBox("Flat")
        layout.addWidget(group)
        hlayout = QHBoxLayout(group)
        lbl = QLabel("Phase shift ({} = 2π):".format(bit_depth))
        self.le_flat = QLineEdit("512")
        hlayout.addWidget(lbl)
        hlayout.addWidget(self.le_flat)

    def phase(self):
        try:
            phi = float(self.le_flat.text()) if self.le_flat.text() != '' else 0
        except ValueError:
            phi = 0
        return np.ones(slm_size) * phi

    def save_(self):
        return {'flat_phase': self.le_flat.text()}

    def load_(self, settings):
        self.le_flat.setText(settings.get('flat_phase', '0'))


class TypeBackground(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = 'Background'
        self.img = None
        layout = QVBoxLayout(self)
        group = QGroupBox("Background Correction File")
        layout.addWidget(group)
        vlayout = QVBoxLayout(group)
        self.btn_open = QPushButton("Open Background file")
        self.btn_open.clicked.connect(self.open_background_file)
        vlayout.addWidget(self.btn_open)
        self.lbl_file = QLabel("")
        self.lbl_file.setWordWrap(True)
        vlayout.addWidget(self.lbl_file)

    def open_background_file(self):
        initial_directory = os.path.join('.', 'ressources', 'background')
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Background File", initial_directory,
                                                  "CSV Files (*.csv);;Image Files (*.bmp);;Text Files (*.txt);;All Files (*)")
        if filepath:
            self._read_file(filepath)
            self.lbl_file.setText(filepath)

    def phase(self):
        return self.img if self.img is not None else np.zeros(slm_size)

    def save_(self):
        return {'filepath': self.lbl_file.text()}

    def load_(self, settings):
        filepath = settings.get('filepath', '')
        self.lbl_file.setText(filepath)
        if filepath:
            self._read_file(filepath)


class TypeLens(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = 'Lens'
        self.updating = False
        layout = QVBoxLayout(self)
        group = QGroupBox("Virtual Lens Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        # Mode selection
        grid.addWidget(QLabel("Mode:"), 0, 0)
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Bending Strength", "Focal Length"])
        grid.addWidget(self.cb_mode, 0, 1)
        self.cb_mode.currentTextChanged.connect(self.toggle_mode)

        # Create parameter fields
        labels = ['Bending Strength (1/f) [1/m]:', 'Focal Length [m]:', 'Wavelength [nm]:',
                  'Calibration Slope [mm * m]:', 'Zero Reference [1/f]:', 'Focus Shift [mm]:']
        self.le_ben = QLineEdit("0")
        self.le_focal = QLineEdit("1")
        self.le_wavelength = QLineEdit("1030")
        self.le_slope = QLineEdit("1.0")
        self.le_zero = QLineEdit("0")
        self.le_focus = QLineEdit("0")

        self.le_ben.textChanged.connect(self.update_from_ben)
        self.le_focus.textChanged.connect(self.update_ben)
        self.le_focal.textChanged.connect(self.update_ben_from_focal)

        self.param_fields = [self.le_ben, self.le_focal, self.le_wavelength,
                             self.le_slope, self.le_zero, self.le_focus]
        for i, (text, le) in enumerate(zip(labels, self.param_fields)):
            grid.addWidget(QLabel(text), i + 1, 0)
            grid.addWidget(le, i + 1, 1)

        self.toggle_mode()  # set initial mode

    def toggle_mode(self):
        mode = self.cb_mode.currentText()
        if mode == "Bending Strength":
            self.le_ben.setEnabled(True)
            self.le_focal.setEnabled(False)
        else:
            self.le_ben.setEnabled(False)
            self.le_focal.setEnabled(True)

    def update_ben(self):
        if self.updating or self.cb_mode.currentText() == "Focal Length":
            return
        self.updating = True
        try:
            slope = float(self.le_slope.text())
            zero_ref = float(self.le_zero.text())
            focus_shift = float(self.le_focus.text())
            bending_strength = zero_ref + focus_shift / slope
            self.le_ben.setText(str(round(bending_strength, 3)))
            self.le_focal.setText(str(round(1 / bending_strength, 3)) if bending_strength != 0 else "inf")
        except ValueError:
            print("Invalid entry in bending strength calculation.")
        self.updating = False

    def update_from_ben(self):
        if self.updating:
            return
        self.updating = True
        try:
            slope = float(self.le_slope.text())
            zero_ref = float(self.le_zero.text())
            bending_strength = float(self.le_ben.text())
            focus_shift = slope * (bending_strength - zero_ref)
            self.le_focus.setText(str(round(focus_shift, 2)))
        except ValueError:
            print("Invalid entry in focus position calculation.")
        self.updating = False

    def update_ben_from_focal(self):
        if self.updating or self.cb_mode.currentText() == "Bending Strength":
            return
        self.updating = True
        try:
            focal_length = float(self.le_focal.text())
            bending_strength = 1 / focal_length if focal_length != 0 else 0
            self.le_ben.setText(str(round(bending_strength, 3)))
        except ValueError:
            print("Invalid entry in focal length calculation.")
        self.updating = False

    def phase(self):
        try:
            bending_strength = float(self.le_ben.text())
            wavelength = float(self.le_wavelength.text()) * 1e-9  # convert to meters
        except ValueError:
            print("Invalid input for bending strength or wavelength.")
            return np.zeros(slm_size)

        if bending_strength == 0:
            return np.zeros(slm_size)

        focal_length = 1 / bending_strength
        x = np.linspace(-chip_width / 2, chip_width / 2, slm_size[1])
        y = np.linspace(-chip_height / 2, chip_height / 2, slm_size[0])
        X, Y = np.meshgrid(x, y)
        R_squared = X ** 2 + Y ** 2

        # Thin lens phase formula (preserves sign)
        phase_profile = (-np.pi * R_squared) / (wavelength * focal_length)

        # Normalize to bit depth (e.g., 8-bit or 16-bit)
        phase_profile = np.mod(phase_profile, 2 * np.pi)  # wrap phase to [0, 2pi]
        phase_profile = phase_profile / (2 * np.pi) * bit_depth

        return phase_profile

    def save_(self):
        return {
            'ben': self.le_ben.text(),
            'focal_length': self.le_focal.text(),
            'wavelength': self.le_wavelength.text(),
            'slope': self.le_slope.text(),
            'zeroref': self.le_zero.text(),
            'focuspos': self.le_focus.text(),
            'mode': self.cb_mode.currentText()
        }

    def load_(self, settings):
        self.le_ben.setText(settings.get('ben', "0"))
        self.le_focal.setText(settings.get('focal_length', "1"))
        self.le_wavelength.setText(settings.get('wavelength', "500"))
        self.le_slope.setText(settings.get('slope', "1.0"))
        self.le_zero.setText(settings.get('zeroref', "0"))
        self.le_focus.setText(settings.get('focuspos', "0"))
        mode = settings.get('mode', 'Bending Strength')
        index = self.cb_mode.findText(mode)
        if index != -1:
            self.cb_mode.setCurrentIndex(index)
        self.toggle_mode()


class TypeZernike(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = 'Zernike'
        self.filepath = ""
        layout = QVBoxLayout(self)
        group = QGroupBox("Zernike Polynomials")
        layout.addWidget(group)
        vlayout = QVBoxLayout(group)

        hbox = QHBoxLayout()
        self.btn_browse = QPushButton("Browse File")
        self.btn_browse.clicked.connect(self.load_file)
        hbox.addWidget(self.btn_browse)
        self.btn_modify = QPushButton("Modify File")
        self.btn_modify.clicked.connect(self.modify_file)
        self.btn_modify.setEnabled(False)
        hbox.addWidget(self.btn_modify)
        self.btn_update = QPushButton("Update Data")
        self.btn_update.clicked.connect(self.update_data)
        self.btn_update.setEnabled(False)
        hbox.addWidget(self.btn_update)
        vlayout.addLayout(hbox)

        self.lbl_file = QLabel("")
        self.lbl_file.setWordWrap(True)
        vlayout.addWidget(self.lbl_file)

        # Matplotlib figure and canvas
        self.fig = Figure(figsize=(3, 1.5))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Zernike Coefficients", fontsize=10)
        self.ax.set_xlabel("Mode (j)", fontsize=8)
        self.ax.set_ylabel("Coef (nm RMS)", fontsize=8)
        self.fig.tight_layout()
        self.canvas = FigureCanvas(self.fig)
        vlayout.addWidget(self.canvas)

    def load_file(self):
        initial_directory = os.path.join('.', 'ressources', 'aberration_correction')
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Zernike Coefficients File", initial_directory,
                                                  "Text Files (*.txt);;All Files (*)")
        if filepath:
            self.filepath = filepath
            self.lbl_file.setText(filepath)
            self.plot_data()
            self.btn_modify.setEnabled(True)
            self.btn_update.setEnabled(True)

    def modify_file(self):
        if os.path.isfile(self.filepath):
            if os.name == 'posix':
                subprocess.Popen(['open', self.filepath])
            else:
                subprocess.Popen(['notepad', self.filepath])

    def update_data(self):
        self.plot_data()

    def plot_data(self):
        if self.filepath:
            try:
                data = np.loadtxt(self.filepath, skiprows=1)
                js = data[:, 0].astype(int)
                coefs = data[:, 1]
                self.update_plot(js, coefs)
            except Exception as e:
                print("Error loading file:", e)

    def update_plot(self, js, coefs):
        self.ax.clear()
        self.ax.bar(js, coefs, color='blue', alpha=0.8)
        self.ax.set_title("Zernike Coefficients", fontsize=10)
        self.ax.set_xlabel("Mode (j)", fontsize=8)
        self.ax.set_ylabel("Coef (nm RMS)", fontsize=8)
        self.ax.set_xticks(np.arange(min(js), max(js) + 1, 2))
        self.ax.grid(True)
        self.canvas.draw()

    def phase(self):
        if not self.filepath:
            print("No file loaded for Zernike coefficients.")
            return np.zeros(slm_size)
        try:
            data = np.loadtxt(self.filepath, skiprows=1)
            js = data[:, 0].astype(int)
            zernike_coefs = data[:, 1]
            size = 1.92  # horizontal size in wL
            x, y = coordinates.make_xy_grid(slm_size[1], diameter=size)
            r, t = coordinates.cart_to_polar(x, y)
            nms = [polynomials.noll_to_nm(j) for j in js]
            zernike_basis = list(polynomials.zernike_nm_sequence(nms, r, t))
            phase = polynomials.sum_of_2d_modes(zernike_basis, zernike_coefs)
            start_row = (slm_size[1] - 1200) // 2
            phase = phase[start_row:start_row + 1200, :]
            return phase
        except Exception as e:
            print("Error computing Zernike phase:", e)
            return np.zeros(slm_size)

    def save_(self):
        return {'filepath': self.lbl_file.text()}

    def load_(self, settings):
        self.filepath = settings.get('filepath', '')
        self.lbl_file.setText(self.filepath)
        self.plot_data()
        if self.filepath:
            self.btn_modify.setEnabled(True)
            self.btn_update.setEnabled(True)

class TypeTilt(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = 'Tilt'
        layout = QVBoxLayout(self)
        group = QGroupBox("Tilt Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Tilt X (rad/m):"), 0, 0)
        self.le_tx = QLineEdit("0.0")
        grid.addWidget(self.le_tx, 0, 1)

        grid.addWidget(QLabel("Tilt Y (rad/m):"), 1, 0)
        self.le_ty = QLineEdit("0.0")
        grid.addWidget(self.le_ty, 1, 1)

    def phase(self):
        try:
            tx = float(self.le_tx.text())
            ty = float(self.le_ty.text())
        except ValueError:
            return np.zeros(slm_size)

        x = np.linspace(-chip_width/2, chip_width/2, slm_size[1])
        y = np.linspace(-chip_height/2, chip_height/2, slm_size[0])
        X, Y = np.meshgrid(x, y)

        ramp = tx * X + ty * Y
        wrapped = np.mod(ramp, 2*np.pi)
        return wrapped * (bit_depth / (2*np.pi))

    def save_(self):
        return {'tilt_x': self.le_tx.text(), 'tilt_y': self.le_ty.text()}

    def load_(self, settings):
        self.le_tx.setText(settings.get('tilt_x', '0.0'))
        self.le_ty.setText(settings.get('tilt_y', '0.0'))

class TypeVortex(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = 'Vortex'
        self.vortices = []
        layout = QVBoxLayout(self)
        group = QGroupBox("Vortex Beam Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Radius (wL):"), 0, 0)
        self.le_radius = QLineEdit("10")
        grid.addWidget(self.le_radius, 0, 1)

        grid.addWidget(QLabel("Vortex Order:"), 1, 0)
        self.le_order = QLineEdit("1")
        grid.addWidget(self.le_order, 1, 1)

        btn_add = QPushButton("Add Vortex")
        btn_add.clicked.connect(self.add_vortex)
        grid.addWidget(btn_add, 2, 0)
        btn_remove = QPushButton("Remove Last Vortex")
        btn_remove.clicked.connect(self.remove_last_vortex)
        grid.addWidget(btn_remove, 2, 1)

        self.lbl_vortices = QLabel("No vortices added")
        self.lbl_vortices.setWordWrap(True)
        layout.addWidget(self.lbl_vortices)

    def add_vortex(self):
        try:
            radius = float(self.le_radius.text())
            order = int(self.le_order.text())
            self.vortices.append((radius, order))
            self.update_vortex_display()
            self.le_radius.clear()
            self.le_order.setText("1")
        except ValueError:
            print("Invalid input for radius or order.")

    def remove_last_vortex(self):
        if self.vortices:
            self.vortices.pop()
            self.update_vortex_display()

    def update_vortex_display(self):
        if not self.vortices:
            self.lbl_vortices.setText("No vortices added")
        else:
            text = "\n".join(["Radius: {:.2f} wL, Order: {}".format(r, o) for r, o in self.vortices])
            self.lbl_vortices.setText(text)

    def phase(self):
        x = np.linspace(-chip_width, chip_width, slm_size[1])
        y = np.linspace(-chip_height, chip_height, slm_size[0])
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X ** 2 + Y ** 2) / 2
        phase_profile = np.zeros(slm_size)
        for radius, order in self.vortices:
            radius_scaled = radius * w_L
            vortex_mask = (rho <= radius_scaled)
            theta = np.arctan2(Y, X)
            vortex_phase = (order * theta) % (2 * np.pi)
            phase_profile[vortex_mask] += vortex_phase[vortex_mask]
        return (phase_profile % (2 * np.pi)) * (bit_depth / (2 * np.pi))

    def save_(self):
        return {'vortices': self.vortices}

    def load_(self, settings):
        self.vortices = settings.get('vortices', [])
        self.update_vortex_display()


class TypeGrating(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = 'Grating'
        layout = QVBoxLayout(self)
        group = QGroupBox("Grating Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Spatial Frequency (lines/mm):"), 0, 0)
        self.le_freq = QLineEdit("0")
        grid.addWidget(self.le_freq, 0, 1)

        grid.addWidget(QLabel("Orientation Angle (degrees):"), 1, 0)
        self.le_angle = QLineEdit("0")
        grid.addWidget(self.le_angle, 1, 1)

    def phase(self):
        try:
            freq = float(self.le_freq.text() or "1000")
            angle = float(self.le_angle.text() or "0")
        except ValueError:
            freq = 1000
            angle = 0
        period = 1 / (freq * 1e-3)
        angle_rad = np.radians(angle)
        x = np.linspace(-chip_width / 2, chip_width / 2, slm_size[1])
        y = np.linspace(-chip_height / 2, chip_height / 2, slm_size[0])
        X, Y = np.meshgrid(x, y)
        grating_pattern = np.sin(2 * np.pi * (X * np.cos(angle_rad) + Y * np.sin(angle_rad)) / period)
        return (grating_pattern + 1) * (bit_depth / 2)

    def save_(self):
        return {'freq': self.le_freq.text(), 'angle': self.le_angle.text()}

    def load_(self, settings):
        self.le_freq.setText(settings.get('freq', '1000'))
        self.le_angle.setText(settings.get('angle', '0'))


class TypeBinary(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = 'Binary'
        layout = QVBoxLayout(self)
        group = QGroupBox("Binary Pattern Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Phase change (π units):"), 0, 0)
        self.le_phi = QLineEdit("1")
        grid.addWidget(self.le_phi, 0, 1)

        grid.addWidget(QLabel("Number of stripes:"), 1, 0)
        self.le_stripes = QLineEdit("2")
        grid.addWidget(self.le_stripes, 1, 1)

        grid.addWidget(QLabel("Angle (degrees):"), 2, 0)
        self.le_angle = QLineEdit("0")
        grid.addWidget(self.le_angle, 2, 1)

    def phase(self):
        try:
            phi = float(self.le_phi.text()) * np.pi
            stripes = int(self.le_stripes.text())
            angle_deg = float(self.le_angle.text())
            angle_rad = np.radians(angle_deg)
        except ValueError:
            print("Invalid parameter values.")
            return np.zeros(slm_size)
        phase_mat = np.zeros(slm_size)
        x = np.linspace(-chip_width / 2, chip_width / 2, slm_size[1])
        y = np.linspace(-chip_height / 2, chip_height / 2, slm_size[0])
        X, Y = np.meshgrid(x, y)
        X_rot = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
        stripe_width = chip_width / stripes
        for i in range(stripes):
            if i % 2 == 0:
                indices = (X_rot >= i * stripe_width - chip_width / 2) & (X_rot < (i + 1) * stripe_width - chip_width / 2)
                phase_mat[indices] = phi
        return (phase_mat % (2 * np.pi)) * (bit_depth / (2 * np.pi))

    def save_(self):
        return {'phi': self.le_phi.text(), 'stripes': self.le_stripes.text(), 'angle': self.le_angle.text()}

    def load_(self, settings):
        self.le_phi.setText(settings.get('phi', '1'))
        self.le_stripes.setText(settings.get('stripes', '2'))
        self.le_angle.setText(settings.get('angle', '0'))


class TypePhaseJumps(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = 'PhaseJumps'
        self.phase_jumps = []
        layout = QVBoxLayout(self)
        group = QGroupBox("Phase Jump Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Distance (wL):"), 0, 0)
        self.le_distance = QLineEdit("1.2")
        grid.addWidget(self.le_distance, 0, 1)

        grid.addWidget(QLabel("Phase Value (π units):"), 1, 0)
        self.le_phase = QLineEdit("1")
        grid.addWidget(self.le_phase, 1, 1)

        btn_add = QPushButton("Add Phase Jump")
        btn_add.clicked.connect(self.add_phase_jump)
        grid.addWidget(btn_add, 2, 0)
        btn_remove = QPushButton("Remove Last Phase Jump")
        btn_remove.clicked.connect(self.remove_last_phase_jump)
        grid.addWidget(btn_remove, 2, 1)

        self.lbl_jumps = QLabel("No jumps added")
        self.lbl_jumps.setWordWrap(True)
        layout.addWidget(self.lbl_jumps)

    def add_phase_jump(self):
        try:
            distance = float(self.le_distance.text())
            phase_value = float(self.le_phase.text()) * np.pi
            self.phase_jumps.append((distance, phase_value))
            self.update_jump_display()
            self.le_distance.clear()
            self.le_phase.clear()
        except ValueError:
            print("Invalid input for distance or phase value.")

    def remove_last_phase_jump(self):
        if self.phase_jumps:
            self.phase_jumps.pop()
            self.update_jump_display()

    def update_jump_display(self):
        if not self.phase_jumps:
            self.lbl_jumps.setText("No jumps added")
        else:
            text = "\n".join(["Distance: {:.2f} wL, Phase: {:.2f} π".format(d, v/np.pi)
                              for d, v in self.phase_jumps])
            self.lbl_jumps.setText(text)

    def phase(self):
        x = np.linspace(-chip_width, chip_width, slm_size[1])
        y = np.linspace(-chip_height, chip_height, slm_size[0])
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X ** 2 + Y ** 2) / 2
        phase_profile = np.zeros_like(X)
        for distance, phase_value in self.phase_jumps:
            indices = (rho <= distance * w_L)
            phase_profile[indices] += phase_value
        return (phase_profile % (2 * np.pi)) * (bit_depth / (2 * np.pi))

    def save_(self):
        return {'phase_jumps': self.phase_jumps}

    def load_(self, settings):
        self.phase_jumps = settings.get('phase_jumps', [])
        self.update_jump_display()


class TypeSquare(BaseTypeWidget):
    """Multiple square phase pattern. Add squares on top of each other."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = 'Square'
        self.squares = []  # Each square: (cx, cy, width, height, phase_value)
        layout = QVBoxLayout(self)
        group = QGroupBox("Square Phase Pattern Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Center X (m):"), 0, 0)
        self.le_cx = QLineEdit("0")
        grid.addWidget(self.le_cx, 0, 1)

        grid.addWidget(QLabel("Center Y (m):"), 1, 0)
        self.le_cy = QLineEdit("0")
        grid.addWidget(self.le_cy, 1, 1)

        grid.addWidget(QLabel("Width (m):"), 2, 0)
        self.le_width = QLineEdit("1")
        grid.addWidget(self.le_width, 2, 1)

        grid.addWidget(QLabel("Height (m):"), 3, 0)
        self.le_height = QLineEdit("1")
        grid.addWidget(self.le_height, 3, 1)

        grid.addWidget(QLabel("Phase value (π units):"), 4, 0)
        self.le_phase = QLineEdit("1")
        grid.addWidget(self.le_phase, 4, 1)

        btn_add = QPushButton("Add Square")
        btn_add.clicked.connect(self.add_square)
        grid.addWidget(btn_add, 5, 0)

        btn_remove = QPushButton("Remove Last Square")
        btn_remove.clicked.connect(self.remove_last_square)
        grid.addWidget(btn_remove, 5, 1)

        self.lbl_squares = QLabel("No squares added")
        self.lbl_squares.setWordWrap(True)
        layout.addWidget(self.lbl_squares)

    def add_square(self):
        try:
            cx = float(self.le_cx.text())
            cy = float(self.le_cy.text())
            width = float(self.le_width.text())
            height = float(self.le_height.text())
            phase_value = float(self.le_phase.text()) * np.pi
            self.squares.append((cx, cy, width, height, phase_value))
            self.update_square_display()
            self.le_cx.clear()
            self.le_cy.clear()
            self.le_width.clear()
            self.le_height.clear()
            self.le_phase.clear()
        except ValueError:
            print("Invalid square parameters.")

    def remove_last_square(self):
        if self.squares:
            self.squares.pop()
            self.update_square_display()

    def update_square_display(self):
        if not self.squares:
            self.lbl_squares.setText("No squares added")
        else:
            text = "\n".join(
                [f"Square {i+1}: Center=({cx:.2f}, {cy:.2f}), Size=({w:.2f}×{h:.2f}), Phase={phase/np.pi:.2f}π"
                 for i, (cx, cy, w, h, phase) in enumerate(self.squares)]
            )
            self.lbl_squares.setText(text)

    def phase(self):
        # Create a coordinate grid based on global chip dimensions and slm_size.
        x = np.linspace(-chip_width / 2, chip_width / 2, slm_size[1])
        y = np.linspace(-chip_height / 2, chip_height / 2, slm_size[0])
        X, Y = np.meshgrid(x, y)
        phase_pattern = np.zeros(slm_size)
        for (cx, cy, width, height, phase_value) in self.squares:
            mask = (np.abs(X - cx) <= width / 2) & (np.abs(Y - cy) <= height / 2)
            phase_pattern[mask] += phase_value
        return (phase_pattern % (2 * np.pi)) * (bit_depth / (2 * np.pi))

    def save_(self):
        return {'squares': self.squares}

    def load_(self, settings):
        self.squares = settings.get('squares', [])
        self.update_square_display()

class TypeAxicon(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = 'Axicon'
        layout = QVBoxLayout(self)
        group = QGroupBox("Axicon Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Cone angle (°):"), 0, 0)
        self.le_angle = QLineEdit("0.01")
        grid.addWidget(self.le_angle, 0, 1)

        grid.addWidget(QLabel("Wavelength (nm):"), 1, 0)
        self.le_wl = QLineEdit("1030")
        grid.addWidget(self.le_wl, 1, 1)

    def phase(self):
        try:
            alpha = np.radians(float(self.le_angle.text()))
            wl = float(self.le_wl.text()) * 1e-9
        except ValueError:
            return np.zeros(slm_size)

        # grille physique
        x = np.linspace(-chip_width/2, chip_width/2, slm_size[1])
        y = np.linspace(-chip_height/2, chip_height/2, slm_size[0])
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)

        # phase conique : φ = (2π/λ) * R * sin(alpha)
        ramp = (2*np.pi/wl) * R * np.sin(alpha)
        wrapped = np.mod(ramp, 2*np.pi)
        return wrapped * (bit_depth/(2*np.pi))

    def save_(self):
        return {'angle': self.le_angle.text(), 'wavelength': self.le_wl.text()}

    def load_(self, settings):
        self.le_angle.setText(settings.get('angle', '1.0'))
        self.le_wl.setText(settings.get('wavelength', '500'))


class TypeCheckerboard(BaseTypeWidget):
    """Checkerboard phase pattern with two phase values."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = 'Checkerboard'
        layout = QVBoxLayout(self)
        group = QGroupBox("Checkerboard Pattern Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Number of columns:"), 0, 0)
        self.le_cols = QLineEdit("8")
        grid.addWidget(self.le_cols, 0, 1)

        grid.addWidget(QLabel("Number of rows:"), 1, 0)
        self.le_rows = QLineEdit("8")
        grid.addWidget(self.le_rows, 1, 1)

        grid.addWidget(QLabel("Phase A (π units):"), 2, 0)
        self.le_phaseA = QLineEdit("0")
        grid.addWidget(self.le_phaseA, 2, 1)

        grid.addWidget(QLabel("Phase B (π units):"), 3, 0)
        self.le_phaseB = QLineEdit("1")
        grid.addWidget(self.le_phaseB, 3, 1)

    def phase(self):
        try:
            ncols = int(self.le_cols.text())
            nrows = int(self.le_rows.text())
            phaseA = float(self.le_phaseA.text()) * np.pi
            phaseB = float(self.le_phaseB.text()) * np.pi
        except ValueError:
            return np.zeros(slm_size)
        x = np.linspace(-chip_width / 2, chip_width / 2, slm_size[1])
        y = np.linspace(-chip_height / 2, chip_height / 2, slm_size[0])
        X, Y = np.meshgrid(x, y)
        square_width = chip_width / ncols
        square_height = chip_height / nrows
        col_indices = np.floor((X + chip_width / 2) / square_width).astype(int)
        row_indices = np.floor((Y + chip_height / 2) / square_height).astype(int)
        checker = (col_indices + row_indices) % 2
        phase_pattern = np.where(checker == 0, phaseA, phaseB)
        return (phase_pattern % (2 * np.pi)) * (bit_depth / (2 * np.pi))

    def save_(self):
        return {
            'cols': self.le_cols.text(),
            'rows': self.le_rows.text(),
            'phaseA': self.le_phaseA.text(),
            'phaseB': self.le_phaseB.text()
        }

    def load_(self, settings):
        self.le_cols.setText(settings.get('cols', '8'))
        self.le_rows.setText(settings.get('rows', '8'))
        self.le_phaseA.setText(settings.get('phaseA', '0'))
        self.le_phaseB.setText(settings.get('phaseB', '1'))
        
        

def new_type(parent, typ):
    types_dict = {
        'Flat': TypeFlat,
        'Binary': TypeBinary,
        'Background': TypeBackground,
        'Lens': TypeLens,
        'Vortex': TypeVortex,
        'Zernike': TypeZernike,
        'PhaseJumps': TypePhaseJumps,
        'Grating': TypeGrating,
        'Square': TypeSquare,
        'Checkerboard': TypeCheckerboard,
        'Tilt': TypeTilt,
        'Axicon': TypeAxicon,
    }
    if typ not in types_dict:
        raise ValueError("Unrecognized type '{}'. Valid types are: {}".format(typ, list(types_dict.keys())))
    return types_dict[typ](parent)

class PhaseSettings:
    types = phase_types
    new_type = staticmethod(new_type)