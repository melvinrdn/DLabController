import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import matplotlib.image as mpimg
from hardware.SLM_driver.SpatialLightModulator import slm_size, bit_depth, chip_width, chip_height
from prysm import coordinates, polynomials
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
from diagnostics.diagnostics_helpers import ColorFormatter
import subprocess
from PIL import Image, ImageTk

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])

types = ['Background', 'Lens', 'Zernike', 'Vortex', 'Grating', 'Binary', 'PhaseJumps', 'Image']
w_L = 3.5e-3  # Beam waist on the SLM


def new_type(frm_mid, typ):
    types_dict = {
        'Flat': TypeFlat,
        'Binary': TypeBinary,
        'Background': TypeBackground,
        'Lens': TypeLens,
        'Vortex': TypeVortex,
        'Zernike': TypeZernike,
        'Image': TypeImage,
        'PhaseJumps': TypePhaseJumps,
        'Grating': TypeGrating
    }
    if typ not in types_dict:
        raise ValueError(f"Unrecognized type '{typ}'. Valid types are: {list(types_dict.keys())}")
    return types_dict[typ](frm_mid)


class BaseType(object):
    def _read_file(self, filepath):
        if not filepath:
            return

        try:
            if filepath[-4:] == '.csv':
                try:
                    self.img = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=np.arange(1920) + 1)
                except:
                    self.img = np.loadtxt(filepath, delimiter=',')
            else:
                self.img = mpimg.imread(filepath)
                if len(self.img.shape) == 3:
                    self.img = self.img.sum(axis=2)
        except:
            print('File "' + filepath + '" not found')

    def open_file(self):
        filepath = askopenfilename(filetypes=[('CSV', '*.csv'), ('Image Files', '*.bmp'), ('All Files', '*.*')])
        self._read_file(filepath)
        self.lbl_file['text'] = f'{filepath}'

    def callback(self, action, P, text):
        # Allow any input when deleting (action '0') or typing valid numbers (action '1')
        return (action == '0') or (action == '1' and text in '0123456789.-+' and (
                P == '' or P.lstrip('-').replace('.', '', 1).isdigit()))

    def name_(self):
        return self.name

    def close_(self):
        self.frm_.destroy()


class TypeFlat(BaseType):
    def __init__(self, parent):
        self.name = 'Flat'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=1, column=0, sticky='nsew')

        # Create a label frame for the flat settings
        lbl_frm = ttk.LabelFrame(self.frm_, text='Flat')
        lbl_frm.grid(row=0, column=0, sticky='ew')

        # Create a label and entry box for the phase shift value
        lbl_phi = ttk.Label(lbl_frm,
                            text='Phase shift (' + str(bit_depth) + '=2pi):')
        vcmd = (parent.register(self.callback))
        self.strvar_flat = tk.StringVar()
        self.ent_flat = tk.Entry(
            lbl_frm, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_flat)
        lbl_phi.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=5)
        self.ent_flat.grid(row=0, column=1, sticky='w', padx=(0, 10))

    def phase(self):
        if self.ent_flat.get() != '':
            phi = float(self.ent_flat.get())
        else:
            phi = 0
        phase = np.ones(slm_size) * phi
        return phase

    def save_(self):
        dict = {'flat_phase': self.ent_flat.get()}
        return dict

    def load_(self, dict):
        self.strvar_flat.set(dict['flat_phase'])


class TypeBackground(BaseType):
    def __init__(self, parent):
        self.name = 'Background'
        self.parent = parent
        self.frm_ = ttk.Frame(self.parent)
        self.frm_.grid(row=0, column=0, sticky='nsew')

        lbl_frm = ttk.LabelFrame(self.frm_, text='Background correction file')
        lbl_frm.grid(row=0, column=0, sticky='ew')

        btn_open = ttk.Button(lbl_frm, text='Open Background file',
                              command=self.open_background_file)
        btn_open.grid(row=0)

        self.lbl_file = ttk.Label(lbl_frm, text='', wraplength=300,
                                  justify='left', foreground='gray')
        self.lbl_file.grid(row=1)

    def open_background_file(self):
        initial_directory = os.path.join('./ressources/background/')
        filepath = askopenfilename(initialdir=initial_directory,
                                   filetypes=[('CSV data arrays', '*.csv'), ('Image Files', '*.bmp'),
                                              ("Text files", "*.txt"), ("All files", "*.*")])
        self._read_file(filepath)
        self.lbl_file['text'] = f'{filepath}'

    def phase(self):
        if self.img is not None:
            phase = self.img
        else:
            phase = np.zeros(slm_size)
        return phase

    def save_(self):
        dict = {'filepath': self.lbl_file['text']}
        return dict

    def load_(self, dict):
        self.lbl_file['text'] = dict['filepath']
        self._read_file(dict['filepath'])


class TypeLens(BaseType):
    def __init__(self, parent):
        self.name = 'Lens'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=4, column=0, sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Virtual Lens Settings')
        lbl_frm.grid(row=0, column=0, sticky='ew', padx=5, pady=10)

        # Label for mode selection
        ttk.Label(lbl_frm, text='Mode:').grid(row=0, column=0, sticky='e', padx=(10, 0), pady=5)

        # Combobox for selecting between bending strength and focal length mode
        self.mode = tk.StringVar(value="Bending Strength")
        self.cbx_mode = ttk.Combobox(lbl_frm, values=["Bending Strength", "Focal Length"],
                                     state='readonly', width=15, textvariable=self.mode)
        self.cbx_mode.grid(row=0, column=1, sticky='w', padx=(0, 10))
        self.cbx_mode.bind("<<ComboboxSelected>>", self.toggle_mode)

        # Define labels and entry fields for lens parameters
        labels = ['Bending Strength (1/f) [1/m]:', 'Focal Length [m]:', 'Wavelength [nm]:',
                  'Calibration Slope [mm * m]:', 'Zero Reference [1/f]:', 'Focus Shift [mm]:']
        vcmd = (parent.register(self.callback))

        # Default values for lens parameters
        self.strvar_ben = tk.StringVar(value="0")  # Bending strength
        self.strvar_focal_length = tk.StringVar(value="1")  # Focal length
        self.strvar_wavelength = tk.StringVar(value="500")  # Wavelength in nm
        self.strvar_slope = tk.StringVar(value="1.0")  # Calibration slope
        self.strvar_zero = tk.StringVar(value="0")  # Zero reference
        self.strvar_focus_position = tk.StringVar(value="0")  # Focus shift

        # Entry fields for each parameter
        self.ent_ben = ttk.Entry(lbl_frm, width=10, validate='all',
                                 validatecommand=(vcmd, '%d', '%P', '%S'),
                                 textvariable=self.strvar_ben)
        self.ent_focal_length = ttk.Entry(lbl_frm, width=10, validate='all',
                                          validatecommand=(vcmd, '%d', '%P', '%S'),
                                          textvariable=self.strvar_focal_length)
        self.ent_wavelength = ttk.Entry(lbl_frm, width=10, validate='all',
                                        validatecommand=(vcmd, '%d', '%P', '%S'),
                                        textvariable=self.strvar_wavelength)
        self.ent_slope = ttk.Entry(lbl_frm, width=10, validate='all',
                                   validatecommand=(vcmd, '%d', '%P', '%S'),
                                   textvariable=self.strvar_slope)
        self.ent_zero = ttk.Entry(lbl_frm, width=10, validate='all',
                                  validatecommand=(vcmd, '%d', '%P', '%S'),
                                  textvariable=self.strvar_zero)
        self.ent_focus_position = ttk.Entry(lbl_frm, width=10, validate='all',
                                            validatecommand=(vcmd, '%d', '%P', '%S'),
                                            textvariable=self.strvar_focus_position)

        # Arrange labels and entries in the grid
        entries = [self.ent_ben, self.ent_focal_length, self.ent_wavelength,
                   self.ent_slope, self.ent_zero, self.ent_focus_position]
        for i, (label_text, entry) in enumerate(zip(labels, entries)):
            ttk.Label(lbl_frm, text=label_text).grid(row=i + 1, column=0, sticky='e', padx=(10, 0), pady=5)
            entry.grid(row=i + 1, column=1, sticky='w', padx=(0, 10))

        # Set up update triggers
        self.updating = False
        self.strvar_focus_position.trace_add("write", self.update_ben)
        self.strvar_ben.trace_add("write", self.update_position)
        self.strvar_focal_length.trace_add("write", self.update_ben_from_focal)

        # Initialize in bending strength mode
        self.toggle_mode()

    def toggle_mode(self, *args):
        # Enable only the selected mode's entry field
        if self.mode.get() == "Bending Strength":
            self.ent_ben.config(state='normal')
            self.ent_focal_length.config(state='disabled')
        else:
            self.ent_ben.config(state='disabled')
            self.ent_focal_length.config(state='normal')

    def update_ben(self, *args):
        if self.updating or self.mode.get() == "Focal Length":
            return
        self.updating = True
        try:
            slope = float(self.strvar_slope.get())
            zero_ref = float(self.strvar_zero.get())
            focus_shift = float(self.strvar_focus_position.get())
            bending_strength = zero_ref + focus_shift / slope
            self.strvar_ben.set(str(round(bending_strength, 3)))
            # Update focal length based on bending strength
            self.strvar_focal_length.set(str(round(1 / bending_strength, 3)) if bending_strength != 0 else "inf")
        except ValueError:
            print("Invalid entry detected in bending strength calculation.")
        finally:
            self.updating = False

    def update_position(self, *args):
        if self.updating:
            return
        self.updating = True
        try:
            slope = float(self.strvar_slope.get())
            zero_ref = float(self.strvar_zero.get())
            bending_strength = float(self.strvar_ben.get())
            focus_shift = slope * (bending_strength - zero_ref)
            self.strvar_focus_position.set(str(round(focus_shift, 2)))
        except ValueError:
            print("Invalid entry detected in focus position calculation.")
        finally:
            self.updating = False

    def update_ben_from_focal(self, *args):
        if self.updating or self.mode.get() == "Bending Strength":
            return
        self.updating = True
        try:
            focal_length = float(self.strvar_focal_length.get())
            bending_strength = 1 / focal_length if focal_length != 0 else 0
            self.strvar_ben.set(str(round(bending_strength, 3)))
        except ValueError:
            print("Invalid entry detected in focal length calculation.")
        finally:
            self.updating = False

    def phase(self):
        try:
            bending_strength = float(self.strvar_ben.get())
            wavelength = float(self.strvar_wavelength.get()) * 1e-9
        except ValueError:
            print("Invalid input for bending strength or wavelength.")
            return np.zeros(slm_size)

        if bending_strength != 0:
            radius = 2 / abs(bending_strength)
        else:
            radius = 1e6  # Effectively flat lens for bending strength 0

        x = np.linspace(-chip_width / 2, chip_width / 2, slm_size[1])
        y = np.linspace(-chip_height / 2, chip_height / 2, slm_size[0])
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X ** 2 + Y ** 2)

        phase_profile = (np.sqrt(radius ** 2 + R ** 2) - radius) / wavelength * bit_depth
        return phase_profile

    def save_(self):
        # Save lens parameters to a dictionary
        return {
            'ben': self.strvar_ben.get(),
            'focal_length': self.strvar_focal_length.get(),
            'wavelength': self.strvar_wavelength.get(),
            'slope': self.strvar_slope.get(),
            'zeroref': self.strvar_zero.get(),
            'focuspos': self.strvar_focus_position.get(),
            'mode': self.mode.get()
        }

    def load_(self, settings):
        # Load settings directly
        self.strvar_ben.set(settings['ben'])
        self.strvar_focal_length.set(settings['focal_length'])
        self.strvar_wavelength.set(settings['wavelength'])
        self.strvar_slope.set(settings['slope'])
        self.strvar_zero.set(settings['zeroref'])
        self.strvar_focus_position.set(settings['focuspos'])
        self.mode.set(settings.get('mode', 'Bending Strength'))
        self.toggle_mode()


class TypeZernike(BaseType):
    def __init__(self, parent):
        self.name = 'Zernike'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=7, column=0, sticky='nsew')

        lbl_frm = ttk.LabelFrame(self.frm_, text='Zernike Polynomials')
        lbl_frm.grid(row=0, column=0, sticky='ew', padx=5, pady=5)

        # File path and preview plot setup
        self.filepath = ''
        self.filepath_var = tk.StringVar()

        # Load file button
        browse_button = ttk.Button(lbl_frm, text="Browse File", command=self.load_file)
        browse_button.grid(row=0, column=0, pady=5, padx=5)

        # Modify file button (initially disabled)
        self.modify_button = ttk.Button(lbl_frm, text="Modify File", command=self.modify_file, state="disabled")
        self.modify_button.grid(row=0, column=1, pady=5, padx=5)

        # Update data button (initially disabled)
        self.update_button = ttk.Button(lbl_frm, text="Update Data", command=self.update_data, state="disabled")
        self.update_button.grid(row=0, column=2, pady=5, padx=5)

        # Display selected file path
        self.lbl_file = ttk.Label(lbl_frm, text='', wraplength=300, justify='left', foreground='gray')
        self.lbl_file.grid(row=1, column=0, columnspan=3, padx=5)

        # Smaller plot area within the interface
        self.fig, self.ax = plt.subplots(figsize=(3, 1.5))  # Reduced size for compact display
        self.ax.set_title("Zernike Coefficients", fontsize=10)
        self.ax.set_xlabel('Mode (j)', fontsize=8)
        self.ax.set_ylabel("Coef (nm RMS)", fontsize=8)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=lbl_frm)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=3, pady=5)
        self.canvas.draw()

    def load_file(self):
        # Load file and display filepath
        initial_directory = os.path.join('./ressources/aberration_correction/')
        filepath = askopenfilename(initialdir=initial_directory,
                                   filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filepath:
            self.filepath = filepath
            self.lbl_file['text'] = f'{filepath}'
            self.plot_data()  # Update plot with loaded data
            self.modify_button.config(state="normal")  # Enable modify button
            self.update_button.config(state="normal")  # Enable update button

    def modify_file(self):
        # Open the text file in the default system editor
        if os.path.isfile(self.filepath):
            subprocess.Popen(['open', self.filepath] if os.name == 'posix' else ['notepad', self.filepath])

    def update_data(self):
        # Reload and plot updated data from the file
        self.plot_data()

    def plot_data(self):
        # Load and plot Zernike coefficient data
        if self.filepath:
            data = np.loadtxt(self.filepath, skiprows=1)
            js = data[:, 0].astype(int)
            coefs = data[:, 1]
            self.update_plot(js, coefs)

    import numpy as np

    def update_plot(self, js, coefs):
        # Clear previous plot and draw new bar plot with small font size
        self.ax.clear()
        self.ax.bar(js, coefs, color='blue', alpha=0.8)
        self.ax.set_title("Zernike Coefficients", fontsize=10)
        self.ax.set_xlabel('Mode (j)', fontsize=8)
        self.ax.set_ylabel("Coef (nm RMS)", fontsize=8)

        # Set the x-axis ticks to every 2 units
        self.ax.set_xticks(np.arange(min(js), max(js) + 1, 2))
        self.ax.grid(True)
        self.canvas.draw()

    def phase(self):
        # Compute phase profile from Zernike polynomials
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

    def save_(self):
        # Save filepath for Zernike coefficients
        return {'filepath': self.lbl_file['text']}

    def load_(self, settings):
        # Load filepath and update plot
        self.filepath = settings.get('filepath', '')
        self.lbl_file['text'] = self.filepath
        self.plot_data()
        if self.filepath:
            self.modify_button.config(state="normal")
            self.update_button.config(state="normal")


class TypeVortex(BaseType):
    def __init__(self, parent):
        self.name = 'Vortex'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=6, column=0, sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Vortex Beam Settings')
        lbl_frm.grid(row=0, column=0, sticky='ew', padx=5, pady=10)

        # Initialize vortex storage
        self.vortices = []  # Store (radius, order) pairs for each vortex

        # Vortex parameter inputs
        lbl_radius = ttk.Label(lbl_frm, text='Radius (wL):')
        lbl_order = ttk.Label(lbl_frm, text='Vortex Order:')
        vcmd = (parent.register(self.callback))
        self.strvar_radius = tk.StringVar()
        self.ent_radius = ttk.Entry(lbl_frm, width=10, validate='all', validatecommand=(vcmd, '%d', '%P', '%S'),
                                    textvariable=self.strvar_radius)
        self.strvar_order = tk.StringVar(value="1")
        self.ent_order = ttk.Entry(lbl_frm, width=10, validate='all', validatecommand=(vcmd, '%d', '%P', '%S'),
                                   textvariable=self.strvar_order)

        # Buttons to add and remove vortices
        btn_add = ttk.Button(lbl_frm, text='Add Vortex', command=self.add_vortex)
        btn_remove = ttk.Button(lbl_frm, text='Remove Last Vortex', command=self.remove_last_vortex)

        # Display added vortices
        self.lbl_vortices = ttk.Label(lbl_frm, text='No vortices added', wraplength=300)

        # Layout widgets
        lbl_radius.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=5)
        self.ent_radius.grid(row=0, column=1, sticky='w', padx=(0, 10))
        lbl_order.grid(row=1, column=0, sticky='e', padx=(10, 0), pady=5)
        self.ent_order.grid(row=1, column=1, sticky='w', padx=(0, 10))
        btn_add.grid(row=2, column=0, pady=10)
        btn_remove.grid(row=2, column=1, pady=10)
        self.lbl_vortices.grid(row=3, column=0, columnspan=2, pady=10)

    def add_vortex(self):
        """Add a new vortex with specified radius and order."""
        try:
            radius = float(self.strvar_radius.get())
            order = int(self.strvar_order.get())
            self.vortices.append((radius, order))
            self.update_vortex_display()
            self.strvar_radius.set('')
            self.strvar_order.set('1')
        except ValueError:
            print("Invalid input for radius or order.")

    def remove_last_vortex(self):
        """Remove the last added vortex."""
        if self.vortices:
            self.vortices.pop()
            self.update_vortex_display()

    def update_vortex_display(self):
        """Update the display of added vortices."""
        if not self.vortices:
            self.lbl_vortices.config(text='No vortices added')
        else:
            vortices_text = "\n".join([f"Radius: {r:.2f} wL, Order: {o}" for r, o in self.vortices])
            self.lbl_vortices.config(text=vortices_text)

    def phase(self):
        """Generate the phase profile based on the configured vortices."""
        x = np.linspace(-chip_width, chip_width, slm_size[1])
        y = np.linspace(-chip_height, chip_height, slm_size[0])
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X ** 2 + Y ** 2) / 2  # Radial distance from the center

        phase_profile = np.zeros(slm_size)

        # Generate each vortex based on its specified radius and order
        for radius, order in self.vortices:
            radius_scaled = radius * w_L  # Scale radius to match actual units
            vortex_mask = rho <= radius_scaled
            theta = np.arctan2(Y, X)
            vortex_phase = (order * theta) % (2 * np.pi)
            phase_profile[vortex_mask] += vortex_phase[vortex_mask]

        # Normalize to SLM bit depth and wrap phase
        phase_profile = (phase_profile % (2 * np.pi)) * (bit_depth / (2 * np.pi))
        return phase_profile

    def save_(self):
        """Save the list of vortices."""
        return {'vortices': self.vortices}

    def load_(self, settings):
        """Load the list of vortices."""
        self.vortices = settings.get('vortices', [])
        self.update_vortex_display()


class TypeGrating(BaseType):
    def __init__(self, parent):
        self.name = 'Grating'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=9, column=0, sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Grating Settings')
        lbl_frm.grid(row=0, column=0, sticky='ew')

        # Labels for grating parameters
        lbl_freq = ttk.Label(lbl_frm, text='Spatial Frequency (lines/mm):')
        lbl_angle = ttk.Label(lbl_frm, text='Orientation Angle (degrees):')

        # Entry fields for grating parameters
        vcmd = (parent.register(self.callback))
        self.strvar_freq = tk.StringVar()
        self.ent_freq = ttk.Entry(lbl_frm, width=10, validate='all',
                                  validatecommand=(vcmd, '%d', '%P', '%S'),
                                  textvariable=self.strvar_freq)

        self.strvar_angle = tk.StringVar()
        self.ent_angle = ttk.Entry(lbl_frm, width=10, validate='all',
                                   validatecommand=(vcmd, '%d', '%P', '%S'),
                                   textvariable=self.strvar_angle)

        # Arrange the labels and entries in the frame
        lbl_freq.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=5)
        self.ent_freq.grid(row=0, column=1, sticky='w', padx=(0, 10))
        lbl_angle.grid(row=1, column=0, sticky='e', padx=(10, 0), pady=5)
        self.ent_angle.grid(row=1, column=1, sticky='w', padx=(0, 10))

    def phase(self):
        # Default grating parameters if none are provided
        freq = float(self.ent_freq.get() or 1000)  # default to 1000 lines/mm if not set
        angle = float(self.ent_angle.get() or 0)  # default to 0 degrees if not set

        # Convert spatial frequency to a phase grating
        period = 1 / (freq * 1e-3)  # period in mm (converted from lines per mm)
        angle_rad = np.radians(angle)

        # Create x, y meshgrid based on the SLM chip dimensions
        x = np.linspace(-chip_width / 2, chip_width / 2, slm_size[1])
        y = np.linspace(-chip_height / 2, chip_height / 2, slm_size[0])
        X, Y = np.meshgrid(x, y)

        # Apply orientation and compute phase pattern for grating
        grating_pattern = np.sin(2 * np.pi * (X * np.cos(angle_rad) + Y * np.sin(angle_rad)) / period)
        phase = (grating_pattern + 1) * (bit_depth / 2)  # Normalize to SLM bit depth

        return phase

    def save_(self):
        # Save the grating's frequency and angle to a dictionary
        return {'freq': self.ent_freq.get(), 'angle': self.ent_angle.get()}

    def load_(self, settings):
        # Load the grating's frequency and angle from a dictionary
        self.strvar_freq.set(settings.get('freq', '1000'))
        self.strvar_angle.set(settings.get('angle', '0'))


class TypeBinary(BaseType):
    def __init__(self, parent):
        self.name = 'Binary'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=3, column=0, sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Binary Pattern Settings')
        lbl_frm.grid(row=0, column=0, sticky='ew', padx=5, pady=10)

        # Parameter labels and input fields
        lbl_phi = ttk.Label(lbl_frm, text='Phase change (pi units):')
        lbl_stripes = ttk.Label(lbl_frm, text='Number of stripes:')
        lbl_angle = ttk.Label(lbl_frm, text='Angle (degrees):')

        # Input fields
        vcmd = (parent.register(self.callback))

        self.strvar_phi = tk.StringVar(value="1")
        self.ent_phi = ttk.Entry(lbl_frm, width=12, validate='all', validatecommand=(vcmd, '%d', '%P', '%S'),
                                 textvariable=self.strvar_phi)

        self.strvar_stripes = tk.StringVar(value="2")
        self.ent_stripes = ttk.Entry(lbl_frm, width=12, validate='all', validatecommand=(vcmd, '%d', '%P', '%S'),
                                     textvariable=self.strvar_stripes)

        self.strvar_angle = tk.StringVar(value="0")
        self.ent_angle = ttk.Entry(lbl_frm, width=12, validate='all', validatecommand=(vcmd, '%d', '%P', '%S'),
                                   textvariable=self.strvar_angle)

        # Arrange widgets in grid
        lbl_phi.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=5)
        self.ent_phi.grid(row=0, column=1, sticky='w', padx=(0, 10))
        lbl_stripes.grid(row=1, column=0, sticky='e', padx=(10, 0), pady=5)
        self.ent_stripes.grid(row=1, column=1, sticky='w', padx=(0, 10))
        lbl_angle.grid(row=2, column=0, sticky='e', padx=(10, 0), pady=5)
        self.ent_angle.grid(row=2, column=1, sticky='w', padx=(0, 10))

    def phase(self):
        # Retrieve and parse parameters
        try:
            phi = float(self.strvar_phi.get()) * np.pi  # Convert pi units to radians
            stripes = int(self.strvar_stripes.get())
            angle_deg = float(self.strvar_angle.get())
            angle_rad = np.radians(angle_deg)
        except ValueError:
            print("Invalid parameter values.")
            return np.zeros(slm_size)

        # Create phase matrix with default zero phase
        phase_mat = np.zeros(slm_size)

        # Set up X and Y meshgrid, rotate by the angle
        x = np.linspace(-chip_width / 2, chip_width / 2, slm_size[1])
        y = np.linspace(-chip_height / 2, chip_height / 2, slm_size[0])
        X, Y = np.meshgrid(x, y)

        # Rotate coordinates based on the angle for stripe pattern
        X_rot = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
        stripe_width = chip_width / stripes

        for i in range(stripes):
            # Apply phase to alternating stripes
            if i % 2 == 0:
                indices = (X_rot >= i * stripe_width - chip_width / 2) & (
                        X_rot < (i + 1) * stripe_width - chip_width / 2)
                phase_mat[indices] = phi

        # Normalize to SLM bit depth and wrap phase
        phase_mat = (phase_mat % (2 * np.pi)) * (bit_depth / (2 * np.pi))
        print("Phase matrix generated with angle:", angle_deg, "degrees")
        return phase_mat

    def save_(self):
        return {
            'phi': self.strvar_phi.get(),
            'stripes': self.strvar_stripes.get(),
            'angle': self.strvar_angle.get()
        }

    def load_(self, settings):
        self.strvar_phi.set(settings.get('phi', '1'))
        self.strvar_stripes.set(settings.get('stripes', '2'))
        self.strvar_angle.set(settings.get('angle', '0'))


class TypePhaseJumps(BaseType):
    def __init__(self, parent):
        self.name = 'PhaseJumps'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=6, column=0, sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Phase Jump Settings')
        lbl_frm.grid(row=0, column=0, sticky='ew')

        # Initialize phase jump storage
        self.phase_jumps = []  # To store (distance, phase value) pairs

        # Widgets for entering phase jump values
        lbl_distance = ttk.Label(lbl_frm, text='Distance (wL):')
        lbl_phase = ttk.Label(lbl_frm, text='Phase Value (pi units):')

        vcmd = (parent.register(self.callback))
        self.strvar_distance = tk.StringVar()
        self.ent_distance = ttk.Entry(lbl_frm, width=11, validate='all',
                                      validatecommand=(vcmd, '%d', '%P', '%S'),
                                      textvariable=self.strvar_distance)

        self.strvar_phase = tk.StringVar()
        self.ent_phase = ttk.Entry(lbl_frm, width=11, validate='all',
                                   validatecommand=(vcmd, '%d', '%P', '%S'),
                                   textvariable=self.strvar_phase)

        # Buttons to add and remove phase jumps
        btn_add = ttk.Button(lbl_frm, text='Add Phase Jump', command=self.add_phase_jump)
        btn_remove = ttk.Button(lbl_frm, text='Remove Last Phase Jump', command=self.remove_last_phase_jump)

        # Display added phase jumps
        self.lbl_jumps = ttk.Label(lbl_frm, text='No jumps added', wraplength=300)

        # Arrange widgets in grid
        lbl_distance.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=5)
        self.ent_distance.grid(row=0, column=1, sticky='w', padx=(0, 10))
        lbl_phase.grid(row=1, column=0, sticky='e', padx=(10, 0), pady=5)
        self.ent_phase.grid(row=1, column=1, sticky='w', padx=(0, 10))
        btn_add.grid(row=2, column=0, pady=10)
        btn_remove.grid(row=2, column=1, pady=10)
        self.lbl_jumps.grid(row=3, column=0, columnspan=2, pady=10)

    def add_phase_jump(self):
        """Add a new phase jump with specified distance and phase value."""
        try:
            distance = float(self.strvar_distance.get())
            phase_value = float(self.strvar_phase.get()) * np.pi  # Convert to radians
            self.phase_jumps.append((distance, phase_value))
            self.update_jump_display()
            self.strvar_distance.set('')
            self.strvar_phase.set('')
        except ValueError:
            print("Invalid input for distance or phase value.")

    def remove_last_phase_jump(self):
        """Remove the last added phase jump."""
        if self.phase_jumps:
            self.phase_jumps.pop()
            self.update_jump_display()

    def update_jump_display(self):
        """Update the display of added phase jumps."""
        if not self.phase_jumps:
            self.lbl_jumps.config(text='No jumps added')
        else:
            jumps_text = "\n".join([f"Distance: {d:.2f} wL, Phase: {v / np.pi:.2f} π" for d, v in self.phase_jumps])
            self.lbl_jumps.config(text=jumps_text)

    def phase(self):
        """Generate the phase profile based on the configured phase jumps."""
        x = np.linspace(-chip_width, chip_width, slm_size[1])
        y = np.linspace(-chip_height, chip_height, slm_size[0])
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X ** 2 + Y ** 2) / 2

        phase_profile = np.zeros_like(X)
        for distance, phase_value in self.phase_jumps:
            indices = np.where(rho <= distance * w_L)
            phase_profile[indices] += phase_value  # Accumulate phase values for each jump

        # Normalize to SLM bit depth, wrap phase if necessary
        phase_profile = (phase_profile % (2 * np.pi)) * (bit_depth / (2 * np.pi))
        return phase_profile

    def save_(self):
        """Save the list of phase jumps."""
        return {'phase_jumps': self.phase_jumps}

    def load_(self, settings):
        """Load the list of phase jumps."""
        self.phase_jumps = settings.get('phase_jumps', [])
        self.update_jump_display()


class TypeImage(TypeBackground):
    def __init__(self, parent):
        self.name = 'Image'
        self.parent = parent
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=8, column=0, sticky='nsew')

        # Frame for the image settings
        lbl_frm = ttk.LabelFrame(self.frm_, text='Image Settings')
        lbl_frm.grid(row=0, column=0, sticky='ew', padx=10, pady=10)

        # Open file button
        btn_open = ttk.Button(lbl_frm, text='Open Phase Profile', command=self.open_file)
        btn_open.grid(row=0, column=0, padx=(10, 5), pady=5, sticky='w')

        # Displaying the file path
        self.lbl_file = ttk.Label(lbl_frm, text='No file selected', wraplength=300, justify='left', foreground='gray')
        self.lbl_file.grid(row=1, column=0, padx=10, pady=(0, 10), sticky='w')

        # Preview thumbnail
        self.preview_canvas = tk.Canvas(lbl_frm, width=100, height=100, bg='white')
        self.preview_canvas.grid(row=0, column=1, rowspan=2, padx=10, pady=5)

        # Refresh button to reload image
        btn_refresh = ttk.Button(lbl_frm, text='Refresh Profile', command=self.refresh_profile)
        btn_refresh.grid(row=2, column=0, columnspan=2, pady=(5, 10))

        self.img = None  # Placeholder for the image data

    def open_file(self):
        filepath = askopenfilename(
            filetypes=[('Image Files', '*.bmp;*.png;*.jpg'), ('CSV data arrays', '*.csv'), ('All Files', '*.*')])

        if filepath:
            self.lbl_file.config(text=filepath)
            self._read_file(filepath)
            self.update_preview(filepath)
        else:
            self.lbl_file.config(text='No file selected')

    def refresh_profile(self):
        """Refreshes the loaded image to update any changes."""
        if self.img is not None:
            self.update_preview(self.lbl_file.cget("text"))
        else:
            logging.info("No image to refresh")

    def update_preview(self, filepath):
        """Displays a thumbnail preview of the loaded image on the Canvas."""
        if self.img is not None:
            img_resized = self.img.resize((100, 100))  # Resize for preview
            self.preview_photo = ImageTk.PhotoImage(img_resized)
            self.preview_canvas.create_image(50, 50, image=self.preview_photo)

    def _read_file(self, filepath):
        """Reads the selected image file and converts it to grayscale if needed."""
        if filepath.endswith('.csv'):
            self.img = np.loadtxt(filepath, delimiter=',')
        else:
            self.img = Image.open(filepath).convert('L')  # Convert image to grayscale

    def phase(self):
        """Returns the image data as a phase profile, normalized to bit depth if applicable."""
        if self.img is not None:
            phase = np.array(self.img) / 255 * bit_depth  # Normalize to SLM bit depth
        else:
            phase = np.zeros(slm_size)
        return phase

# class TypeMultibeam(BaseType):
#     def __init__(self, parent):
#         self.name = 'Multi'
#         self.frm_ = ttk.Frame(parent)
#         self.frm_.grid(row=5, column=0, sticky='nsew')
#         lbl_frm = ttk.LabelFrame(self.frm_, text='Multibeam')
#         lbl_frm.grid(row=0, column=0, sticky='ew')
#
#         # creating frames
#         frm_n = ttk.Frame(lbl_frm)
#         frm_sprrad = ttk.Frame(lbl_frm)
#         frm_spr = ttk.Frame(frm_sprrad)
#         frm_rad = ttk.Frame(frm_sprrad)
#         frm_int = ttk.Frame(lbl_frm)
#         frm_pxsiz = ttk.Frame(lbl_frm)
#
#         # creating labels
#         lbl_n = ttk.Label(frm_n, text='n^2; n=:')
#         lbl_hor = ttk.Label(frm_int, text='Hor:')
#         lbl_vert = ttk.Label(frm_int, text='Vert:')
#         lbl_intil = ttk.Label(frm_int, text='Intensity tilt')
#         lbl_insqr = ttk.Label(frm_int, text='Intensity curve')
#         lbl_horspr = ttk.Label(frm_spr, text='Horizontal spread:')
#         lbl_verspr = ttk.Label(frm_spr, text='Vertical spread:')
#         lbl_cph = ttk.Label(frm_sprrad, text='Hyp.phase diff')
#         lbl_rad = ttk.Label(frm_rad, text='Phase[' + str(bit_depth) + ']:')
#         lbl_amp = ttk.Label(frm_rad, text='Choose beam:')
#         lbl_pxsiz = ttk.Label(frm_pxsiz, text='pixel size:')
#
#         # creating entries
#         vcmd = (parent.register(self.callback))
#         self.strvar_n = tk.StringVar()
#         self.ent_n = ttk.Entry(frm_n, width=5, validate='all',
#                                validatecommand=(vcmd, '%d', '%P', '%S'),
#                                textvariable=self.strvar_n)
#         self.strvar_hpt = tk.StringVar()
#         self.ent_hpt = ttk.Entry(frm_spr, width=5, validate='all',
#                                  validatecommand=(vcmd, '%d', '%P', '%S'),
#                                  textvariable=self.strvar_hpt)
#         self.strvar_vpt = tk.StringVar()
#         self.ent_vpt = ttk.Entry(frm_spr, width=5, validate='all',
#                                  validatecommand=(vcmd, '%d', '%P', '%S'),
#                                  textvariable=self.strvar_vpt)
#         self.strvar_rad = tk.StringVar()
#         self.ent_rad = ttk.Entry(frm_rad, width=5, validate='all',
#                                  validatecommand=(vcmd, '%d', '%P', '%S'),
#                                  textvariable=self.strvar_rad)
#         self.strvar_amp = tk.StringVar()
#         self.ent_amp = ttk.Entry(frm_rad, width=5, validate='all',
#                                  validatecommand=(vcmd, '%d', '%P', '%S'),
#                                  textvariable=self.strvar_amp)
#         self.strvar_hit = tk.StringVar()
#         self.ent_hit = ttk.Entry(frm_int, width=5, validate='all',
#                                  validatecommand=(vcmd, '%d', '%P', '%S'),
#                                  textvariable=self.strvar_hit)
#         self.strvar_vit = tk.StringVar()
#         self.ent_vit = ttk.Entry(frm_int, width=5, validate='all',
#                                  validatecommand=(vcmd, '%d', '%P', '%S'),
#                                  textvariable=self.strvar_vit)
#         self.strvar_his = tk.StringVar()
#         self.ent_his = ttk.Entry(frm_int, width=5, validate='all',
#                                  validatecommand=(vcmd, '%d', '%P', '%S'),
#                                  textvariable=self.strvar_his)
#         self.strvar_vis = tk.StringVar()
#         self.ent_vis = ttk.Entry(frm_int, width=5, validate='all',
#                                  validatecommand=(vcmd, '%d', '%P', '%S'),
#                                  textvariable=self.strvar_vis)
#         self.strvar_pxsiz = tk.StringVar()
#         self.ent_pxsiz = ttk.Entry(frm_pxsiz, width=5, validate='all',
#                                    validatecommand=(vcmd, '%d', '%P', '%S'),
#                                    textvariable=self.strvar_pxsiz)
#
#         # setup
#         frm_n.grid(row=0, sticky='nsew')
#         frm_sprrad.grid(row=1, sticky='nsew')
#         frm_int.grid(row=2, sticky='nsew')
#         frm_pxsiz.grid(row=3, sticky='nsew')
#
#         frm_spr.grid(row=1, column=0, sticky='nsew')
#         frm_rad.grid(row=1, column=1, sticky='ew')
#
#         lbl_n.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=(5, 10))
#         self.ent_n.grid(row=0, column=1, sticky='w',
#                         padx=(0, 10), pady=(5, 10))
#
#         lbl_horspr.grid(row=0, column=0, sticky='e', padx=(10, 0))
#         lbl_verspr.grid(row=1, column=0, sticky='e', padx=(10, 0))
#         self.ent_hpt.grid(row=0, column=1, sticky='w')
#         self.ent_vpt.grid(row=1, column=1, sticky='w')
#
#         lbl_cph.grid(row=0, column=1, sticky='ew')
#
#         lbl_rad.grid(row=0, column=0, sticky='e', padx=(15, 0))
#         lbl_amp.grid(row=1, column=0, sticky='e', padx=(15, 0))
#         self.ent_rad.grid(row=0, column=1, sticky='w', padx=(0, 5))
#         self.ent_amp.grid(row=1, column=1, sticky='w', padx=(0, 5))
#
#         lbl_hor.grid(row=1, column=0, sticky='e', padx=(10, 0))
#         lbl_vert.grid(row=2, column=0, sticky='e', padx=(10, 0))
#         lbl_intil.grid(row=0, column=1, padx=5, pady=(10, 0))
#         lbl_insqr.grid(row=0, column=2, padx=(0, 5), pady=(10, 0))
#         self.ent_hit.grid(row=1, column=1)
#         self.ent_his.grid(row=1, column=2, padx=(0, 5))
#         self.ent_vit.grid(row=2, column=1, pady=(0, 5))
#         self.ent_vis.grid(row=2, column=2, padx=(0, 5), pady=(0, 5))
#
#         lbl_pxsiz.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=5)
#         self.ent_pxsiz.grid(row=0, column=1, sticky='w')
#
#     def phase(self):
#
#         if self.ent_n.get() != '':
#             n = int(self.ent_n.get())
#         else:
#             n = 1
#
#         if self.ent_hpt.get() != '':
#             xtilt = float(self.ent_hpt.get())
#         else:
#             xtilt = 0
#         if self.ent_vpt.get() != '':
#             ytilt = float(self.ent_vpt.get())
#         else:
#             ytilt = 0
#         tilts = np.arange(-n + 1, n + 1, 2)
#         xtilts = tilts * xtilt / 2
#         ytilts = tilts * ytilt / 2
#         phases = np.zeros([slm_size[0], slm_size[1], n * n])
#         ind = 0
#         for xdir in xtilts:
#             for ydir in ytilts:
#                 phases[:, :, ind] = self.phase_tilt(xdir, ydir)
#                 ind += 1
#
#         if self.ent_rad.get() != '':
#             tmprad = float(self.ent_rad.get())
#         else:
#             tmprad = 0
#         if self.ent_amp.get() != '':
#             amp = int(self.ent_amp.get())
#         else:
#             amp = 0
#         if tmprad != 0:
#             phases[:, :, amp] = tmprad + phases[:, :, amp]
#             phases[:, :, amp + 1] = tmprad + phases[:, :, amp + 1]
#
#         if self.ent_hit.get() != '':
#             xit = float(self.ent_hit.get())
#         else:
#             xit = 0
#         if self.ent_vit.get() != '':
#             yit = float(self.ent_vit.get())
#         else:
#             yit = 0
#         if self.ent_his.get() != '':
#             xis = float(self.ent_his.get())
#         else:
#             xis = 0
#         if self.ent_vis.get() != '':
#             yis = float(self.ent_vis.get())
#         else:
#             yis = 0
#         intensities = np.ones(n ** 2)
#         totnum = np.ceil((slm_size[0] * (slm_size[1] + n) / (n ** 2)))
#         phase_nbr = np.outer(np.arange(n ** 2), np.ones([int(totnum)]))
#
#         if xit != 0:
#             xits = np.linspace(-((n - 1) / 2 * xit), ((n - 1) / 2 * xit), num=n)
#         else:
#             xits = np.zeros(n)
#         if yit != 0:
#             yits = np.linspace(-((n - 1) / 2 * yit), ((n - 1) / 2 * yit), num=n)
#         else:
#             yits = np.zeros(n)
#         ii = 0
#         for tmpx in xits:
#             intensities[n * ii:n * (ii + 1)] = (tmpx + yits + 1)
#             ii += 1
#
#         spread = tilts
#         xiss = -xis * (spread ** 2 - spread[0] ** 2)
#         yiss = -yis * (spread ** 2 - spread[0] ** 2)
#         ii = 0
#         for tmpx in xiss:
#             intensities[n * ii:n * (ii + 1)] += (tmpx + yiss + 1)
#             ii += 1
#
#         intensities[intensities < 0] = 0
#         intensities = intensities / np.sum(intensities) * n ** 2
#         tmpint = intensities - 1
#         beam = 0
#         strong_beams = []
#         strong_beams_int = []
#         weak_beams = []
#         weak_beams_int = []
#         for intens in tmpint:
#             if intens > 0:
#                 strong_beams.append(beam)
#                 strong_beams_int.append(intens)
#             elif intens < 0:
#                 weak_beams.append(beam)
#                 weak_beams_int.append(intens)
#             beam += 1
#
#         strong_beams_int = strong_beams_int / np.sum(strong_beams_int)
#         for wbeam, wbeam_int in zip(weak_beams, weak_beams_int):
#             nbr_pixels = np.ceil(np.abs(wbeam_int) * totnum)
#             nbr_each = np.ceil(nbr_pixels * strong_beams_int)
#             strt = 0
#             for nbr, sbeam in zip(nbr_each, strong_beams):
#                 phase_nbr[int(wbeam), strt:(strt + int(nbr))] = sbeam
#                 strt += int(nbr)
#         rng = np.random.default_rng()
#         rng.shuffle(phase_nbr, axis=1)
#         if self.ent_pxsiz.get() != '':
#             pxsiz = int(self.ent_pxsiz.get())
#         else:
#             pxsiz = 1
#
#         xrange = np.arange(0, slm_size[1], 1)
#         yrange = np.arange(0, slm_size[0], 1)
#         tot_phase = np.zeros(slm_size)
#         [X, Y] = np.meshgrid(xrange, yrange)
#         ind_phase_tmp = (np.floor(X / pxsiz) % n) * n + (np.floor(Y / pxsiz) % n)
#         ind_phase = ind_phase_tmp.astype(int)
#
#         ind_phase2 = ind_phase.copy()
#         for ii in range(n ** 2):
#             max_nbr = np.count_nonzero(ind_phase == ii)
#             if max_nbr <= phase_nbr[0, :].size:
#                 ind_phase2[ind_phase == ii] = phase_nbr[ii, 0:max_nbr]
#             else:
#                 extra = ii * np.ones([max_nbr - phase_nbr[0, :].size])
#                 ind_phase2[ind_phase == ii] = np.append(phase_nbr[ii, :], extra)
#
#         for ii in range(n ** 2):
#             ii_phase = phases[:, :, ii]
#             tot_phase[ind_phase2 == ii] = ii_phase[ind_phase2 == ii]
#
#         return tot_phase
#
#     def phase_tilt(self, xdir, ydir):
#         if xdir != '' and float(xdir) != 0:
#             lim = np.ones(slm_size[0]) * float(xdir) * (slm_size[0] - 1) / 2
#             phx = np.linspace(-lim, +lim, slm_size[1], axis=1)
#         else:
#             phx = np.zeros(slm_size)
#
#         if ydir != '' and float(ydir) != 0:
#             lim = np.ones(slm_size[1]) * float(ydir) * (slm_size[1] - 1) / 2
#             phy = np.linspace(-lim, +lim, slm_size[0], axis=0)
#         else:
#             phy = np.zeros(slm_size)
#
#         return phx + phy
#
#     def save_(self):
#         dict = {'n': self.ent_n.get(),
#                 'hpt': self.ent_hpt.get(),
#                 'rad': self.ent_rad.get(),
#                 'hit': self.ent_hit.get(),
#                 'his': self.ent_his.get(),
#                 'vpt': self.ent_vpt.get(),
#                 'amp': self.ent_amp.get(),
#                 'vit': self.ent_vit.get(),
#                 'vis': self.ent_vis.get(),
#                 'pxsiz': self.ent_pxsiz.get()}
#         return dict
#
#     def load_(self, dict):
#         self.strvar_n.set(dict['n'])
#         self.strvar_hpt.set(dict['hpt'])
#         self.strvar_rad.set(dict['rad'])
#         self.strvar_hit.set(dict['hit'])
#         self.strvar_his.set(dict['his'])
#         self.strvar_vpt.set(dict['vpt'])
#         self.strvar_amp.set(dict['amp'])
#         self.strvar_vit.set(dict['vit'])
#         self.strvar_vis.set(dict['vis'])
#         self.strvar_pxsiz.set(dict['pxsiz'])
