import json
import os
import logging
import tkinter as tk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
from diagnostics.diagnostics_helpers import ColorFormatter

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("from HHGView: %(levelname)s: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])

class WPCalib:
    """
    A graphical interface for waveplate calibration using Tkinter and Matplotlib.
    """
    colors = ['r', 'g', 'b', 'k']
    config_file = './diagnostics/WaveplateCalib/calib_path.json'  # JSON file to store calibration paths

    def __init__(self):
        """
        Initializes the WPCalib instance, setting up the Tkinter window,
        frames, and default calibration files.
        """
        self.win = tk.Toplevel()
        self.win.title("D-Lab Controller - WP Calibration")
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

        self.frm_options = tk.LabelFrame(self.win, text='Options')
        self.frm_plot = tk.LabelFrame(self.win, text='Plots')

        self.frm_options.grid(row=0, column=0, sticky='nsew')
        self.frm_plot.grid(row=0, column=1, sticky='nsew')

        self.initialize_variables()
        self.create_figures()
        self.create_widgets()
        self.default_calib = self.load_config()
        self.load_default_calibrations()

    def initialize_variables(self):
        """Initializes waveplate maximum and offset values and associated Tkinter StringVars."""
        for i in range(1, 5):
            setattr(self, f'max_wp_{i}', 0)
            setattr(self, f'offset_wp_{i}', 0)
            setattr(self, f'strvar_wp_{i}_max', tk.StringVar(value="0"))
            setattr(self, f'strvar_wp_{i}_offset', tk.StringVar(value="0"))

    def create_figures(self):
        """Creates the Matplotlib figure and axes for each waveplate plot."""
        self.fig, self.axes = Figure(figsize=(4, 3), dpi=100), []
        for i in range(4):
            ax = self.fig.add_subplot(4, 1, i + 1)
            self.axes.append(ax)

        self.img = FigureCanvasTkAgg(self.fig, self.frm_plot)
        self.img.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self.img.draw()

    def create_widgets(self):
        """Creates labels, entries, and buttons within the options frame for each waveplate."""
        self.create_option_labels()
        self.create_buttons()

        self.selected_wp = tk.StringVar(self.win)
        self.selected_wp.set("Select WP")
        self.wp_dropdown = tk.OptionMenu(self.frm_options, self.selected_wp, "1", "2", "3", "4")
        self.wp_dropdown.grid(row=9, column=0, padx=2, pady=2, sticky='nsew')

        self.btn_update_calib = tk.Button(
            self.frm_options, text='Update Selected WP Calibration File',
            command=self.update_selected_calibration_file
        )
        self.btn_update_calib.grid(row=9, column=1, padx=2, pady=2, sticky='nsew')

        self.btn_exit = tk.Button(
            self.frm_options, text='Exit', command=self.on_close
        )
        self.btn_exit.grid(row=10, column=0, columnspan=2, padx=2, pady=10, sticky='nsew')

    def create_option_labels(self):
        """Generates labels and entries for each waveplate's maximum and offset values."""
        for i in range(1, 5):
            max_label = f"Max WP{i} (W):" if i != 2 else f"Max WP{i} (mW):"
            offset_label = f"WP{i} offset phase (deg):"
            row_offset = (i - 1) * 2

            tk.Label(self.frm_options, text=max_label).grid(row=row_offset, column=1, padx=2, pady=2)
            tk.Entry(self.frm_options, width=10, textvariable=getattr(self, f'strvar_wp_{i}_max')).grid(
                row=row_offset, column=2, padx=2, pady=2
            )

            tk.Label(self.frm_options, text=offset_label).grid(row=row_offset + 1, column=1, padx=2, pady=2)
            tk.Entry(self.frm_options, width=10, textvariable=getattr(self, f'strvar_wp_{i}_offset')).grid(
                row=row_offset + 1, column=2, padx=2, pady=2
            )

    def create_buttons(self):
        """Adds buttons for loading calibration files for each waveplate."""
        tk.Label(self.frm_options, text='Open calibration file:').grid(row=0, column=0, padx=2, pady=2, sticky='nsew')

        for i in range(1, 5):
            button = tk.Button(
                self.frm_options, text=f'Update WP{i}',
                command=lambda i=i: self.open_calibration_file(i, self.axes[i - 1], self.colors[i - 1])
            )
            button.grid(row=i, column=0, padx=2, pady=2, sticky='nsew')
            setattr(self, f'but_calibration_power_wp_{i}', button)

    def load_config(self):
        """Loads calibration configuration from a JSON file, ensuring valid paths exist."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                return {k: v for k, v in config.items() if os.path.isfile(v)}
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning("Calibration configuration not found or invalid. Using default values.")
            return {str(i): f"default_wp_{i}_calib.txt" for i in range(1, 5)}

    def open_calibration_file(self, wp_index, axis, color):
        """
        Loads the calibration file for the specified waveplate index from the JSON-configured path and plots the data.
        """
        filename = self.default_calib.get(str(wp_index))
        if not filename or not os.path.isfile(filename):
            logging.error(f"Calibration file for WP{wp_index} is missing or invalid in the JSON configuration.")
            return

        try:
            angles, powers = np.loadtxt(filename, delimiter='\t', unpack=True)
            amplitude, phase = self.cos_fit(angles, powers)
            self.plot_waveplate_data(axis, angles, powers, color, amplitude, phase)

            self.update_waveplate_data(wp_index, amplitude, phase)
            getattr(self, f'but_calibration_power_wp_{wp_index}').config(fg='green')
            logging.info(f"Calibration data for WP{wp_index} loaded from {filename}.")
        except Exception as e:
            logging.error(f"Failed to load the calibration data for WP{wp_index}: {e}")
            getattr(self, f'but_calibration_power_wp_{wp_index}').config(fg='red')

    def load_default_calibrations(self):
        """Loads calibration data from default files for each waveplate based on current default_calib."""
        for wp_index_str, filepath in self.default_calib.items():
            wp_index = int(wp_index_str)  # Convert key to integer
            self.open_calibration_file(wp_index, self.axes[wp_index - 1], self.colors[wp_index - 1])

    def save_config(self):
        """Saves the current calibration configuration to a JSON file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.default_calib, f)
        logging.info("Saved calibration configuration to file.")

    def update_selected_calibration_file(self):
        """Updates the calibration file path for the selected waveplate."""
        wp_index = self.selected_wp.get()
        if wp_index not in {"1", "2", "3", "4"}:
            logging.warning("Please select a valid waveplate number.")
            return

        init_dir = os.path.join('./ressources/calibration/')
        filename = tk.filedialog.askopenfilename(initialdir=init_dir,title=f"Select calibration file for WP{wp_index}")
        if filename:
            self.default_calib[wp_index] = filename
            logging.info(f"Updated WP{wp_index} calibration file path to {filename}")

            self.save_config()

    def plot_waveplate_data(self, axis, angles, powers, color, amplitude, phase):
        """Plots waveplate calibration data."""
        axis.clear()
        axis.plot(angles, powers, f'{color}o')
        x = np.linspace(0, 180, 100)
        axis.plot(x, self.cos_func(x, amplitude, phase), color)
        self.img.draw()

    def update_waveplate_data(self, wp_index, amplitude, phase):
        """
        Updates the maximum power and offset phase for a specific waveplate.
        """
        max_value = 2 * amplitude
        setattr(self, f'max_wp_{wp_index}', max_value)
        setattr(self, f'offset_wp_{wp_index}', phase)

        getattr(self, f'strvar_wp_{wp_index}_max').set(f"{max_value:.2f}")
        getattr(self, f'strvar_wp_{wp_index}_offset').set(f"{phase:.2f}")

    def cos_func(self, x, amplitude, phase):
        """
        Cosine function used for fitting calibration data.
        """
        return amplitude * np.cos(2 * np.pi / 90 * x - 2 * np.pi / 90 * phase) + amplitude

    def cos_fit(self, x, y):
        """
        Fits a cosine function to waveplate calibration data.
        """
        initial_guess = (np.max(y) / 2, 0)
        popt, _ = curve_fit(self.cos_func, x, y, p0=initial_guess)
        return popt

    def angle_to_power(self, angle, wp_index):
        """
        Computes power from waveplate angle using cosine fit parameters.
        """
        return self.cos_func(angle, getattr(self, f'max_wp_{wp_index}') / 2, getattr(self, f'offset_wp_{wp_index}'))

    def power_to_angle(self, power, wp_index):
        """
        Computes the waveplate angle for a given power.
        """
        max_wp = getattr(self, f'max_wp_{wp_index}')
        offset_wp = getattr(self, f'offset_wp_{wp_index}')
        return -(45 * np.arccos(power / max_wp - 1)) / np.pi + offset_wp

    def on_close(self):
        """Closes the Tkinter window."""
        self.win.destroy()
