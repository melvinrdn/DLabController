import datetime
import threading
import tkinter as tk
from collections import deque
from datetime import date
from tkinter import ttk
from tkinter.filedialog import asksaveasfile, askopenfilename, asksaveasfilename
from tkinter.scrolledtext import ScrolledText
import logging
import cv2
import h5py
import numpy as np
import pylablib as pll
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from pylablib.devices import Andor
import thorlabs_apt as apt

import hardware.SLM_driver._slm_py as slm
import diagnostics.diagnostics_helpers as help
from diagnostics.WaveplateCalib import WaveplateCalib
import time

from diagnostics.diagnostics_helpers import ColorFormatter
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("from WPCalib: %(levelname)s: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler])

class HHGView(object):

    def __init__(self, parent):

        self.parent = parent
        self.initialize_window()
        self.initialize_variables()
        self.initialize_frames()

        if self.ANDOR_cam is True:
            self.name_cam = 'ANDOR_cam'

        self.open_calibrator_on_start()

    def initialize_window(self):
        self.win = tk.Toplevel()
        self.win.title('D-Lab Controller - HHGView')
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

    ## Frames initialization
    def initialize_variables(self):
        pll.par["devices/dlls/andor_sdk2"] = "hardware/andor_driver/"
        self.cam = None
        self.slm_lib = slm
        self.calibration_image = np.zeros([512, 512])
        self.calibration_image_update = np.zeros([512, 512])
        self.calibration_image_update_energy = np.zeros([512, 512])
        self.eaxis = None
        self.eaxis_correct = None
        self.roix = [0, 512]
        self.roiy = [0, 512]
        self.live_is_pressed = False
        self.measurement_counter = 0
        self.measurement_running = 0
        self.stop_acquire = 0
        self.ANDOR_cam = False

        #self.saving_folder = 'C:/data/' + str(date.today()) + '/' + str(date.today())
        #self.autolog = self.saving_folder + '-' + 'auto-log.txt'
        self.autolog = './ressources/dummy_log'
        self.f = open(self.autolog, "a+")

    def initialize_frames(self):
        self.frm_control = ttk.Frame(self.win)
        self.frm_control.grid(row=0, column=1)

        self.frm_mcpview = ttk.Frame(self.win)
        self.frm_mcpview.grid(row=0, column=2)

        self.initialize_control_panel_1()
        self.initialize_control_panel_2()
        self.initialize_control_panel_3()

        self.initialize_mcp_frame()

    def initialize_control_panel_1(self):
        frm_control_panel_1 = ttk.LabelFrame(self.frm_control, text='Acquisition control')

        # Create buttons
        self.but_mcp_single_image = tk.Button(frm_control_panel_1, text='Single Image', command=self.enabl_mcp_simple)
        self.but_mcp_liveview = tk.Button(frm_control_panel_1, text='Live View', command=self.enabl_mcp_live)

        # Dictionary to hold StringVars, Entries, and Labels
        self.mcp_controls = {
            'mcp_cam_choice': {'label': 'MCP Camera:', 'default': '', 'widget': 'combobox'},
            'mcp_voltage': {'label': 'Neg. MCP voltage (V):', 'default': '-1400'},
            'mcp_avgs': {'label': 'MCP averages:', 'default': '1'},
            'mcp_exposure_time': {'label': 'MCP exposure (µs):', 'default': '100000'},
            'comment': {'label': 'Comment:', 'default': ''}
        }

        for i, (key, properties) in enumerate(self.mcp_controls.items()):
            self.mcp_controls[key]['var'] = tk.StringVar(self.win, properties['default'])

            label = tk.Label(frm_control_panel_1, text=properties['label'])
            if properties.get('widget') == 'combobox':
                self.mcp_controls[key]['entry'] = ttk.Combobox(frm_control_panel_1, textvariable=self.mcp_controls[key]['var'])
                self.mcp_controls[key]['entry']['values'] = ('Andor',)
                self.mcp_controls[key]['entry'].bind("<<ComboboxSelected>>", self.change_mcp_cam)
            else:
                self.mcp_controls[key]['entry'] = tk.Entry(frm_control_panel_1, textvariable=self.mcp_controls[key]['var'])

            label.grid(row=i, column=0, padx=2, pady=2, sticky='w')
            self.mcp_controls[key]['entry'].grid(row=i, column=1, padx=2, pady=2, sticky='nsew')

        frm_control_panel_1.grid(row=0, column=0, sticky='nsew')

        lbl_acquisition = tk.Label(frm_control_panel_1, text='Acquisition:')
        lbl_acquisition.grid(row=0, column=2, pady=2, padx=2, sticky='nsew')
        self.but_mcp_single_image.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        self.but_mcp_liveview.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')

    def initialize_control_panel_2(self):
        self.frm_control_panel_2 = ttk.Notebook(self.frm_control)
        self.frm_waveplates = ttk.Frame(self.frm_control)
        self.frm_stages = ttk.Frame(self.frm_control)
        frm_wp_power_cal = ttk.Frame(self.frm_control)
        self.frm_control_panel_2.add(self.frm_waveplates, text="Waveplate control")
        self.frm_control_panel_2.add(self.frm_stages, text="Stage control")
        self.frm_control_panel_2.add(frm_wp_power_cal, text="Power calibration")
        self.frm_control_panel_2.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')

        # Waveplate control tab
        lbl_waveplate = tk.Label(self.frm_waveplates, text='Waveplate')
        lbl_waveplate.grid(row=0, column=0, pady=2, padx=2, sticky='nsew')

        lbl_nr = tk.Label(self.frm_waveplates, text='#')
        lbl_nr.grid(row=0, column=1, pady=2, padx=2, sticky='nsew')

        lbl_is = tk.Label(self.frm_waveplates, text='is (deg)')
        lbl_is.grid(row=0, column=2, pady=2, padx=2, sticky='nsew')

        lbl_should = tk.Label(self.frm_waveplates, text='should')
        lbl_should.grid(row=0, column=3, pady=2, padx=2, sticky='nsew')

        # Waveplate 1
        lbl_wp_1 = tk.Label(self.frm_waveplates, text='ω field', fg='red')
        lbl_wp_1.grid(row=1, column=0, pady=2, padx=2, sticky='nsew')

        self.strvar_wp_1_is = tk.StringVar(self.win, '')
        self.ent_wp_1_is = tk.Entry(self.frm_waveplates, width=10, state='readonly', textvariable=self.strvar_wp_1_is)
        self.ent_wp_1_is.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_1_should = tk.StringVar(self.win, '')
        self.ent_wp_1_should = tk.Entry(self.frm_waveplates, width=10, validate='all', textvariable=self.strvar_wp_1_should, state='disabled')
        self.ent_wp_1_should.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_1_nr = tk.StringVar(self.win, '83837725')
        self.ent_wp_1_nr = tk.Entry(self.frm_waveplates, width=10, validate='all', textvariable=self.strvar_wp_1_nr)
        self.ent_wp_1_nr.grid(row=1, column=1, pady=2, padx=2, sticky='nsew')

        self.but_wp_1_init = tk.Button(self.frm_waveplates, text='Init', command=lambda: self.init_motor_thorlabs('wp_1'))
        self.but_wp_1_init.grid(row=1, column=4, padx=2, pady=2, sticky='nsew')

        self.but_wp_1_home = tk.Button(self.frm_waveplates, text='Home', command=lambda: self.home_motor_thorlabs('wp_1'), state='disabled')
        self.but_wp_1_home.grid(row=1, column=5, padx=2, pady=2, sticky='nsew')

        self.but_wp_1_read = tk.Button(self.frm_waveplates, text='Read', command=lambda: self.read_motor_thorlabs('wp_1'), state='disabled')
        self.but_wp_1_read.grid(row=1, column=6, padx=2, pady=2, sticky='nsew')

        self.but_wp_1_move = tk.Button(self.frm_waveplates, text='Move', command=lambda: self.move_motor_thorlabs('wp_1', float(self.strvar_wp_1_should.get())), state='disabled')
        self.but_wp_1_move.grid(row=1, column=7, padx=2, pady=2, sticky='nsew')

        self.var_wp_1_power = tk.IntVar()
        self.cb_wp_1_power = tk.Checkbutton(self.frm_waveplates, text='Power', variable=self.var_wp_1_power, onvalue=1,
                                            offvalue=0, command=lambda: logging.info(f"Power mode on wp_1 set to: {self.var_wp_1_power.get()}"))
        self.cb_wp_1_power.grid(row=1, column=8, padx=2, pady=2, sticky='nsew')

        self.but_wp_1_disable = tk.Button(self.frm_waveplates, text='Disable', command=lambda: self.disable_motor_thorlabs('wp_1'), state='disabled')
        self.but_wp_1_disable.grid(row=1, column=9, padx=2, pady=2, sticky='nsew')

        # Waveplate 2
        lbl_wp_2 = tk.Label(self.frm_waveplates, text='2ω field', fg='green')
        lbl_wp_2.grid(row=2, column=0, pady=2, padx=2, sticky='nsew')

        self.strvar_wp_2_is = tk.StringVar(self.win, '')
        self.ent_wp_2_is = tk.Entry(self.frm_waveplates, width=10, state='readonly', textvariable=self.strvar_wp_2_is)
        self.ent_wp_2_is.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_2_should = tk.StringVar(self.win, '')
        self.ent_wp_2_should = tk.Entry(self.frm_waveplates, width=10, validate='all', textvariable=self.strvar_wp_2_should)
        self.ent_wp_2_should.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_2_nr = tk.StringVar(self.win, '0000000')
        self.ent_wp_2_nr = tk.Entry(self.frm_waveplates, width=10, validate='all', textvariable=self.strvar_wp_2_nr)
        self.ent_wp_2_nr.grid(row=2, column=1, pady=2, padx=2, sticky='nsew')

        self.var_wp_2_power = tk.IntVar()
        self.wb_wp_2_power = tk.Checkbutton(self.frm_waveplates, text='Power', variable=self.var_wp_2_power, onvalue=1,
                                            offvalue=0, command=None)
        self.wb_wp_2_power.grid(row=2, column=8, padx=2, pady=2, sticky='nsew')

        self.but_wp_2_init = tk.Button(self.frm_waveplates, text='Init', command=None)
        self.but_wp_2_init.grid(row=2, column=4, padx=2, pady=2, sticky='nsew')

        self.but_wp_2_home = tk.Button(self.frm_waveplates, text='Home', command=None)
        self.but_wp_2_home.grid(row=2, column=5, padx=2, pady=2, sticky='nsew')

        self.but_wp_2_read = tk.Button(self.frm_waveplates, text='Read', command=None)
        self.but_wp_2_read.grid(row=2, column=6, padx=2, pady=2, sticky='nsew')

        self.but_wp_2_move = tk.Button(self.frm_waveplates, text='Move', command=None)
        self.but_wp_2_move.grid(row=2, column=7, padx=2, pady=2, sticky='nsew')

        # Waveplate 3
        lbl_wp_3 = tk.Label(self.frm_waveplates, text='3ω field', fg='blue')
        lbl_wp_3.grid(row=3, column=0, pady=2, padx=2, sticky='nsew')

        self.strvar_wp_3_is = tk.StringVar(self.win, '')
        self.ent_wp_3_is = tk.Entry(self.frm_waveplates, width=10, state='readonly', textvariable=self.strvar_wp_3_is)
        self.ent_wp_3_is.grid(row=3, column=2, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_3_should = tk.StringVar(self.win, '')
        self.ent_wp_3_should = tk.Entry(self.frm_waveplates, width=10, validate='all', textvariable=self.strvar_wp_3_should)
        self.ent_wp_3_should.grid(row=3, column=3, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_3_nr = tk.StringVar(self.win, '0000000')
        self.ent_wp_3_nr = tk.Entry(self.frm_waveplates, width=10, validate='all', textvariable=self.strvar_wp_3_nr)
        self.ent_wp_3_nr.grid(row=3, column=1, pady=2, padx=2, sticky='nsew')

        self.var_wp_3_power = tk.IntVar()
        self.wb_wp_3_power = tk.Checkbutton(self.frm_waveplates, text='Power', variable=self.var_wp_3_power, onvalue=1,
                                            offvalue=0, command=None)
        self.wb_wp_3_power.grid(row=3, column=8, padx=2, pady=2, sticky='nsew')

        self.but_wp_3_init = tk.Button(self.frm_waveplates, text='Init', command=None)
        self.but_wp_3_init.grid(row=3, column=4, padx=2, pady=2, sticky='nsew')

        self.but_wp_3_home = tk.Button(self.frm_waveplates, text='Home', command=None)
        self.but_wp_3_home.grid(row=3, column=5, padx=2, pady=2, sticky='nsew')

        self.but_wp_3_read = tk.Button(self.frm_waveplates, text='Read', command=None)
        self.but_wp_3_read.grid(row=3, column=6, padx=2, pady=2, sticky='nsew')

        self.but_wp_3_move = tk.Button(self.frm_waveplates, text='Move', command=None)
        self.but_wp_3_move.grid(row=3, column=7, padx=2, pady=2, sticky='nsew')

        # Waveplate 4
        lbl_wp_4 = tk.Label(self.frm_waveplates, text='Waveplate 4')
        lbl_wp_4.grid(row=4, column=0, pady=2, padx=2, sticky='nsew')

        self.strvar_wp_4_is = tk.StringVar(self.win, '')
        self.ent_wp_4_is = tk.Entry(self.frm_waveplates, width=10, state='readonly', textvariable=self.strvar_wp_4_is)
        self.ent_wp_4_is.grid(row=4, column=2, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_4_should = tk.StringVar(self.win, '')
        self.ent_wp_4_should = tk.Entry(self.frm_waveplates, width=10, validate='all', textvariable=self.strvar_wp_4_should)
        self.ent_wp_4_should.grid(row=4, column=3, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_4_nr = tk.StringVar(self.win, '0000000')
        self.ent_wp_4_nr = tk.Entry(self.frm_waveplates, width=10, validate='all', textvariable=self.strvar_wp_4_nr)
        self.ent_wp_4_nr.grid(row=4, column=1, pady=2, padx=2, sticky='nsew')

        self.var_wp_4_power = tk.IntVar()
        self.wb_wp_4_power = tk.Checkbutton(self.frm_waveplates, text='Power', variable=self.var_wp_4_power, onvalue=1,
                                            offvalue=0, command=None)
        self.wb_wp_4_power.grid(row=4, column=8, padx=2, pady=2, sticky='nsew')

        self.but_wp_4_init = tk.Button(self.frm_waveplates, text='Init', command=None)
        self.but_wp_4_init.grid(row=4, column=4, padx=2, pady=2, sticky='nsew')

        self.but_wp_4_home = tk.Button(self.frm_waveplates, text='Home', command=None)
        self.but_wp_4_home.grid(row=4, column=5, padx=2, pady=2, sticky='nsew')

        self.but_wp_4_read = tk.Button(self.frm_waveplates, text='Read', command=None)
        self.but_wp_4_read.grid(row=4, column=6, padx=2, pady=2, sticky='nsew')

        self.but_wp_4_move = tk.Button(self.frm_waveplates, text='Move', command=None)
        self.but_wp_4_move.grid(row=4, column=7, padx=2, pady=2, sticky='nsew')

        # Stage control tab
        lbl_stage = tk.Label(self.frm_stages, text='Stage')
        lbl_stage.grid(row=0, column=0, pady=2, padx=2, sticky='nsew')

        lbl_nr = tk.Label(self.frm_stages, text='#')
        lbl_nr.grid(row=0, column=1, pady=2, padx=2, sticky='nsew')

        lbl_is = tk.Label(self.frm_stages, text='is (mm)')
        lbl_is.grid(row=0, column=2, pady=2, padx=2, sticky='nsew')

        lbl_should = tk.Label(self.frm_stages, text='should')
        lbl_should.grid(row=0, column=3, pady=2, padx=2, sticky='nsew')

        # Stage 1
        lbl_stage_1 = tk.Label(self.frm_stages, text='Stage 1:')
        lbl_stage_1.grid(row=3, column=0, padx=2, pady=2, sticky='nsew')

        self.strvar_stage_1_is = tk.StringVar(self.win, '')
        self.ent_stage_1_is = tk.Entry(self.frm_stages, width=10, state='readonly', textvariable=self.strvar_stage_1_is)
        self.ent_stage_1_is.grid(row=3, column=2, padx=2, pady=2, sticky='nsew')

        self.strvar_stage_1_should = tk.StringVar(self.win, '')
        self.ent_stage_1_should = tk.Entry(self.frm_stages, width=10, validate='all', textvariable=self.strvar_stage_1_should)
        self.ent_stage_1_should.grid(row=3, column=3, padx=2, pady=2, sticky='nsew')

        self.strvar_stage_1_nr = tk.StringVar(self.win, '000000')
        self.ent_stage_1_nr = tk.Entry(self.frm_stages, width=10, validate='all', textvariable=self.strvar_stage_1_nr)
        self.ent_stage_1_nr.grid(row=3, column=1, padx=2, pady=2, sticky='nsew')

        self.but_stage_1_init = tk.Button(self.frm_stages, text='Init', command=None)
        self.but_stage_1_init.grid(row=3, column=4, padx=2, pady=2, sticky='nsew')

        self.but_stage_1_home = tk.Button(self.frm_stages, text='Home', command=None)
        self.but_stage_1_home.grid(row=3, column=5, padx=2, pady=2, sticky='nsew')

        self.but_stage_1_read = tk.Button(self.frm_stages, text='Read', command=None)
        self.but_stage_1_read.grid(row=3, column=6, padx=2, pady=2, sticky='nsew')

        self.but_stage_1_move = tk.Button(self.frm_stages, text='Move', command=None)
        self.but_stage_1_move.grid(row=3, column=7, padx=2, pady=2, sticky='nsew')

        # Stage 2
        lbl_stage_2 = tk.Label(self.frm_stages, text='Stage 2:')
        lbl_stage_2.grid(row=4, column=0, padx=2, pady=2, sticky='nsew')

        self.strvar_stage_2_is = tk.StringVar(self.win, '')
        self.ent_stage_2_is = tk.Entry(self.frm_stages, width=10, state='readonly', textvariable=self.strvar_stage_2_is)
        self.ent_stage_2_is.grid(row=4, column=2, padx=2, pady=2, sticky='nsew')

        self.strvar_stage_2_should = tk.StringVar(self.win, '')
        self.ent_stage_2_should = tk.Entry(self.frm_stages, width=10, validate='all', textvariable=self.strvar_stage_2_should)
        self.ent_stage_2_should.grid(row=4, column=3, padx=2, pady=2, sticky='nsew')

        self.strvar_stage_2_nr = tk.StringVar(self.win, '000000')
        self.ent_stage_2_nr = tk.Entry(self.frm_stages, width=10, validate='all', textvariable=self.strvar_stage_2_nr)
        self.ent_stage_2_nr.grid(row=4, column=1, padx=2, pady=2, sticky='nsew')

        self.but_stage_2_init = tk.Button(self.frm_stages, text='Init', command=None)
        self.but_stage_2_init.grid(row=4, column=4, padx=2, pady=2, sticky='nsew')

        self.but_stage_2_home = tk.Button(self.frm_stages, text='Home', command=None)
        self.but_stage_2_home.grid(row=4, column=5, padx=2, pady=2, sticky='nsew')

        self.but_stage_2_read = tk.Button(self.frm_stages, text='Read', command=None)
        self.but_stage_2_read.grid(row=4, column=6, padx=2, pady=2, sticky='nsew')

        self.but_stage_2_move = tk.Button(self.frm_stages, text='Move', command=None)
        self.but_stage_2_move.grid(row=4, column=7, padx=2, pady=2, sticky='nsew')

        # Stage 3
        lbl_stage_3 = tk.Label(self.frm_stages, text='Stage 3:')
        lbl_stage_3.grid(row=5, column=0, padx=2, pady=2, sticky='nsew')

        self.strvar_stage_3_is = tk.StringVar(self.win, '')
        self.ent_stage_3_is = tk.Entry(self.frm_stages, width=10, state='readonly', textvariable=self.strvar_stage_3_is)
        self.ent_stage_3_is.grid(row=5, column=2, padx=2, pady=2, sticky='nsew')

        self.strvar_stage_3_should = tk.StringVar(self.win, '')
        self.ent_stage_3_should = tk.Entry(self.frm_stages, width=10, validate='all', textvariable=self.strvar_stage_3_should)
        self.ent_stage_3_should.grid(row=5, column=3, padx=2, pady=2, sticky='nsew')

        self.strvar_stage_3_nr = tk.StringVar(self.win, '000000')
        self.ent_stage_3_nr = tk.Entry(self.frm_stages, width=10, validate='all', textvariable=self.strvar_stage_3_nr)
        self.ent_stage_3_nr.grid(row=5, column=1, padx=2, pady=2, sticky='nsew')

        self.but_stage_3_init = tk.Button(self.frm_stages, text='Init', command=None)
        self.but_stage_3_init.grid(row=5, column=4, padx=2, pady=2, sticky='nsew')

        self.but_stage_3_home = tk.Button(self.frm_stages, text='Home', command=None)
        self.but_stage_3_home.grid(row=5, column=5, padx=2, pady=2, sticky='nsew')

        self.but_stage_3_read = tk.Button(self.frm_stages, text='Read', command=None)
        self.but_stage_3_read.grid(row=5, column=6, padx=2, pady=2, sticky='nsew')

        self.but_stage_3_move = tk.Button(self.frm_stages, text='Move', command=None)
        self.but_stage_3_move.grid(row=5, column=7, padx=2, pady=2, sticky='nsew')

        # Stage 4
        lbl_stage_4 = tk.Label(self.frm_stages, text='Stage 4:')
        lbl_stage_4.grid(row=6, column=0, padx=2, pady=2, sticky='nsew')

        self.strvar_stage_4_is = tk.StringVar(self.win, '')
        self.ent_stage_4_is = tk.Entry(self.frm_stages, width=10, state='readonly', textvariable=self.strvar_stage_4_is)
        self.ent_stage_4_is.grid(row=6, column=2, padx=2, pady=2, sticky='nsew')

        self.strvar_stage_4_should = tk.StringVar(self.win, '')
        self.ent_stage_4_should = tk.Entry(self.frm_stages, width=10, validate='all', textvariable=self.strvar_stage_4_should)
        self.ent_stage_4_should.grid(row=6, column=3, padx=2, pady=2, sticky='nsew')

        self.strvar_stage_4_nr = tk.StringVar(self.win, '000000')
        self.ent_stage_4_nr = tk.Entry(self.frm_stages, width=10, validate='all', textvariable=self.strvar_stage_4_nr)
        self.ent_stage_4_nr.grid(row=6, column=1, padx=2, pady=2, sticky='nsew')

        self.but_stage_4_init = tk.Button(self.frm_stages, text='Init', command=None)
        self.but_stage_4_init.grid(row=6, column=4, padx=2, pady=2, sticky='nsew')

        self.but_stage_4_home = tk.Button(self.frm_stages, text='Home', command=None)
        self.but_stage_4_home.grid(row=6, column=5, padx=2, pady=2, sticky='nsew')

        self.but_stage_4_read = tk.Button(self.frm_stages, text='Read', command=None)
        self.but_stage_4_read.grid(row=6, column=6, padx=2, pady=2, sticky='nsew')

        self.but_stage_4_move = tk.Button(self.frm_stages, text='Move', command=None)
        self.but_stage_4_move.grid(row=6, column=7, padx=2, pady=2, sticky='nsew')

        # Power calibration tab
        self.but_open_calibrator = tk.Button(frm_wp_power_cal, text='Open calibrator', command=self.enable_calibrator)
        self.but_open_calibrator.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')

        # Laser parameter
        lbl_laser_param = tk.Label(frm_wp_power_cal, text='Laser')
        lbl_laser_param.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')

        lbl_laser_att = tk.Label(frm_wp_power_cal, text='Attenuation:')
        lbl_laser_att.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')

        self.strvar_laser_att = tk.StringVar(self.win, '100')
        self.ent_laser_att = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_laser_att)
        self.ent_laser_att.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')

        lbl_laser_pp = tk.Label(frm_wp_power_cal, text='Pulse picker:')
        lbl_laser_pp.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')

        self.strvar_laser_pp = tk.StringVar(self.win, '1')
        self.ent_laser_pp = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_laser_pp)
        self.ent_laser_pp.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')

        # Power parameter
        lbl_omega_power = tk.Label(frm_wp_power_cal, text='Max power (W):')
        lbl_omega_power.grid(row=1, column=5, padx=2, pady=2, sticky='nsew')

        lbl_omega_offset = tk.Label(frm_wp_power_cal, text='Offset phase (deg):')
        lbl_omega_offset.grid(row=2, column=5, padx=2, pady=2, sticky='nsew')

        lbl_omega_current_power = tk.Label(frm_wp_power_cal, text='Current power (W):')
        lbl_omega_current_power.grid(row=3, column=5, padx=2, pady=2, sticky='nsew')

        # Omega
        lbl_omega = tk.Label(frm_wp_power_cal, text='ω field', fg='red')
        lbl_omega.grid(row=0, column=6, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_1_power = tk.StringVar(self.win, '0')
        self.ent_wp_1_power = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_wp_1_power)
        self.ent_wp_1_power.grid(row=1, column=6, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_1_offset = tk.StringVar(self.win, '0')
        self.ent_wp_1_offset = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_wp_1_offset)
        self.ent_wp_1_offset.grid(row=2, column=6, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_1_current_power = tk.StringVar(self.win, '0')
        self.ent_wp_1_current_power = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_wp_1_current_power)
        self.ent_wp_1_current_power.grid(row=3, column=6, padx=2, pady=2, sticky='nsew')

        # 2 Omega
        lbl_2omega = tk.Label(frm_wp_power_cal, text='2ω field', fg='green')
        lbl_2omega.grid(row=0, column=7, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_2_power = tk.StringVar(self.win, '0')
        self.ent_wp_2_power = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_wp_2_power)
        self.ent_wp_2_power.grid(row=1, column=7, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_2_offset = tk.StringVar(self.win, '0')
        self.ent_wp_2_offset = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_wp_2_offset)
        self.ent_wp_2_offset.grid(row=2, column=7, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_2_current_power = tk.StringVar(self.win, '0')
        self.ent_wp_2_current_power = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_wp_2_current_power)
        self.ent_wp_2_current_power.grid(row=3, column=7, padx=2, pady=2, sticky='nsew')

        # 3 Omega
        lbl_3omega = tk.Label(frm_wp_power_cal, text='3ω field', fg='blue')
        lbl_3omega.grid(row=0, column=8, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_3_power = tk.StringVar(self.win, '0')
        self.ent_wp_3_power = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_wp_3_power)
        self.ent_wp_3_power.grid(row=1, column=8, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_3_offset = tk.StringVar(self.win, '0')
        self.ent_wp_3_offset = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_wp_3_offset)
        self.ent_wp_3_offset.grid(row=2, column=8, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_3_current_power = tk.StringVar(self.win, '0')
        self.ent_wp_3_current_power = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_wp_3_current_power)
        self.ent_wp_3_current_power.grid(row=3, column=8, padx=2, pady=2, sticky='nsew')

        # Waveplate 4
        lbl_wp_4_calib = tk.Label(frm_wp_power_cal, text='Waveplate 4')
        lbl_wp_4_calib.grid(row=0, column=9, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_4_power = tk.StringVar(self.win, '0')
        self.ent_wp_4_power = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_wp_4_power)
        self.ent_wp_4_power.grid(row=1, column=9, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_4_offset = tk.StringVar(self.win, '0')
        self.ent_wp_4_offset = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_wp_4_offset)
        self.ent_wp_4_offset.grid(row=2, column=9, padx=2, pady=2, sticky='nsew')

        self.strvar_wp_4_current_power = tk.StringVar(self.win, '0')
        self.ent_wp_4_current_power = tk.Entry(frm_wp_power_cal, width=10, validate='all', textvariable=self.strvar_wp_4_current_power)
        self.ent_wp_4_current_power.grid(row=3, column=9, padx=2, pady=2, sticky='nsew')

    def initialize_control_panel_3(self):
        self.frm_control_panel_3 = ttk.Notebook(self.frm_control)
        frm_scans = ttk.Frame(self.frm_control)
        self.frm_control_panel_3.add(frm_scans, text="Scans")
        self.frm_control_panel_3.grid(row=3, column=0, padx=2, pady=2, sticky='nsew')

    def initialize_mcp_frame(self):
        self.frm_mcp_panel = ttk.Notebook(self.frm_mcpview)
        frm_mcp_image = ttk.Frame(self.frm_mcpview)
        frm_mcp_calibrate = ttk.Frame(self.frm_mcpview)
        frm_mcp_calibrate_energy = ttk.Frame(self.frm_mcpview)
        frm_mcp_treated = ttk.Frame(self.frm_mcpview)

        self.frm_mcp_panel.add(frm_mcp_image, text='MCP raw')
        self.frm_mcp_panel.add(frm_mcp_calibrate, text='Calibrate Spatial')
        self.frm_mcp_panel.add(frm_mcp_calibrate_energy, text='Calibrate Energy')
        self.frm_mcp_panel.add(frm_mcp_treated, text='MCP treated')
        self.frm_mcp_panel.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')

        frm_mcp_calibrate_options = ttk.LabelFrame(frm_mcp_calibrate, text='Calibration Options')
        frm_mcp_calibrate_image = ttk.LabelFrame(frm_mcp_calibrate, text='Calibration Image')
        frm_mcp_calibrate_options_energy = ttk.LabelFrame(frm_mcp_calibrate_energy, text='Calibration Options')
        frm_mcp_calibrate_image_energy = ttk.LabelFrame(frm_mcp_calibrate_energy, text='Calibration Image')
        frm_mcp_treated_options = ttk.LabelFrame(frm_mcp_treated, text='Options')
        frm_mcp_treated_image = ttk.LabelFrame(frm_mcp_treated, text='Images')

        frm_mcp_calibrate_options.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_calibrate_image.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_calibrate_options_energy.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_calibrate_image_energy.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_treated_options.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_treated_image.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')

        self.figure_mcp = Figure(figsize=(5, 6), dpi=100)
        self.ax_mcp = self.figure_mcp.add_subplot(211)
        self.ax_harmonics = self.figure_mcp.add_subplot(212)
        self.ax_mcp.set_xlim(0, 1600)
        self.ax_mcp.set_ylim(0, 1000)
        self.ax_harmonics.set_xlim(0, 1600)
        self.figure_mcp.tight_layout()
        self.figure_mcp.canvas.draw()
        self.image_mcp = FigureCanvasTkAgg(self.figure_mcp, frm_mcp_image)
        self.tk_widget_figure_mcp = self.image_mcp.get_tk_widget()
        self.tk_widget_figure_mcp.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.image_mcp.draw()

        self.figure_mcp_calib = Figure(figsize=(5, 4), dpi=100)
        self.ax_mcp_calib = self.figure_mcp_calib.add_subplot(211)
        self.ax_harmonics_calib = self.figure_mcp_calib.add_subplot(212)
        self.ax_mcp_calib.set_xlim(0, 515)
        self.ax_mcp_calib.set_ylim(0, 515)
        self.ax_harmonics_calib.set_xlim(0, 515)
        self.figure_mcp_calib.tight_layout()
        self.figure_mcp_calib.canvas.draw()
        self.image_mcp_calib = FigureCanvasTkAgg(self.figure_mcp_calib, frm_mcp_calibrate_image)
        self.tk_widget_figure_mcp_calib = self.image_mcp_calib.get_tk_widget()
        self.tk_widget_figure_mcp_calib.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.image_mcp_calib.draw()

        self.but_load_mcp_calib_image = tk.Button(frm_mcp_calibrate_options, text="Load Image",
                                                  command=self.load_test_calibration_image_thread)
        self.but_load_mcp_calib_image.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.but_take_mcp_calib_image = tk.Button(frm_mcp_calibrate_options, text="Take Image",
                                                  command=self.load_test_calibration_image_take_thread)
        self.but_take_mcp_calib_image.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calib_shear_val = tk.Label(frm_mcp_calibrate_options, text="Shear:")
        self.var_mcp_calib_shear_val = tk.StringVar(self.win, "0")
        self.var_mcp_calib_shear_val.trace_add("write", self.update_calib_mcp)
        self.ent_mcp_calib_shear_val = tk.Entry(frm_mcp_calibrate_options,
                                                textvariable=self.var_mcp_calib_shear_val,
                                                width=4, validate='all')
        lbl_mcp_calib_shear_val.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calib_shear_val.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calib_roix_val = tk.Label(frm_mcp_calibrate_options, text="ROIX:")
        self.var_mcp_calib_roix_1_val = tk.StringVar(self.win, "0")
        self.var_mcp_calib_roix_1_val.trace_add("write", self.update_calib_mcp)
        self.var_mcp_calib_roix_2_val = tk.StringVar(self.win, "512")
        self.var_mcp_calib_roix_2_val.trace_add("write", self.update_calib_mcp)
        self.ent_mcp_calib_roix_1_val = tk.Entry(frm_mcp_calibrate_options,
                                                 textvariable=self.var_mcp_calib_roix_1_val,
                                                 width=4, validate='all')
        self.ent_mcp_calib_roix_2_val = tk.Entry(frm_mcp_calibrate_options,
                                                 textvariable=self.var_mcp_calib_roix_2_val,
                                                 width=4, validate='all')
        lbl_mcp_calib_roix_val.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calib_roix_1_val.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calib_roix_2_val.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calib_roiy_val = tk.Label(frm_mcp_calibrate_options, text="ROIY:")
        self.var_mcp_calib_roiy_1_val = tk.StringVar(self.win, "0")
        self.var_mcp_calib_roiy_1_val.trace_add("write", self.update_calib_mcp)
        self.var_mcp_calib_roiy_2_val = tk.StringVar(self.win, "512")
        self.var_mcp_calib_roiy_2_val.trace_add("write", self.update_calib_mcp)
        self.ent_mcp_calib_roiy_1_val = tk.Entry(frm_mcp_calibrate_options,
                                                 textvariable=self.var_mcp_calib_roiy_1_val,
                                                 width=4, validate='all')
        self.ent_mcp_calib_roiy_2_val = tk.Entry(frm_mcp_calibrate_options,
                                                 textvariable=self.var_mcp_calib_roiy_2_val,
                                                 width=4, validate='all')
        lbl_mcp_calib_roiy_val.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calib_roiy_1_val.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calib_roiy_2_val.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calib_background_val = tk.Label(frm_mcp_calibrate_options, text="Background:")
        self.var_mcp_calib_background_val = tk.StringVar(self.win, "0")
        self.var_mcp_calib_background_val.trace_add("write", self.update_calib_mcp)
        self.ent_mcp_calib_background_val = tk.Entry(frm_mcp_calibrate_options,
                                                     textvariable=self.var_mcp_calib_background_val,
                                                     width=6, validate='all')
        lbl_mcp_calib_background_val.grid(row=0, column=4, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calib_background_val.grid(row=0, column=5, padx=2, pady=2, sticky='nsew')

        self.figure_mcp_calib_energy = Figure(figsize=(5, 4), dpi=100)
        self.ax_harmonics_calib_energy_1 = self.figure_mcp_calib_energy.add_subplot(211)
        self.ax_harmonics_calib_energy_2 = self.figure_mcp_calib_energy.add_subplot(212)
        self.ax_harmonics_calib_energy_1.set_xlim(0, 515)
        self.ax_harmonics_calib_energy_2.set_xlim(0, 515)
        self.figure_mcp_calib_energy.tight_layout()
        self.figure_mcp_calib_energy.canvas.draw()
        self.image_mcp_calib_energy = FigureCanvasTkAgg(self.figure_mcp_calib_energy,
                                                        frm_mcp_calibrate_image_energy)
        self.tk_widget_figure_mcp_calib_energy = self.image_mcp_calib_energy.get_tk_widget()
        self.tk_widget_figure_mcp_calib_energy.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.image_mcp_calib_energy.draw()

        lbl_mcp_calib_energy_smooth = tk.Label(frm_mcp_calibrate_options_energy, text="Smooth:")
        self.var_mcp_calib_energy_smooth = tk.StringVar(self.win, "5")
        self.var_mcp_calib_energy_smooth.trace_add("write", self.update_calib_energy_mcp)
        self.ent_mcp_calib_energy_smooth = tk.Entry(frm_mcp_calibrate_options_energy,
                                                    textvariable=self.var_mcp_calib_energy_smooth,
                                                    width=4, validate='all')
        lbl_mcp_calib_energy_smooth.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calib_energy_smooth.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calib_energy_prom = tk.Label(frm_mcp_calibrate_options_energy, text="Peak prominence:")
        self.var_mcp_calib_energy_prom = tk.StringVar(self.win, "20")
        self.var_mcp_calib_energy_prom.trace_add("write", self.update_calib_energy_mcp)
        self.ent_mcp_calib_energy_prom = tk.Entry(frm_mcp_calibrate_options_energy,
                                                  textvariable=self.var_mcp_calib_energy_prom,
                                                  width=6, validate='all')
        lbl_mcp_calib_energy_prom.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calib_energy_prom.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calib_energy_ignore = tk.Label(frm_mcp_calibrate_options_energy, text="Ignore peaks from/to:")
        self.var_mcp_calib_energy_ignore_1 = tk.StringVar(self.win, "0")
        self.var_mcp_calib_energy_ignore_1.trace_add("write", self.update_calib_energy_mcp)
        self.var_mcp_calib_energy_ignore_2 = tk.StringVar(self.win, "512")
        self.var_mcp_calib_energy_ignore_2.trace_add("write", self.update_calib_energy_mcp)
        self.ent_mcp_calib_energy_ignore_1 = tk.Entry(frm_mcp_calibrate_options_energy,
                                                      textvariable=self.var_mcp_calib_energy_ignore_1,
                                                      width=4, validate='all')
        self.ent_mcp_calib_energy_ignore_2 = tk.Entry(frm_mcp_calibrate_options_energy,
                                                      textvariable=self.var_mcp_calib_energy_ignore_2,
                                                      width=4, validate='all')
        lbl_mcp_calib_energy_ignore.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calib_energy_ignore_1.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calib_energy_ignore_2.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calib_energy_ignore_list = tk.Label(frm_mcp_calibrate_options_energy,
                                                          text="Ignore peaks around:")
        self.var_mcp_calib_energy_ignore_list = tk.StringVar(self.win, "")
        self.var_mcp_calib_energy_ignore_list.trace_add("write", self.update_calib_energy_mcp)
        self.ent_mcp_calib_energy_ignore_list = tk.Entry(frm_mcp_calibrate_options_energy,
                                                         textvariable=self.var_mcp_calib_energy_ignore_list,
                                                         width=10)
        lbl_mcp_calib_energy_ignore_list.grid(row=2, column=4, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calib_energy_ignore_list.grid(row=2, column=5, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calib_energy_firstharmonic = tk.Label(frm_mcp_calibrate_options_energy, text="First Harmonic")
        self.var_mcp_calib_energy_first_harmonic = tk.StringVar(self.win, "17")
        self.var_mcp_calib_energy_first_harmonic.trace_add("write", self.update_calib_energy_mcp)
        self.ent_mcp_calib_energy_first_harmonic = tk.Entry(frm_mcp_calibrate_options_energy,
                                                            textvariable=self.var_mcp_calib_energy_first_harmonic,
                                                            width=4, validate='all')
        lbl_mcp_calib_energy_firstharmonic.grid(row=0, column=5, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calib_energy_first_harmonic.grid(row=0, column=6, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calib_energy_order = tk.Label(frm_mcp_calibrate_options_energy, text="Harmonic Spacing")
        self.var_mcp_calib_energy_order = tk.StringVar(self.win, "2")
        self.var_mcp_calib_energy_order.trace_add("write", self.update_calib_energy_mcp)
        self.ent_mcp_calib_energy_order = tk.Entry(frm_mcp_calibrate_options_energy,
                                                   textvariable=self.var_mcp_calib_energy_order,
                                                   width=4, validate='all')
        lbl_mcp_calib_energy_order.grid(row=1, column=5, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calib_energy_order.grid(row=1, column=6, padx=2, pady=2, sticky='nsew')

        self.figure_mcp_treated = Figure(figsize=(5, 4), dpi=100)
        self.ax_mcp_treated = self.figure_mcp_treated.add_subplot(211)
        self.ax_harmonics_treated = self.figure_mcp_treated.add_subplot(212)
        self.figure_mcp_treated.tight_layout()
        self.figure_mcp_treated.canvas.draw()
        self.image_mcp_treated = FigureCanvasTkAgg(self.figure_mcp_treated, frm_mcp_treated_image)
        self.tk_widget_figure_mcp_treated = self.image_mcp_treated.get_tk_widget()
        self.tk_widget_figure_mcp_treated.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.image_mcp_treated.draw()

        self.var_show_treated_live = tk.IntVar()
        self.cb_show_treated_live = tk.Checkbutton(frm_mcp_treated_options, text='Show Treated Image Live',
                                                   variable=self.var_show_treated_live, onvalue=1,
                                                   offvalue=0,
                                                   command=None)
        self.cb_show_treated_live.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')

        self.but_show_treated_image = tk.Button(frm_mcp_treated_options, text="Show Treated Image",
                                                command=self.treat_image_test)

        self.but_export_image_treatment_parameters = tk.Button(frm_mcp_treated_options,
                                                               text="Export treatment parameters",
                                                               command=self.export_mcp_treatment_parameters)
        self.but_import_image_treatment_parameters = tk.Button(frm_mcp_treated_options,
                                                               text="Import treatment parameters",
                                                               command=self.import_mcp_treatment_parameters)
        self.but_show_treated_image.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        self.but_export_image_treatment_parameters.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')
        self.but_import_image_treatment_parameters.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')


        frm_mcp_options = ttk.LabelFrame(self.win, text='MCP options')
        self.but_fixyaxis_mcp_options = tk.Button(frm_mcp_options, text='Update Y Axis!', command=self.fix_y_axis_mcp)
        self.var_fixyaxis_mcp_options = tk.IntVar()
        self.cb_fixyaxis_mcp_options = tk.Checkbutton(frm_mcp_options, text='Fix Y axis', variable=self.var_fixyaxis_mcp_options, onvalue=1,
                                                      offvalue=0,
                                                      command=None)

        self.but_get_background_mcp_options = tk.Button(frm_mcp_options, text='Record Background', command=self.get_background_mcp)
        self.but_remove_background_mcp_options = tk.Button(frm_mcp_options, text='Reset Background',
                                                           command=self.reset_background_mcp)

        frm_mcp_options.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        self.cb_fixyaxis_mcp_options.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.but_fixyaxis_mcp_options.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        self.but_get_background_mcp_options.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.but_remove_background_mcp_options.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')

    ##
    def final_image_treatment(self, im):
        bg = int(self.var_mcp_calib_background_val.get())
        x1 = int(self.var_mcp_calib_roix_1_val.get())
        x2 = int(self.var_mcp_calib_roix_2_val.get())
        y1 = int(self.var_mcp_calib_roiy_1_val.get())
        y2 = int(self.var_mcp_calib_roiy_2_val.get())
        shear = float(self.var_mcp_calib_shear_val.get())
        correct_E_axis, treated = help.treat_image_new(im, self.eaxis, x1, x2, y1, y2, bg, shear)
        return correct_E_axis, treated

    def export_mcp_treatment_parameters(self):
        bg = int(self.var_mcp_calib_background_val.get())
        x1 = int(self.var_mcp_calib_roix_1_val.get())
        x2 = int(self.var_mcp_calib_roix_2_val.get())
        y1 = int(self.var_mcp_calib_roiy_1_val.get())
        y2 = int(self.var_mcp_calib_roiy_2_val.get())
        shear = float(self.var_mcp_calib_shear_val.get())
        energy_axis = self.eaxis_correct

        try:
            filepath = asksaveasfilename(
                defaultextension='h5',
                filetypes=[('h5 Files', '*.h5'), ('All Files', '*.*')]
            )

            filename = filepath

            with h5py.File(filename, 'w') as hf:
                hf.create_dataset('bg', data=bg)
                hf.create_dataset('x1', data=x1)
                hf.create_dataset('x2', data=x2)
                hf.create_dataset('y1', data=y1)
                hf.create_dataset('y2', data=y2)
                hf.create_dataset('shear', data=shear)
                hf.create_dataset('energy_axis', data=energy_axis)
                hf.create_dataset('energy_axis_temp', data=self.eaxis)
        except:
            message = f'Exporting data failed'

    def import_mcp_treatment_parameters(self):
        filepath = askopenfilename(
            filetypes=[('h5 Files', '*.h5'), ('All Files', '*.*')]
        )
        try:
            hfr = h5py.File(filepath, 'r')
            energy_axis = np.asarray(hfr.get('energy_axis'))
            energy_axis_temp = np.asarray(hfr.get('energy_axis_temp'))
            bg = int(np.asarray(hfr.get('bg')))
            x1 = int(np.asarray(hfr.get('x1')))
            x2 = int(np.asarray(hfr.get('x2')))
            y1 = int(np.asarray(hfr.get('y1')))
            y2 = int(np.asarray(hfr.get('y2')))
            shear = float(np.asarray(hfr.get('shear')))

            self.var_mcp_calib_background_val.set(bg)
            self.var_mcp_calib_roix_1_val.set(x1)
            self.var_mcp_calib_roix_2_val.set(x2)
            self.var_mcp_calib_roiy_1_val.set(y1)
            self.var_mcp_calib_roiy_2_val.set(y2)
            self.var_mcp_calib_shear_val.set(shear)
            self.eaxis_correct = energy_axis
            self.eaxis = energy_axis_temp

            self.but_import_image_treatment_parameters.config(fg='green')

        except Exception as e:
            message = f'Chose a proper filename'
            self.but_import_image_treatment_parameters.config(fg='red')

    def update_calib_energy_mcp(self):
        im = np.flipud(self.calibration_image_update)
        profile = np.sum(im, axis=1)
        try:
            smooth = int(self.var_mcp_calib_energy_smooth.get())
            prom = int(self.var_mcp_calib_energy_prom.get())
            order = int(self.var_mcp_calib_energy_order.get())
            firstharmonic = int(self.var_mcp_calib_energy_first_harmonic.get())

            data, peaks = help.fit_energy_calibration_peaks(profile, prom=prom, smoothing=smooth)
            condition = (peaks > int(self.var_mcp_calib_energy_ignore_1.get())) & (
                    peaks < int(self.var_mcp_calib_energy_ignore_2.get()))
            peaks = peaks[condition]
            try:
                ignore_list = [int(x) for x in self.ent_mcp_calib_energy_ignore_list.get().split(',') if
                               x.strip().isdigit()]
                if ignore_list:
                    for num in ignore_list:
                        range_value = 5
                        peaks = peaks[~((num - range_value <= peaks) & (peaks <= num + range_value))]



            except:
                a = 1
            h = 6.62607015e-34
            c = 299792458
            qe = 1.60217662e-19
            lam = 1030e-9
            Eq = h * c / lam

            first_harmonic = firstharmonic
            E = np.ones_like(peaks) * first_harmonic * Eq / qe + np.arange(0, np.size(peaks)) * order * Eq / qe
            p = np.polyfit(peaks, E, 3)
            x_axis = np.arange(0, np.shape(im)[1])
            scale_x_axis = np.polyval(p, x_axis)
            E_axis = scale_x_axis
            self.eaxis = E_axis

            self.plot_calibration_image_energy(profile, data, peaks, E_axis)
        except:
            message = f'Enter odd number for smooth! and something reasonable for peak prominence!'

    def update_calib_mcp(self):
        im = self.calibration_image
        try:
            im = im - int(self.var_mcp_calib_background_val.get())
        except:
            message = f'Enter something reasonable for the background!!'

        try:
            im = help.shear_image(im, float(self.var_mcp_calib_shear_val.get()), axis=1)
        except:
            message = f'Enter a reasonable value for the shear!'
        try:
            x_cut1 = int(self.var_mcp_calib_roix_1_val.get())
            x_cut2 = int(self.var_mcp_calib_roix_2_val.get())
            y_cut1 = int(self.var_mcp_calib_roiy_1_val.get())
            y_cut2 = int(self.var_mcp_calib_roiy_2_val.get())
        except:
            x_cut1 = 0
            x_cut2 = 512
            y_cut1 = 0
            y_cut2 = 512
        try:
            mask = np.zeros_like(im)
            mask[512 - x_cut2:512 - x_cut1, y_cut1:y_cut2] = 1
            im = im * mask
        except:
            message = f'Enter a reasonable value for the ROI!'

        self.calibration_image_update = im
        self.plot_calibration_image(self.calibration_image_update)

    def load_test_calibration_image_take_thread(self):
        self.load_calib_thread = threading.Thread(target=self.load_test_calibration_image_take)
        self.load_calib_thread.daemon = True
        self.load_calib_thread.start()

    def load_test_calibration_image_thread(self):
        self.load_calib_thread = threading.Thread(target=self.load_test_calibration_image)
        self.load_calib_thread.daemon = True
        self.load_calib_thread.start()

    def load_test_calibration_image_take(self):
        im_temp = self.take_image(int(self.mcp_controls['mcp_avgs']['var'].get()))
        self.calibration_image = im_temp
        self.plot_calibration_image(self.calibration_image)

    def load_test_calibration_image(self):
        file_path = askopenfilename(filetypes=[("TIFF files", "*.tiff;*.tif")])
        im_temp = np.asarray(Image.open(file_path))
        self.calibration_image = im_temp
        self.plot_calibration_image(self.calibration_image)

    def fix_y_axis_mcp(self):
        if self.var_fixyaxis_mcp_options.get() == 1:
            self.ymin_harmonics = self.current_harmonics_profile_min
            self.ymax_harmonics = self.current_harmonics_profile_max + 0.1 * (
                    self.current_harmonics_profile_max - self.current_harmonics_profile_min)
            self.ymin_harmonics_calibrate = self.current_harmonics_profile_min_calibrate
            self.ymax_harmonics_calibrate = self.current_harmonics_profile_max_calibrate + 0.1 * (
                    self.current_harmonics_profile_max_calibrate - self.current_harmonics_profile_min_calibrate)

    ## MCP Cam (ANDOR) communication

    def change_mcp_cam(self, event):
        selected_value = self.mcp_controls['mcp_cam_choice']['var'].get()
        if selected_value == 'Andor':
            self.cam = Andor.AndorSDK2Camera(fan_mode="full", amp_mode=None)
            self.ANDOR_cam = True
            self.name_cam = 'ANDOR_cam'
            self.background_mcp = np.zeros([512, 512])

    def get_background_mcp(self):
        try:
            im = self.take_image(int(self.mcp_controls['mcp_avgs']['var'].get()))
            self.background_mcp = im
            logging.info("MCP Background has been taken")
        except:
            logging.warning("Impossible to take the MCP background")

    def reset_background_mcp(self):
        self.background_mcp = np.zeros([512, 512])
        logging.info("MCP Background has been reset")

    ## Thorlabs Stage communication
    def init_motor_thorlabs(self, motor_attr):
        motor = getattr(self, motor_attr, None)
        if motor is not None:
            logging.info(f"{motor_attr} is already initialized.")
            return

        try:
            entry_widget = getattr(self, f"ent_{motor_attr}_nr", None)
            button_widget = getattr(self, f"but_{motor_attr}_init", None)

            logging.info(f"Attempting to connect to {motor_attr}...")

            motor_id = entry_widget.get() if entry_widget else None
            if motor_id is None:
                raise ValueError(f"Motor ID for {motor_attr} is not specified.")

            setattr(self, motor_attr, apt.Motor(int(motor_id)))

            if button_widget:
                button_widget.config(fg='green')
            logging.info(f"{motor_attr} connected successfully.")

            getattr(self, f"ent_{motor_attr}_should", None).config(state='normal')
            getattr(self, f"but_{motor_attr}_home", None).config(state='normal')
            getattr(self, f"but_{motor_attr}_read", None).config(state='normal')
            getattr(self, f"but_{motor_attr}_move", None).config(state='normal')
            getattr(self, f"but_{motor_attr}_disable", None).config(state='normal')
            getattr(self, f"ent_{motor_attr}_nr", None).config(state='readonly')
            getattr(self, f"but_{motor_attr}_init", None).config(state='disable')

        except Exception as e:
            logging.warning(f"Unexpected error connecting {motor_attr}: {e}")
            if button_widget:
                button_widget.config(fg='red')
            return

        position = self.read_motor_thorlabs(motor_attr)
        strvar = getattr(self, f"strvar_{motor_attr}_is", None)
        if strvar is not None:
            strvar.set(f"{position:.3f}" if position is not None else "#ERROR")

    def home_motor_thorlabs(self, motor_attr):
        try:
            logging.info(f"Homing {motor_attr}...")
            motor = getattr(self, motor_attr)
            motor.move_home(blocking=True)
            logging.info(f"{motor_attr} homed")
        except Exception as e:
            logging.warning(f"Unable to home {motor_attr}: {e}")

        position = self.read_motor_thorlabs(motor_attr)
        strvar = getattr(self, f"strvar_{motor_attr}_is", None)
        strvar.set(f"{position:.3f}")

    def read_motor_thorlabs(self, motor_attr):
        motor = getattr(self, motor_attr, None)
        if motor is None:
            logging.warning(f"{motor_attr} does not exist or is not initialized.")
            return None

        try:
            position = motor.position
            logging.info(f"{motor_attr} position: {position:.3f}")
        except Exception as e:
            logging.warning(f"Could not read position of {motor_attr}: {e}")
            position = None

        strvar = getattr(self, f"strvar_{motor_attr}_is", None)
        strvar.set(f"{position:.3f}" if position is not None else "#ERROR")

        amplitude = float(getattr(self, f'ent_{motor_attr}_power', 0).get())
        offset = float(getattr(self, f'ent_{motor_attr}_offset', 0).get())
        power = self.angle_to_power(position, amplitude, offset)
        strvar_current_power = getattr(self, f"strvar_{motor_attr}_current_power", None)
        strvar_current_power.set(f"{power :.3f}" if power is not None else "#ERROR")

        return position

    def move_motor_thorlabs(self, motor_attr, target_position):
        if not isinstance(target_position, (float, int)):
            logging.warning(f"Invalid target position for {motor_attr}: {target_position}. Must be a number.")
            return

        motor = getattr(self, motor_attr, None)
        if motor is None:
            logging.warning(f"{motor_attr} does not exist or is not initialized.")
            return

        power_mode = getattr(self, f'var_{motor_attr}_power', None)
        if power_mode.get() == 1:
            try:
                max_power = float(getattr(self, f'ent_{motor_attr}_power', 0).get())
                amplitude = max_power
                offset = float(getattr(self, f'ent_{motor_attr}_offset', 0).get())
                if target_position > max_power:
                    logging.warning(
                        f"Requested power {target_position:.3f} exceeds max of {max_power:.3f}. Setting to max.")
                    target_position = max_power
                target_position = self.power_to_angle(target_position, amplitude, offset) + 90
                logging.info(f"Converted target position with power mode for {motor_attr}: {target_position:.3f}")

            except (ValueError, AttributeError) as e:
                logging.warning(f"Failed to convert power to angle for {motor_attr}: {e}")
                return

        try:
            logging.info(f"Moving {motor_attr} to position: {target_position:.3f}")
            motor.move_to(target_position, blocking=True)
            logging.info(f"{motor_attr} moved to position {target_position:.3f}")
            self.read_motor_thorlabs(motor_attr)

        except Exception as e:
            logging.warning(f"Could not move {motor_attr} to position {target_position}: {e}")
            self.read_motor_thorlabs(motor_attr)

    def disable_motor_thorlabs(self, motor_attr):
        motor = getattr(self, motor_attr, None)
        if motor is None:
            logging.info(f"{motor_attr} is already disabled.")
            return

        try:
            motor.disable() if hasattr(motor, 'disable') else motor.close()
            setattr(self, motor_attr, None)
            logging.info(f"{motor_attr} successfully disabled.")

            button_widget = getattr(self, f"but_{motor_attr}_init", None)
            if button_widget:
                button_widget.config(fg='black')

            getattr(self, f"ent_{motor_attr}_should", None).config(state='disabled')
            getattr(self, f"but_{motor_attr}_home", None).config(state='disabled')
            getattr(self, f"but_{motor_attr}_read", None).config(state='disabled')
            getattr(self, f"but_{motor_attr}_move", None).config(state='disabled')
            getattr(self, f"but_{motor_attr}_disable", None).config(state='disable')
            getattr(self, f"but_{motor_attr}_init", None).config(state='normal')
            getattr(self, f"ent_{motor_attr}_nr", None).config(state='normal')

        except Exception as e:
            logging.warning(f"Error disabling {motor_attr}: {str(e)}")

    ## Calibration of the waveplates
    def enable_calibrator(self):
        self.stop_calibrator = False
        self.calibrator_thread = threading.Thread(target=self.open_calibrator)
        self.calibrator_thread.daemon = True
        self.calibrator_thread.start()

    def open_calibrator(self):
        """
        Opens the waveplate calibrator, retrieves maximum power and offset values
        for each waveplate, and updates the corresponding Tkinter StringVars.
        """
        self.calibrator = WaveplateCalib.WPCalib()
        for i in range(1, 5):
            max_power = round(getattr(self.calibrator, f"max_wp_{i}"), 2)
            offset = round(getattr(self.calibrator, f"offset_wp_{i}"), 2)
            getattr(self, f"strvar_wp_{i}_power").set(str(max_power))
            getattr(self, f"strvar_wp_{i}_offset").set(str(offset))

    def open_calibrator_on_start(self):
        """
        Opens the waveplate calibrator on startup, retrieves and updates the
        maximum power and phase offset values for each waveplate, then closes
        the calibrator. Logs information and errors during the process.
        """
        try:
            self.open_calibrator()
            self.calibrator.on_close()
            logging.info("Waveplate calibrator values successfully loaded.")

        except AttributeError as e:
            logging.error(f"Failed to access waveplate attributes: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in open_calibrator_on_start: {e}")

    def angle_to_power(self, angle, amplitude, offset):
        power = amplitude / 2 * np.cos(2 * np.pi / 90 * angle - 2 * np.pi / 90 * offset) + amplitude / 2
        return power

    def power_to_angle(self, power, amplitude, offset):
        A = amplitude / 2
        angle = -(45 * np.arccos(power / A - 1)) / np.pi + offset
        return angle

    ##

    def take_image(self, avgs):
        if self.ANDOR_cam is True:
            self.cam.set_exposure(float(self.mcp_controls['mcp_exposure_time']['var'].get()) * 1e-6)
            self.cam.setup_shutter('open')
            self.d_phase = deque()
            image = np.zeros([512, 512])
            self.cam.start_acquisition()
            for i in range(avgs):
                self.cam.wait_for_frame(timeout=20)
                frame = self.cam.read_oldest_image()
                image += frame
            image /= avgs
            self.cam.stop_acquisition()
        else:
            message = 'No cam'

        return image - self.background_mcp

    def save_im(self, image):

        nr = self.get_start_image_from_autolog()
        filename = self.saving_folder + '-' + str(int(nr)) + '.tif'

        image_16bit = image.astype(np.uint16)
        cv2.imwrite(filename, image_16bit, [cv2.IMWRITE_PXM_BINARY, 1])

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        try:
            if self.parent.vars_red[1].get() == 1:
                rl = np.round(float(self.lens_red.strvar_ben.get()), 3)
                rl_pos = np.round(float(self.lens_red.strvar_focus_position.get()), 3)
            else:
                rl = 0
                rl_pos = 0
        except:
            rl = np.nan
            rl_pos = np.nan

        try:
            if self.parent.vars_green[1].get() == 1:
                gl = np.round(float(self.lens_green.strvar_ben.get()), 3)
                gl_pos = np.round(float(self.lens_green.strvar_focus_position.get()), 3)
            else:
                gl = 0
                gl_pos = 0
        except:
            gl = np.nan
            gl_pos = np.nan

        log_entry = str(
            int(nr)) + '\t' + self.ent_wp_1_current_power.get() + '\t' + self.ent_wp_2_current_power.get() + '\t' + str(
            np.round(float(self.strvar_setp.get()), 2)) + '\t' + str(
            np.round(np.mean(np.unwrap(self.d_phase)), 2)) + '\t' + str(
            np.round(np.std(np.unwrap(self.d_phase)),
                     2)) + '\t' + self.mcp_controls['mcp_voltage']['var'].get() + '\t' + str(
            self.name_cam) + '\t' + self.mcp_controls['mcp_avgs']['var'].get() + '\t' + self.mcp_controls['mcp_exposure_time']['var'].get() + '\t' + str(
            gl) + '\t' + str(
            gl_pos) + '\t' + str(
            rl) + '\t' + str(
            rl_pos) + '\t' + timestamp + '\n'
        self.f.write(log_entry)

    def enabl_mcp_live(self):
        self.live_is_pressed = not self.live_is_pressed
        self.update_live_but_mcp()

        self.stop_mcp = False
        self.mcp_thread = threading.Thread(target=self.view_live_mcp_cam)
        self.mcp_thread.daemon = True
        self.mcp_thread.start()

    def update_live_but_mcp(self):
        if self.live_is_pressed:
            self.but_mcp_liveview.config(relief="sunken")
            self.but_mcp_liveview.config(fg='red')
        else:
            self.but_mcp_liveview.config(relief="raised")
            self.but_mcp_liveview.config(fg='green')

    def enabl_mcp_simple(self):
        self.stop_mcp = False
        self.mcp_thread = threading.Thread(target=self.measure_simple)
        self.mcp_thread.daemon = True
        self.mcp_thread.start()


    def view_live_mcp_cam(self):
        while self.live_is_pressed:
            im = self.take_image(int(self.mcp_controls['mcp_avgs']['var'].get()))
            self.plot_MCP(im)


    def measure_simple(self):
        self.but_mcp_single_image.config(fg='red')
        self.f.write("# SingleImage, " + self.mcp_controls['comment']['var'].get() + '\n')
        im = self.take_image(int(self.mcp_controls['mcp_avgs']['var'].get()))
        self.save_im(im)
        self.plot_MCP(im)
        self.but_mcp_single_image.config(fg='green')

    def treat_image_test(self):
        im = self.calibration_image
        E_new, im_new = self.final_image_treatment(im)
        self.eaxis_correct = E_new
        self.plot_treated_image(im_new)

    def plot_treated_image(self, image):
        self.ax_mcp_treated.clear()
        pcm = self.ax_mcp_treated.pcolormesh(self.eaxis_correct, np.arange(0, 512), image.T)
        cbar = self.figure_mcp_treated.colorbar(pcm, ax=self.ax_mcp_treated)
        self.ax_mcp_treated.set_xlim(20, 45)
        self.ax_mcp_treated.set_xlabel("Energy (eV)")
        self.ax_mcp_treated.set_ylabel(" y (px) ")
        self.ax_harmonics_treated.clear()
        self.ax_harmonics_treated.plot(self.eaxis_correct, np.sum(image, 1), color='k')
        for har in np.arange(15, 35):
            self.ax_harmonics_treated.axvline(har * 1.2037300291262136, color='r', alpha=0.4)
        self.ax_harmonics_treated.set_xlim(20, 45)
        self.ax_harmonics_treated.set_xlabel("Energy (eV)")
        self.figure_mcp_treated.tight_layout()
        self.image_mcp_treated.draw()
        cbar.remove()

    def plot_calibration_image(self, image):
        image = np.flipud(image)
        self.ax_mcp_calib.clear()
        self.ax_mcp_calib.imshow(image.T, cmap='turbo')

        self.ax_mcp_calib.set_xlabel("Energy equivalent")
        self.ax_mcp_calib.set_ylabel("Y (px)")
        self.ax_mcp_calib.set_xlim(0, 512)
        self.ax_mcp_calib.set_ylim(0, 512)

        self.ax_harmonics_calib.clear()
        self.ax_harmonics_calib.plot(np.arange(512), np.sum(image, 1))
        self.ax_harmonics_calib.axhline(0, color='k', alpha=0.5)
        self.ax_harmonics_calib.set_xlabel("Energy equivalent")
        self.ax_harmonics_calib.set_ylabel("Counts (arb.u.)")

        self.ax_harmonics_calib.set_xlim(0, 512)
        self.current_harmonics_profile_max_calibrate = np.max(np.sum(image, 1))
        self.current_harmonics_profile_min_calibrate = np.min(np.sum(image, 1))
        if self.var_fixyaxis_mcp_options.get() == 1:
            self.ax_harmonics_calib.set_ylim(self.ymin_harmonics_calibrate, self.ymax_harmonics_calibrate)

        self.ax_harmonics_calib.set_title("Sum: {}, Max: {}".format(int(np.sum(np.sum(image))), int(np.max(image))))
        self.figure_mcp_calib.tight_layout()
        self.image_mcp_calib.draw()

    def plot_calibration_image_energy(self, profile, data, peaks, E_axis):

        self.ax_harmonics_calib_energy_1.clear()
        self.ax_harmonics_calib_energy_1.plot(np.arange(512), profile, color='g')
        self.ax_harmonics_calib_energy_1.plot(np.arange(512), data, color='k')
        self.ax_harmonics_calib_energy_1.scatter(peaks, data[peaks], color='r')

        self.ax_harmonics_calib_energy_1.set_xlabel("Energy equivalent")
        self.ax_harmonics_calib_energy_1.set_ylabel("Counts (arb.u.)")

        self.ax_harmonics_calib_energy_1.set_xlim(0, 512)

        self.ax_harmonics_calib_energy_2.clear()
        self.ax_harmonics_calib_energy_2.plot(E_axis, profile, color='r')
        self.ax_harmonics_calib_energy_2.set_xlabel("Energy (eV)")
        self.ax_harmonics_calib_energy_2.set_ylabel("Counts (arb.u.)")
        self.figure_mcp_calib_energy.tight_layout()
        self.image_mcp_calib_energy.draw()

    def plot_MCP(self, mcpimage):

        if self.ANDOR_cam is True:
            self.ax_mcp.clear()
            pcm = self.ax_mcp.pcolormesh(np.arange(0, 512), np.arange(0, 512), mcpimage.T, cmap='turbo')
            cbar = self.figure_mcp.colorbar(pcm, ax=self.ax_mcp)

            self.ax_mcp.set_xlabel("X (px)")
            self.ax_mcp.set_ylabel("Y (px)")
            self.ax_mcp.set_xlim(0, 512)
            self.ax_mcp.set_ylim(0, 512)

            self.ax_harmonics.clear()
            self.ax_harmonics.plot(np.arange(512), np.sum(mcpimage, 1))
            self.ax_harmonics.set_xlabel("X (px)")
            self.ax_harmonics.set_ylabel("Counts (arb.u.)")

            self.ax_harmonics.set_xlim(0, 512)
            self.current_harmonics_profile_max = np.max(np.sum(mcpimage, 1))
            self.current_harmonics_profile_min = np.min(np.sum(mcpimage, 1))
            if self.var_fixyaxis_mcp_options.get() == 1:
                self.ax_harmonics.set_ylim(self.ymin_harmonics, self.ymax_harmonics)

            self.ax_harmonics.set_title("Sum: {}, Max: {}".format(int(np.sum(np.sum(mcpimage))), int(np.max(mcpimage))))
            self.figure_mcp.tight_layout()
            self.image_mcp.draw()
            cbar.remove()
        else:
            self.ax_mcp.clear()
            pcm = self.ax_mcp.pcolormesh(np.arange(0, 512), np.arange(0, 512), mcpimage.T)
            cbar = self.figure_mcp.colorbar(pcm, ax=self.ax_mcp)

            self.ax_mcp.set_xlabel("X (px)")
            self.ax_mcp.set_ylabel("Y (px)")
            self.ax_mcp.set_xlim(0, 512)
            self.ax_mcp.set_ylim(0, 512)
            self.ax_harmonics.clear()
            self.ax_harmonics.plot(np.arange(512), np.sum(mcpimage, 1))
            self.ax_harmonics.set_xlabel("X (px)")
            self.ax_harmonics.set_ylabel("Counts (arb.u.)")

            self.ax_harmonics.set_xlim(0, 512)
            self.current_harmonics_profile_max = np.max(np.sum(mcpimage, 1))
            self.current_harmonics_profile_min = np.min(np.sum(mcpimage, 1))
            if self.var_fixyaxis_mcp_options.get() == 1:
                self.ax_harmonics.set_ylim(self.ymin_harmonics, self.ymax_harmonics)

            self.ax_harmonics.set_title("Sum: {}, Max: {}".format(int(np.sum(np.sum(mcpimage))), int(np.max(mcpimage))))
            self.figure_mcp.tight_layout()
            self.image_mcp.draw()
            cbar.remove()


            if self.var_show_treated_live.get() == 1:
                Etemp, treated = self.final_image_treatment(mcpimage)
                self.plot_treated_image(treated)

    # Autolog file writting

    def get_start_image_from_autolog(self):
        self.f.seek(0)
        lines = np.loadtxt(self.autolog, comments="#", delimiter="\t", unpack=False, usecols=(0,))
        if lines.size > 0:
            try:
                start_image = lines[-1] + 1
            except:
                start_image = lines + 1
            message = "The last image had index " + str(int(start_image - 1))
        else:
            start_image = 0
        return start_image

    def on_close(self):
        self.f.close()
        self.disable_motor_thorlabs('wp_1')
        if self.cam is not None:
            self.cam.close()
        self.spec_deactivate()
        self.win.destroy()
        self.parent.feedback_win = None
