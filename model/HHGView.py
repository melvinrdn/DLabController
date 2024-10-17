import datetime
import threading
import time
import tkinter as tk
from collections import deque
from datetime import date
from tkinter import ttk
from tkinter.filedialog import asksaveasfile, askopenfilename, asksaveasfilename
from tkinter.scrolledtext import ScrolledText
import pygame
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap

import cv2
import h5py
import matplotlib
import numpy as np
import pylablib as pll
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from pylablib.devices import Andor
from simple_pid import PID
import drivers.zaber_binary.zaber_binary as zb

from diagnostic_board.diagnostics_helpers import process_image
import drivers.avaspec_driver._avs_py as avs
import drivers.jena_piezo.jena_piezo_V3 as jena
import drivers.santec_driver._slm_py as slm
import model.helpers as help
from drivers.thorlabs_apt_driver import core as apt
from drivers.vimba_driver import *
from ressources.calibration import waveplate_calibrator as cal
from ressources.slm_infos import slm_size, bit_depth

colors = [
    (1, 1, 1),  # white
    (0, 0, 0.5),  # dark blue
    (0, 0, 1),  # clearer blue
    (0, 1, 1),  # turquoise
    (0, 1, 0),  # green
    (1, 1, 0),  # yellow
    (1, 0.5, 0),  # orange
    (1, 0, 0),   # red
    (0.5, 0, 0)  # darker red
]

custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=512)



class HHGView(object):
    """
    A class for controlling the overlap between the green and the red, using spectral fringes.
    """

    def __init__(self, parent):
        """
        Initialize the object.

        Parameters
        ----------
        parent : object
            The parent object.

        Returns
        -------
        None

        """

        matplotlib.use("TkAgg")
        self.cam = None
        self.parent = parent
        self.lens_green = self.parent.phase_refs_green[1]
        self.lens_red = self.parent.phase_refs_red[1]
        self.slm_lib = slm
        self.win = tk.Toplevel()
        self.set_point = 0

        title = 'D-Lab Controller - Feedbacker'
        print('Opening feedbacker...')

        self.win.title(title)
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)
        self.rect_id = 0

        pll.par["devices/dlls/andor_sdk2"] = "drivers/andor_driver/"

        self.WPG = None
        self.WPR = None
        self.cam_stage = None
        self.delay_stage = None
        self.lens_stage = None
        self.MPC_wp = None
        self.MPC_grating = None
        self.abort = 0

        self.pid_stage = None
        self.pid_stage_initialized = False

        self.meas_has_started = False

        self.scan_is_done = True

        self.scan_is_done_threading = threading.Event()

        self.ymin_harmonics = None
        self.ymax_harmonics = None

        self.ymin_harmonics_calibrate = None
        self.ymax_harmonics_calibrate = None

        self.current_harmonics_profile_max = None
        self.current_harmonics_profile_min = None

        self.current_harmonics_profile_max_calibrate = None
        self.current_harmonics_profile_min_calibrate = None

        self.background = None

        self.calibration_image = np.zeros([512, 512])
        self.calibration_image_update = np.zeros([512, 512])
        self.calibration_image_update_energy = np.zeros([512, 512])
        self.eaxis = None
        self.eaxis_correct = None

        self.roix = [0, 512]
        self.roiy = [0, 512]

        self.live_is_pressed = False

        self.measurement_array = None
        self.measurement_array_flat = None
        self.measurement_treated_array = None
        self.measurement_treated_array_flat = None
        self.focus_image_array = None
        self.focus_image_array_flat = None
        self.measurement_counter = 0
        self.measurement_running = 0
        self.phase_array = None
        self.phase_meas_array = None
        self.phase_meas_array_flat = None
        self.phase_std_array = None
        self.phase_std_array_flat = None
        self.ratio_array = None
        self.pi_radius_array = None
        self.red_power_array = None
        self.green_power_array = None
        self.mcp_voltage = None
        self.time_stamps_array = None
        self.time_stamps_array_flat = None
        self.aquisition_time = None
        self.averages = None

        self.calibrator = None

        self.saving_folder = 'C:/data/' + str(date.today()) + '/' + str(date.today())

        self.autolog = self.saving_folder + '-' + 'auto-log.txt'
        # self.autolog = 'C:/data/' + '2024-01-25-TEMP' + '/' + str(date.today()) + '-' + 'auto-log.txt'
        self.f = open(self.autolog, "a+")

        # creating frames

        frm_mid = ttk.Frame(self.win)
        frm_bot = ttk.Frame(self.win)
        frm_scans = ttk.Frame(self.win)
        frm_mcp_all = ttk.Frame(self.win)

        # Spectrometer
        self.frm_plt = ttk.LabelFrame(self.win, text='Spectrometer')

        self.frm_notebook_param_spc = ttk.Notebook(self.win)
        frm_spc_settings = ttk.Frame(self.frm_notebook_param_spc)
        frm_spc_export = ttk.Frame(self.frm_notebook_param_spc)
        frm_spc_plt_set = ttk.Frame(self.frm_notebook_param_spc)
        frm_spc_ratio = ttk.Frame(self.frm_notebook_param_spc)
        frm_spc_pid = ttk.Frame(self.frm_notebook_param_spc)
        self.frm_notebook_param_spc.add(frm_spc_settings, text="Acquisition parameters")
        self.frm_notebook_param_spc.add(frm_spc_pid, text="PID options")
        self.frm_notebook_param_spc.add(frm_spc_plt_set, text="Plot options")
        self.frm_notebook_param_spc.add(frm_spc_ratio, text="Phase extraction options")
        self.frm_notebook_param_spc.add(frm_spc_export, text="Export options")

        frm_measure = ttk.LabelFrame(frm_scans, text='Measurement')

        self.frm_notebook_scans = ttk.Notebook(frm_scans)
        frm_wp_scans = ttk.Frame(frm_scans)
        frm_phase_scan = ttk.Frame(frm_scans)
        frm_const_intensity_scan = ttk.Frame(frm_scans)
        frm_mpc_campaign = ttk.Frame(frm_scans)
        frm_beam_shaping = ttk.Frame(frm_scans)
        self.frm_notebook_scans.add(frm_wp_scans, text="Power scan")
        self.frm_notebook_scans.add(frm_phase_scan, text="Two-color phase scan")
        # self.frm_notebook_scans.add(frm_const_intensity_scan, text="I=cst z-scan")
        self.frm_notebook_scans.add(frm_mpc_campaign, text="MPC")
        self.frm_notebook_scans.add(frm_beam_shaping, text="Beam Shaping scan")

        frm_mpc_campaign_stages = ttk.LabelFrame(frm_mpc_campaign, text='Stage control')
        frm_mpc_campaign_stages.grid(row=0, column=0)
        frm_mpc_campaign_scans = ttk.LabelFrame(frm_mpc_campaign, text='Scan')
        frm_mpc_campaign_scans.grid(row=1, column=0)
        frm_mpc_campaign_current = ttk.Frame(frm_mpc_campaign)
        frm_mpc_campaign_current.grid(row=1, column=1)

        frm_beam_shaping_scans = ttk.LabelFrame(frm_beam_shaping, text='Scan')
        frm_beam_shaping_scans.grid(row=0, column=0)



        self.frm_notebook_waveplate = ttk.Notebook(frm_scans)
        frm_stage = ttk.Frame(frm_scans)
        frm_wp_power_cal = ttk.Frame(frm_scans)
        frm_calculator = ttk.Frame(frm_scans)
        self.frm_notebook_waveplate.add(frm_stage, text="Stage control")
        self.frm_notebook_waveplate.add(frm_wp_power_cal, text="Power calibration")
        self.frm_notebook_waveplate.add(frm_calculator, text="Calculator")

        self.output_console = ScrolledText(self.win, height=10, state='disabled')
        self.output_console.grid(row=1, column=1, columnspan=4, sticky='ew')

        self.frm_notebook_mcp = ttk.Notebook(frm_mcp_all)
        frm_mcp_image = ttk.Frame(frm_mcp_all)
        frm_mcp_calibrate = ttk.Frame(frm_mcp_all)
        frm_mcp_calibrate_energy = ttk.Frame(frm_mcp_all)
        frm_mcp_treated = ttk.Frame(frm_mcp_all)
        frm_mcp_analysis = ttk.Frame(frm_mcp_all)
        self.frm_notebook_mcp.add(frm_mcp_image, text='MCP raw')
        self.frm_notebook_mcp.add(frm_mcp_calibrate, text='Calibrate Spatial')
        self.frm_notebook_mcp.add(frm_mcp_calibrate_energy, text='Calibrate Energy')
        self.frm_notebook_mcp.add(frm_mcp_treated, text='MCP treated')
        self.frm_notebook_mcp.add(frm_mcp_analysis, text='Analysis')

        frm_mcp_calibrate_options = ttk.LabelFrame(frm_mcp_calibrate, text='Calibration Options')
        frm_mcp_calibrate_image = ttk.LabelFrame(frm_mcp_calibrate, text='Calibration Image')
        frm_mcp_calibrate_options_energy = ttk.LabelFrame(frm_mcp_calibrate_energy, text='Calibration Options')
        frm_mcp_calibrate_image_energy = ttk.LabelFrame(frm_mcp_calibrate_energy, text='Calibration Image')
        frm_mcp_treated_options = ttk.LabelFrame(frm_mcp_treated, text='Options')
        frm_mcp_treated_image = ttk.LabelFrame(frm_mcp_treated, text='Images')
        frm_mcp_analysis_options = ttk.LabelFrame(frm_mcp_analysis, text='Options')
        frm_mcp_analysis_results = ttk.LabelFrame(frm_mcp_analysis, text='Results')

        # frm_mcp_image = ttk.LabelFrame(self.win, text='MCP')
        frm_mcp_options = ttk.LabelFrame(self.win, text='MCP options')

        vcmd = (self.win.register(self.parent.callback))

        # creating buttons n labels
        but_exit = tk.Button(frm_bot, text='EXIT', command=self.on_close)
        but_feedback = tk.Button(frm_bot, text='Feedback', command=self.feedback)

        lbl_spc_ind = tk.Label(frm_spc_settings, text='Spectrometer index:')
        self.strvar_spc_ind = tk.StringVar(self.win, '1')
        self.ent_spc_ind = tk.Entry(
            frm_spc_settings, width=9, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_spc_ind)
        lbl_spc_exp = tk.Label(frm_spc_settings, text='Exposure time (ms):')
        self.strvar_spc_exp = tk.StringVar(self.win, '50')
        self.ent_spc_exp = tk.Entry(
            frm_spc_settings, width=9, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_spc_exp)
        lbl_spc_gain = tk.Label(frm_spc_settings, text='Nbr. of averages:')
        self.strvar_spc_avg = tk.StringVar(self.win, '1')
        self.ent_spc_avg = tk.Entry(
            frm_spc_settings, width=9, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_spc_avg)
        but_spc_activate = tk.Button(frm_spc_settings, text='Activate',
                                     command=self.spec_activate, width=8)
        but_spc_deactivate = tk.Button(frm_spc_settings, text='Desactivate',
                                       command=self.spec_deactivate, width=8)
        but_spc_start = tk.Button(frm_spc_settings, text='Start',
                                  command=self.spc_img)
        but_spc_stop = tk.Button(frm_spc_settings, text='Stop',
                                 command=self.stop_measure)
        but_spc_phi = tk.Button(frm_spc_settings, text='Fast 2pi',
                                command=self.fast_scan)

        self.but_spc_export_fringes = tk.Button(frm_spc_export, text='Save spectral fringes',
                                                command=self.enable_save_spc_fringes, width=20)
        self.but_spc_export_phase_stab = tk.Button(frm_spc_export, text='Save phase stability',
                                                   command=self.enable_save_spc_phase_stability, width=20)

        but_auto_scale = tk.Button(frm_spc_plt_set, text='Auto-scale',
                                   command=self.auto_scale_spec_axis, width=13)
        but_bck = tk.Button(frm_spc_plt_set, text='Take background',
                            command=self.take_background, width=13)
        lbl_phi = tk.Label(frm_spc_ratio, text='Phase shift:')
        lbl_phi_2 = tk.Label(frm_spc_ratio, text='pi')
        self.strvar_flat = tk.StringVar()
        self.ent_flat = tk.Entry(
            frm_spc_ratio, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_flat)

        text = '20'
        self.strvar_indexfft = tk.StringVar(self.win, text)
        lbl_indexfft = tk.Label(frm_spc_ratio, text='Index fft:')
        lbl_angle = tk.Label(frm_spc_ratio, text='Phase:')
        self.ent_indexfft = tk.Entry(
            frm_spc_ratio, width=11,
            textvariable=self.strvar_indexfft)
        self.lbl_angle = tk.Label(frm_spc_ratio, text='angle')

        text = '1950'
        self.strvar_area1x = tk.StringVar(self.win, text)
        self.ent_area1x = tk.Entry(
            frm_spc_ratio, width=11,
            textvariable=self.strvar_area1x)

        text = '2100'
        self.strvar_area1y = tk.StringVar(self.win, text)
        self.ent_area1y = tk.Entry(
            frm_spc_ratio, width=11,
            textvariable=self.strvar_area1y)

        lbl_setp = tk.Label(frm_spc_pid, text='Setpoint:')
        self.strvar_setp = tk.StringVar(self.win, '0')
        self.ent_setp = tk.Entry(
            frm_spc_pid, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_setp)
        lbl_pidp = tk.Label(frm_spc_pid, text='P-value:')
        self.strvar_pidp = tk.StringVar(self.win, '0')
        self.ent_pidp = tk.Entry(
            frm_spc_pid, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_pidp)
        lbl_pidi = tk.Label(frm_spc_pid, text='I-value:')
        self.strvar_pidi = tk.StringVar(self.win, '0')  # -6 is nice
        self.ent_pidi = tk.Entry(
            frm_spc_pid, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_pidi)
        lbl_pidd = tk.Label(frm_spc_pid, text='D-value:')
        self.strvar_pidd = tk.StringVar(self.win, '0')
        self.ent_pidd = tk.Entry(
            frm_spc_pid, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_pidd)
        lbl_std = tk.Label(frm_spc_pid, text='std:', width=6)
        self.lbl_std_val = tk.Label(frm_spc_pid, text='None', width=6)

        but_pid_setp = tk.Button(frm_spc_pid, text='Setpoint', command=self.set_setpoint)
        but_pid_enbl = tk.Button(frm_spc_pid, text='Start PID', command=self.enbl_pid)
        but_pid_stop = tk.Button(frm_spc_pid, text='Stop PID', command=self.pid_stop)
        but_pid_setk = tk.Button(frm_spc_pid, text='Set PID values', command=self.set_pid_val)


        self.var_pid_cl = tk.IntVar()
        self.cb_pid_cl = tk.Checkbutton(frm_spc_pid, text='Closed Loop', variable=self.var_pid_cl, onvalue=1,
                                        offvalue=0,
                                        command=None)
        self.var_pid_stage_enable = tk.IntVar()
        self.cb_pid_stage_enable = tk.Checkbutton(frm_spc_pid, text='PID with Piezo',
                                                  variable=self.var_pid_stage_enable, onvalue=1,
                                                  offvalue=0,
                                                  command=None)
        self.but_pid_stage_init = tk.Button(frm_spc_pid, text='Init', command=self.init_pid_piezo)

        lbl_pid_stage_move = tk.Label(frm_spc_pid, text='Position (V):')
        but_pid_stage_move = tk.Button(frm_spc_pid, text='Move', command=self.move_pid_piezo)
        but_pid_stage_home = tk.Button(frm_spc_pid, text='Set to 0V', command=self.home_pid_piezo)
        lbl_pid_stage_com = tk.Label(frm_spc_pid, text='COM Port:')
        self.strvar_stage_pid_com = tk.StringVar(self.win, 'COM9')
        self.ent_pid_stage_com = tk.Entry(
            frm_spc_pid, width=8, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_stage_pid_com)
        self.strvar_pid_stage_actual_position = tk.StringVar(self.win, '')
        self.ent_pid_stage_actual_position = tk.Entry(
            frm_spc_pid, width=8, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_pid_stage_actual_position)
        self.strvar_pid_stage_set_position = tk.StringVar(self.win, '0.00')
        self.ent_pid_stage_set_position = tk.Entry(
            frm_spc_pid, width=8, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_pid_stage_set_position)

        lbl_from = tk.Label(frm_phase_scan, text='From:')
        self.strvar_from = tk.StringVar(self.win, '-3.14')
        self.ent_from = tk.Entry(
            frm_phase_scan, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_from)

        lbl_to = tk.Label(frm_phase_scan, text='To:')
        self.strvar_to = tk.StringVar(self.win, '3.14')
        self.ent_to = tk.Entry(
            frm_phase_scan, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_to)

        lbl_steps = tk.Label(frm_phase_scan, text='Steps:')
        self.strvar_steps = tk.StringVar(self.win, '10')
        self.ent_steps = tk.Entry(
            frm_phase_scan, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_steps)





        self.var_phasescan = tk.IntVar()
        self.cb_phasescan = tk.Checkbutton(frm_phase_scan, text='Scan', variable=self.var_phasescan, onvalue=1,
                                           offvalue=0,
                                           command=None)

        # MEASUREMENT FRAME
        self.but_meas_simple = tk.Button(frm_measure, text='Single Image', command=self.enabl_mcp_simple)
        self.but_meas_scan = tk.Button(frm_measure, text='Phase Scan', command=self.enabl_mcp)
        self.but_meas_all = tk.Button(frm_measure, text='Measurement Series', command=self.enabl_mcp_all)
        self.but_view_live = tk.Button(frm_measure, text='Live View!', command=self.enabl_mcp_live)

        self.but_hide_twocolor = tk.Button(frm_measure, text='Hide/Show Spectrometer', command=self.hide_show_spectrometer)
        self.frm_plt_visible = False


        self.var_background = tk.IntVar()
        self.cb_background = tk.Checkbutton(frm_measure, text='Background', variable=self.var_background, onvalue=1,
                                            offvalue=0,
                                            command=None)
        self.var_saveh5 = tk.IntVar()
        self.cb_saveh5 = tk.Checkbutton(frm_measure, text='Save as h5', variable=self.var_saveh5, onvalue=1,
                                        offvalue=0,
                                        command=None)

        self.var_export_treated_image = tk.IntVar()
        self.cb_export_treated_image = tk.Checkbutton(frm_measure, text='Export Treated Image',
                                                      variable=self.var_export_treated_image, onvalue=1,
                                                      offvalue=0,
                                                      command=None)

        lbl_avgs = tk.Label(frm_measure, text='Averages:')
        self.strvar_avgs = tk.StringVar(self.win, '1')
        self.ent_avgs = tk.Entry(
            frm_measure, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_avgs)

        lbl_mcp_cam_choice = tk.Label(frm_measure, text='MCP Camera selected :')
        self.strvar_mcp_cam_choice = tk.StringVar(self.win, '')
        self.cbox_mcp_cam_choice = ttk.Combobox(frm_measure, textvariable=self.strvar_mcp_cam_choice)
        # self.cbox_mcp_cam_choice['values'] = ('Pike Camera', 'Andor Camera')
        self.cbox_mcp_cam_choice['values'] = ('Andor')

        lbl_exposure_time = tk.Label(frm_measure, text='Exposure (us):')
        self.strvar_exposure_time = tk.StringVar(self.win, '100000')
        self.ent_exposure_time = tk.Entry(
            frm_measure, width=25, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_exposure_time)

        lbl_temperature = tk.Label(frm_measure, text='Temp (C):')
        lbl_temperature_status = tk.Label(frm_measure, text='Status:')

        self.strvar_temperature = tk.StringVar(self.win, 'none')
        self.strvar_temperature_status = tk.StringVar(self.win, 'none')
        lbl_actual_temperature = tk.Label(frm_measure, textvariable=self.strvar_temperature)
        lbl_actual_temperature_status = tk.Label(frm_measure, textvariable=self.strvar_temperature_status)

        lbl_mcp = tk.Label(frm_measure, text='Neg. MCP value (V):')
        self.strvar_mcp = tk.StringVar(self.win, '-1400')
        self.ent_mcp = tk.Entry(
            frm_measure, width=25, validate='none',
            textvariable=self.strvar_mcp)

        lbl_comment = tk.Label(frm_measure, text='Comment:')
        self.strvar_comment = tk.StringVar(self.win, '')
        self.ent_comment = tk.Entry(
            frm_measure, width=25, validate='none',
            textvariable=self.strvar_comment)

        lbl_Stage = tk.Label(frm_stage, text='Stage')
        lbl_Nr = tk.Label(frm_stage, text='#')
        lbl_is = tk.Label(frm_stage, text='is (deg)')
        lbl_should = tk.Label(frm_stage, text='should')

        lbl_WPR = tk.Label(frm_stage, text='WP red:')
        self.strvar_WPR_is = tk.StringVar(self.win, '')
        self.ent_WPR_is = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPR_is)
        self.strvar_WPR_should = tk.StringVar(self.win, '')
        self.ent_WPR_should = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPR_should)
        self.strvar_WPR_Nr = tk.StringVar(self.win, '83837724')
        self.ent_WPR_Nr = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPR_Nr)

        # buttons
        self.but_WPR_Ini = tk.Button(frm_stage, text='Init', command=self.init_WPR)
        self.but_WPR_Home = tk.Button(frm_stage, text='Home', command=self.home_WPR)
        self.but_WPR_Read = tk.Button(frm_stage, text='Read', command=self.read_WPR)
        self.but_WPR_Move = tk.Button(frm_stage, text='Move', command=self.move_WPR)

        self.var_wprpower = tk.IntVar()
        self.cb_wprpower = tk.Checkbutton(frm_stage, text='Power', variable=self.var_wprpower, onvalue=1, offvalue=0,
                                          command=None)

        lbl_WPG = tk.Label(frm_stage, text='WP green:')
        self.strvar_WPG_is = tk.StringVar(self.win, '')
        self.ent_WPG_is = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPG_is)
        self.strvar_WPG_should = tk.StringVar(self.win, '')
        self.ent_WPG_should = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPG_should)
        # self.strvar_WPG_Nr = tk.StringVar(self.win, '83837725')
        self.strvar_WPG_Nr = tk.StringVar(self.win, '83837714')
        self.ent_WPG_Nr = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPG_Nr)
        self.but_WPG_Ini = tk.Button(frm_stage, text='Init', command=self.init_WPG)
        self.but_WPG_Home = tk.Button(frm_stage, text='Home', command=self.home_WPG)
        self.but_WPG_Read = tk.Button(frm_stage, text='Read', command=self.read_WPG)
        self.but_WPG_Move = tk.Button(frm_stage, text='Move', command=self.move_WPG)

        self.var_wpgpower = tk.IntVar()
        self.cb_wpgpower = tk.Checkbutton(frm_stage, text='Power', variable=self.var_wpgpower, onvalue=1, offvalue=0,
                                          command=None)

        lbl_cam_stage = tk.Label(frm_stage, text='Camera in focus:')
        self.strvar_cam_stage_is = tk.StringVar(self.win, '')
        self.ent_cam_stage_is = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_cam_stage_is)
        self.strvar_cam_stage_should = tk.StringVar(self.win, '')
        self.ent_cam_stage_should = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_cam_stage_should)
        self.strvar_cam_stage_Nr = tk.StringVar(self.win, '83837725')
        self.ent_cam_stage_Nr = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_cam_stage_Nr)
        self.but_cam_stage_Ini = tk.Button(frm_stage, text='Init', command=self.init_cam_stage)
        self.but_cam_stage_Home = tk.Button(frm_stage, text='Home', command=self.home_cam_stage)
        self.but_cam_stage_Read = tk.Button(frm_stage, text='Read', command=self.read_cam_stage)
        self.but_cam_stage_Move = tk.Button(frm_stage, text='Move', command=self.move_cam_stage)

        lbl_delay_stage = tk.Label(frm_stage, text='Delay:')
        self.strvar_delay_stage_is = tk.StringVar(self.win, '')
        self.ent_delay_stage_is = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_delay_stage_is)
        self.strvar_delay_stage_should = tk.StringVar(self.win, '')
        self.ent_delay_stage_should = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_delay_stage_should)
        self.strvar_delay_stage_Nr = tk.StringVar(self.win, '83837719')
        self.ent_delay_stage_Nr = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_delay_stage_Nr)
        # scan parameters
        self.strvar_delay_stage_from = tk.StringVar(self.win, '6.40')
        self.ent_delay_stage_from = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_delay_stage_from)
        self.strvar_delay_stage_to = tk.StringVar(self.win, '6.45')
        self.ent_delay_stage_to = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_delay_stage_to)
        self.strvar_delay_stage_steps = tk.StringVar(self.win, '10')
        self.ent_delay_stage_steps = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_delay_stage_steps)
        self.var_delayscan = tk.IntVar()
        self.cb_delayscan = tk.Checkbutton(frm_stage, text='Scan', variable=self.var_delayscan, onvalue=1, offvalue=0,
                                           command=None)
        self.but_delay_stage_Ini = tk.Button(frm_stage, text='Init', command=self.init_delay_stage)
        self.but_delay_stage_Home = tk.Button(frm_stage, text='Home', command=self.home_delay_stage)
        self.but_delay_stage_Read = tk.Button(frm_stage, text='Read', command=self.read_delay_stage)
        self.but_delay_stage_Move = tk.Button(frm_stage, text='Move', command=self.move_delay_stage)

        lbl_Stage_MPC = tk.Label(frm_mpc_campaign_stages, text='Stage')
        lbl_Nr_MPC = tk.Label(frm_mpc_campaign_stages, text='#')
        lbl_is_MPC = tk.Label(frm_mpc_campaign_stages, text='is')
        lbl_should_MPC = tk.Label(frm_mpc_campaign_stages, text='should')

        lbl_mpc_to = tk.Label(frm_mpc_campaign_scans, text='to')
        lbl_mpc_from = tk.Label(frm_mpc_campaign_scans, text='from')
        lbl_mpc_steps = tk.Label(frm_mpc_campaign_scans, text='steps')

        lbl_mpc_stage_label = tk.Label(frm_mpc_campaign_scans, text='Stage')
        lbl_mpc_wp_label = tk.Label(frm_mpc_campaign_scans, text='WP')
        lbl_mpc_lens_label = tk.Label(frm_mpc_campaign_scans, text='Lens')
        lbl_mpc_grating_label = tk.Label(frm_mpc_campaign_scans, text='Grating')


        lbl_beam_shaping_to = tk.Label(frm_beam_shaping_scans, text='to')
        lbl_beam_shaping_from = tk.Label(frm_beam_shaping_scans, text='from')
        lbl_beam_shaping_steps = tk.Label(frm_beam_shaping_scans, text='steps')

        lbl_beam_shaping_stage_label = tk.Label(frm_beam_shaping_scans, text='Stage')
        lbl_beam_shaping_lens_label = tk.Label(frm_beam_shaping_scans, text='Lens')

        lbl_lens_stage = tk.Label(frm_stage, text='Lens:')
        lbl_MPC_wp = tk.Label(frm_mpc_campaign_stages, text='WP')
        lbl_zaber_grating = tk.Label(frm_mpc_campaign_stages, text='Grating')

        lbl_MPC_maxpower = tk.Label(frm_mpc_campaign_current, text='Max Power (W)')
        lbl_MPC_minpower = tk.Label(frm_mpc_campaign_current, text='Min Power (W)')

        lbl_MPC_maxangle = tk.Label(frm_mpc_campaign_current, text='Max Angle (deg)')
        lbl_MPC_currentpower = tk.Label(frm_mpc_campaign_current, text='Current Power (W)')

        self.strvar_mpc_lens_nr = tk.StringVar(self.win, '83838295')
        self.ent_mpc_lens_nr = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_lens_nr)
        self.strvar_mpc_lens_is = tk.StringVar(self.win, '')
        self.ent_mpc_lens_is = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_lens_is)
        self.strvar_mpc_lens_should = tk.StringVar(self.win, '')
        self.ent_mpc_lens_should = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_lens_should)

        self.strvar_mpc_wp_nr = tk.StringVar(self.win, '83837724')
        self.ent_mpc_wp_nr = tk.Entry(
            frm_mpc_campaign_stages, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_wp_nr)
        self.strvar_mpc_wp_is = tk.StringVar(self.win, '')
        self.ent_mpc_wp_is = tk.Entry(
            frm_mpc_campaign_stages, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_wp_is)
        self.strvar_mpc_wp_should = tk.StringVar(self.win, '')
        self.ent_mpc_wp_should = tk.Entry(
            frm_mpc_campaign_stages, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_wp_should)

        self.strvar_zaber_grating_nr = tk.StringVar(self.win, 'COM11')
        self.ent_zaber_grating_nr = tk.Entry(
            frm_mpc_campaign_stages, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_zaber_grating_nr)
        self.strvar_zaber_grating_is = tk.StringVar(self.win, '')
        self.ent_zaber_grating_is = tk.Entry(
            frm_mpc_campaign_stages, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_zaber_grating_is)
        self.strvar_zaber_grating_should = tk.StringVar(self.win, '')
        self.ent_zaber_grating_should = tk.Entry(
            frm_mpc_campaign_stages, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_zaber_grating_should)

        self.strvar_mpc_lens_from = tk.StringVar(self.win, '5')
        self.ent_mpc_lens_from = tk.Entry(
            frm_mpc_campaign_scans, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_lens_from)
        self.strvar_mpc_lens_to = tk.StringVar(self.win, '10')
        self.ent_mpc_lens_to = tk.Entry(
            frm_mpc_campaign_scans, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_lens_to)
        self.strvar_mpc_lens_steps = tk.StringVar(self.win, '5')
        self.ent_mpc_lens_steps = tk.Entry(
            frm_mpc_campaign_scans, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_lens_steps)

        self.strvar_mpc_wp_from = tk.StringVar(self.win, '1')
        self.ent_mpc_wp_from = tk.Entry(
            frm_mpc_campaign_scans, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_wp_from)
        self.strvar_mpc_wp_to = tk.StringVar(self.win, '2')
        self.ent_mpc_wp_to = tk.Entry(
            frm_mpc_campaign_scans, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_wp_to)
        self.strvar_mpc_wp_steps = tk.StringVar(self.win, '5')
        self.ent_mpc_wp_steps = tk.Entry(
            frm_mpc_campaign_scans, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_wp_steps)

        self.strvar_mpc_grating_from = tk.StringVar(self.win, '0')
        self.ent_mpc_grating_from = tk.Entry(
            frm_mpc_campaign_scans, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_grating_from)
        self.strvar_mpc_grating_to = tk.StringVar(self.win, '10')
        self.ent_mpc_grating_to = tk.Entry(
            frm_mpc_campaign_scans, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_grating_to)
        self.strvar_mpc_grating_steps = tk.StringVar(self.win, '10')
        self.ent_mpc_grating_steps = tk.Entry(
            frm_mpc_campaign_scans, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_grating_steps)

        self.strvar_beam_shaping_lens_from = tk.StringVar(self.win, '5')
        self.ent_beam_shaping_lens_from = tk.Entry(
            frm_beam_shaping_scans, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_lens_from)
        self.strvar_beam_shaping_lens_to = tk.StringVar(self.win, '10')
        self.ent_beam_shaping_lens_to = tk.Entry(
            frm_beam_shaping_scans, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_lens_to)
        self.strvar_beam_shaping_lens_steps = tk.StringVar(self.win, '5')
        self.ent_beam_shaping_lens_steps = tk.Entry(
            frm_beam_shaping_scans, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_lens_steps)

        self.strvar_mpc_maxpower = tk.StringVar(self.win, '5')
        self.ent_mpc_maxpower = tk.Entry(
            frm_mpc_campaign_current, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_maxpower)

        self.strvar_mpc_minpower = tk.StringVar(self.win, '5')
        self.ent_mpc_minpower = tk.Entry(
            frm_mpc_campaign_current, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_minpower)

        self.strvar_mpc_maxangle = tk.StringVar(self.win, '42')
        self.ent_mpc_maxangle = tk.Entry(
            frm_mpc_campaign_current, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_maxangle)
        self.strvar_mpc_currentpower = tk.StringVar(self.win, '')
        self.ent_mpc_currentpower = tk.Entry(
            frm_mpc_campaign_current, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_mpc_currentpower)

        self.but_MPC_measure = tk.Button(frm_mpc_campaign_scans, text='Measure The MPC', command=self.enabl_mpc_meas)
        self.but_MPC_abort = tk.Button(frm_mpc_campaign_scans, text='ABORT!!!', command=self.abort_mpc_measurement)
        self.but_MPC_test_scan = tk.Button(frm_mpc_campaign_scans, text='Test scan', command=self.enabl_mpc_test_scan)

        self.but_beam_shaping_measure = tk.Button(frm_beam_shaping_scans, text='Measure', command=self.enabl_beam_shaping_meas)
        self.but_beam_shaping_abort = tk.Button(frm_beam_shaping_scans, text='Abort', command=self.abort_beam_shaping_measurement)

        self.but_lens_stage_Ini = tk.Button(frm_stage, text='Init', command=self.init_lens_stage)
        self.but_lens_stage_Home = tk.Button(frm_stage, text='Home', command=self.home_lens_stage)
        self.but_lens_stage_Read = tk.Button(frm_stage, text='Read', command=self.read_lens_stage)
        self.but_lens_stage_Move = tk.Button(frm_stage, text='Move', command=self.move_lens_stage)

        self.but_MPC_wp_Ini = tk.Button(frm_mpc_campaign_stages, text='Init', command=self.init_MPC_wp)
        self.but_MPC_wp_Home = tk.Button(frm_mpc_campaign_stages, text='Home', command=self.home_MPC_wp)
        self.but_MPC_wp_Read = tk.Button(frm_mpc_campaign_stages, text='Read', command=self.read_MPC_wp)
        self.but_MPC_wp_Move = tk.Button(frm_mpc_campaign_stages, text='Move', command=self.move_MPC_wp)

        self.but_zaber_grating_Ini = tk.Button(frm_mpc_campaign_stages, text='Init', command=self.init_zaber_stage)
        self.but_zaber_grating_Home = tk.Button(frm_mpc_campaign_stages, text='Home', command=self.home_zaber_stage)
        self.but_zaber_grating_Read = tk.Button(frm_mpc_campaign_stages, text='Read', command=self.read_zaber_stage)
        self.but_zaber_grating_Move = tk.Button(frm_mpc_campaign_stages, text='Move', command=self.move_zaber_stage)

        self.var_mpc_wp_power = tk.IntVar()
        self.cb_mpc_wp_power = tk.Checkbutton(frm_mpc_campaign_stages, text='Power', variable=self.var_mpc_wp_power,
                                              onvalue=1,
                                              offvalue=0,
                                              command=None)
        self.var_mpc_scan_lens = tk.IntVar()
        self.cb_mpc_scan_lens = tk.Checkbutton(frm_mpc_campaign_scans, text='Scan Lens',
                                               variable=self.var_mpc_scan_lens,
                                               onvalue=1,
                                               offvalue=0,
                                               command=None)
        self.var_mpc_scan_wp = tk.IntVar()
        self.cb_mpc_scan_wp = tk.Checkbutton(frm_mpc_campaign_scans, text='Scan Power', variable=self.var_mpc_scan_wp,
                                             onvalue=1,
                                             offvalue=0,
                                             command=None)

        self.var_mpc_scan_grating = tk.IntVar()
        self.cb_mpc_scan_grating = tk.Checkbutton(frm_mpc_campaign_scans, text='Scan Grating',
                                                  variable=self.var_mpc_scan_grating,
                                                  onvalue=1,
                                                  offvalue=0,
                                                  command=None)

        self.var_beam_shaping_scan_lens = tk.IntVar()
        self.cb_beam_shaping_scan_lens = tk.Checkbutton(frm_beam_shaping_scans, text='Scan Lens',
                                               variable=self.var_beam_shaping_scan_lens,
                                               onvalue=1,
                                               offvalue=0,
                                               command=None)

        lbl_Stage_MPC.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')

        lbl_MPC_wp.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        lbl_zaber_grating.grid(row=3, column=0, padx=2, pady=2, sticky='nsew')
        lbl_Nr_MPC.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        lbl_is_MPC.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')
        lbl_should_MPC.grid(row=0, column=3, padx=2, pady=2, sticky='nsew')

        self.ent_mpc_lens_nr.grid(row=5, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_lens_is.grid(row=5, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_lens_should.grid(row=5, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_wp_nr.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_wp_is.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_wp_should.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_zaber_grating_nr.grid(row=3, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_zaber_grating_is.grid(row=3, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_zaber_grating_should.grid(row=3, column=3, padx=2, pady=2, sticky='nsew')

        lbl_mpc_stage_label.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        lbl_mpc_from.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        lbl_mpc_to.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')
        lbl_mpc_steps.grid(row=0, column=3, padx=2, pady=2, sticky='nsew')
        lbl_mpc_lens_label.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        lbl_mpc_wp_label.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        lbl_mpc_grating_label.grid(row=3, column=0, padx=2, pady=2, sticky='nsew')

        lbl_beam_shaping_stage_label.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        lbl_beam_shaping_from.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        lbl_beam_shaping_to.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')
        lbl_beam_shaping_steps.grid(row=0, column=3, padx=2, pady=2, sticky='nsew')
        lbl_beam_shaping_lens_label.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')

        self.ent_beam_shaping_lens_from.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_beam_shaping_lens_to.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_beam_shaping_lens_steps.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')

        self.but_beam_shaping_measure.grid(row=1, column=5, padx=2, pady=2, sticky='nsew')
        self.but_beam_shaping_abort.grid(row=2, column=5, padx=2, pady=2, sticky='nsew')

        self.cb_beam_shaping_scan_lens.grid(row=1, column=4, padx=2, pady=2, sticky='nsew')

        self.ent_mpc_lens_from.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_lens_to.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_lens_steps.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_wp_from.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_wp_to.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_wp_steps.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_grating_from.grid(row=3, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_grating_to.grid(row=3, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_grating_steps.grid(row=3, column=3, padx=2, pady=2, sticky='nsew')
        self.cb_mpc_scan_lens.grid(row=1, column=4, padx=2, pady=2, sticky='nsew')
        self.cb_mpc_scan_wp.grid(row=2, column=4, padx=2, pady=2, sticky='nsew')
        self.cb_mpc_scan_grating.grid(row=3, column=4, padx=2, pady=2, sticky='nsew')
        self.but_MPC_measure.grid(row=1, column=5, padx=2, pady=2, sticky='nsew')
        self.but_MPC_abort.grid(row=2, column=5, padx=2, pady=2, sticky='nsew')
        self.but_MPC_test_scan.grid(row=3, column=5, padx=2, pady=2, sticky='nsew')

        self.but_lens_stage_Ini.grid(row=5, column=4, padx=2, pady=2, sticky='nsew')
        self.but_lens_stage_Home.grid(row=5, column=5, padx=2, pady=2, sticky='nsew')
        self.but_lens_stage_Read.grid(row=5, column=6, padx=2, pady=2, sticky='nsew')
        self.but_lens_stage_Move.grid(row=5, column=7, padx=2, pady=2, sticky='nsew')
        self.but_MPC_wp_Ini.grid(row=2, column=4, padx=2, pady=2, sticky='nsew')
        self.but_MPC_wp_Home.grid(row=2, column=5, padx=2, pady=2, sticky='nsew')
        self.but_MPC_wp_Read.grid(row=2, column=6, padx=2, pady=2, sticky='nsew')
        self.but_MPC_wp_Move.grid(row=2, column=7, padx=2, pady=2, sticky='nsew')
        self.cb_mpc_wp_power.grid(row=2, column=8, padx=2, pady=2, sticky='nsew')

        self.but_zaber_grating_Ini.grid(row=3, column=4, padx=2, pady=2, sticky='nsew')
        self.but_zaber_grating_Home.grid(row=3, column=5, padx=2, pady=2, sticky='nsew')
        self.but_zaber_grating_Read.grid(row=3, column=6, padx=2, pady=2, sticky='nsew')
        self.but_zaber_grating_Move.grid(row=3, column=7, padx=2, pady=2, sticky='nsew')

        lbl_MPC_maxpower.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        lbl_MPC_minpower.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        lbl_MPC_maxangle.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        lbl_MPC_currentpower.grid(row=3, column=0, padx=2, pady=2, sticky='nsew')

        self.ent_mpc_maxpower.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_minpower.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_maxangle.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mpc_currentpower.grid(row=3, column=1, padx=2, pady=2, sticky='nsew')

        lbl_MPC_fitA = tk.Label(frm_mpc_campaign_current, text='Amplitude')
        lbl_MPC_fitf = tk.Label(frm_mpc_campaign_current, text='Frequency')
        lbl_MPC_fitph = tk.Label(frm_mpc_campaign_current, text='Phase offset')
        lbl_MPC_fito = tk.Label(frm_mpc_campaign_current, text='Amp offset')

        lbl_MPC_fitA.grid(row=6, column=0, padx=2, pady=2, sticky='nsew')
        lbl_MPC_fitf.grid(row=6, column=1, padx=2, pady=2, sticky='nsew')
        lbl_MPC_fitph.grid(row=6, column=2, padx=2, pady=2, sticky='nsew')
        lbl_MPC_fito.grid(row=6, column=3, padx=2, pady=2, sticky='nsew')

        self.strvar_MPC_fitA = tk.StringVar(self.win, '5')
        self.ent_MPC_fitA = tk.Entry(
            frm_mpc_campaign_current, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_MPC_fitA)

        self.strvar_MPC_fitf = tk.StringVar(self.win, '0.070')
        self.ent_MPC_fitf = tk.Entry(
            frm_mpc_campaign_current, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_MPC_fitf)

        self.strvar_MPC_fitph = tk.StringVar(self.win, '57.000')
        self.ent_MPC_fitph = tk.Entry(
            frm_mpc_campaign_current, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_MPC_fitph)

        self.strvar_MPC_fito = tk.StringVar(self.win, '1.000')
        self.ent_MPC_fito = tk.Entry(
            frm_mpc_campaign_current, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_MPC_fito)

        self.ent_MPC_fitA.grid(row=7, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_MPC_fitf.grid(row=7, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_MPC_fitph.grid(row=7, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_MPC_fito.grid(row=7, column=3, padx=2, pady=2, sticky='nsew')

        # power wp calibration
        lbl_pharos_att = tk.Label(frm_wp_power_cal, text='Pharos Att:')
        self.strvar_pharos_att = tk.StringVar(self.win, '100')
        self.ent_pharos_att = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_pharos_att)

        lbl_pharos_pp = tk.Label(frm_wp_power_cal, text='Pharos PP:')
        self.strvar_pharos_pp = tk.StringVar(self.win, '1')
        self.ent_pharos_pp = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',

            textvariable=self.strvar_pharos_pp)

        self.strvar_red_power = tk.StringVar(self.win, '')
        self.ent_red_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_red_power)

        self.but_calibrator_open = tk.Button(frm_wp_power_cal, text='Open calibrator', command=self.enable_calibrator)

        lbl_red_power = tk.Label(frm_wp_power_cal, text='Red max power (W):')

        self.strvar_red_power = tk.StringVar(self.win, '0')

        self.ent_red_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_red_power)

        lbl_red_phase = tk.Label(frm_wp_power_cal, text='Red offset phase (deg):')
        self.strvar_red_phase = tk.StringVar(self.win, '')
        self.ent_red_phase = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_red_phase)

        lbl_red_current_power = tk.Label(frm_wp_power_cal, text='Red current power (W):')
        self.strvar_red_current_power = tk.StringVar(self.win, '')
        self.ent_red_current_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_red_current_power)

        lbl_green_power = tk.Label(frm_wp_power_cal, text='Green max power (mW):')

        self.strvar_green_power = tk.StringVar(self.win, '0')

        self.ent_green_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_green_power)

        lbl_green_phase = tk.Label(frm_wp_power_cal, text='Green offset phase (deg):')
        self.strvar_green_phase = tk.StringVar(self.win, '')
        self.ent_green_phase = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_green_phase)

        lbl_green_current_power = tk.Label(frm_wp_power_cal, text='Green current power (mW):')
        self.strvar_green_current_power = tk.StringVar(self.win, '')
        self.ent_green_current_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_green_current_power)

        # calcualtor
        lbl_pulse_length = tk.Label(frm_calculator, text='Pulse length (fs):')
        self.strvar_pulse_length = tk.StringVar(self.win, '180')
        self.ent_pulse_length = tk.Entry(
            frm_calculator, width=8, validate='all',
            textvariable=self.strvar_pulse_length)

        lbl_rep_rate = tk.Label(frm_calculator, text='Repetition rate (kHz):')
        self.strvar_rep_rate = tk.StringVar(self.win, '10')
        self.ent_rep_rate = tk.Entry(
            frm_calculator, width=8, validate='all',
            textvariable=self.strvar_rep_rate)

        lbl_power = tk.Label(frm_calculator, text='Power (W):')
        self.strvar_power = tk.StringVar(self.win, '1')
        self.ent_power = tk.Entry(
            frm_calculator, width=8, validate='all',
            textvariable=self.strvar_power)

        lbl_beam_radius = tk.Label(frm_calculator, text='Beam radius (µm):')
        self.strvar_beam_radius = tk.StringVar(self.win, '30')
        self.ent_beam_radius = tk.Entry(
            frm_calculator, width=8, validate='all',
            textvariable=self.strvar_beam_radius)

        lbl_pulse_energy = tk.Label(frm_calculator, text='Pulse energy (µJ):')
        self.strvar_pulse_energy = tk.StringVar(self.win, '')
        self.ent_pulse_energy = tk.Entry(
            frm_calculator, width=8, validate='all',
            textvariable=self.strvar_pulse_energy)

        lbl_peak_intensity = tk.Label(frm_calculator, text='Peak intensity (1e14 W/cm2):')
        self.strvar_peak_intensity = tk.StringVar(self.win, '')
        self.ent_peak_intensity = tk.Entry(
            frm_calculator, width=8, validate='all',
            textvariable=self.strvar_peak_intensity)

        lbl_hhg_cutoff = tk.Label(frm_calculator, text='HHG cutoff (eV, q):')
        self.strvar_hhg_cutoff = tk.StringVar(self.win, '')
        self.ent_hhg_cutoff = tk.Entry(
            frm_calculator, width=8, validate='all',
            textvariable=self.strvar_hhg_cutoff)

        self.strvar_hhg_cutoff_q = tk.StringVar(self.win, '')
        self.ent_hhg_cutoff_q = tk.Entry(
            frm_calculator, width=8, validate='all',
            textvariable=self.strvar_hhg_cutoff_q)

        # frm_wp_scans
        lbl_wp_scan_info = tk.Label(frm_wp_scans, text="Choose your fighter!")
        self.var_scan_wp_option = tk.StringVar(self.win, "Nothing")
        self.rb_int_ratio = tk.Radiobutton(frm_wp_scans, variable=self.var_scan_wp_option, value="Red/Green Ratio",
                                           text="Red/Green Ratio")
        self.rb_wpr = tk.Radiobutton(frm_wp_scans, variable=self.var_scan_wp_option, value="Only Red", text="Only Red")
        self.rb_wpg = tk.Radiobutton(frm_wp_scans, variable=self.var_scan_wp_option, value="Only Green",
                                     text="Only Green")
        self.rb_nothing = tk.Radiobutton(frm_wp_scans, variable=self.var_scan_wp_option, value="Nothing",
                                         text="Nothing")
        self.rb_green_focus_SLM = tk.Radiobutton(frm_wp_scans, variable=self.var_scan_wp_option, value="Green Focus",
                                                 text="Green Focus")
        self.rb_red_focus_SLM = tk.Radiobutton(frm_wp_scans, variable=self.var_scan_wp_option, value="Red Focus",
                                               text="Red Focus")

        lbl_stage_scan_from = tk.Label(frm_wp_scans, text='from:')
        lbl_stage_scan_to = tk.Label(frm_wp_scans, text='to:')
        lbl_stage_scan_steps = tk.Label(frm_wp_scans, text='steps:')

        self.strvar_int_ratio_focus = tk.StringVar(self.win, '2')
        self.ent_int_ratio_focus = tk.Entry(
            frm_wp_scans, width=4, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_int_ratio_focus)
        self.strvar_int_ratio_constant = tk.StringVar(self.win, '4.4')
        self.ent_int_ratio_constant = tk.Entry(
            frm_wp_scans, width=4, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_int_ratio_constant)

        lbl_int_ratio_focus = tk.Label(frm_wp_scans, text='Focus size ratio:')
        self.lbl_int_ratio_constant = tk.Label(frm_wp_scans,
                                               text='Pr+{:.2f}*PG='.format(
                                                   (float(self.ent_int_ratio_focus.get())) ** 2))
        lbl_int_green_ratio = tk.Label(frm_wp_scans, text="Ratio of green intensity: ")

        # scan paramters RATIO
        self.strvar_ratio_from = tk.StringVar(self.win, '0')
        self.ent_ratio_from = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_ratio_from)
        x = float(self.ent_int_ratio_focus.get()) ** 2
        c = float(self.ent_int_ratio_constant.get())
        maxG = float(self.ent_green_power.get()) * 1e-3
        self.strvar_ratio_to = tk.StringVar(self.win, str(np.round(x * maxG / (c), 3)))
        self.ent_ratio_to = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_ratio_to)
        self.strvar_ratio_steps = tk.StringVar(self.win, '10')
        self.ent_ratio_steps = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_ratio_steps)

        self.strvar_int_ratio_constant.trace_add('write', self.update_maxgreenratio)
        self.strvar_int_ratio_focus.trace_add('write', self.update_maxgreenratio)

        lbl_int_red = tk.Label(frm_wp_scans, text="Red Power (W)")
        # scan parameters ONLY RED
        self.strvar_WPR_from = tk.StringVar(self.win, '0')
        self.ent_WPR_from = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPR_from)
        self.strvar_WPR_to = tk.StringVar(self.win, self.ent_red_power.get())
        self.ent_WPR_to = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPR_to)
        self.strvar_WPR_steps = tk.StringVar(self.win, '10')
        self.ent_WPR_steps = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPR_steps)
        # self.var_wprscan = tk.IntVar()
        # self.cb_wprscan = tk.Checkbutton(frm_stage, text='Scan', variable=self.var_wprscan, onvalue=1, offvalue=0,
        #                                 command=None)

        lbl_int_green = tk.Label(frm_wp_scans, text="Green Power (mW)")
        # scan parameters ONLY GREEN
        self.strvar_WPG_from = tk.StringVar(self.win, '0')
        self.ent_WPG_from = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPG_from)
        self.strvar_WPG_to = tk.StringVar(self.win, self.ent_green_power.get())
        self.ent_WPG_to = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPG_to)
        self.strvar_WPG_steps = tk.StringVar(self.win, '10')
        self.ent_WPG_steps = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPG_steps)

        lbl_GFP_green = tk.Label(frm_wp_scans, text="Green Focus Position (mm)")
        # scan parameters GREEN FOCUS POSITIOM
        self.strvar_GFP_from = tk.StringVar(self.win, '0.02')
        self.ent_GFP_from = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_GFP_from)
        self.strvar_GFP_to = tk.StringVar(self.win, '0.05')
        self.ent_GFP_to = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_GFP_to)
        self.strvar_GFP_steps = tk.StringVar(self.win, '10')
        self.ent_GFP_steps = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_GFP_steps)

        lbl_RFP_red = tk.Label(frm_wp_scans, text="Red Focus Position (mm)")
        # scan parameters RED FOCUS POSITIOM
        self.strvar_RFP_from = tk.StringVar(self.win, '-0.15')
        self.ent_RFP_from = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_RFP_from)
        self.strvar_RFP_to = tk.StringVar(self.win, '-0.05')
        self.ent_RFP_to = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_RFP_to)
        self.strvar_RFP_steps = tk.StringVar(self.win, '10')
        self.ent_RFP_steps = tk.Entry(
            frm_wp_scans, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_RFP_steps)

        # self.var_wpgscan = tk.IntVar()
        # self.cb_wpgscan = tk.Checkbutton(frm_stage, text='Scan', variable=self.var_wpgscan, onvalue=1, offvalue=0,
        # command=None)

        self.but_fixyaxis = tk.Button(frm_mcp_options, text='Update Y Axis!', command=self.fixyaxis)
        self.var_fixyaxis = tk.IntVar()
        self.cb_fixyaxis = tk.Checkbutton(frm_mcp_options, text='Fix Y axis', variable=self.var_fixyaxis, onvalue=1,
                                          offvalue=0,
                                          command=None)

        self.but_get_background = tk.Button(frm_mcp_options, text='Record Background', command=self.get_background)
        self.but_remove_background = tk.Button(frm_mcp_options, text='Remove Background',
                                               command=self.remove_background)

        #self.frm_notebook_param_spc.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        #self.frm_plt.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')

        frm_mcp_all.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')
        frm_mcp_calibrate_options.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_calibrate_image.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_calibrate_options_energy.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_calibrate_image_energy.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_treated_options.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_treated_image.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_analysis_options.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_analysis_results.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.frm_notebook_mcp.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')

        frm_mcp_options.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        frm_scans.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        frm_measure.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.frm_notebook_waveplate.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.frm_notebook_scans.grid(row=3, column=0, padx=2, pady=2, sticky='nsew')

        frm_mid.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        frm_bot.grid(row=3, column=0, padx=2, pady=2, sticky='nsew')

        # setting up buttons frm_spc

        but_spc_start.grid(row=0, column=3, padx=2, pady=2, sticky='nsew')
        but_spc_stop.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')
        but_spc_phi.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')
        lbl_spc_ind.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_spc_ind.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        but_spc_activate.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')
        but_spc_deactivate.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        lbl_spc_exp.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_spc_exp.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        lbl_spc_gain.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_spc_avg.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')

        self.but_spc_export_fringes.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.but_spc_export_phase_stab.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')

        # setting up frm_spc_set
        but_auto_scale.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        but_bck.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')

        # setting up buttons frm_bot
        #but_exit.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        #but_feedback.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')

        # setting up frm_spc_pid
        lbl_setp.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        lbl_pidp.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        lbl_pidi.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        lbl_pidd.grid(row=3, column=0, padx=2, pady=2, sticky='nsew')

        self.ent_setp.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_pidp.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_pidi.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_pidd.grid(row=3, column=1, padx=2, pady=2, sticky='nsew')

        but_pid_setp.grid(row=4, column=0, padx=2, pady=2, sticky='nsew')
        but_pid_setk.grid(row=4, column=1, padx=2, pady=2, sticky='nsew')
        but_pid_enbl.grid(row=4, column=2, padx=2, pady=2, sticky='nsew')
        but_pid_stop.grid(row=4, column=3, padx=2, pady=2, sticky='nsew')

        lbl_pid_stage_com.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')
        lbl_pid_stage_move.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')

        self.ent_pid_stage_com.grid(row=0, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_pid_stage_set_position.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')
        #self.ent_pid_stage_actual_position.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')

        self.but_pid_stage_init.grid(row=0, column=4, padx=2, pady=2, sticky='nsew')
        but_pid_stage_move.grid(row=1, column=4, padx=2, pady=2, sticky='nsew')
        but_pid_stage_home.grid(row=2, column=4, padx=2, pady=2, sticky='nsew')

        self.cb_pid_stage_enable.grid(row=0, column=5, padx=2, pady=2, sticky='nsew')
        self.cb_pid_cl.grid(row=1, column=5, padx=2, pady=2, sticky='nsew')

        lbl_std.grid(row=3, column=5, padx=2, pady=2, sticky='nsew')
        self.lbl_std_val.grid(row=3, column=6, padx=2, pady=2, sticky='nsew')


        # setting up frm_measure
        lbl_mcp.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_mcp.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')

        lbl_mcp_cam_choice.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.cbox_mcp_cam_choice.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')

        lbl_avgs.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_avgs.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')

        lbl_exposure_time.grid(row=3, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_exposure_time.grid(row=3, column=1, padx=2, pady=2, sticky='nsew')

        lbl_temperature.grid(row=3, column=3, padx=2, pady=2, sticky='nsew')
        lbl_temperature_status.grid(row=4, column=3, padx=2, pady=2, sticky='nsew')

        lbl_actual_temperature.grid(row=3, column=4, padx=2, pady=2, sticky='nsew')
        lbl_actual_temperature_status.grid(row=4, column=4, padx=2, pady=2, sticky='nsew')

        lbl_comment.grid(row=4, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_comment.grid(row=4, column=1, padx=2, pady=2, sticky='nsew')

        self.cb_background.grid(row=5, column=0, padx=2, pady=2, sticky='nsew')
        self.cb_saveh5.grid(row=5, column=1, padx=2, pady=2, sticky='nsew')
        self.cb_export_treated_image.grid(row=5, column=2, padx=2, pady=2, sticky='nsew')

        self.but_meas_all.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')
        self.but_hide_twocolor.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        self.but_meas_scan.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')
        self.but_meas_simple.grid(row=3, column=2, padx=2, pady=2, sticky='nsew')
        self.but_view_live.grid(row=4, column=2, padx=2, pady=2, sticky='nsew')
        # setting up frm_phase_scan
        lbl_from.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        lbl_to.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        lbl_steps.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_from.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_to.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_steps.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')
        self.cb_phasescan.grid(row=5, column=1, padx=2, pady=2, sticky='nsew')

        # setting up frm_stage
        lbl_Stage.grid(row=0, column=0, pady=2, padx=2,sticky='nsew')
        lbl_Nr.grid(row=0, column=1, pady=2, padx=2,sticky='nsew')
        lbl_is.grid(row=0, column=2, pady=2, padx=2,sticky='nsew')
        lbl_should.grid(row=0, column=3, pady=2, padx=2,sticky='nsew')

        lbl_WPR.grid(row=1, column=0, pady=2, padx=2,sticky='nsew')
        lbl_WPG.grid(row=2, column=0, pady=2, padx=2,sticky='nsew')
        lbl_delay_stage.grid(row=3, column=0, padx=2,pady=2, sticky='nsew')
        lbl_cam_stage.grid(row=4, column=0, padx=2,pady=2, sticky='nsew')
        lbl_lens_stage.grid(row=5, column=0, padx=2, pady=2, sticky='nsew')

        self.ent_WPR_Nr.grid(row=1, column=1, pady=2, padx=2,sticky='nsew')
        self.ent_WPG_Nr.grid(row=2, column=1, pady=2, padx=2,sticky='nsew')
        self.ent_delay_stage_Nr.grid(row=3, column=1, padx=2,pady=2, sticky='nsew')
        self.ent_cam_stage_Nr.grid(row=4, column=1,padx=2, pady=2, sticky='nsew')

        self.ent_WPR_is.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_WPG_is.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_delay_stage_is.grid(row=3, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_cam_stage_is.grid(row=4, column=2, padx=2, pady=2, sticky='nsew')

        self.ent_WPR_should.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_WPG_should.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_delay_stage_should.grid(row=3, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_cam_stage_should.grid(row=4, column=3, padx=2, pady=2, sticky='nsew')

        self.but_WPR_Ini.grid(row=1, column=4, padx=2, pady=2, sticky='nsew')
        self.but_WPR_Home.grid(row=1, column=5, padx=2, pady=2, sticky='nsew')
        self.but_WPR_Read.grid(row=1, column=6, padx=2, pady=2, sticky='nsew')
        self.but_WPR_Move.grid(row=1, column=7, padx=2, pady=2, sticky='nsew')

        self.but_WPG_Ini.grid(row=2, column=4, padx=2, pady=2, sticky='nsew')
        self.but_WPG_Home.grid(row=2, column=5, padx=2, pady=2, sticky='nsew')
        self.but_WPG_Read.grid(row=2, column=6, padx=2, pady=2, sticky='nsew')
        self.but_WPG_Move.grid(row=2, column=7, padx=2, pady=2, sticky='nsew')

        self.but_cam_stage_Ini.grid(row=4, column=4, padx=2, pady=2, sticky='nsew')
        self.but_cam_stage_Home.grid(row=4, column=5, padx=2, pady=2, sticky='nsew')
        self.but_cam_stage_Read.grid(row=4, column=6, padx=2, pady=2, sticky='nsew')
        self.but_cam_stage_Move.grid(row=4, column=7, padx=2, pady=2, sticky='nsew')

        self.but_delay_stage_Ini.grid(row=3, column=4, padx=2, pady=2, sticky='nsew')
        self.but_delay_stage_Home.grid(row=3, column=5, padx=2, pady=2, sticky='nsew')
        self.but_delay_stage_Read.grid(row=3, column=6, padx=2, pady=2, sticky='nsew')
        self.but_delay_stage_Move.grid(row=3, column=7, padx=2, pady=2, sticky='nsew')

        self.cb_wprpower.grid(row=1, column=8, padx=2, pady=2, sticky='nsew')
        self.cb_wpgpower.grid(row=2, column=8, padx=2, pady=2, sticky='nsew')

        # setting up frm_wp_power_calibration
        self.but_calibrator_open.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')

        lbl_pharos_att.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_pharos_att.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')
        lbl_pharos_pp.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_pharos_pp.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')

        lbl_red_power.grid(row=0, column=5, padx=2, pady=2, sticky='nsew')
        self.ent_red_power.grid(row=0, column=6, padx=2, pady=2, sticky='nsew')
        lbl_red_phase.grid(row=1, column=5, padx=2, pady=2, sticky='nsew')
        self.ent_red_phase.grid(row=1, column=6, padx=2, pady=2, sticky='nsew')
        lbl_red_current_power.grid(row=2, column=5, padx=2, pady=2, sticky='nsew')
        self.ent_red_current_power.grid(row=2, column=6, padx=2, pady=2, sticky='nsew')

        lbl_green_power.grid(row=0, column=7, padx=2, pady=2, sticky='nsew')
        self.ent_green_power.grid(row=0, column=8, padx=2, pady=2, sticky='nsew')
        lbl_green_phase.grid(row=1, column=7, padx=2, pady=2, sticky='nsew')
        self.ent_green_phase.grid(row=1, column=8, padx=2, pady=2, sticky='nsew')
        lbl_green_current_power.grid(row=2, column=7, padx=2, pady=2, sticky='nsew')
        self.ent_green_current_power.grid(row=2, column=8, padx=2, pady=2, sticky='nsew')

        # setting up frm_calculator

        lbl_pulse_length.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_pulse_length.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        lbl_rep_rate.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_rep_rate.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        lbl_power.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_power.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')
        lbl_beam_radius.grid(row=3, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_beam_radius.grid(row=3, column=1, padx=2, pady=2, sticky='nsew')

        self.but_calculate = tk.Button(frm_calculator, text='Calculate',
                                       command=self.calculate_energy_and_peak_intensity)
        self.but_calculate.grid(row=4, column=0, padx=2, pady=2, sticky='nsew')

        lbl_pulse_energy.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_pulse_energy.grid(row=0, column=3, padx=2, pady=2, sticky='nsew')
        lbl_peak_intensity.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_peak_intensity.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')
        lbl_hhg_cutoff.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_hhg_cutoff.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_hhg_cutoff_q.grid(row=2, column=4, padx=2, pady=2, sticky='nsew')

        # setting up frm_wp_scans
        lbl_wp_scan_info.grid(row=0, column=0, padx=2, pady=2)
        self.rb_int_ratio.grid(row=1, column=0)
        self.rb_wpr.grid(row=2, column=0)
        self.rb_wpg.grid(row=3, column=0)
        self.rb_green_focus_SLM.grid(row=4, column=0)
        self.rb_red_focus_SLM.grid(row=5, column=0)
        self.rb_nothing.grid(row=6, column=0)

        lbl_int_ratio_focus.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.lbl_int_ratio_constant.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')

        lbl_stage_scan_from.grid(row=0, column=8, padx=2, pady=2, sticky='nsew')
        lbl_stage_scan_to.grid(row=0, column=9, padx=2, pady=2, sticky='nsew')
        lbl_stage_scan_steps.grid(row=0, column=10, padx=2, pady=2, sticky='nsew')

        self.ent_int_ratio_focus.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_int_ratio_constant.grid(row=1, column=4, padx=2, pady=2, sticky='nsew')

        lbl_int_green_ratio.grid(row=1, column=7, padx=2, pady=2, sticky='nsew')

        self.ent_ratio_from.grid(row=1, column=8, padx=2, pady=2, sticky='nsew')
        self.ent_ratio_to.grid(row=1, column=9, padx=2, pady=2, sticky='nsew')
        self.ent_ratio_steps.grid(row=1, column=10, padx=2, pady=2, sticky='nsew')

        lbl_int_red.grid(row=2, column=7, padx=2, pady=2, sticky='nsew')
        self.ent_WPR_from.grid(row=2, column=8, padx=2, pady=2, sticky='nsew')
        self.ent_WPR_to.grid(row=2, column=9, padx=2, pady=2, sticky='nsew')
        self.ent_WPR_steps.grid(row=2, column=10, padx=2, pady=2, sticky='nsew')

        lbl_int_green.grid(row=3, column=7, padx=2, pady=2, sticky='nsew')
        self.ent_WPG_from.grid(row=3, column=8, padx=2, pady=2, sticky='nsew')
        self.ent_WPG_to.grid(row=3, column=9, padx=2, pady=2, sticky='nsew')
        self.ent_WPG_steps.grid(row=3, column=10, padx=2, pady=2, sticky='nsew')

        lbl_GFP_green.grid(row=4, column=7, padx=2, pady=2, sticky='nsew')
        self.ent_GFP_from.grid(row=4, column=8, padx=2, pady=2, sticky='nsew')
        self.ent_GFP_to.grid(row=4, column=9, padx=2, pady=2, sticky='nsew')
        self.ent_GFP_steps.grid(row=4, column=10, padx=2, pady=2, sticky='nsew')

        lbl_RFP_red.grid(row=5, column=7, padx=2, pady=2, sticky='nsew')
        self.ent_RFP_from.grid(row=5, column=8, padx=2, pady=2, sticky='nsew')
        self.ent_RFP_to.grid(row=5, column=9, padx=2, pady=2, sticky='nsew')
        self.ent_RFP_steps.grid(row=5, column=10, padx=2, pady=2, sticky='nsew')

        self.figrMCP = Figure(figsize=(5, 6), dpi=100)
        self.axMCP = self.figrMCP.add_subplot(211)
        self.axHarmonics = self.figrMCP.add_subplot(212)
        self.axMCP.set_xlim(0, 1600)
        self.axMCP.set_ylim(0, 1000)
        self.axHarmonics.set_xlim(0, 1600)
        self.figrMCP.tight_layout()
        self.figrMCP.canvas.draw()
        self.imgMCP = FigureCanvasTkAgg(self.figrMCP, frm_mcp_image)
        self.tk_widget_figrMCP = self.imgMCP.get_tk_widget()
        self.tk_widget_figrMCP.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.imgMCP.draw()

        self.figrMCP_calibrate = Figure(figsize=(5, 4), dpi=100)
        self.axMCP_calibrate = self.figrMCP_calibrate.add_subplot(211)
        self.axHarmonics_calibrate = self.figrMCP_calibrate.add_subplot(212)
        self.axMCP_calibrate.set_xlim(0, 515)
        self.axMCP_calibrate.set_ylim(0, 515)
        self.axHarmonics_calibrate.set_xlim(0, 515)
        self.figrMCP_calibrate.tight_layout()
        self.figrMCP_calibrate.canvas.draw()
        self.imgMCP_calibrate = FigureCanvasTkAgg(self.figrMCP_calibrate, frm_mcp_calibrate_image)
        self.tk_widget_figrMCP_calibrate = self.imgMCP_calibrate.get_tk_widget()
        self.tk_widget_figrMCP_calibrate.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.imgMCP_calibrate.draw()

        self.but_mcp_calibration = tk.Button(frm_mcp_calibrate_options, text="Load Test Image",
                                             command=self.load_test_calibration_image_thread)
        self.but_mcp_calibration.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.but_mcp_calibration_take = tk.Button(frm_mcp_calibrate_options, text="Take Image",
                                                  command=self.load_test_calibration_image_take_thread)
        self.but_mcp_calibration_take.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calibration_shear_val = tk.Label(frm_mcp_calibrate_options, text="Shear:")
        self.var_mcp_calibration_shear_val = tk.StringVar(self.win, "0")
        self.var_mcp_calibration_shear_val.trace_add("write", self.update_calibration)
        self.ent_mcp_calibration_shear_val = tk.Entry(frm_mcp_calibrate_options,
                                                      textvariable=self.var_mcp_calibration_shear_val,
                                                      width=4, validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        lbl_mcp_calibration_shear_val.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calibration_shear_val.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calibration_ROIX_val = tk.Label(frm_mcp_calibrate_options, text="ROIX:")
        self.var_mcp_calibration_ROIX1_val = tk.StringVar(self.win, "0")
        self.var_mcp_calibration_ROIX1_val.trace_add("write", self.update_calibration)
        self.var_mcp_calibration_ROIX2_val = tk.StringVar(self.win, "512")
        self.var_mcp_calibration_ROIX2_val.trace_add("write", self.update_calibration)
        self.ent_mcp_calibration_ROIX1_val = tk.Entry(frm_mcp_calibrate_options,
                                                      textvariable=self.var_mcp_calibration_ROIX1_val,
                                                      width=4, validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_mcp_calibration_ROIX2_val = tk.Entry(frm_mcp_calibrate_options,
                                                      textvariable=self.var_mcp_calibration_ROIX2_val,
                                                      width=4, validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        lbl_mcp_calibration_ROIX_val.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calibration_ROIX1_val.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calibration_ROIX2_val.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calibration_ROIY_val = tk.Label(frm_mcp_calibrate_options, text="ROIY:")
        self.var_mcp_calibration_ROIY1_val = tk.StringVar(self.win, "0")
        self.var_mcp_calibration_ROIY1_val.trace_add("write", self.update_calibration)
        self.var_mcp_calibration_ROIY2_val = tk.StringVar(self.win, "512")
        self.var_mcp_calibration_ROIY2_val.trace_add("write", self.update_calibration)
        self.ent_mcp_calibration_ROIY1_val = tk.Entry(frm_mcp_calibrate_options,
                                                      textvariable=self.var_mcp_calibration_ROIY1_val,
                                                      width=4, validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_mcp_calibration_ROIY2_val = tk.Entry(frm_mcp_calibrate_options,
                                                      textvariable=self.var_mcp_calibration_ROIY2_val,
                                                      width=4, validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        lbl_mcp_calibration_ROIY_val.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calibration_ROIY1_val.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calibration_ROIY2_val.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calibration_background_val = tk.Label(frm_mcp_calibrate_options, text="Background:")
        self.var_mcp_calibration_background_val = tk.StringVar(self.win, "0")
        self.var_mcp_calibration_background_val.trace_add("write", self.update_calibration)
        self.ent_mcp_calibration_background_val = tk.Entry(frm_mcp_calibrate_options,
                                                           textvariable=self.var_mcp_calibration_background_val,
                                                           width=6, validate='all',
                                                           validatecommand=(vcmd, '%d', '%P', '%S'))
        lbl_mcp_calibration_background_val.grid(row=0, column=4, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calibration_background_val.grid(row=0, column=5, padx=2, pady=2, sticky='nsew')

        self.figrMCP_calibrate_energy = Figure(figsize=(5, 4), dpi=100)
        self.axHarmonics_calibrate_energy1 = self.figrMCP_calibrate_energy.add_subplot(211)
        self.axHarmonics_calibrate_energy2 = self.figrMCP_calibrate_energy.add_subplot(212)
        self.axHarmonics_calibrate_energy1.set_xlim(0, 515)
        self.axHarmonics_calibrate_energy2.set_xlim(0, 515)
        self.figrMCP_calibrate_energy.tight_layout()
        self.figrMCP_calibrate_energy.canvas.draw()
        self.imgMCP_calibrate_energy = FigureCanvasTkAgg(self.figrMCP_calibrate_energy, frm_mcp_calibrate_image_energy)
        self.tk_widget_figrMCP_calibrate_energy = self.imgMCP_calibrate_energy.get_tk_widget()
        self.tk_widget_figrMCP_calibrate_energy.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.imgMCP_calibrate_energy.draw()

        lbl_mcp_calibration_energy_smooth = tk.Label(frm_mcp_calibrate_options_energy, text="Smooth:")
        self.var_mcp_calibration_energy_smooth = tk.StringVar(self.win, "5")
        self.var_mcp_calibration_energy_smooth.trace_add("write", self.update_calibration_energy)
        self.ent_mcp_calibration_energy_smooth = tk.Entry(frm_mcp_calibrate_options_energy,
                                                          textvariable=self.var_mcp_calibration_energy_smooth,
                                                          width=4, validate='all',
                                                          validatecommand=(vcmd, '%d', '%P', '%S'))
        lbl_mcp_calibration_energy_smooth.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calibration_energy_smooth.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calibration_energy_prom = tk.Label(frm_mcp_calibrate_options_energy, text="Peak prominence:")
        self.var_mcp_calibration_energy_prom = tk.StringVar(self.win, "20")
        self.var_mcp_calibration_energy_prom.trace_add("write", self.update_calibration_energy)
        self.ent_mcp_calibration_energy_prom = tk.Entry(frm_mcp_calibrate_options_energy,
                                                        textvariable=self.var_mcp_calibration_energy_prom,
                                                        width=6, validate='all',
                                                        validatecommand=(vcmd, '%d', '%P', '%S'))
        lbl_mcp_calibration_energy_prom.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calibration_energy_prom.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calibration_energy_ignore = tk.Label(frm_mcp_calibrate_options_energy, text="Ignore peaks from/to:")
        self.var_mcp_calibration_energy_ignore1 = tk.StringVar(self.win, "0")
        self.var_mcp_calibration_energy_ignore1.trace_add("write", self.update_calibration_energy)
        self.var_mcp_calibration_energy_ignore2 = tk.StringVar(self.win, "512")
        self.var_mcp_calibration_energy_ignore2.trace_add("write", self.update_calibration_energy)
        self.ent_mcp_calibration_energy_ignore1 = tk.Entry(frm_mcp_calibrate_options_energy,
                                                           textvariable=self.var_mcp_calibration_energy_ignore1,
                                                           width=4, validate='all',
                                                           validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_mcp_calibration_energy_ignore2 = tk.Entry(frm_mcp_calibrate_options_energy,
                                                           textvariable=self.var_mcp_calibration_energy_ignore2,
                                                           width=4, validate='all',
                                                           validatecommand=(vcmd, '%d', '%P', '%S'))
        lbl_mcp_calibration_energy_ignore.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calibration_energy_ignore1.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calibration_energy_ignore2.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calibration_energy_ignore_list = tk.Label(frm_mcp_calibrate_options_energy, text="Ignore peaks around:")
        self.var_mcp_calibration_energy_ignore_list = tk.StringVar(self.win, "")
        self.var_mcp_calibration_energy_ignore_list.trace_add("write", self.update_calibration_energy)
        self.ent_mcp_calibration_energy_ignore_list = tk.Entry(frm_mcp_calibrate_options_energy,
                                                               textvariable=self.var_mcp_calibration_energy_ignore_list,
                                                               width=10)
        lbl_mcp_calibration_energy_ignore_list.grid(row=2, column=4, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calibration_energy_ignore_list.grid(row=2, column=5, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calibration_energy_firstharmonic = tk.Label(frm_mcp_calibrate_options_energy, text="First Harmonic")
        self.var_mcp_calibration_energy_firstharmonic = tk.StringVar(self.win, "17")
        self.var_mcp_calibration_energy_firstharmonic.trace_add("write", self.update_calibration_energy)
        self.ent_mcp_calibration_energy_firstharmonic = tk.Entry(frm_mcp_calibrate_options_energy,
                                                                 textvariable=self.var_mcp_calibration_energy_firstharmonic,
                                                                 width=4, validate='all',
                                                                 validatecommand=(vcmd, '%d', '%P', '%S'))
        lbl_mcp_calibration_energy_firstharmonic.grid(row=0, column=5, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calibration_energy_firstharmonic.grid(row=0, column=6, padx=2, pady=2, sticky='nsew')

        lbl_mcp_calibration_energy_order = tk.Label(frm_mcp_calibrate_options_energy, text="Harmonic Spacing")
        self.var_mcp_calibration_energy_order = tk.StringVar(self.win, "2")
        self.var_mcp_calibration_energy_order.trace_add("write", self.update_calibration_energy)
        self.ent_mcp_calibration_energy_order = tk.Entry(frm_mcp_calibrate_options_energy,
                                                         textvariable=self.var_mcp_calibration_energy_order,
                                                         width=4, validate='all',
                                                         validatecommand=(vcmd, '%d', '%P', '%S'))
        lbl_mcp_calibration_energy_order.grid(row=1, column=5, padx=2, pady=2, sticky='nsew')
        self.ent_mcp_calibration_energy_order.grid(row=1, column=6, padx=2, pady=2, sticky='nsew')

        self.figrMCP_treated = Figure(figsize=(5, 4), dpi=100)
        self.axMCP_treated = self.figrMCP_treated.add_subplot(211)
        self.axHarmonics_treated = self.figrMCP_treated.add_subplot(212)
        # self.axMCP_treated.set_xlim(0, 515)
        # self.axMCP_treated.set_ylim(0, 515)
        # self.axHarmonics_treated.set_xlim(0, 515)
        self.figrMCP_treated.tight_layout()
        self.figrMCP_treated.canvas.draw()
        self.imgMCP_treated = FigureCanvasTkAgg(self.figrMCP_treated, frm_mcp_treated_image)
        self.tk_widget_figrMCP_treated = self.imgMCP_treated.get_tk_widget()
        self.tk_widget_figrMCP_treated.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.imgMCP_treated.draw()

        self.var_show_treated = tk.IntVar()
        self.cb_show_treated = tk.Checkbutton(frm_mcp_treated_options, text='Show Treated Image Live',
                                              variable=self.var_show_treated, onvalue=1,
                                              offvalue=0,
                                              command=None)
        self.cb_show_treated.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')

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

        # analysis frame options
        self.open_h5_file_analysis = tk.Button(frm_mcp_analysis_options, text='Open h5 file',
                                               command=self.open_h5_file_analysis)
        self.lbl_mcp_analysis_info = tk.Label(frm_mcp_analysis_options, text=" ")
        self.var_log_scale = tk.IntVar()
        self.cb_log_scale = tk.Checkbutton(frm_mcp_analysis_options, text='Log Scale',
                                           variable=self.var_log_scale, onvalue=1,
                                           offvalue=0)
        self.var_log_scale.trace_add("write", self.update_mcp_analysis)

        lbl_mcp_analysis_harmonic_order = tk.Label(frm_mcp_analysis_options, text="Look at Harmonic Order: ")
        lbl_mcp_analysis_energy_lim = tk.Label(frm_mcp_analysis_options, text="Energy Axis (eV): ")
        self.var_mcp_analysis_harmonic_order = tk.StringVar(self.win, "21")
        self.var_mcp_analysis_harmonic_order.trace_add("write", self.update_mcp_analysis)
        self.ent_mcp_analysis_harmonic_order = tk.Entry(frm_mcp_analysis_options,
                                                        textvariable=self.var_mcp_analysis_harmonic_order,
                                                        width=4, validate='all',
                                                        validatecommand=(vcmd, '%d', '%P', '%S'))

        self.var_mcp_analysis_emin = tk.StringVar(self.win, "20")
        self.var_mcp_analysis_emin.trace_add("write", self.update_mcp_analysis)
        self.var_mcp_analysis_emax = tk.StringVar(self.win, "40")
        self.var_mcp_analysis_emax.trace_add("write", self.update_mcp_analysis)

        self.ent_mcp_analysis_emax = tk.Entry(frm_mcp_analysis_options,
                                              textvariable=self.var_mcp_analysis_emax,
                                              width=4, validate='all',
                                              validatecommand=(vcmd, '%d', '%P', '%S'))

        self.ent_mcp_analysis_emin = tk.Entry(frm_mcp_analysis_options,
                                              textvariable=self.var_mcp_analysis_emin,
                                              width=4, validate='all',
                                              validatecommand=(vcmd, '%d', '%P', '%S'))

        self.open_h5_file_analysis.grid(row=0, column=0)
        lbl_mcp_analysis_harmonic_order.grid(row=0, column=1)
        self.lbl_mcp_analysis_info.grid(row=1, column=0)
        self.ent_mcp_analysis_harmonic_order.grid(row=0, column=2)
        self.cb_log_scale.grid(row=1, column=2)
        lbl_mcp_analysis_energy_lim.grid(row=2, column=0)
        self.ent_mcp_analysis_emin.grid(row=2, column=1)
        self.ent_mcp_analysis_emax.grid(row=2, column=2)

        # analysis frame
        self.figrAnalysis = Figure(figsize=(5, 4), dpi=100)
        self.axAnalysis_1 = self.figrAnalysis.add_subplot(221)
        self.axAnalysis_2 = self.figrAnalysis.add_subplot(222)
        self.axAnalysis_3 = self.figrAnalysis.add_subplot(223)
        self.axAnalysis_4 = self.figrAnalysis.add_subplot(224)
        self.figrAnalysis.tight_layout()
        self.figrAnalysis.canvas.draw()
        self.canvas_results = FigureCanvasTkAgg(self.figrAnalysis, frm_mcp_analysis_results)
        self.canvas_widget = self.canvas_results.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Draw canvas
        self.canvas_results.draw()

        sizefactor = 1

        self.figr = Figure(figsize=(5 * sizefactor, 3 * sizefactor), dpi=100)
        self.ax1r = self.figr.add_subplot(211)
        self.ax2r = self.figr.add_subplot(212)
        self.trace_line, = self.ax1r.plot([])
        self.fourier_line, = self.ax2r.plot([])
        self.fourier_indicator = self.ax2r.plot([], 'v')[0]
        self.fourier_text = self.ax2r.text(0.4, 0.5, "")
        self.ax1r.set_xlim(510, 520)
        self.ax1r.set_ylim(0, 3000)
        self.ax1r.grid()
        self.ax2r.set_xlim(0, 50)
        self.ax2r.set_ylim(0, .6)
        self.figr.tight_layout()
        self.figr.canvas.draw()
        self.img1r = FigureCanvasTkAgg(self.figr, self.frm_plt)
        self.tk_widget_figr = self.img1r.get_tk_widget()
        self.tk_widget_figr.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.img1r.draw()
        self.ax1r_blit = self.figr.canvas.copy_from_bbox(self.ax1r.bbox)
        self.ax2r_blit = self.figr.canvas.copy_from_bbox(self.ax2r.bbox)

        self.figp = Figure(figsize=(5 * sizefactor, 2 * sizefactor), dpi=100)
        self.ax1p = self.figp.add_subplot(111)
        self.phase_line, = self.ax1p.plot([], '.', ms=1)
        self.ax1p.set_xlim(0, 1000)
        self.ax1p.set_ylim([-np.pi, np.pi])
        self.ax1p.set_ylabel('Phase $\phi$')
        self.ax1p.grid()
        self.figp.tight_layout()
        self.figp.canvas.draw()
        self.img1p = FigureCanvasTkAgg(self.figp, self.frm_plt)
        self.tk_widget_figp = self.img1p.get_tk_widget()
        self.tk_widget_figp.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.img1p.draw()
        self.ax1p_blit = self.figp.canvas.copy_from_bbox(self.ax1p.bbox)

        self.figV = Figure(figsize=(5, 2), dpi=100)
        self.ax1V = self.figV.add_subplot(111)
        self.V_line, = self.ax1V.plot([], '.', ms=1)
        self.ax1V.set_xlim(0, 1000)
        self.ax1V.set_ylabel('Voltage (V)')
        self.ax1V.grid()
        self.figV.tight_layout()
        self.figV.canvas.draw()
        self.img1V = FigureCanvasTkAgg(self.figV, self.frm_plt)
        self.tk_widget_figV = self.img1V.get_tk_widget()
        self.tk_widget_figV.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        self.img1V.draw()
        self.ax1V_blit = self.figV.canvas.copy_from_bbox(self.ax1V.bbox)

        # setting up frm_mcp_options
        self.cb_fixyaxis.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.but_fixyaxis.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        self.but_get_background.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.but_remove_background.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')

        # setting up frm_spc_ratio
        self.ent_area1x.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_area1y.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')

        lbl_indexfft.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_indexfft.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        lbl_angle.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        self.lbl_angle.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')

        lbl_phi.grid(row=3, column=0, padx=2, pady=2, sticky='nsew')

        self.ent_flat.grid(row=3, column=1, padx=2, pady=2, sticky='nsew')
        lbl_phi_2.grid(row=3, column=2, padx=2, pady=2, sticky='nsew')

        self.im_phase = np.zeros(1000)
        self.im_voltage = np.zeros(1000)
        self.pid = PID(0, 0, 0, setpoint=0)

        self.stop_acquire = 0
        self.stop_pid = False

        self.spec_interface_initialized = False
        self.active_spec_handle = None

        self.PIKE_cam = False
        self.ANDOR_cam = False

        if self.ANDOR_cam is True:
            self.name_cam = 'ANDOR_cam'

        self.cbox_mcp_cam_choice.bind("<<ComboboxSelected>>", self.change_mcp_cam)

        self.open_calibrator_on_start()

    def hide_show_spectrometer(self):
        if self.frm_plt_visible:
            self.frm_plt.grid_remove()
            self.frm_notebook_param_spc.grid_remove()
        else:
            self.frm_plt.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
            self.frm_notebook_param_spc.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.frm_plt_visible = not self.frm_plt_visible

    def insert_message(self, message):
        self.output_console.configure(state='normal')
        self.output_console.insert(tk.END, message + "\n")
        self.output_console.configure(state='disabled')
        self.output_console.see(tk.END)

    def open_h5_file_analysis(self):
        filepath = tk.filedialog.askopenfilename()
        message = f'Opening {filepath}'
        self.insert_message(message)

        hfr = h5py.File(filepath, 'r')
        self.current_scan_type = np.asarray(hfr.get('scan_type'))
        self.current_treated_images = np.asarray(hfr.get('treated_images'))
        self.current_lens_position_array = np.asarray(hfr.get('lens_position'))
        self.current_grating_array = np.asarray(hfr.get('grating_position'))
        self.current_power_array = np.asarray(hfr.get('power'))
        self.current_E = np.asarray(hfr.get('e_axis'))
        self.lbl_mcp_analysis_info.config(
            text="ScanType: {}, Data analyzed: {} \n Filepath: {}".format(self.current_scan_type,
                                                                          self.current_E is not None, filepath))
        self.plot_analysis(self.current_scan_type, self.current_treated_images,
                           parameter1=self.current_lens_position_array,
                           parameter2=self.current_power_array,
                           parameter3=self.current_grating_array,
                           energy_axis=self.current_E)

    def open_h5_file(self):
        filepath = tk.filedialog.askopenfilename()
        message = f'Opening {filepath}'
        self.insert_message(message)
        hfr = h5py.File(filepath, 'r')
        self.images = np.asarray(hfr.get('images'))
        self.positions = np.asarray(hfr.get('positions'))
        self.green_lens = np.asarray(hfr.get('green_lens'))
        self.red_lens = np.asarray(hfr.get('red_lens'))
        message = 'Success'
        self.insert_message(message)
        processed_images, som_x, som_y = self.process_images_dict()
        zero_position = 8
        zmin = (self.positions[0] - zero_position) * 1e-3
        zmax = (self.positions[-1] - zero_position) * 1e-3
        self.z_range = np.linspace(zmin, zmax, len(self.positions))
        self.params_m2_x, self.params_m2_y = self.get_M_sq(som_x, som_y, self.z_range, 1030e-9, 3.45e-6)

    def set_target_intensity(self):

        def beam_quality_factor_fit(z, w0, M2, z0):
            return w0 * np.sqrt(1 + (z - z0) ** 2 * (M2 * 1030e-9 / (np.pi * w0 ** 2)) ** 2)

        z_fit = np.linspace(self.z_range[0], self.z_range[-1], int(self.ent_p_steps.get()))

        C = 2 * 0.94 / (
                (float(self.ent_pulse_length_m2.get()) * 1e-15 * np.pi) * (float(self.ent_rep_rate_m2.get()) * 1e3))
        self.power_x_list = float(self.ent_target_intensity.get()) * 1e18 / C * beam_quality_factor_fit(z_fit,
                                                                                                        self.params_m2_x[
                                                                                                            0],
                                                                                                        self.params_m2_x[
                                                                                                            1],
                                                                                                        self.params_m2_x[
                                                                                                            2]) ** 2  # W/m2
        self.power_y_list = float(self.ent_target_intensity.get()) * 1e18 / C * beam_quality_factor_fit(z_fit,
                                                                                                        self.params_m2_y[
                                                                                                            0],
                                                                                                        self.params_m2_y[
                                                                                                            1],
                                                                                                        self.params_m2_y[
                                                                                                            2]) ** 2

        # I_test_x = C * Px / beam_quality_factor_fit(z_fit, self.params_m2_x[0], self.params_m2_x[1], self.params_m2_x[2]) ** 2
        # I_test_y = C * Py / beam_quality_factor_fit(z_fit, self.params_m2_y[0], self.params_m2_y[1], self.params_m2_y[2]) ** 2

        self.ax2r_m2.clear()
        self.ax2r_m2.grid(True)

        self.ax2r_m2.plot(z_fit, self.power_x_list, linestyle='None', marker='o', color='blue')
        self.ax2r_m2.plot(z_fit, self.power_y_list, linestyle='None', marker='x', color='red')

        self.ax2r_m2.set_ylabel('P (W)')
        self.ax2r_m2.set_xlabel('x [mm]')
        self.ax2r_m2.legend()
        self.figr_m2.tight_layout()
        self.img1r_m2.draw()

    def get_M_sq(self, som_x, som_y, z, lambda_0, dx):
        def beam_quality_factor_fit(z, w0, M2, z0):
            return w0 * np.sqrt(1 + (z - z0) ** 2 * (M2 * lambda_0 / (np.pi * w0 ** 2)) ** 2)

        p0 = [dx, 1, 0]

        params_x, _ = curve_fit(beam_quality_factor_fit, z, som_x, p0=p0)
        w0_x_fit, M_sq_x_fit, z0_x_fit = params_x
        print(f'M_sq_x: {abs(M_sq_x_fit):.4f},w0_x: {w0_x_fit * 1e6:.4f} µm, z0_x: {z0_x_fit * 1e3:.4f} mm')

        params_y, _ = curve_fit(beam_quality_factor_fit, z, som_y, p0=p0)
        w0_y_fit, M_sq_y_fit, z0_y_fit = params_y
        print(f'M_sq_y: {abs(M_sq_y_fit):.4f},w0_y: {w0_y_fit * 1e6:.4f} µm, z0_y: {z0_y_fit * 1e3:.4f} mm')

        z_fit = np.linspace(z[0], z[-1], 100)

        self.ax1r_m2.clear()
        self.ax1r_m2.grid(True)

        self.ax1r_m2.plot(z, som_x, linestyle='None', marker='x', color='blue')
        self.ax1r_m2.plot(z, som_y, linestyle='None', marker='x', color='red')
        self.ax1r_m2.plot(z_fit, beam_quality_factor_fit(z_fit, w0_y_fit, M_sq_y_fit, z0_y_fit),
                          label=f'M_sq_y: {abs(params_y[1]):.2f}, '
                                f'w0_y: {params_y[0] * 1e6:.2f} µm, '
                                f'z0_y: {params_y[2] * 1e3:.2f} mm', color='red')
        self.ax1r_m2.plot(z_fit, beam_quality_factor_fit(z_fit, w0_x_fit, M_sq_x_fit, z0_x_fit),
                          label=f'M_sq_x: {abs(params_x[1]):.2f}, '
                                f'w0_x: {params_x[0] * 1e6:.2f} µm, '
                                f'z0_x: {params_x[2] * 1e3:.2f} mm', color='blue')

        self.ax1r_m2.set_ylabel('z [m]')
        self.ax1r_m2.set_xlabel('x [mm]')
        self.ax1r_m2.legend()
        self.figr_m2.tight_layout()
        self.img1r_m2.draw()

        return params_x, params_y

    def process_images_dict(self):
        processed_images = {}
        dz = self.images.shape[2]
        print(f'Number of steps: {dz}')
        som_x = np.zeros(dz, dtype=float)
        som_y = np.zeros(dz, dtype=float)

        for i in range(dz):
            processed_image, som_x[i], som_y[i] = process_image(self.images[:, :, i])
            processed_images[f'processed_image_{i}'] = processed_image

        som_y *= 3.45e-6
        som_x *= 3.45e-6
        return processed_images, som_x, som_y

    def init_pid_piezo(self):
        try:
            if self.pid_stage_initialized == False:
                self.pid_stage = jena.NV40(self.strvar_stage_pid_com.get(), closed_loop=self.var_pid_cl.get())
                self.pid_stage_initialized = True
                self.home_pid_piezo()
                self.but_pid_stage_init.config(fg='green')
            else:
                self.pid_stage = None
                self.but_pid_stage_init.config(fg='black')
                self.pid_stage_initialized = False
        except:
            message = f'Initialization of Piezo went wrong'
            self.insert_message(message)
            self.but_pid_stage_init.config(fg='red')

    def read_pid_piezo(self):
        if self.pid_stage_initialized == True:
            try:
                pos = self.pid_stage.get_position()
                self.strvar_pid_stage_actual_position.set(str(np.round(pos, 2)))
            except:
                message = f'Readout of Piezo went wrong'
                self.insert_message(message)
        else:
            message = f'Initialize the Piezo stage first'
            self.insert_message(message)

    def move_pid_piezo(self):
        if self.pid_stage_initialized == True:
            try:
                self.pid_stage.set_position(float(self.strvar_pid_stage_set_position.get()))
                # self.read_pid_piezo()
            except:
                message = f'Moving of the Piezo went wrong'
                self.insert_message(message)
        else:
            message = f'Initialize the Piezo stage first'
            self.insert_message(message)

    def home_pid_piezo(self):
        if self.pid_stage_initialized == True:
            try:
                self.strvar_pid_stage_set_position.set(str(0))
                self.move_pid_piezo()
                message = f'Piezo homed'
                self.insert_message(message)
                # self.read_pid_piezo()
            except:
                message = f'Homing of the Piezo went wrong'
                self.insert_message(message)
        else:
            message = f'Initialize the Piezo stage first'
            self.insert_message(message)

    def final_image_treatment(self, im):
        bg = int(self.var_mcp_calibration_background_val.get())
        x1 = int(self.var_mcp_calibration_ROIX1_val.get())
        x2 = int(self.var_mcp_calibration_ROIX2_val.get())
        y1 = int(self.var_mcp_calibration_ROIY1_val.get())
        y2 = int(self.var_mcp_calibration_ROIY2_val.get())
        shear = float(self.var_mcp_calibration_shear_val.get())
        correct_E_axis, treated = help.treat_image_new(im, self.eaxis, x1, x2, y1, y2, bg, shear)
        return correct_E_axis, treated

    def export_mcp_treatment_parameters(self):
        bg = int(self.var_mcp_calibration_background_val.get())
        x1 = int(self.var_mcp_calibration_ROIX1_val.get())
        x2 = int(self.var_mcp_calibration_ROIX2_val.get())
        y1 = int(self.var_mcp_calibration_ROIY1_val.get())
        y2 = int(self.var_mcp_calibration_ROIY2_val.get())
        shear = float(self.var_mcp_calibration_shear_val.get())
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
            self.insert_message(message)

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

            self.var_mcp_calibration_background_val.set(bg)
            self.var_mcp_calibration_ROIX1_val.set(x1)
            self.var_mcp_calibration_ROIX2_val.set(x2)
            self.var_mcp_calibration_ROIY1_val.set(y1)
            self.var_mcp_calibration_ROIY2_val.set(y2)
            self.var_mcp_calibration_shear_val.set(shear)
            self.eaxis_correct = energy_axis
            self.eaxis = energy_axis_temp

            self.but_import_image_treatment_parameters.config(fg='green')

        except Exception as e:
            message = f'Chose a proper filename'
            self.insert_message(message)
            self.but_import_image_treatment_parameters.config(fg='red')

    def update_calibration_energy(self, var, index, mode):
        im = np.flipud(self.calibration_image_update)
        profile = np.sum(im, axis=1)
        try:
            smooth = int(self.var_mcp_calibration_energy_smooth.get())
            prom = int(self.var_mcp_calibration_energy_prom.get())
            order = int(self.var_mcp_calibration_energy_order.get())
            firstharmonic = int(self.var_mcp_calibration_energy_firstharmonic.get())

            data, peaks = help.fit_energy_calibration_peaks(profile, prom=prom, smoothing=smooth)
            condition = (peaks > int(self.var_mcp_calibration_energy_ignore1.get())) & (
                    peaks < int(self.var_mcp_calibration_energy_ignore2.get()))
            peaks = peaks[condition]
            try:
                ignore_list = [int(x) for x in self.ent_mcp_calibration_energy_ignore_list.get().split(',') if
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
            self.insert_message(message)

    def update_calibration(self, var, index, mode):
        im = self.calibration_image
        try:
            im = im - int(self.var_mcp_calibration_background_val.get())
        except:
            message = f'Enter something reasonable for the background!!'
            self.insert_message(message)

        try:
            im = help.shear_image(im, float(self.var_mcp_calibration_shear_val.get()), axis=1)
        except:
            message = f'Enter a reasonable value for the shear!'
            self.insert_message(message)
        try:
            x_cut1 = int(self.var_mcp_calibration_ROIX1_val.get())
            x_cut2 = int(self.var_mcp_calibration_ROIX2_val.get())
            y_cut1 = int(self.var_mcp_calibration_ROIY1_val.get())
            y_cut2 = int(self.var_mcp_calibration_ROIY2_val.get())
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
            self.insert_message(message)

        self.calibration_image_update = im
        self.plot_calibration_image(self.calibration_image_update)

    def update_mcp_analysis(self, var, index, mode):
        message = f'The update function has been called'
        self.insert_message(message)
        try:
            self.plot_analysis(self.current_scan_type, self.current_treated_images,
                               parameter1=self.current_lens_position_array,
                               parameter2=self.current_power_array, parameter3=self.current_grating_array,
                               energy_axis=self.current_E)
        except:
            "Something is Nan in the plot :("

    def load_test_calibration_image_take_thread(self):
        self.load_calib_thread = threading.Thread(target=self.load_test_calibration_image_take)
        self.load_calib_thread.daemon = True
        self.load_calib_thread.start()

    def load_test_calibration_image_thread(self):
        self.load_calib_thread = threading.Thread(target=self.load_test_calibration_image)
        self.load_calib_thread.daemon = True
        self.load_calib_thread.start()

    def load_test_calibration_image_take(self):
        im_temp = self.take_image(int(self.ent_avgs.get()))
        self.calibration_image = im_temp
        self.plot_calibration_image(self.calibration_image)

    def load_test_calibration_image(self):
        file_path = askopenfilename(filetypes=[("TIFF files", "*.tiff;*.tif")])
        im_temp = np.asarray(Image.open(file_path))
        self.calibration_image = im_temp
        self.plot_calibration_image(self.calibration_image)

    def calculate_com(self, im):
        # Convert the image data to a NumPy array
        image_array = np.array(im)

        # Create a grid of coordinates
        y_coords, x_coords = np.indices(image_array.shape)

        # Calculate the weighted sum of x and y coordinates
        sum_x = np.sum(x_coords * image_array)
        sum_y = np.sum(y_coords * image_array)

        # Calculate the total sum of pixel values
        total_sum = np.sum(image_array)

        # Calculate the center of mass
        center_x = sum_x / total_sum
        center_y = sum_y / total_sum
        return center_x, center_y

    def get_background(self):
        im = self.take_image(int(self.ent_avgs.get()))
        self.background = im

    def remove_background(self):
        self.background = np.zeros([512, 512])

    def fixyaxis(self):
        if self.var_fixyaxis.get() == 1:
            # ymin, ymax = self.axHarmonics.get_ylim()
            self.ymin_harmonics = self.current_harmonics_profile_min
            self.ymax_harmonics = self.current_harmonics_profile_max + 0.1 * (
                    self.current_harmonics_profile_max - self.current_harmonics_profile_min)
            self.ymin_harmonics_calibrate = self.current_harmonics_profile_min_calibrate
            self.ymax_harmonics_calibrate = self.current_harmonics_profile_max_calibrate + 0.1 * (
                    self.current_harmonics_profile_max_calibrate - self.current_harmonics_profile_min_calibrate)

    def change_mcp_cam(self, event):
        selected_value = self.strvar_mcp_cam_choice.get()

        if selected_value == 'Pike Camera':
            if self.cam is not None:
                self.cam.stop_acquisition()
                self.cam.close()
            self.PIKE_cam = True
            self.ANDOR_cam = False
            self.name_cam = 'PIKE_cam'
        elif selected_value == 'Andor':
            self.cam = Andor.AndorSDK2Camera(fan_mode="full", amp_mode=None)
            self.PIKE_cam = False
            self.ANDOR_cam = True
            self.name_cam = 'ANDOR_cam'
            self.background = np.zeros([512, 512])
            # self.strvar_temperature_status.set(str(self.cam.get_temperature_status()))
            # self.strvar_temperature.set(str(self.cam.get_temperature()))
            self.update_camera_status_thread()
        print(f"PIKE_cam: {self.PIKE_cam}")
        print(f"ANDOR_cam: {self.ANDOR_cam}")

    def update_camera_status_thread(self):
        self.temp_thread = threading.Thread(target=self.update_camera_status)
        self.temp_thread.daemon = True
        self.temp_thread.start()

    def update_camera_status(self):
        while self.ANDOR_cam:
            self.strvar_temperature_status.set(str(self.cam.get_temperature_status()))
            self.strvar_temperature.set(str(np.round(float(self.cam.get_temperature()), 2)))
            time.sleep(1)

    def update_maxgreenratio(self, var, index, mode):
        try:
            x = float(self.ent_int_ratio_focus.get()) ** 2
            c = float(self.ent_int_ratio_constant.get())
            maxG = float(self.ent_green_power.get()) * 1e-3
            self.lbl_int_ratio_constant.config(text='Pr+{:.2f}*PG='.format(x))
            self.strvar_ratio_to.set(str(np.round(x * maxG / c, 3)))
        except:
            message = f'Enter a reasonable value'
            self.insert_message(message)

    def angle_to_power(self, angle, maxA, phase):
        power = maxA / 2 * np.cos(2 * np.pi / 90 * angle - 2 * np.pi / 90 * phase) + maxA / 2
        return power

    def power_to_angle_new(self, power, amplitude, frequency, phase,offset):
        return (-np.arccos((power-offset)/amplitude) + frequency*phase)/frequency

    def angle_to_power_new(self, angle, amplitude, frequency, phase,offset):
        return amplitude*np.cos(frequency*(angle-phase))+ offset

    def power_to_angle(self, power, maxA, phase):
        A = maxA / 2
        angle = -(45 * np.arccos(power / A - 1)) / np.pi + phase
        return angle

    def calculate_energy_and_peak_intensity(self):

        E_pulse = round(float(self.ent_power.get()) / (float(self.ent_rep_rate.get()) * 1e3) * 1e6, 3)  # in µJ
        I_peak = round(0.94 * 2 * E_pulse * 1e-6 / ((float(self.ent_pulse_length.get()) * 1e-15 * np.pi) * (
                float(self.ent_beam_radius.get()) * 1e-6) ** 2) * 1e-4 * 1e-14, 3)

        U_p = 9.337 * 1.03 ** 2 * I_peak  # in eV
        I_p_Argon = 15.7596  # in eV
        E_cut = round(U_p * 3.2 + I_p_Argon, 3)  # in eV

        self.strvar_pulse_energy.set(E_pulse)
        self.strvar_peak_intensity.set(I_peak)
        self.strvar_hhg_cutoff.set(E_cut)
        self.strvar_hhg_cutoff_q.set(int(E_cut / 1.2))

    def init_zaber_stage(self):
        port = self.ent_zaber_grating_nr.get()
        try:
            self.MPC_grating = zb.ZaberStage(port)
            self.but_zaber_grating_Ini.config(fg='green')
            message = f'Zaber stage initialized'
            self.insert_message(message)
            current_position = self.MPC_grating.get_position()
            self.strvar_zaber_grating_is.set(str(current_position))

        except Exception as e:
            self.but_zaber_grating_Ini.config(fg='red')
            message = f'Failed to initialize Zaber stage'
            self.insert_message(message)

    def move_zaber_stage(self):

        target_position = float(self.ent_zaber_grating_should.get())
        try:
            self.MPC_grating.set_position(target_position)
            message = f'Zaber stage moved to position: {target_position} mm'
            self.insert_message(message)
            current_position = self.MPC_grating.get_position()
            self.strvar_zaber_grating_is.set(current_position)
        except Exception as e:
            message = f'Failed to move Zaber stage to position: {e}'
            self.insert_message(message)

    def read_zaber_stage(self):
        try:
            current_position = self.MPC_grating.get_position()
            self.strvar_zaber_grating_is.set(current_position)
            message = f'Current Zaber stage position: {current_position} mm'
            self.insert_message(message)
        except Exception as e:
            message = f'Failed to read Zaber stage to position: {e}'
            self.insert_message(message)

    def home_zaber_stage(self):
        message = f'This feature is desactivated'
        self.insert_message(message)

    def init_WPR(self):
        """
        Initializes the red waveplate motor object.

        Raises
        ------
        Exception
            If the motor fails to initialize.

        Returns
        -------
        None
        """
        try:
            self.WPR = apt.Motor(int(self.ent_WPR_Nr.get()))
            self.but_WPR_Ini.config(fg='green')
            message = f'WPR connected'
            self.insert_message(message)
        except:
            self.but_WPR_Ini.config(fg='red')
            message = f'Not able to initalize WPR'
            self.insert_message(message)

    def home_WPR(self):
        """
        Homes the red waveplate motor object.

        Raises
        ------
        Exception
            If the motor fails to home.

        Returns
        -------
        None
        """
        try:
            self.WPR.move_home(blocking=True)
            self.but_WPR_Home.config(fg='green')
            message = f'WPR homed'
            self.insert_message(message)
            self.read_WPR()
        except:
            self.but_WPR_Home.config(fg='red')
            message = f'Not able to home WPR'
            self.insert_message(message)

    def read_WPR(self):
        """
        Reads the current position of the red waveplate motor.

        Raises
        ------
        Exception
            If the position cannot be read.

        Returns
        -------
        None
        """
        try:
            pos = self.WPR.position
            self.strvar_WPR_is.set(pos)
            self.strvar_red_current_power.set(
                np.round(self.angle_to_power(pos, float(self.ent_red_power.get()), float(self.ent_red_phase.get())), 3))
        except:
            message = f'Impossible to read WPR position'
            self.insert_message(message)

    def move_WPR(self):
        """
        Moves the red waveplate motor to the desired position.

        Raises
        ------
        Exception
            If the motor fails to move to the desired position.

        Returns
        -------
        None
        """
        try:
            if self.var_wprpower.get() == 1:
                power = float(self.strvar_WPR_should.get())
                if power > float(self.ent_red_power.get()):
                    power = float(self.ent_red_power.get())
                    message = f'Value above maximum! Desired power set to maximum instead'
                    self.insert_message(message)
                pos = self.power_to_angle(power, float(self.ent_red_power.get()), float(self.ent_red_phase.get())) + 90
            else:
                pos = float(self.strvar_WPR_should.get())

            message = f'WPR is moving...'
            self.insert_message(message)
            self.WPR.move_to(pos, True)
            message = f"WPR moved to {str(self.WPR.position)}"
            self.insert_message(message)
            self.read_WPR()
        except Exception as e:
            message = f'Impossible to move WPR'
            self.insert_message(message)

    def init_WPG(self):
        """
        Initializes the green waveplate motor object.

        Raises
        ------
        Exception
            If the motor fails to initialize.

        Returns
        -------
        None
        """
        try:
            self.WPG = apt.Motor(int(self.ent_WPG_Nr.get()))
            self.but_WPG_Ini.config(fg='green')
            message = f'WPG connected'
            self.insert_message(message)
        except:
            self.but_WPG_Ini.config(fg='red')
            message = f'Not able to initialize WPG'
            self.insert_message(message)

    def home_WPG(self):
        """
        Homes the green waveplate motor object.
        Raises
        ------
        Exception
            If the motor fails to home.

        Returns
        -------
        None
        """

        message = f'This feature is currently desactivated'
        self.insert_message(message)

    def read_WPG(self):
        """
        Reads the current position of the green waveplate motor.

        Raises
        ------
        Exception
            If the position cannot be read.

        Returns
        -------
        None
        """
        try:
            pos = self.WPG.position
            self.strvar_WPG_is.set(pos)
            self.strvar_green_current_power.set(
                np.round(self.angle_to_power(pos, float(self.ent_green_power.get()), float(self.ent_green_phase.get())),
                         3))

        except:
            message = 'Impossible to read WPR position'
            self.insert_message(message)

    def move_WPG(self):
        """
        Moves the green waveplate motor to the desired position.

        Raises
        ------
        Exception
            If the motor fails to move to the desired position.

        Returns
        -------
        None
        """
        try:
            if self.var_wpgpower.get() == 1:
                power = float(self.strvar_WPG_should.get())
                if power > float(self.ent_green_power.get()):
                    power = float(self.ent_green_power.get())
                    message = f'Value above maximum! Desired power set to maximum instead'
                    self.insert_message(message)
                pos = self.power_to_angle(power, float(self.ent_green_power.get()),
                                          float(self.ent_green_phase.get())) + 90
            else:
                pos = float(self.strvar_WPG_should.get())

            message = f'WPG is moving...'
            self.insert_message(message)
            self.WPG.move_to(pos, True)
            message = f"WPG moved to {str(self.WPG.position)}"
            self.insert_message(message)
            self.read_WPG()

        except Exception as e:
            message = f'Impossible to move WPG'
            self.insert_message(message)

    def init_cam_stage(self):
        """
        Initializes the Dummy waveplate motor object.

        Raises
        ------
        Exception
            If the motor fails to initialize.

        Returns
        -------
        None
        """
        try:
            self.cam_stage = apt.Motor(int(self.ent_cam_stage_Nr.get()))
            self.but_cam_stage_Ini.config(fg='green')
            message = f'cam_stage connected'
            self.insert_message(message)
        except:
            self.but_cam_stage_Ini.config(fg='red')
            message = f'Not able to initialize cam_stage'
            self.insert_message(message)

    def home_cam_stage(self):
        """
        Homes the Dummy waveplate motor object.
        Raises
        ------
        Exception
            If the motor fails to home.

        Returns
        -------
        None
        """

        try:
            self.cam_stage.move_home(blocking=True)
            self.but_cam_stage_Home.config(fg='green')
            message = f'cam_stage homed'
            self.insert_message(message)
            self.read_cam_stage()
        except:
            self.but_cam_stage_Home.config(fg='red')
            message = f'Not able to home cam_stage'
            self.insert_message(message)

    def read_cam_stage(self):
        """
        Reads the current position of the green waveplate motor.

        Raises
        ------
        Exception
            If the position cannot be read.

        Returns
        -------
        None
        """
        try:
            pos = self.cam_stage.position
            self.strvar_cam_stage_is.set(pos)

        except:
            message = f'Impossible to read cam_stage position'
            self.insert_message(message)

    def move_cam_stage(self):
        """
        Moves the Dummy waveplate motor to the desired position.

        Raises
        ------
        Exception
            If the motor fails to move to the desired position.

        Returns
        -------
        None
        """
        try:
            pos = float(self.strvar_cam_stage_should.get())
            message = "cam_stage is moving ..."
            self.insert_message(message)
            self.cam_stage.move_to(pos, True)
            message = f"cam_stage moved to {str(self.cam_stage.position)}"
            self.insert_message(message)
            self.read_cam_stage()
        except Exception as e:
            message = f'Impossible to move cam_stage'
            self.insert_message(message)

    def init_delay_stage(self):
        """
        Initializes the Delay motor object.

        Raises
        ------
        Exception
            If the motor fails to initialize.

        Returns
        -------
        None
        """
        try:
            self.delay_stage = apt.Motor(int(self.ent_delay_stage_Nr.get()))
            message = "delay_stage connected"
            self.insert_message(message)
            self.but_delay_stage_Ini.config(fg='green')
        except:
            self.but_delay_stage_Ini.config(fg='red')
            message = "Not able to initialize the delay_stage"
            self.insert_message(message)

    def home_delay_stage(self):
        """
        Homes the delay waveplate motor object.

        Raises
        ------
        Exception
            If the motor fails to home.

        Returns
        -------
        None
        """
        try:
            self.delay_stage.move_home(blocking=True)
            self.but_delay_stage_Home.config(fg='green')
            message = "delay_stage stage homed"
            self.insert_message(message)
            self.read_delay_stage()
        except:
            self.but_delay_stage_Home.config(fg='red')
            message = "Not able to home the delay_stage"
            self.insert_message(message)

    def read_delay_stage(self):
        """
        Reads the current position of the Delay motor.

        Raises
        ------
        Exception
            If the position cannot be read.

        Returns
        -------
        None
        """
        try:
            pos = self.delay_stage.position
            self.strvar_delay_stage_is.set(pos)
        except:
            message = "Not able to red the Delay stage position"
            self.insert_message(message)

    def move_delay_stage(self):
        """
        Moves the Delay motor to the desired position.

        Raises
        ------
        Exception
            If the motor fails to move to the desired position.

        Returns
        -------
        None
        """
        try:
            pos = float(self.strvar_delay_stage_should.get())
            message = "delay_stage is moving.."
            self.insert_message(message)
            self.delay_stage.move_to(pos, True)
            message = f"delay_stage moved to {str(self.delay_stage.position)}"
            self.insert_message(message)
            self.read_delay_stage()
        except:
            message = "Impossible to move delay_stage"
            self.insert_message(message)

    # def scan(self):

    def init_lens_stage(self):
        try:
            self.lens_stage = apt.Motor(int(self.ent_mpc_lens_nr.get()))
            message = "lens_stage connected"
            self.insert_message(message)
            self.but_lens_stage_Ini.config(fg='green')
        except:
            self.but_lens_stage_Ini.config(fg='red')
            message = "Not able to initialize lens_stage"
            self.insert_message(message)

    def read_lens_stage(self):
        try:
            pos = self.lens_stage.position
            self.strvar_mpc_lens_is.set(pos)
        except:
            message = "Not able to read lens_stage position"
            self.insert_message(message)

    def move_lens_stage(self):
        try:
            pos = float(self.strvar_mpc_lens_should.get())
            message = "lens_stage is moving"
            self.insert_message(message)
            self.lens_stage.move_to(pos, True)
            message = f"lens_stage moved to {str(self.lens_stage.position)}"
            self.insert_message(message)
            self.read_lens_stage()
        except:
            message = f"Impossible to move lens_stage"
            self.insert_message(message)

    def home_lens_stage(self):
        try:
            self.lens_stage.move_home(blocking=True)
            self.but_lens_stage_Home.config(fg='green')
            message = "lens_stage homed"
            self.insert_message(message)
            self.read_lens_stage()
        except:
            self.but_lens_stage_Home.config(fg='red')
            message = "Not able to home lens_stage"
            self.insert_message(message)

    def init_MPC_wp(self):
        try:
            self.MPC_wp = apt.Motor(int(self.ent_mpc_wp_nr.get()))
            message = "MPC_WP connected"
            self.insert_message(message)
            self.but_MPC_wp_Ini.config(fg='green')
        except:
            self.but_MPC_wp_Ini.config(fg='red')
            message = "Not able to initialize MPC_WP"
            self.insert_message(message)

    def read_MPC_wp(self):
        try:
            pos = self.MPC_wp.position
            self.strvar_mpc_wp_is.set(pos)
            self.strvar_mpc_currentpower.set(
                np.round(
                    self.angle_to_power_new(pos, float(self.ent_MPC_fitA.get()), float(self.ent_MPC_fitf.get()),float(self.ent_MPC_fitph.get()), float(self.ent_MPC_fito.get())),
                    3))
        except:
            message = "Not able to read MPC_WP position"
            self.insert_message(message)

    def move_MPC_wp(self):
        try:
            if self.var_mpc_wp_power.get() == 1:
                power = float(self.strvar_mpc_wp_should.get())
                maxp = float(self.ent_MPC_fitA.get()) + float(self.ent_MPC_fito.get())
                if power > maxp:
                    power = maxp
                    message = "Value above maximum! Desired power set to maximum instead"
                    self.insert_message(message)
                pos = self.power_to_angle_new(power, float(self.ent_MPC_fitA.get()), float(self.ent_MPC_fitf.get()),float(self.ent_MPC_fitph.get()), float(self.ent_MPC_fito.get()))
            else:
                pos = float(self.strvar_mpc_wp_should.get())
            message = "MPC_WP is moving..."
            self.insert_message(message)
            self.MPC_wp.move_to(pos, True)
            message = f"WP moved to {str(self.MPC_wp.position)}"
            self.insert_message(message)
            self.read_MPC_wp()
        except:
            message = "Not able to move MPC_WP"
            self.insert_message(message)

    def home_MPC_wp(self):
        try:
            self.MPC_wp.move_home(blocking=True)
            self.but_MPC_wp_Home.config(fg='green')
            message = "MPC_WP homed"
            self.insert_message(message)
            self.read_MPC_wp()
        except:
            self.but_MPC_wp_Home.config(fg='red')
            message = "Not able to home MPC_WP"
            self.insert_message(message)

    def disable_motors(self):
        """
        Disconnect all the motors.

        Returns
        -------
        None
        """
        if self.WPG is not None:
            self.WPG.disable()
        if self.WPR is not None:
            self.WPR.disable()
        if self.delay_stage is not None:
            self.delay_stage.disable()
        if self.lens_stage is not None:
            self.lens_stage.disable()
        if self.MPC_wp is not None:
            self.MPC_wp.disable()
        if self.MPC_grating is not None:
            self.MPC_grating.close()
            self.MPC_grating = None

        message = 'Stages are disconnected'
        self.insert_message(message)

    def enable_calibrator(self):
        self.stop_calib = False
        self.calib_thread = threading.Thread(target=self.open_calibrator)
        self.calib_thread.daemon = True
        self.calib_thread.start()

    def open_calibrator(self):
        """
        Open the file where the power calibration is

        Returns
        -------
        None
        """
        # try:
        self.calibrator = cal.Calibrator()
        self.strvar_red_power.set(str(self.calibrator.max_red))
        self.strvar_green_power.set(str(self.calibrator.max_green))
        self.strvar_mpc_maxpower.set(str(self.calibrator.max_mpc))
        self.strvar_red_phase.set(str(self.calibrator.phase_red))
        self.strvar_green_phase.set(str(self.calibrator.phase_green))
        self.strvar_mpc_maxangle.set(str(self.calibrator.phase_mpc))

    def open_calibrator_on_start(self):
        """
        Open the calibrator to initialize current calibration value

        Returns
        -------
        None
        """
        # try:
        self.calibrator = cal.Calibrator()
        self.strvar_red_power.set(str(self.calibrator.max_red))
        self.strvar_green_power.set(str(self.calibrator.max_green))
        self.strvar_red_phase.set(str(self.calibrator.phase_red))
        self.strvar_mpc_maxpower.set(str(self.calibrator.max_mpc))
        self.strvar_green_phase.set(str(self.calibrator.phase_green))
        self.strvar_mpc_maxangle.set(str(self.calibrator.phase_mpc))
        self.calibrator.on_close()

    def read_red_power(self):
        """
        Reads the corresponding red power if one knows the attenuation and the pulse picker on the Pharos.

        Raises
        ------
        Exception
            If the power cannot be read.

        Returns
        -------
        None
        """
        try:
            given_att = float(self.ent_pharos_att.get())
            given_pp = float(self.ent_pharos_pp.get())
            red_power_indice = np.where((self.pharos_att == given_att) & (self.pharos_pp == given_pp))
            red_power = self.red_p[red_power_indice]
            self.strvar_red_power.set(str(red_power[0]))
        except:
            message = 'Impossible to read red power'
            self.insert_message(message)

    def take_image(self, avgs):
        """
        Takes an image from the camera.

        This method takes an image from the camera using the specified number of averages,
        and returns the captured image.

        Parameters
        ----------
        avgs : int
            The number of images to average over.

        Returns
        -------
        numpy.ndarray
            The captured image.

        """
        if self.PIKE_cam is True:
            with Vimba.get_instance() as vimba:
                cams = vimba.get_all_cameras()
                image = np.zeros([1000, 1600])
                self.d_phase = deque()
                self.meas_has_started = True
                nr = avgs
                with cams[0] as cam:
                    exposure_time = cam.ExposureTime
                    exposure_time.set(float(self.ent_exposure_time.get()))
                    for frame in cam.get_frame_generator(limit=avgs):
                        frame = cam.get_frame()
                        frame.convert_pixel_format(PixelFormat.Mono8)
                        img = frame.as_opencv_image()
                        img = np.squeeze(frame.as_opencv_image())
                        numpy_image = img
                        image = image + numpy_image
                    image = image / nr
                    self.meas_has_started = False

        elif self.ANDOR_cam is True:
            self.cam.set_exposure(float(self.ent_exposure_time.get()) * 1e-6)
            self.cam.setup_shutter('open')
            self.d_phase = deque()
            self.meas_has_started = True
            image = np.zeros([512, 512])
            self.cam.start_acquisition()
            for i in range(avgs):
                self.cam.wait_for_frame(timeout=20)
                frame = self.cam.read_oldest_image()
                image += frame
            image /= avgs
            self.cam.stop_acquisition()
            self.meas_has_started = False

        else:
            message = 'No cam'
            self.insert_message(message)

        return image - self.background

    def save_h5(self):
        nr = self.get_start_image()
        filename = self.saving_folder + '-' + str(int(nr)) + '.h5'

        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('raw_images', data=self.measurement_array)
            hf.create_dataset('treated_images', data=self.measurement_treated_array)
            hf.create_dataset('e_axis', data=self.eaxis_correct)
            hf.create_dataset('phases', data=self.phase_array)
            hf.create_dataset('ratios', data=self.ratio_array)
            hf.create_dataset('phase_meas', data=self.phase_meas_array)
            hf.create_dataset('phase_std', data=self.phase_std_array)
            hf.create_dataset('power_red', data=self.red_power_array)
            hf.create_dataset('power_green', data=self.green_power_array)
            hf.create_dataset('exposure_time', data=self.aquisition_time)
            hf.create_dataset('averages', data=self.averages)
            hf.create_dataset('voltage', data=self.mcp_voltage)
            time_stamps_array_converted = np.array(self.time_stamps_array, dtype='S')
            hf.create_dataset('timestamps', data=time_stamps_array_converted)

        log_entry = str(int(nr)) + '\n'
        self.f.write(log_entry)

    def save_im(self, image):
        """
        Saves the captured image to a file and writes in the log file

        Parameters
        ----------
        image : numpy.ndarray
            The captured image.

        Returns
        -------
        None
        """
        nr = self.get_start_image()
        # self.f = open(self.autolog, "a+")
        filename = self.saving_folder + '-' + str(int(nr)) + '.tif'
        # filename = 'C:/data/' +  '2024-01-25-TEMP' + '/' + str(date.today()) + '-' + str(int(nr)) + '.tif'

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
            int(nr)) + '\t' + self.ent_red_current_power.get() + '\t' + self.ent_green_current_power.get() + '\t' + str(
            np.round(float(self.strvar_setp.get()), 2)) + '\t' + str(
            np.round(np.mean(np.unwrap(self.d_phase)), 2)) + '\t' + str(
            np.round(np.std(np.unwrap(self.d_phase)),
                     2)) + '\t' + self.ent_mcp.get() + '\t' + str(
            self.name_cam) + '\t' + self.ent_avgs.get() + '\t' + self.ent_exposure_time.get() + '\t' + str(
            gl) + '\t' + str(
            gl_pos) + '\t' + str(
            rl) + '\t' + str(
            rl_pos) + '\t' + timestamp + '\n'
        self.f.write(log_entry)

    def split_threading(self):
        total_steps = int(self.ent_ratio_steps.get())
        inital_start = float(self.ent_ratio_from.get())
        initial_end = float(self.ent_ratio_to.get())
        # now, split WP positions in steps of 10!

        all_steps = np.linspace(inital_start, initial_end, total_steps)
        nr_blocks = int(np.ceil(total_steps / 10))
        last_block_length = int(np.mod(total_steps, 10))
        self.f.write(
            '# this scan is split in {} blocks, from {} to {} with {} steps: \n'.format(nr_blocks, inital_start,
                                                                                        initial_end, total_steps))
        print("I will measure ", nr_blocks, " blocks")
        if nr_blocks > 1:
            for i in range(nr_blocks):
                self.scan_is_done_threading.clear()
                self.f.write('# PART {} \n'.format(i))
                if i < (nr_blocks - 1):
                    print("I am in the first block!")
                    print(str(all_steps[int(i * 10)]))
                    self.strvar_ratio_from.set(str(all_steps[int(i * 10)]))
                    self.strvar_ratio_to.set(str(all_steps[int((i + 1) * 10 - 1)]))
                    self.strvar_ratio_steps.set(str(10))
                else:
                    print("I am in the last block!")
                    self.strvar_ratio_from.set(str(all_steps[int(i * 10)]))
                    self.strvar_ratio_to.set(str(all_steps[-1]))
                    if last_block_length == 0:
                        self.strvar_ratio_steps.set(str(10))
                    else:
                        self.strvar_ratio_steps.set(str(last_block_length))

                self.stop_mcp = False
                self.mcp_thread = threading.Thread(target=self.measure_all)
                self.mcp_thread.daemon = True
                self.mcp_thread.start()
                print("I started the measure_all thread ")
                self.scan_is_done = False
                # while not self.scan_is_done:
                #    pass
                self.scan_is_done_threading.wait()

        self.strvar_ratio_from.set(str(inital_start))
        self.strvar_ratio_to.set(str(initial_end))
        self.strvar_ratio_steps.set(str(total_steps))
        print("Measurement Done!!!")

    def enabl_mcp_live(self):
        self.live_is_pressed = not self.live_is_pressed
        self.update_live_button()

        self.stop_mcp = False
        self.mcp_thread = threading.Thread(target=self.view_live)
        self.mcp_thread.daemon = True
        self.mcp_thread.start()

    def update_live_button(self):
        if self.live_is_pressed:
            self.but_view_live.config(relief="sunken")
            self.but_view_live.config(fg='red')
        else:
            self.but_view_live.config(relief="raised")
            self.but_view_live.config(fg='green')

    def enabl_mcp_all(self):
        self.stop_mcp = False
        self.mcp_thread = threading.Thread(target=self.measure_all)
        self.mcp_thread.daemon = True
        self.mcp_thread.start()

    def enabl_mcp(self):

        self.stop_mcp = False
        self.mcp_thread = threading.Thread(target=self.measure)
        self.mcp_thread.daemon = True
        self.mcp_thread.start()

    def enabl_mcp_simple(self):

        self.stop_mcp = False
        self.mcp_thread = threading.Thread(target=self.measure_simple)
        self.mcp_thread.daemon = True
        self.mcp_thread.start()

    def enabl_mpc_meas(self):

        self.stop_mcp = False
        self.mcp_thread = threading.Thread(target=self.measure_mpc)
        self.mcp_thread.daemon = True
        self.mcp_thread.start()

    def enabl_beam_shaping_meas(self):

        self.stop_beam_shaping = False
        self.beam_shaping_thread = threading.Thread(target=self.measure_beam_shaping)
        self.beam_shaping_thread.daemon = True
        self.beam_shaping_thread.start()

    def enabl_mpc_test_scan(self):

        self.stop_mcp_test = False
        self.mcp_thread_test = threading.Thread(target=self.test_mpc_scan)
        self.mcp_thread_test.daemon = True
        self.mcp_thread_test.start()

    def get_start_image(self):
        self.f.seek(0)
        lines = np.loadtxt(self.autolog, comments="#", delimiter="\t", unpack=False, usecols=(0,))
        if lines.size > 0:
            try:
                start_image = lines[-1] + 1
            except:
                start_image = lines + 1
            message = "The last image had index " + str(int(start_image - 1))
            self.insert_message(message)
        else:
            start_image = 0
        return start_image

    def red_only_scan(self):
        self.var_wprpower.set(1)
        WPR_steps = int(self.ent_WPR_steps.get())
        WPR_scan_list = np.linspace(float(self.ent_WPR_from.get()), float(self.ent_WPR_to.get()), WPR_steps)

        for i in np.arange(0, WPR_steps):
            r = WPR_scan_list[i]
            self.strvar_WPR_should.set(str(r))
            self.move_WPR()
            im = self.take_image(int(self.ent_avgs.get()))
            self.save_im(im)
            self.plot_MCP(im)

    def green_only_scan(self):
        self.var_wpgpower.set(1)
        WPG_steps = int(self.ent_WPG_steps.get())
        WPG_scan_list = np.linspace(float(self.ent_WPG_from.get()), float(self.ent_WPG_to.get()), WPG_steps)

        for i in np.arange(0, WPG_steps):
            g = WPG_scan_list[i]
            self.strvar_WPG_should.set(str(g))
            self.move_WPG()
            im = self.take_image(int(self.ent_avgs.get()))
            self.save_im(im)
            self.plot_MCP(im)

    def red_green_ratio_scan(self):
        steps = int(self.ent_ratio_steps.get())
        pr, pg = self.get_power_values_for_ratio_scan()
        self.var_wprpower.set(1)
        self.var_wpgpower.set(1)

        message = f'Power values for red: {pr}'
        self.insert_message(message)
        message = f'Power values for green: {pg}'
        self.insert_message(message)

        for i in np.arange(0, steps):
            r = pr[i]
            g = pg[i]

            self.strvar_WPG_should.set(str(1e3 * g))
            self.move_WPG()
            time.sleep(0.2)

            self.strvar_WPR_should.set(str(r))
            self.move_WPR()
            time.sleep(0.2)

            if self.var_phasescan.get() == 1 and self.var_background.get() == 0:
                self.phase_scan()
            else:
                im = self.take_image(int(self.ent_avgs.get()))
                self.save_im(im)
                self.plot_MCP(im)

    def get_power_values_for_ratio_scan(self):
        c = float(self.strvar_int_ratio_constant.get())
        x = float(self.ent_int_ratio_focus.get()) ** 2
        ratios = np.linspace(float(self.ent_ratio_from.get()), float(self.ent_ratio_to.get()),
                             int(self.ent_ratio_steps.get()))
        pr = c - c * ratios
        pg = c * ratios / x
        return pr, pg

    def focus_position_scan_green(self):
        start = float(self.ent_GFP_from.get())
        end = float(self.ent_GFP_to.get())
        steps = float(self.ent_GFP_steps.get())

        for ind, b in enumerate(np.linspace(start, end, int(steps))):
            # things go to hell if division by zero
            # if b == 0:
            #    b = 0.00001
            # self.lens_green.strvar_ben.set(str(b))
            self.lens_green.strvar_focus_position.set(str(b))
            self.parent.open_pub_green()
            if self.var_phasescan.get() == 1 and self.var_background.get() == 0:
                self.phase_scan()
            else:
                im = self.take_image(int(self.ent_avgs.get()))
                self.save_im(im)
                self.plot_MCP(im)
                # self.cam_mono_acq()

    def focus_position_scan_red(self):
        start = float(self.ent_RFP_from.get())
        end = float(self.ent_RFP_to.get())
        steps = float(self.ent_RFP_steps.get())

        for ind, b in enumerate(np.linspace(start, end, int(steps))):
            # things go to hell if division by zero
            # if b == 0:
            #    b = 0.00001
            # self.lens_red.strvar_ben.set(str(b))
            self.lens_red.strvar_focus_position.set(str(b))
            self.parent.open_pub_red()
            if self.var_phasescan.get() == 1 and self.var_background.get() == 0:
                self.phase_scan()
            else:
                im = self.take_image(int(self.ent_avgs.get()))
                self.save_im(im)
                self.plot_MCP(im)

    def view_live(self):
        while self.live_is_pressed:
            im = self.take_image(int(self.ent_avgs.get()))
            self.plot_MCP(im)

    def phase_scan(self):
        start_image = self.get_start_image()
        self.phis = np.linspace(float(self.ent_from.get()), float(self.ent_to.get()), int(self.ent_steps.get()))
        self.strvar_setp.set(self.phis[0])
        self.set_setpoint()
        time.sleep(0.05)
        for ind, phi in enumerate(self.phis):
            start_time = time.time()
            self.strvar_setp.set(phi)
            self.set_setpoint()
            im = self.take_image(int(self.ent_avgs.get()))
            if self.measurement_running and self.var_saveh5.get():
                self.measurement_array_flat[self.measurement_counter, :, :] = im
                if self.var_export_treated_image.get():
                    E_new, im_new = self.final_image_treatment(im)
                    self.measurement_treated_array_flat[self.measurement_counter, :, :] = im_new
                self.time_stamps_array_flat[self.measurement_counter] = str(
                    datetime.datetime.now().strftime("%H:%M:%S"))
                self.phase_meas_array_flat[self.measurement_counter] = np.round(np.mean(np.unwrap(self.d_phase)), 2)
                self.phase_std_array_flat[self.measurement_counter] = np.round(np.std(np.unwrap(self.d_phase)), 2)
                self.insert_message(self.measurement_counter)
                self.measurement_counter = self.measurement_counter + 1
            else:
                self.save_im(im)
            self.plot_MCP(im)

            end_time = time.time()
            elapsed_time = end_time - start_time
            #message = "Imagenr ", (start_image + ind), " Phase: ", round(phi, 2), " Elapsed time: ", round(elapsed_time, 2)
            #self.insert_message(str(message))

    def abort_mpc_measurement(self):
        self.abort = 1

    def abort_beam_shaping_measurement(self):
        self.abort = 1


    def test_mpc_scan(self):
        grating_pos_array = np.linspace(float(self.ent_mpc_grating_from.get()), float(self.ent_mpc_grating_to.get()),
                                        int(self.ent_mpc_grating_steps.get()))

        for ind_pos, pos in enumerate(grating_pos_array):
            message = f'Current index: {ind_pos}'
            self.insert_message(message)
            self.strvar_zaber_grating_should.set(str(pos))
            self.move_zaber_stage()

    def measure_mpc(self):
        lens_pos_array = np.linspace(float(self.ent_mpc_lens_from.get()), float(self.ent_mpc_lens_to.get()),
                                     int(self.ent_mpc_lens_steps.get()))
        self.current_lens_position_array = lens_pos_array
        power_array = np.linspace(float(self.ent_mpc_wp_from.get()), float(self.ent_mpc_wp_to.get()),
                                  int(self.ent_mpc_wp_steps.get()))
        self.current_power_array = power_array

        grating_array = np.linspace(float(self.ent_mpc_grating_from.get()), float(self.ent_mpc_grating_to.get()),
                                    int(self.ent_mpc_grating_steps.get()))
        self.current_grating_array = grating_array

        self.but_MPC_measure.config(fg='red')
        pygame.mixer.init()
        pygame.mixer.music.load("ressources/ok_lets_go.mp3")
        message = "OKKKKKKKK LET'S GO"
        self.insert_message(message)
        pygame.mixer.music.play()
        if self.var_mpc_scan_grating.get() == 1:
            self.scan_type = 3
            self.current_scan_type = 3
            message = "Now we scan the MPC grating"
            self.insert_message(message)
            res = np.zeros([512, 512, grating_array.size]) * np.nan
            res_treated = np.zeros([512, 512, grating_array.size]) * np.nan
            self.abort = 0
            if self.MPC_grating is not None:
                for ind_pos, pos in enumerate(grating_array):
                    if self.abort == 1:
                        break
                    self.strvar_zaber_grating_should.set(str(pos))
                    self.move_zaber_stage()
                    if self.ANDOR_cam == True:
                        im = self.take_image(int(self.ent_avgs.get()))
                    else:
                        im = np.random.rand(512, 512)
                    self.plot_MCP(im)
                    res[:, :, ind_pos] = im
                    if self.eaxis is not None:
                        E_new, im_new = self.final_image_treatment(im)
                        res_treated[:, :, ind_pos] = im_new
                        self.current_E = self.eaxis_correct
                        self.current_treated_images = res_treated
                        self.plot_analysis(3, res_treated, parameter1=lens_pos_array, parameter2=power_array,
                                           parameter3=grating_array, energy_axis=self.current_E)

            else:
                message = "Check if the grating stage is initalized"
                self.insert_message(message)


        elif self.var_mpc_scan_wp.get() == 1 and self.var_mpc_scan_lens.get() == 1:
            self.scan_type = 0
            self.current_scan_type = 0
            self.var_mpc_wp_power.set(1)
            message = "Now we scan the lens position and the power"
            self.insert_message(message)
            res = np.zeros([512, 512, lens_pos_array.size, power_array.size]) * np.nan
            res_treated = np.zeros([512, 512, lens_pos_array.size, power_array.size]) * np.nan

            for ind_pos, pos in enumerate(lens_pos_array):
                if self.abort == 1:
                    break
                self.strvar_mpc_lens_should.set(str(pos))
                self.move_lens_stage()
                for ind_power, power in enumerate(power_array):
                    if self.abort == 1:
                        break
                    self.strvar_mpc_wp_should.set(str(power))
                    self.move_MPC_wp()
                    if self.ANDOR_cam == True:
                        im = self.take_image(int(self.ent_avgs.get()))
                    else:
                        im = np.random.rand(512, 512)
                    self.plot_MCP(im)
                    res[:, :, ind_pos, ind_power] = im
                    if self.eaxis is not None:
                        E_new, im_new = self.final_image_treatment(im)
                        res_treated[:, :, ind_pos, ind_power] = im_new
                        self.current_E = self.eaxis_correct
                        self.current_treated_images = res_treated
                        self.plot_analysis(0, res_treated, parameter1=lens_pos_array, parameter2=power_array,
                                           energy_axis=self.current_E)


        elif self.var_mpc_scan_wp.get() == 1 and self.var_mpc_scan_lens.get() == 0:
            self.scan_type = 1
            self.current_scan_type = 1
            self.var_mpc_wp_power.set(1)
            message = "Now we scan the the power only"
            self.insert_message(message)
            lens_pos_array = 0
            res = np.zeros([512, 512, power_array.size]) * np.nan
            res_treated = np.zeros([512, 512, power_array.size]) * np.nan

            for ind_power, power in enumerate(power_array):
                if self.abort == 1:
                    break
                self.strvar_mpc_wp_should.set(str(power))
                self.move_MPC_wp()
                if self.ANDOR_cam == True:
                    im = self.take_image(int(self.ent_avgs.get()))
                else:
                    im = np.random.rand(512, 512)
                self.plot_MCP(im)
                res[:, :, ind_power] = im
                if self.eaxis is not None:
                    E_new, im_new = self.final_image_treatment(im)
                    res_treated[:, :, ind_power] = im_new
                    self.current_E = self.eaxis_correct
                    self.current_treated_images = res_treated
                    self.plot_analysis(1, res_treated, parameter1=lens_pos_array, parameter2=power_array,
                                       energy_axis=self.current_E)

        elif self.var_mpc_scan_wp.get() == 0 and self.var_mpc_scan_lens.get() == 1:
            self.scan_type = 2
            self.current_scan_type = 2
            message = "Now we scan the the lens position only"
            self.insert_message(message)
            power_array = 0
            res = np.zeros([512, 512, lens_pos_array.size]) * np.nan
            res_treated = np.zeros([512, 512, lens_pos_array.size]) * np.nan

            for ind_pos, pos in enumerate(lens_pos_array):
                if self.abort == 1:
                    break
                self.strvar_mpc_lens_should.set(str(pos))
                self.move_lens_stage()
                if self.ANDOR_cam == True:
                    im = self.take_image(int(self.ent_avgs.get()))
                else:
                    im = np.random.rand(512, 512)
                self.plot_MCP(im)
                res[:, :, ind_pos] = im
                if self.eaxis is not None:
                    E_new, im_new = self.final_image_treatment(im)
                    res_treated[:, :, ind_pos] = im_new
                    self.current_E = self.eaxis_correct
                    self.current_treated_images = res_treated
                    self.plot_analysis(2, res_treated, parameter1=lens_pos_array, parameter2=power_array,
                                       energy_axis=self.current_E)

        else:
            message = "Now we scan absolutely nothing"
            self.insert_message(message)
            self.abort = 1

        if self.abort == 1:
            self.but_MPC_measure.config(fg='magenta')
            self.abort = 0
        else:
            nr = self.get_start_image()
            filename = self.saving_folder + '-' + str(int(nr)) + '.h5'

            with h5py.File(filename, 'w') as hf:
                hf.create_dataset('raw_images', data=res)
                hf.create_dataset('lens_position', data=lens_pos_array)
                hf.create_dataset('grating_position', data=grating_array)
                hf.create_dataset('power', data=power_array)
                hf.create_dataset('scan_type', data=self.scan_type)
                if self.eaxis is not None:
                    hf.create_dataset('treated_images', data=res_treated)
                    hf.create_dataset('e_axis', data=self.eaxis_correct)

                hf.create_dataset('exposure_time', data=int(self.ent_exposure_time.get()))
                hf.create_dataset('averages', data=int(self.ent_avgs.get()))
                hf.create_dataset('voltage', data=int(self.ent_mcp.get()))

            log_entry = str(int(nr)) + '\t' + str(self.scan_type) + '\t' + str(self.ent_comment.get()) + '\n'
            self.f.write(log_entry)
            self.but_MPC_measure.config(fg='green')




    def measure_beam_shaping(self):
        lens_pos_array = np.linspace(float(self.ent_beam_shaping_lens_from.get()), float(self.ent_beam_shaping_lens_to.get()),
                                     int(self.ent_beam_shaping_lens_steps.get()))
        self.current_lens_position_array = lens_pos_array
        self.but_beam_shaping_measure.config(fg='red')
        message = "LET'S GO"
        self.insert_message(message)

        if self.var_beam_shaping_scan_lens.get() == 1:
            self.scan_type = 2
            self.current_scan_type = 2
            message = "Scanning the the lens position"
            self.insert_message(message)
            power_array = 0
            res = np.zeros([512, 512, lens_pos_array.size]) * np.nan
            res_treated = np.zeros([512, 512, lens_pos_array.size]) * np.nan

            for ind_pos, pos in enumerate(lens_pos_array):
                if self.abort == 1:
                    break
                self.strvar_mpc_lens_should.set(str(pos))
                self.move_lens_stage()
                if self.ANDOR_cam == True:
                    im = self.take_image(int(self.ent_avgs.get()))
                else:
                    im = np.random.rand(512, 512)
                self.plot_MCP(im)
                res[:, :, ind_pos] = im
                if self.eaxis is not None:
                    E_new, im_new = self.final_image_treatment(im)
                    res_treated[:, :, ind_pos] = im_new
                    self.current_E = self.eaxis_correct
                    self.current_treated_images = res_treated
                    self.plot_analysis(2, res_treated, parameter1=lens_pos_array, parameter2=power_array,
                                       energy_axis=self.current_E)

        else:
            message = "Now we scan absolutely nothing"
            self.insert_message(message)
            self.abort = 1

        if self.abort == 1:
            self.but_MPC_measure.config(fg='magenta')
            self.abort = 0
        else:
            nr = self.get_start_image()
            filename = self.saving_folder + '-' + str(int(nr)) + '.h5'

            with h5py.File(filename, 'w') as hf:
                hf.create_dataset('raw_images', data=res)
                hf.create_dataset('lens_position', data=lens_pos_array)
                #hf.create_dataset('scan_type', data=self.scan_type)
                if self.eaxis is not None:
                    hf.create_dataset('treated_images', data=res_treated)
                    hf.create_dataset('e_axis', data=self.eaxis_correct)
                hf.create_dataset('exposure_time', data=int(self.ent_exposure_time.get()))
                hf.create_dataset('averages', data=int(self.ent_avgs.get()))
                hf.create_dataset('voltage', data=int(self.ent_mcp.get()))

            log_entry = str(int(nr)) + '\t' + str(self.scan_type) + '\t' + str(self.ent_comment.get()) + '\n'
            self.f.write(log_entry)
            self.but_MPC_measure.config(fg='green')

    def plot_analysis(self, scan_type, treated_images, parameter1=None, parameter2=None, parameter3=None,
                      energy_axis=None):
        har = int(self.var_mcp_analysis_harmonic_order.get())
        message = f"Harmonics order: {har}"
        self.insert_message(message)
        if energy_axis is not None:
            ind = int(np.argmin(abs(energy_axis - har * 1.2037300291262136)))
        else:
            message = "There is no energy axis"
            self.insert_message(message)
            ind = 0

        if scan_type == 0:
            self.axAnalysis_1.clear()
            if self.var_log_scale.get() == 1:
                imm = self.axAnalysis_1.imshow(np.flipud(np.nansum(treated_images, axis=(0, 1)).T),
                                               extent=[parameter1[0], parameter1[-1], parameter2[0],
                                                       parameter2[-1]], aspect='auto', norm=LogNorm())
                cbar1 = self.figrAnalysis.colorbar(imm, ax=self.axAnalysis_1)
            else:
                imm = self.axAnalysis_1.imshow(np.flipud(np.nansum(treated_images, axis=(0, 1)).T),
                                               extent=[parameter1[0], parameter1[-1], parameter2[0],
                                                       parameter2[-1]], aspect='auto', norm='linear')
                cbar1 = self.figrAnalysis.colorbar(imm, ax=self.axAnalysis_1)

            self.axAnalysis_1.set_xlabel("Lens position (mm)")
            self.axAnalysis_1.set_ylabel("Power (W)")
            self.axAnalysis_1.set_title("Yield: whole image")

            self.axAnalysis_2.clear()
            if self.var_log_scale.get() == 1:
                imm2 = self.axAnalysis_2.imshow(
                    np.flipud(np.nansum(treated_images[ind - 8:ind + 8, :, :], axis=(0, 1)).T),
                    extent=[parameter1[0], parameter1[-1], parameter2[0],
                            parameter2[-1]], aspect='auto', norm=LogNorm())
                cbar2 = self.figrAnalysis.colorbar(imm2, ax=self.axAnalysis_2)
            else:
                imm2 = self.axAnalysis_2.imshow(
                    np.flipud(np.nansum(treated_images[ind - 8:ind + 8, :, :], axis=(0, 1)).T),
                    extent=[parameter1[0], parameter1[-1], parameter2[0],
                            parameter2[-1]], aspect='auto', norm='linear')
                cbar2 = self.figrAnalysis.colorbar(imm2, ax=self.axAnalysis_2)
            self.axAnalysis_2.set_xlabel("Lens position (mm)")
            self.axAnalysis_2.set_ylabel("Power (W)")
            self.axAnalysis_2.set_title("Yield: H {}".format(har))

            self.figrAnalysis.tight_layout()
            self.canvas_results.draw()
            cbar1.remove()
            cbar2.remove()

            self.axAnalysis_3.clear()
        elif scan_type == 1:
            message = "Plotting the power scan..."
            self.insert_message(message)
            self.axAnalysis_1.clear()
            self.axAnalysis_1.plot(parameter2, np.nansum(treated_images, axis=(0, 1)).ravel())
            self.axAnalysis_1.set_xlabel("Power (W)")
            self.axAnalysis_1.set_ylabel("Total signal")
            self.axAnalysis_1.set_title("Yield: whole image")

            self.axAnalysis_2.clear()
            self.axAnalysis_2.plot(parameter2, np.nansum(treated_images[ind - 8:ind + 8, :, :], axis=(0, 1)).ravel())
            self.axAnalysis_2.set_xlabel("Power (W)")
            self.axAnalysis_2.set_ylabel("Total signal")
            self.axAnalysis_2.set_title("Yield: H {}".format(har))

            self.axAnalysis_3.clear()
            profiles = np.sum(treated_images, 1)
            if self.var_log_scale.get() == 1:
                imm = self.axAnalysis_3.imshow(np.flipud(profiles.T),
                                               extent=[energy_axis[0], energy_axis[-1], parameter2[0],
                                                       parameter2[-1]], aspect='auto', norm=LogNorm())

                cbar1 = self.figrAnalysis.colorbar(imm, ax=self.axAnalysis_3)
                self.axAnalysis_1.set_yscale('log')
                self.axAnalysis_2.set_yscale('log')
            else:
                imm = self.axAnalysis_3.imshow(np.flipud(profiles.T),
                                               extent=[energy_axis[0], energy_axis[-1], parameter2[0],
                                                       parameter2[-1]], aspect='auto')
                cbar1 = self.figrAnalysis.colorbar(imm, ax=self.axAnalysis_3)
                self.axAnalysis_1.set_yscale('linear')
                self.axAnalysis_2.set_yscale('linear')

            self.axAnalysis_3.set_xlabel("Energy (eV)")
            self.axAnalysis_3.set_ylabel("Power (W)")
            self.axAnalysis_3.set_title("Profiles")
            try:
                self.axAnalysis_3.set_xlim(float(self.var_mcp_analysis_emin.get()),
                                           float(self.var_mcp_analysis_emax.get()))
            except:
                self.axAnalysis_3.set_xlim(auto=True)

            self.figrAnalysis.tight_layout()
            self.canvas_results.draw()
            cbar1.remove()

        elif scan_type == 2:
            message = "Plotting the lens scan..."
            self.insert_message(message)

            self.axAnalysis_1.clear()
            self.axAnalysis_1.plot(parameter1, np.nansum(treated_images, axis=(0, 1)).ravel())
            self.axAnalysis_1.set_xlabel("Lens position (mm)")
            self.axAnalysis_1.set_ylabel("Total signal")
            self.axAnalysis_1.set_title("Yield: whole image")

            self.axAnalysis_2.clear()
            self.axAnalysis_2.plot(parameter1, np.nansum(treated_images[ind - 8:ind + 8, :, :], axis=(0, 1)).ravel())
            self.axAnalysis_2.set_xlabel("Lens position (mm)")
            self.axAnalysis_2.set_ylabel("Total signal")
            self.axAnalysis_2.set_title("Yield: H {}".format(har))

            self.axAnalysis_3.clear()
            profiles = np.sum(treated_images, 1)
            if self.var_log_scale.get() == 1:
                imm = self.axAnalysis_3.imshow(np.flipud(profiles.T),
                                               extent=[energy_axis[0], energy_axis[-1], parameter1[0],
                                                       parameter1[-1]], aspect='auto', norm=LogNorm())
                cbar1 = self.figrAnalysis.colorbar(imm, ax=self.axAnalysis_3)
                self.axAnalysis_1.set_yscale('log')
                self.axAnalysis_2.set_yscale('log')
            else:
                imm = self.axAnalysis_3.imshow(np.flipud(profiles.T),
                                               extent=[energy_axis[0], energy_axis[-1], parameter1[0],
                                                       parameter1[-1]], aspect='auto')
                cbar1 = self.figrAnalysis.colorbar(imm, ax=self.axAnalysis_3)
                self.axAnalysis_1.set_yscale('linear')
                self.axAnalysis_2.set_yscale('linear')

            self.axAnalysis_3.set_xlabel("Energy (eV)")
            self.axAnalysis_3.set_ylabel("Lens position (mm)")
            self.axAnalysis_3.set_title("Profiles")
            try:
                self.axAnalysis_3.set_xlim(float(self.var_mcp_analysis_emin.get()),
                                           float(self.var_mcp_analysis_emax.get()))
            except:
                self.axAnalysis_3.set_xlim(auto=True)

            self.figrAnalysis.tight_layout()
            self.canvas_results.draw()
            cbar1.remove()

        elif scan_type == 3:
            message = "Pltting the grating scan..."
            self.insert_message(message)

            self.axAnalysis_1.clear()
            self.axAnalysis_1.plot(parameter3, np.nansum(treated_images, axis=(0, 1)).ravel())
            self.axAnalysis_1.set_xlabel("Grating Position (mm)")
            self.axAnalysis_1.set_ylabel("Total signal")
            self.axAnalysis_1.set_title("Yield: whole image")

            self.axAnalysis_2.clear()
            self.axAnalysis_2.plot(parameter3, np.nansum(treated_images[ind - 8:ind + 8, :, :], axis=(0, 1)).ravel())
            self.axAnalysis_2.set_xlabel("Grating position (mm)")
            self.axAnalysis_2.set_ylabel("Total signal")
            self.axAnalysis_2.set_title("Yield: H {}".format(har))

            self.axAnalysis_3.clear()
            profiles = np.sum(treated_images, 1)
            if self.var_log_scale.get() == 1:
                imm = self.axAnalysis_3.imshow(np.flipud(profiles.T),
                                               extent=[energy_axis[0], energy_axis[-1], parameter3[0],
                                                       parameter3[-1]], aspect='auto', norm=LogNorm())
                cbar1 = self.figrAnalysis.colorbar(imm, ax=self.axAnalysis_3)
                self.axAnalysis_1.set_yscale('log')
                self.axAnalysis_2.set_yscale('log')
            else:
                imm = self.axAnalysis_3.imshow(np.flipud(profiles.T),
                                               extent=[energy_axis[0], energy_axis[-1], parameter3[0],
                                                       parameter3[-1]], aspect='auto')
                cbar1 = self.figrAnalysis.colorbar(imm, ax=self.axAnalysis_3)
                self.axAnalysis_1.set_yscale('linear')
                self.axAnalysis_2.set_yscale('linear')

            self.axAnalysis_3.set_xlabel("Energy (eV)")
            self.axAnalysis_3.set_ylabel("Grating position (mm)")
            self.axAnalysis_3.set_title("Profiles")
            try:
                self.axAnalysis_3.set_xlim(float(self.var_mcp_analysis_emin.get()),
                                           float(self.var_mcp_analysis_emax.get()))
            except:
                self.axAnalysis_3.set_xlim(auto=True)
            divs = np.zeros_like(parameter1)
            for ind_pos, pos in enumerate(parameter1):
                x_data = np.arange(0, 512)
                y_data = np.nansum(treated_images[:, ind-8:ind+8,ind_pos], axis = 1)
                try:
                    A_fit, mu_fit, sigma_fit, B_fit = help.fit_gaussian(x_data,y_data)
                    divs[ind_pos] = sigma_fit
                except:
                    divs[ind_pos] = np.nan

            self.axAnalysis_3.clear()
            self.axAnalysis_3.plot(parameter1, divs)
            self.axAnalysis_3.set_xlabel("Lens position (mm)")
            self.axAnalysis_3.set_xlim(parameter1[0],parameter1[-1])
            self.axAnalysis_3.set_ylim(0,200)

            self.axAnalysis_3.set_ylabel("Divergence (px)")
            self.axAnalysis_3.set_title("Div: H {}".format(har))

            self.figrAnalysis.tight_layout()
            self.canvas_results.draw()
            cbar1.remove()

    def measure_all(self):
        self.but_meas_all.config(fg='red')

        status = self.var_scan_wp_option.get()
        self.insert_message(status)

        if status == "Green Focus":
            if self.var_phasescan.get() == 1:
                if self.var_background.get() == 1:
                    self.f.write("# BACKGROUND FocusPositionScan, " + self.ent_comment.get() + "\n")
                    self.focus_position_scan_green()
                else:
                    self.f.write("# FocusPositionScan, " + self.ent_comment.get() + "\n")
                    self.focus_position_scan_green()
            else:
                message = "Are you sure you do not want to scan the phase for each focus position?"
                self.insert_message(message)
                if self.var_background.get() == 1:
                    self.f.write("# BACKGROUND FocusPositionScan, " + self.ent_comment.get() + "\n")
                    self.focus_position_scan_green()
                else:
                    self.f.write("# FocusPositionScan, " + self.ent_comment.get() + "\n")
                    self.focus_position_scan_green()

        if status == "Red Focus":
            if self.var_phasescan.get() == 1:
                if self.var_background.get() == 1:
                    self.f.write("# BACKGROUND FocusPositionScan, " + self.ent_comment.get() + "\n")
                    self.focus_position_scan_red()
                else:
                    self.f.write("# FocusPositionScan, " + self.ent_comment.get() + "\n")
                    self.focus_position_scan_red()
            else:
                message = "Are you sure you do not want to scan the phase for each focus position?"
                self.insert_message(message)
                if self.var_background.get() == 1:
                    self.f.write("# BACKGROUND FocusPositionScan, " + self.ent_comment.get() + "\n")
                    self.focus_position_scan_red()
                else:
                    self.f.write("# FocusPositionScan, " + self.ent_comment.get() + "\n")
                    self.focus_position_scan_red()

        elif status == "Nothing":
            if self.var_phasescan.get() == 1:
                if self.var_background.get() == 1:
                    self.f.write(
                        "# BACKGROUND PhaseScan, " + self.ent_comment.get() + "\n")
                    im = self.take_image(int(self.ent_avgs.get()))
                    self.save_im(im)
                    self.plot_MCP(im)
                else:
                    self.f.write(
                        "# PhaseScan, " + self.ent_comment.get() + "\n")
                    self.phase_scan()
            else:
                message = "Select something to scan"
                self.insert_message(message)

        elif status == "Red/Green Ratio":
            if self.var_phasescan.get() == 1:
                if self.var_background.get() == 1:
                    self.f.write("# BACKGROUND RedGreenRatioScan, " + self.ent_comment.get() + "\n")
                    self.red_green_ratio_scan()
                else:
                    self.f.write("# RedGreenRatioScan, " + self.ent_comment.get() + "\n")
                    ratio_steps = int(self.ent_ratio_steps.get())
                    phase_steps = int(self.ent_steps.get())
                    self.measurement_counter = 0
                    self.measurement_array = np.zeros([ratio_steps, phase_steps]) * np.nan
                    self.measurement_treated_array = np.zeros([ratio_steps, phase_steps]) * np.nan

                    self.phase_std_array = np.zeros([ratio_steps, phase_steps]) * np.nan
                    self.phase_meas_array = np.zeros([ratio_steps, phase_steps]) * np.nan
                    self.time_stamps_array = np.zeros([ratio_steps, phase_steps]) * np.nan
                    self.time_stamps_array = self.time_stamps_array.astype('str')

                    self.phase_array = np.linspace(float(self.ent_from.get()), float(self.ent_to.get()),
                                                   int(self.ent_steps.get()))
                    self.ratio_array = np.linspace(float(self.ent_ratio_from.get()), float(self.ent_ratio_to.get()),
                                                   int(self.ent_ratio_steps.get()))
                    pr, pg = self.get_power_values_for_ratio_scan()
                    self.red_power_array = pr
                    self.green_power_array = pg
                    self.aquisition_time = int(self.ent_exposure_time.get())
                    self.averages = int(self.ent_avgs.get())
                    self.mcp_voltage = float(self.ent_mcp.get())

                    self.measurement_array_flat = self.measurement_array.flatten()
                    self.measurement_array_flat = np.zeros([self.measurement_array_flat.size, 512, 512]) * np.nan
                    self.measurement_treated_array_flat = self.measurement_array.flatten()
                    self.measurement_treated_array_flat = np.zeros(
                        [self.measurement_treated_array_flat.size, 512, 512]) * np.nan
                    self.phase_std_array_flat = self.phase_std_array.flatten()
                    self.phase_meas_array_flat = self.phase_std_array.flatten()
                    self.time_stamps_array_flat = self.time_stamps_array.flatten()

                    self.measurement_running = 1
                    self.red_green_ratio_scan()
            else:
                message = "Are you sure you do not want to scan the phase for each ratio?"
                self.insert_message(message)

        elif status == "Only Red":
            self.f.write("# RedOnlyScan, " + self.ent_comment.get() + "\n")
            self.red_only_scan()
            self.insert_message(status)
        elif status == "Only Green":
            self.f.write("# GreenOnlyScan, " + self.ent_comment.get() + "\n")
            self.green_only_scan()
            self.insert_message(status)
        else:
            self.insert_message("Bruh")

        self.but_meas_all.config(fg='green')
        if self.measurement_running:
            self.measurement_running = 0
            self.measurement_counter = 0
            self.measurement_array = self.measurement_array_flat.reshape(
                [self.measurement_array.shape[0], self.measurement_array.shape[1], 512, 512])

            self.measurement_treated_array = self.measurement_treated_array_flat.reshape(
                [self.measurement_treated_array.shape[0], self.measurement_treated_array.shape[1], 512, 512])

            self.phase_std_array = self.phase_std_array_flat.reshape(self.phase_std_array.shape)
            self.phase_meas_array = self.phase_meas_array_flat.reshape(self.phase_std_array.shape)
            self.time_stamps_array = self.time_stamps_array_flat.reshape(self.phase_std_array.shape)

            self.save_h5()

        # Indicate completion by setting the event
        self.scan_is_done = True
        self.scan_is_done_threading.set()

    def measure(self):

        self.but_meas_scan.config(fg='red')
        if self.var_background.get() == 1:
            self.f.write(
                "# BACKGROUND PhaseScan, " + self.ent_comment.get() + "\n")
            im = self.take_image(int(self.ent_avgs.get()))
            self.save_im(im)
            self.plot_MCP(im)
        else:
            self.f.write(
                "# PhaseScan, " + self.ent_comment.get() + "\n")
            self.phase_scan()

        self.but_meas_scan.config(fg='green')

    def measure_simple(self):

        self.but_meas_simple.config(fg='red')


        if self.var_background.get() == 1:
            self.f.write("# BACKGROUND SingleImage, " + self.ent_comment.get() + '\n')
        else:
            self.f.write("# SingleImage, " + self.ent_comment.get() + '\n')

        im = self.take_image(int(self.ent_avgs.get()))
        self.save_im(im)
        self.plot_MCP(im)
        self.but_meas_simple.config(fg='green')


    def feedback(self):
        if self.ent_flat.get() != '':
            phi = float(self.ent_flat.get())
        else:
            phi = 0
        if self.var_pid_stage_enable.get() == 1:
            self.move_pid_piezo()
        else:
            phase_map = self.parent.phase_map_green + phi / 2 * bit_depth

            self.slm_lib.SLM_Disp_Open(int(self.parent.ent_scr_green.get()))
            self.slm_lib.SLM_Disp_Data(int(self.parent.ent_scr_green.get()), phase_map,
                                       slm_size[1], slm_size[0])

    def eval_spec(self):

        while True:
            time.sleep(0.01)

            # get raw trace
            timestamp, data = avs.get_spectrum(self.active_spec_handle)
            wavelength = avs.AVS_GetLambda(self.active_spec_handle)

            start = int(self.ent_area1x.get())
            stop = int(self.ent_area1y.get())
            self.trace = data[start:stop]
            self.wavelength = wavelength[start:stop]



            im_fft = np.fft.fft(self.trace)
            self.abs_im_fft = np.abs(im_fft)
            self.abs_im_fft = self.abs_im_fft / np.max(self.abs_im_fft)
            ind = round(float(self.ent_indexfft.get()))
            try:
                self.im_angl = np.angle(im_fft[ind])
            except:
                self.im_angl = 0
            self.lbl_angle.config(text=np.round(self.im_angl, 6))

            self.im_phase[:-1] = self.im_phase[1:]
            self.im_phase[-1] = self.im_angl

            # creating the phase vector


            # calculating standard deviation
            mean = np.mean(self.im_phase)
            std = np.sqrt(np.sum((self.im_phase - mean) ** 2) / (len(self.im_phase) - 1))
            if std < 0.12:
                self.lbl_std_val.config(text=np.round(std, 4), fg='green')
            else:
                self.lbl_std_val.config(text=np.round(std, 4), fg='red')

            if self.stop_acquire == 1:
                self.stop_acquire = 0
                break
            if self.meas_has_started:
                self.d_phase.append(self.im_angl)

            self.plot_fft_blit()

    def spc_img(self):
        """
        Starts spectrometer image acquisition and plots the acquired image.

        Returns:
        --------
        None
        """
        self.render_thread = threading.Thread(target=self.start_measure)
        self.render_thread.daemon = True
        self.render_thread.start()
        self.plot_phase()
        self.plot_voltage()

    def take_background(self):
        """
        Take a spectrum to be used as background for subsequent measurements.

        Raises
        ------
        AttributeError
            If no spectrum has been taken yet.

        Returns
        -------
        None
        """
        if not hasattr(self, 'trace'):
            raise AttributeError('Take a spectrum to be used as background')
        self.background = self.trace

    def auto_scale_spec_axis(self):
        """
        Automatically scale the spectrum axis.

        Raises
        ------
        AttributeError
            If no spectrum has been taken yet.

        Returns
        -------
        None
        """
        # this may take ~200ms, do not add it to the mainloop!
        if not hasattr(self, 'trace'):
            raise AttributeError('Take a spectrum before trying autoscale')
        self.ax1r.clear()
        self.trace_line, = self.ax1r.plot([])
        self.ax1r.set_xlim(np.min(self.wavelength), np.max(self.wavelength))
        self.ax1r.set_ylim(0, np.max(self.trace) * 1.2)
        self.ax1r.grid('both')
        self.figr.canvas.draw()
        self.img1r.draw()
        self.ax1r_blit = self.figr.canvas.copy_from_bbox(self.ax1r.bbox)

    def treat_image_test(self):
        im = self.calibration_image
        E_new, im_new = self.final_image_treatment(im)
        self.eaxis_correct = E_new
        self.plot_treated_image(im_new)

    def plot_treated_image(self, image):
        self.axMCP_treated.clear()
        pcm = self.axMCP_treated.pcolormesh(self.eaxis_correct, np.arange(0, 512), image.T)
        cbar = self.figrMCP_treated.colorbar(pcm, ax=self.axMCP_treated)
        # pcm.set_clim(vmin=0, vmax=1234)  # Set your desired limits
        self.axMCP_treated.set_xlim(20, 45)
        self.axMCP_treated.set_xlabel("Energy (eV)")
        self.axMCP_treated.set_ylabel(" y (px) ")
        self.axHarmonics_treated.clear()
        self.axHarmonics_treated.plot(self.eaxis_correct, np.sum(image, 1), color='k')
        for har in np.arange(15, 35):
            self.axHarmonics_treated.axvline(har * 1.2037300291262136, color='r', alpha=0.4)
        self.axHarmonics_treated.set_xlim(20, 45)
        self.axHarmonics_treated.set_xlabel("Energy (eV)")
        self.figrMCP_treated.tight_layout()
        self.imgMCP_treated.draw()
        cbar.remove()

       # with h5py.File('data.h5', 'w') as hf:
            #hf.create_dataset('e_axis', data=self.eaxis_correct)
            #hf.create_dataset('y_axis', data=np.arange(0, 512))
            #hf.create_dataset('image_T', data=image.T)
            #print('done')

    def plot_calibration_image(self, image):
        image = np.flipud(image)
        self.axMCP_calibrate.clear()
        self.axMCP_calibrate.imshow(image.T,cmap=custom_cmap)
        # self.axMCP.set_aspect('equal')

        self.axMCP_calibrate.set_xlabel("Energy equivalent")
        self.axMCP_calibrate.set_ylabel("Y (px)")
        self.axMCP_calibrate.set_xlim(0, 512)
        self.axMCP_calibrate.set_ylim(0, 512)

        self.axHarmonics_calibrate.clear()
        self.axHarmonics_calibrate.plot(np.arange(512), np.sum(image, 1))
        self.axHarmonics_calibrate.axhline(0, color='k', alpha=0.5)
        self.axHarmonics_calibrate.set_xlabel("Energy equivalent")
        self.axHarmonics_calibrate.set_ylabel("Counts (arb.u.)")

        self.axHarmonics_calibrate.set_xlim(0, 512)
        self.current_harmonics_profile_max_calibrate = np.max(np.sum(image, 1))
        self.current_harmonics_profile_min_calibrate = np.min(np.sum(image, 1))
        if self.var_fixyaxis.get() == 1:
            self.axHarmonics_calibrate.set_ylim(self.ymin_harmonics_calibrate, self.ymax_harmonics_calibrate)

        self.axHarmonics_calibrate.set_title("Sum: {}, Max: {}".format(int(np.sum(np.sum(image))), int(np.max(image))))
        self.figrMCP_calibrate.tight_layout()
        self.imgMCP_calibrate.draw()

    def plot_calibration_image_energy(self, profile, data, peaks, E_axis):
        # image = np.flipud(image)
        # self.axMCP_calibrate.clear()
        # self.axMCP_calibrate.imshow(image.T)
        # self.axMCP.set_aspect('equal')

        # self.axMCP_calibrate.set_xlabel("Energy equivalent")
        # self.axMCP_calibrate.set_ylabel("Y (px)")
        # self.axMCP_calibrate.set_xlim(0, 512)
        # self.axMCP_calibrate.set_ylim(0, 512)

        self.axHarmonics_calibrate_energy1.clear()
        self.axHarmonics_calibrate_energy1.plot(np.arange(512), profile, color='g')
        self.axHarmonics_calibrate_energy1.plot(np.arange(512), data, color='k')
        self.axHarmonics_calibrate_energy1.scatter(peaks, data[peaks], color='r')

        self.axHarmonics_calibrate_energy1.set_xlabel("Energy equivalent")
        self.axHarmonics_calibrate_energy1.set_ylabel("Counts (arb.u.)")

        self.axHarmonics_calibrate_energy1.set_xlim(0, 512)

        self.axHarmonics_calibrate_energy2.clear()
        self.axHarmonics_calibrate_energy2.plot(E_axis, profile, color='r')
        self.axHarmonics_calibrate_energy2.set_xlabel("Energy (eV)")
        self.axHarmonics_calibrate_energy2.set_ylabel("Counts (arb.u.)")
        # self.current_harmonics_profile_max_calibrate = np.max(np.sum(image, 1))
        # self.current_harmonics_profile_min_calibrate = np.min(np.sum(image, 1))
        # if self.var_fixyaxis.get() == 1:
        #    self.axHarmonics_calibrate.set_ylim(self.ymin_harmonics_calibrate, self.ymax_harmonics_calibrate)

        # self.axHarmonics_calibrate.set_title("Sum: {}, Max: {}".format(int(np.sum(np.sum(image))), int(np.max(image))))
        self.figrMCP_calibrate_energy.tight_layout()
        self.imgMCP_calibrate_energy.draw()

    def plot_MCP(self, mcpimage):
        """
        Plot the MCP image and harmonics plot.

        Parameters
        ----------
        mcpimage : array_like
            MCP image data.

        Returns
        -------
        None
        """

        # mcpimage = mcpimage - self.background

        if self.PIKE_cam is True:
            self.axMCP.clear()
            self.axMCP.imshow(mcpimage, vmin=0, vmax=2, extent=[0, 1600, 0, 1000])
            # self.axMCP.set_aspect('equal')

            self.axMCP.set_xlabel("X (px)")
            self.axMCP.set_ylabel("Y (px)")
            self.axMCP.set_xlim(0, 1600)
            self.axMCP.set_ylim(0, 1000)

            self.axHarmonics.clear()
            self.axHarmonics.plot(np.arange(1600), np.sum(mcpimage, 0))
            self.axHarmonics.set_xlabel("X (px)")
            self.axHarmonics.set_ylabel("Counts (arb.u.)")

            self.axHarmonics.set_xlim(0, 1600)

            self.figrMCP.tight_layout()
            self.imgMCP.draw()

        elif self.ANDOR_cam is True:
            self.axMCP.clear()
            pcm = self.axMCP.pcolormesh(np.arange(0, 512), np.arange(0, 512), mcpimage.T,cmap='turbo')
            cbar = self.figrMCP.colorbar(pcm, ax=self.axMCP)

            # self.axMCP.set_aspect('equal')

            self.axMCP.set_xlabel("X (px)")
            self.axMCP.set_ylabel("Y (px)")
            self.axMCP.set_xlim(0, 512)
            self.axMCP.set_ylim(0, 512)

            self.axHarmonics.clear()
            self.axHarmonics.plot(np.arange(512), np.sum(mcpimage, 1))
            self.axHarmonics.set_xlabel("X (px)")
            self.axHarmonics.set_ylabel("Counts (arb.u.)")

            self.axHarmonics.set_xlim(0, 512)
            self.current_harmonics_profile_max = np.max(np.sum(mcpimage, 1))
            self.current_harmonics_profile_min = np.min(np.sum(mcpimage, 1))
            if self.var_fixyaxis.get() == 1:
                self.axHarmonics.set_ylim(self.ymin_harmonics, self.ymax_harmonics)

            self.axHarmonics.set_title("Sum: {}, Max: {}".format(int(np.sum(np.sum(mcpimage))), int(np.max(mcpimage))))
            self.figrMCP.tight_layout()
            self.imgMCP.draw()
            cbar.remove()
        else:
            self.axMCP.clear()
            pcm = self.axMCP.pcolormesh(np.arange(0, 512), np.arange(0, 512), mcpimage.T)
            cbar = self.figrMCP.colorbar(pcm, ax=self.axMCP)

            # self.axMCP.set_aspect('equal')

            self.axMCP.set_xlabel("X (px)")
            self.axMCP.set_ylabel("Y (px)")
            self.axMCP.set_xlim(0, 512)
            self.axMCP.set_ylim(0, 512)
            self.axHarmonics.clear()
            self.axHarmonics.plot(np.arange(512), np.sum(mcpimage, 1))
            self.axHarmonics.set_xlabel("X (px)")
            self.axHarmonics.set_ylabel("Counts (arb.u.)")

            self.axHarmonics.set_xlim(0, 512)
            self.current_harmonics_profile_max = np.max(np.sum(mcpimage, 1))
            self.current_harmonics_profile_min = np.min(np.sum(mcpimage, 1))
            if self.var_fixyaxis.get() == 1:
                self.axHarmonics.set_ylim(self.ymin_harmonics, self.ymax_harmonics)

            self.axHarmonics.set_title("Sum: {}, Max: {}".format(int(np.sum(np.sum(mcpimage))), int(np.max(mcpimage))))
            self.figrMCP.tight_layout()
            self.imgMCP.draw()
            cbar.remove()

            # print("Test")

            if self.var_show_treated.get() == 1:
                #    print("This was pressed!!!")
                Etemp, treated = self.final_image_treatment(mcpimage)
                self.plot_treated_image(treated)

            #   print(np.sum(treated))
            # self.axMCP_treated.clear()
            # self.axMCP_treated.pcolormesh(self.eaxis_correct,np.arange(0,512),treated.T)
            # self.axHarmonics_treated.clear()
            # self.axHarmonics_treated.plot(self.eaxis_correct,np.sum(treated,1))
            # self.figrMCP_treated.tight_layout()
            # self.imgMCP_treated.draw()

    def plot_fft(self):
        """
        Plot the Fourier transform of the spectrum.

        Returns
        -------
        None
        """
        # find maximum in the fourier trace
        maxindex = np.where(self.abs_im_fft == np.max(self.abs_im_fft[3:50]))[0][0]

        self.ax1r.clear()
        self.ax1r.plot(self.wavelength, self.trace)
        self.ax2r.clear()
        self.ax2r.plot(self.abs_im_fft)
        self.ax2r.plot(maxindex, self.abs_im_fft[maxindex] * 1.2, 'v')
        self.ax2r.text(maxindex - 1, self.abs_im_fft[maxindex] * 1.5, str(maxindex))
        self.ax2r.set_xlim(0, 50)
        self.img1r.draw()

    def plot_fft_blit(self):
        """
        Plot the Fourier transform of the spectrum with blitting.

        Returns
        -------
        None
        """
        # find maximum in the fourier trace
        maxindex = np.where(self.abs_im_fft == np.max(self.abs_im_fft[5:50]))[0][0]

        self.figr.canvas.restore_region(self.ax1r_blit)
        self.figr.canvas.restore_region(self.ax2r_blit)
        self.trace_line.set_data(self.wavelength, self.trace)
        self.trace_line.set_color('green')
        self.ax1r.draw_artist(self.trace_line)
        self.fourier_line.set_data(np.arange(50), self.abs_im_fft[:50])
        self.fourier_line.set_color('green')
        self.ax1r.draw_artist(self.fourier_line)
        self.fourier_indicator.set_data([maxindex], [self.abs_im_fft[maxindex] + 0.05])
        self.ax1r.draw_artist(self.fourier_indicator)
        self.fourier_text.set_text(str(maxindex))
        self.fourier_text.set_position((maxindex - 1, self.abs_im_fft[maxindex] + 0.09))
        self.ax1r.draw_artist(self.fourier_text)
        self.figr.canvas.blit()
        self.figr.canvas.flush_events()

    def plot_phase(self):
        """
        Plot the phase image using blitting.

        Updates the plot element with the new phase data, blits the canvas,
        and uses recursion to call itself after 50 milliseconds.

        Returns
        -------
        None
        """
        self.figp.canvas.restore_region(self.ax1p_blit)
        self.phase_line.set_data(np.arange(1000), self.im_phase)
        self.V_line.set_color('blue')
        self.ax1p.draw_artist(self.phase_line)
        self.figp.canvas.blit(self.ax1p.bbox)
        self.figp.canvas.flush_events()
        self.win.after(50, self.plot_phase)


    def plot_voltage(self):
        """
        Plot the phase image using blitting.

        Updates the plot element with the new phase data, blits the canvas,
        and uses recursion to call itself after 50 milliseconds.

        Returns
        -------
        None
        """
        mean_value = np.mean(self.im_voltage[-100:])
        lower_limit =  -0.25 + mean_value
        upper_limit =  0.25 + mean_value
        self.ax1V.set_ylim(lower_limit, upper_limit)


        self.figV.canvas.restore_region(self.ax1V_blit)
        self.V_line.set_data(np.arange(1000), self.im_voltage)
        self.V_line.set_color('red')
        self.ax1V.draw_artist(self.V_line)
        self.figV.canvas.draw()
        self.figV.canvas.blit(self.ax1V.bbox)
        self.figV.canvas.flush_events()
        self.win.after(50, self.plot_voltage)

    def spec_activate(self):
        """
        Activate the spectrometer.

        Initializes the spectrometer interface if necessary, activates the
        spectrometer handle, and disables the state of the `ent_spc_ind` widget.

        Returns
        -------
        None
        """
        try:
            if not self.spec_interface_initialized:
                avs.AVS_Init()
            if self.active_spec_handle is None:
                speclist = avs.AVS_GetList()
                message = str(len(speclist)) + ' spectrometer(s) found.'
                self.insert_message(message)
                self.active_spec_handle = avs.AVS_Activate(speclist[0])
                self.ent_spc_ind.config(state='disabled')
        except:
            message = 'There was no spectrometer found!'
            self.insert_message(message)

    def spec_deactivate(self):
        """
        Deactivate the spectrometer.

        Stops the spectrometer measurement, deactivates the spectrometer handle,
        and enables the `ent_spc_ind` widget.

        Returns
        -------
        None
        """
        if self.active_spec_handle is not None:
            avs.AVS_StopMeasure(self.active_spec_handle)
            avs.AVS_Deactivate(self.active_spec_handle)
            self.ent_spc_ind.config(state='normal')
            self.active_spec_handle = None

    def start_measure(self):
        """
        Start the spectrometer measurement.

        Activates the spectrometer, sets the measurement parameters, and starts
        the measurement. Calls the `eval_spec()` method to evaluate the measured data.

        Returns
        -------
        None
        """
        try:
            self.spec_activate()
            int_time = float(self.ent_spc_exp.get())
            no_avg = int(self.ent_spc_avg.get())
            avs.set_measure_params(self.active_spec_handle, int_time, no_avg)
            avs.AVS_Measure(self.active_spec_handle)
            self.eval_spec()
        except:
            message = 'No spectrometer found'
            self.insert_message(message )

    def stop_measure(self):
        """
        Stop the spectrometer measurement.

        Returns
        -------
        None
        """
        if self.active_spec_handle is not None:
            avs.AVS_StopMeasure(self.active_spec_handle)

    def fast_scan(self):
        """
        Perform a fast scan.

        Creates a linspace of 60 values between 0 and 2*pi, sets the `phi_ind` to 0,
        and calls the `fast_scan_loop()` method.

        Returns
        -------
        None
        """
        self.phis = np.linspace(0, 2 * np.pi, 60)
        self.phi_ind = 0
        self.fast_scan_loop()

    def fast_scan_loop(self):
        """
        Perform a fast scan loop.

        Sets the `strvar_setp` to the current `phi_ind` value, calls the
        `set_setpoint()` method, increments the `phi_ind` by 1, and calls itself
        after 100 milliseconds if the `phi_ind` is less than 60.

        Returns
        -------
        None
        """
        self.strvar_setp.set(self.phis[self.phi_ind])
        self.set_setpoint()
        self.phi_ind = self.phi_ind + 1
        if self.phi_ind < 60:
            self.win.after(100, self.fast_scan_loop)

    def set_setpoint(self):
        """
        Set the set point.

        Sets the `set_point` attribute to the float value of the `ent_setp` widget.

        Returns
        -------
        None
        """
        self.set_point = float(self.ent_setp.get())

    def set_pid_val(self):
        """
        Set the PID values.

        Sets the `Kp` and `Ki` attributes of the `pid` object to the float values
        of the `ent_pidp` and `ent_pidi` widgets, respectively.

        Returns
        -------
        None
        """
        self.pid.Kp = float(self.ent_pidp.get())
        self.pid.Ki = float(self.ent_pidi.get())
        self.pid.Kd = float(self.ent_pidd.get())

        # print(self.pid.tunings)

    def pid_strt(self):
        """
        Start the PID control loop.

        Sets the set point and PID values, and then enters a loop that calculates
        the correction based on the difference between the image angle and the set point.
        Sets the `strvar_flat` to the correction, calls the `feedback()` method, and
        breaks the loop if `stop_pid` is `True`.

        Returns
        -------
        None
        """
        self.home_pid_piezo()
        self.set_setpoint()
        self.set_pid_val()
        self.pid.sample_time = float(self.ent_spc_exp.get())*1e-3
        self.pid.output_limits = (-10, 10)

        while True:
            #time.sleep(float(self.ent_spc_exp.get())*1e-3)
            # deviation_slm = (self.im_angl - self.set_point + np.pi) % (2 * np.pi) - np.pi
            # correction_slm = self.pid(deviation_slm)
            # self.strvar_flat.set(str(correction_slm))

            deviation = (self.im_angl - self.set_point + np.pi) % (2 * np.pi) - np.pi  # -pi to pi
            deviation_in_V = deviation / (2 * np.pi) * 0.38

            correction_piezo = self.pid(deviation_in_V)
            self.strvar_pid_stage_set_position.set(str(correction_piezo))

            self.im_voltage[:-1] = self.im_voltage[1:]
            self.im_voltage[-1] = correction_piezo

            self.feedback()

            # print(self.pid.components)
            if self.stop_pid:
                correction_piezo = self.pid(0)
                self.strvar_pid_stage_set_position.set(str(correction_piezo))
                break

    def enbl_pid(self):
        """
        Enable the PID control loop.

        Sets the `stop_pid` to `False` and starts a new thread running the `pid_strt()` method.

        Returns
        -------
        None
        """
        self.stop_pid = False
        self.pid_thread = threading.Thread(target=self.pid_strt)
        self.pid_thread.daemon = True
        self.pid_thread.start()

    def pid_stop(self):
        """
        Stop the PID control loop.

        Sets the `stop_pid` to `True`.

        Returns
        -------
        None
        """
        self.stop_pid = True

    def enable_save_spc_fringes(self):
        """
        Run the thread to save the spectrometer data (spectral fringe and phase stability) into a .txt file.

        Returns
        -------
        None
        """
        self.save_fringe_thread = threading.Thread(target=self.save_spc_fringes)
        self.save_fringe_thread.daemon = True
        self.save_fringe_thread.start()

    def save_spc_fringes(self):
        """
        Save the spectral fringes into a .txt file.

        Returns
        -------
        None
        """
        self.but_spc_export_fringes.config(fg='red')
        file_path = asksaveasfile(initialfile=f'fringes_{date.today()}', defaultextension=".txt",
                                  filetypes=[("Text Files", "*.txt")])
        fringes = np.array([self.wavelength, self.trace]).T

        if file_path is not None:
            header = "Wavelength (nm)\tCounts (arb.u)"
            np.savetxt(file_path.name, fringes, delimiter="\t", header=header)
            file_path.close()
            message = f"Spectral fringes saved to {file_path.name}"
            self.insert_message(message )
            self.but_spc_export_fringes.config(fg='green')

    def enable_save_spc_phase_stability(self):
        """
        Run the thread to save the spectrometer data (spectral fringe and phase stability) into a .txt file.

        Returns
        -------
        None
        """
        self.save_phase_stability_thread = threading.Thread(target=self.save_spc_phase_stability)
        self.save_phase_stability_thread.daemon = True
        self.save_phase_stability_thread.start()

    def save_spc_phase_stability(self):
        """
        Save the spectral fringes into a .txt file.

        Returns
        -------
        None
        """
        self.but_spc_export_phase_stab.config(fg='red')
        file_path = asksaveasfile(initialfile=f'phase_stability_{date.today()}', defaultextension=".txt",
                                  filetypes=[("Text Files", "*.txt")])
        phase_stability = np.array([np.arange(1000), self.im_phase]).T

        if file_path is not None:
            header = "Time (arb.u)\tTwo-color phase (rad)"
            np.savetxt(file_path.name, phase_stability, delimiter="\t", header=header)
            file_path.close()
            message = f"Phase stability saved to {file_path.name}"
            self.insert_message(message )
        self.but_spc_export_phase_stab.config(fg='green')

    def on_close(self):
        """
        Close the program.

        Closes the figures, cleans up the APT module, deactivates the spectrometer,
        and destroys the window.

        Returns
        -------
        None
        """
        self.f.close()
        # self.g.close()
        plt.close(self.figr)
        plt.close(self.figp)
        self.disable_motors()
        if self.cam is not None:
            self.cam.close()
        self.spec_deactivate()
        avs.AVS_Done()
        self.win.destroy()
        self.parent.feedback_win = None
        print('Feedbacker closed')
