import os
import threading
import time
import tkinter as tk
from tkinter.filedialog import asksaveasfile, askopenfilename
from tkinter import ttk
from collections import deque
from datetime import date
import datetime
import h5py

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from simple_pid import PID

import drivers.avaspec_driver._avs_py as avs
from drivers import gxipy_driver as gx
from drivers.thorlabs_apt_driver import core as apt
from drivers.vimba_driver import *
import drivers.santec_driver._slm_py as slm
from ressources.slm_infos import slm_size, bit_depth
from stages_and_sensors import waveplate_calibrator as cal
from pylablib.devices import Andor
import pylablib as pll

import views.focus_diagnostic as dh
from ressources.slm_infos import slm_size, bit_depth, chip_width, chip_height
import model.helpers as help


class Feedbacker(object):
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
        self.WPDummy = None
        self.Delay = None

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

        self.daheng_active = False
        self.daheng_camera = None
        self.daheng_is_live = False
        self.current_daheng_image = None
        self.daheng_zoom = None

        # This opens the autologfile from the start! closes it on close command
        self.autolog = 'C:/data/' + str(date.today()) + '/' + str(date.today()) + '-' + 'auto-log.txt'
        self.f = open(self.autolog, "a+")

        self.autolog_images = 'C:/data/' + str(date.today()) + '/' + str(date.today()) + '-' + 'auto-log-images.txt'
        self.g = open(self.autolog_images, "a+")

        self.autolog_slm_param_scan = 'C:/data/' + str(date.today()) + '/' + str(
            date.today()) + '-' + 'auto-log-slm_param_scan.txt'
        self.slmps = open(self.autolog_slm_param_scan, "a+")

        # creating frames

        frm_mid = ttk.Frame(self.win)
        frm_bot = ttk.Frame(self.win)
        frm_scans = ttk.Frame(self.win)
        frm_mcp_all = ttk.Frame(self.win)

        # Spectrometer
        frm_plt = ttk.LabelFrame(self.win, text='Spectrometer')

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
        frm_slm_param_scan = ttk.Frame(frm_scans)
        self.frm_notebook_scans.add(frm_wp_scans, text="Power scan")
        self.frm_notebook_scans.add(frm_phase_scan, text="Two-color phase scan")
        self.frm_notebook_scans.add(frm_slm_param_scan, text="SLM parameters scan")

        self.frm_notebook_waveplate = ttk.Notebook(frm_scans)
        frm_stage = ttk.Frame(frm_scans)
        frm_wp_power_cal = ttk.Frame(frm_scans)
        self.frm_notebook_waveplate.add(frm_stage, text="Waveplate control")
        self.frm_notebook_waveplate.add(frm_wp_power_cal, text="Power calibration")

        frm_daheng_camera = ttk.LabelFrame(self.win, text='Daheng Camera')
        self.frm_daheng_camera_settings = ttk.LabelFrame(frm_daheng_camera, text='Settings')
        self.frm_daheng_camera_image = ttk.Frame(frm_daheng_camera)

        self.frm_notebook_mcp = ttk.Notebook(frm_mcp_all)
        frm_mcp_image = ttk.Frame(frm_mcp_all)
        frm_mcp_calibrate = ttk.Frame(frm_mcp_all)
        frm_mcp_calibrate_energy = ttk.Frame(frm_mcp_all)
        frm_mcp_treated = ttk.Frame(frm_mcp_all)
        self.frm_notebook_mcp.add(frm_mcp_image, text='MCP raw')
        self.frm_notebook_mcp.add(frm_mcp_calibrate, text='Calibrate Spatial')
        self.frm_notebook_mcp.add(frm_mcp_calibrate_energy, text='Calibrate Energy')
        self.frm_notebook_mcp.add(frm_mcp_treated, text='MCP treated')

        frm_mcp_calibrate_options = ttk.LabelFrame(frm_mcp_calibrate, text='Calibration Options')
        frm_mcp_calibrate_image = ttk.LabelFrame(frm_mcp_calibrate, text='Calibration Image')
        frm_mcp_calibrate_options_energy = ttk.LabelFrame(frm_mcp_calibrate_energy, text='Calibration Options')
        frm_mcp_calibrate_image_energy = ttk.LabelFrame(frm_mcp_calibrate_energy, text='Calibration Image')
        frm_mcp_treated_options = ttk.LabelFrame(frm_mcp_treated, text='Options')
        frm_mcp_treated_image = ttk.LabelFrame(frm_mcp_treated, text='Images')

        # frm_mcp_image = ttk.LabelFrame(self.win, text='MCP')
        frm_mcp_options = ttk.LabelFrame(self.win, text='MCP options')

        vcmd = (self.win.register(self.parent.callback))

        self.but_hide_frm_daheng = ttk.Button(frm_daheng_camera, text="Hide/Show", command=self.hide_frm_daheng)
        self.but_hide_frm_daheng.grid(row=0, column=0)

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
        lbl_std = tk.Label(frm_spc_plt_set, text='sigma:', width=6)
        self.lbl_std_val = tk.Label(frm_spc_plt_set, text='None', width=6)
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
        self.strvar_pidp = tk.StringVar(self.win, '-0.17')
        self.ent_pidp = tk.Entry(
            frm_spc_pid, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_pidp)
        lbl_pidi = tk.Label(frm_spc_pid, text='I-value:')
        self.strvar_pidi = tk.StringVar(self.win, '-1.5')
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
        but_pid_setp = tk.Button(frm_spc_pid, text='Setpoint', command=self.set_setpoint)
        but_pid_enbl = tk.Button(frm_spc_pid, text='Start PID', command=self.enbl_pid)
        but_pid_stop = tk.Button(frm_spc_pid, text='Stop PID', command=self.pid_stop)
        but_pid_setk = tk.Button(frm_spc_pid, text='Set PID values', command=self.set_pid_val)

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

        # Daheng camera

        self.var_camera_1 = tk.IntVar()
        self.cb_camera_1 = tk.Checkbutton(self.frm_daheng_camera_settings, text='Nozzle 1', variable=self.var_camera_1,
                                          onvalue=1,
                                          offvalue=0,
                                          command=None)

        self.var_camera_3 = tk.IntVar()
        self.cb_camera_3 = tk.Checkbutton(self.frm_daheng_camera_settings, text='Nozzle 2', variable=self.var_camera_3,
                                          onvalue=1,
                                          offvalue=0,
                                          command=None)

        self.var_camera_2 = tk.IntVar()
        self.cb_camera_2 = tk.Checkbutton(self.frm_daheng_camera_settings, text='Focus', variable=self.var_camera_2,
                                          onvalue=1,
                                          offvalue=0,
                                          command=None)

        # MEASUREMENT FRAME
        self.but_meas_simple = tk.Button(frm_measure, text='Single Image', command=self.enabl_mcp_simple)
        self.but_meas_scan = tk.Button(frm_measure, text='Phase Scan', command=self.enabl_mcp)
        self.but_meas_all = tk.Button(frm_measure, text='Measurement Series', command=self.enabl_mcp_all)
        self.but_meas_slm_param = tk.Button(frm_measure, text='SLM param Scan', command=self.enabl_scan_slm_param)
        self.but_view_live = tk.Button(frm_measure, text='Live View!', command=self.enabl_mcp_live)

        self.var_split_scan = tk.IntVar()
        self.cb_split_scan = tk.Checkbutton(frm_measure, text='Split Scan', variable=self.var_split_scan, onvalue=1,
                                            offvalue=0,
                                            command=None)

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

        lbl_avgs = tk.Label(frm_measure, text='Avgs:')
        self.strvar_avgs = tk.StringVar(self.win, '1')
        self.ent_avgs = tk.Entry(
            frm_measure, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_avgs)

        lbl_mcp_cam_choice = tk.Label(frm_measure, text='MCP Camera selected :')
        self.strvar_mcp_cam_choice = tk.StringVar(self.win, 'Pike camera')
        self.cbox_mcp_cam_choice = ttk.Combobox(frm_measure, textvariable=self.strvar_mcp_cam_choice)
        self.cbox_mcp_cam_choice['values'] = ('Pike Camera', 'Andor Camera')

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
        self.strvar_mcp = tk.StringVar(self.win, '-1550')
        self.ent_mcp = tk.Entry(
            frm_measure, width=25, validate='none',
            textvariable=self.strvar_mcp)

        lbl_comment = tk.Label(frm_measure, text='comment:')
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

        lbl_WPDummy = tk.Label(frm_stage, text='Focus Stage:')
        self.strvar_WPDummy_is = tk.StringVar(self.win, '')
        self.ent_WPDummy_is = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPDummy_is)
        self.strvar_WPDummy_should = tk.StringVar(self.win, '')
        self.ent_WPDummy_should = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPDummy_should)
        # self.strvar_WPDummy_Nr = tk.StringVar(self.win, '83837725')
        self.strvar_WPDummy_Nr = tk.StringVar(self.win, '83837725')
        self.ent_WPDummy_Nr = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPDummy_Nr)
        self.but_WPDummy_Ini = tk.Button(frm_stage, text='Init', command=self.init_WPDummy)
        self.but_WPDummy_Home = tk.Button(frm_stage, text='Home', command=self.home_WPDummy)
        self.but_WPDummy_Read = tk.Button(frm_stage, text='Read', command=self.read_WPDummy)
        self.but_WPDummy_Move = tk.Button(frm_stage, text='Move', command=self.move_WPDummy)

        lbl_Delay = tk.Label(frm_stage, text='Delay:')
        self.strvar_Delay_is = tk.StringVar(self.win, '')
        self.ent_Delay_is = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_Delay_is)
        self.strvar_Delay_should = tk.StringVar(self.win, '')
        self.ent_Delay_should = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_Delay_should)
        self.strvar_Delay_Nr = tk.StringVar(self.win, '83837719')
        self.ent_Delay_Nr = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_Delay_Nr)
        # scan parameters
        self.strvar_Delay_from = tk.StringVar(self.win, '6.40')
        self.ent_Delay_from = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_Delay_from)
        self.strvar_Delay_to = tk.StringVar(self.win, '6.45')
        self.ent_Delay_to = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_Delay_to)
        self.strvar_Delay_steps = tk.StringVar(self.win, '10')
        self.ent_Delay_steps = tk.Entry(
            frm_stage, width=10, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_Delay_steps)
        self.var_delayscan = tk.IntVar()
        self.cb_delayscan = tk.Checkbutton(frm_stage, text='Scan', variable=self.var_delayscan, onvalue=1, offvalue=0,
                                           command=None)
        self.but_Delay_Ini = tk.Button(frm_stage, text='Init', command=self.init_Delay)
        self.but_Delay_Home = tk.Button(frm_stage, text='Home', command=self.home_Delay)
        self.but_Delay_Read = tk.Button(frm_stage, text='Read', command=self.read_Delay)
        self.but_Delay_Move = tk.Button(frm_stage, text='Move', command=self.move_Delay)

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

        self.strvar_red_power = tk.StringVar(self.win, '4.34')

        self.ent_red_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_red_power)

        lbl_red_phase = tk.Label(frm_wp_power_cal, text='Red offset phase (deg):')
        self.strvar_red_phase = tk.StringVar(self.win, '-27.76')
        self.ent_red_phase = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_red_phase)

        lbl_red_current_power = tk.Label(frm_wp_power_cal, text='Red current power (W):')
        self.strvar_red_current_power = tk.StringVar(self.win, '')
        self.ent_red_current_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_red_current_power)

        lbl_green_power = tk.Label(frm_wp_power_cal, text='Green max power (mW):')

        self.strvar_green_power = tk.StringVar(self.win, '307.8')

        self.ent_green_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_green_power)

        lbl_green_phase = tk.Label(frm_wp_power_cal, text='Green offset phase (deg):')
        self.strvar_green_phase = tk.StringVar(self.win, '42.08')
        self.ent_green_phase = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_green_phase)

        lbl_green_current_power = tk.Label(frm_wp_power_cal, text='Green current power (mW):')
        self.strvar_green_current_power = tk.StringVar(self.win, '')
        self.ent_green_current_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_green_current_power)

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

        self.frm_notebook_param_spc.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')

        frm_plt.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        # frm_mcp_image.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')
        frm_mcp_all.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')
        frm_mcp_calibrate_options.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_calibrate_image.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_calibrate_options_energy.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_calibrate_image_energy.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_treated_options.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        frm_mcp_treated_image.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.frm_notebook_mcp.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')

        frm_mcp_options.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        frm_scans.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        frm_measure.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.frm_notebook_waveplate.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.frm_notebook_scans.grid(row=3, column=0, padx=2, pady=2, sticky='nsew')

        frm_daheng_camera.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.frm_daheng_camera_settings.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.frm_daheng_camera_image.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')

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
        lbl_std.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        self.lbl_std_val.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')

        # setting up buttons frm_bot
        but_exit.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        but_feedback.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')

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
        but_pid_enbl.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        but_pid_stop.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')

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

        self.cb_background.grid(row=0, column=3, padx=2, pady=2, sticky='nsew')
        self.cb_saveh5.grid(row=0, column=4, padx=2, pady=2, sticky='nsew')
        self.cb_export_treated_image.grid(row=1, column=4, padx=2, pady=2, sticky='nsew')

        self.but_meas_all.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')
        self.but_meas_slm_param.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')
        self.but_meas_scan.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')
        self.but_meas_simple.grid(row=3, column=2, padx=2, pady=2, sticky='nsew')
        self.but_view_live.grid(row=4, column=2, padx=2, pady=2, sticky='nsew')
        self.cb_split_scan.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')

        # setting up frm_phase_scan
        lbl_from.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        lbl_to.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        lbl_steps.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_from.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_to.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_steps.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')
        self.cb_phasescan.grid(row=5, column=1, padx=2, pady=2, sticky='nsew')

        # setting up frm_stage
        lbl_Stage.grid(row=0, column=1, pady=2, sticky='nsew')
        lbl_Nr.grid(row=0, column=2, pady=2, sticky='nsew')
        lbl_is.grid(row=0, column=3, pady=2, sticky='nsew')
        lbl_should.grid(row=0, column=4, pady=2, sticky='nsew')

        lbl_WPR.grid(row=1, column=1, pady=2, sticky='nsew')
        lbl_WPG.grid(row=2, column=1, pady=2, sticky='nsew')
        lbl_Delay.grid(row=3, column=1, pady=2, sticky='nsew')
        lbl_WPDummy.grid(row=4, column=1, pady=2, sticky='nsew')

        self.ent_WPR_Nr.grid(row=1, column=2, pady=2, sticky='nsew')
        self.ent_WPG_Nr.grid(row=2, column=2, pady=2, sticky='nsew')
        self.ent_Delay_Nr.grid(row=3, column=2, pady=2, sticky='nsew')

        self.ent_WPDummy_Nr.grid(row=4, column=2, pady=2, sticky='nsew')

        self.ent_WPR_is.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_WPG_is.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_Delay_is.grid(row=3, column=3, padx=2, pady=2, sticky='nsew')

        self.ent_WPDummy_is.grid(row=4, column=3, padx=2, pady=2, sticky='nsew')

        self.ent_WPR_should.grid(row=1, column=4, padx=2, pady=2, sticky='nsew')
        self.ent_WPG_should.grid(row=2, column=4, padx=2, pady=2, sticky='nsew')
        self.ent_Delay_should.grid(row=3, column=4, padx=2, pady=2, sticky='nsew')

        self.ent_WPDummy_should.grid(row=4, column=4, padx=2, pady=2, sticky='nsew')

        self.but_WPR_Ini.grid(row=1, column=5, padx=2, pady=2, sticky='nsew')
        self.but_WPR_Home.grid(row=1, column=6, padx=2, pady=2, sticky='nsew')
        self.but_WPR_Read.grid(row=1, column=7, padx=2, pady=2, sticky='nsew')
        self.but_WPR_Move.grid(row=1, column=8, padx=2, pady=2, sticky='nsew')

        self.but_WPG_Ini.grid(row=2, column=5, padx=2, pady=2, sticky='nsew')
        self.but_WPG_Home.grid(row=2, column=6, padx=2, pady=2, sticky='nsew')
        self.but_WPG_Read.grid(row=2, column=7, padx=2, pady=2, sticky='nsew')
        self.but_WPG_Move.grid(row=2, column=8, padx=2, pady=2, sticky='nsew')

        self.but_WPDummy_Ini.grid(row=4, column=5, padx=2, pady=2, sticky='nsew')
        self.but_WPDummy_Home.grid(row=4, column=6, padx=2, pady=2, sticky='nsew')
        self.but_WPDummy_Read.grid(row=4, column=7, padx=2, pady=2, sticky='nsew')
        self.but_WPDummy_Move.grid(row=4, column=8, padx=2, pady=2, sticky='nsew')

        self.but_Delay_Ini.grid(row=3, column=5, padx=2, pady=2, sticky='nsew')
        self.but_Delay_Home.grid(row=3, column=6, padx=2, pady=2, sticky='nsew')
        self.but_Delay_Read.grid(row=3, column=7, padx=2, pady=2, sticky='nsew')
        self.but_Delay_Move.grid(row=3, column=8, padx=2, pady=2, sticky='nsew')

        self.cb_wprpower.grid(row=1, column=9, padx=2, pady=2, sticky='nsew')
        self.cb_wpgpower.grid(row=2, column=9, padx=2, pady=2, sticky='nsew')

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

        # setting up Daheng stuff
        self.but_initialize_daheng = tk.Button(self.frm_daheng_camera_settings, text="Initialize",
                                               command=self.initialize_daheng)
        self.but_initialize_daheng.grid(row=0, column=0)
        self.but_close_daheng = tk.Button(self.frm_daheng_camera_settings, text="Disconnect", command=self.close_daheng)
        self.but_close_daheng.grid(row=0, column=1)
        self.var_index_camera = tk.StringVar(self.win, value="1")
        self.ent_default_cam_index = tk.Entry(self.frm_daheng_camera_settings, textvariable=self.var_index_camera,
                                              width=3,
                                              validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_default_cam_index.grid(row=1, column=0)

        lbl_daheng_exposure = tk.Label(self.frm_daheng_camera_settings, text="Exp:")
        lbl_daheng_exposure.grid(row=2, column=0)
        lbl_daheng_gain = tk.Label(self.frm_daheng_camera_settings, text="Gain:")
        lbl_daheng_gain.grid(row=3, column=0)
        lbl_daheng_avg = tk.Label(self.frm_daheng_camera_settings, text="Avg:")
        lbl_daheng_avg.grid(row=4, column=0)

        self.var_daheng_exposure = tk.StringVar(self.win, value="100000")
        self.ent_daheng_exposure = tk.Entry(self.frm_daheng_camera_settings, textvariable=self.var_daheng_exposure,
                                            width=10,
                                            validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_daheng_exposure.grid(row=2, column=1)
        self.ent_daheng_exposure.bind("<KeyRelease>", self.exp_gain_value_changed)

        self.var_daheng_gain = tk.StringVar(self.win, value="0")
        self.ent_daheng_gain = tk.Entry(self.frm_daheng_camera_settings, textvariable=self.var_daheng_gain, width=10,
                                        validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_daheng_gain.grid(row=3, column=1)
        self.ent_daheng_gain.bind("<KeyRelease>", self.exp_gain_value_changed)

        self.var_daheng_avg = tk.StringVar(self.win, value="1")
        self.ent_daheng_avg = tk.Entry(self.frm_daheng_camera_settings, textvariable=self.var_daheng_avg, width=10,
                                       validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_daheng_avg.grid(row=4, column=1)

        self.but_live_daheng = tk.Button(self.frm_daheng_camera_settings, text="Live",
                                         command=self.live_daheng_thread)
        self.but_live_daheng.grid(row=5, column=0)

        self.but_single_daheng = tk.Button(self.frm_daheng_camera_settings, text="Single",
                                           command=self.single_daheng_thread)
        self.but_single_daheng.grid(row=5, column=1)

        self.but_scan_daheng = tk.Button(self.frm_daheng_camera_settings, text="Stage Scan!",
                                         command=self.scan_daheng_thread)
        self.but_scan_daheng.grid(row=5, column=2)
        self.but_com_daheng = tk.Button(self.frm_daheng_camera_settings, text="Zoom in COM",
                                        command=self.zoom_around_com)
        self.but_com_daheng.grid(row=6, column=0)

        self.but_reset_daheng = tk.Button(self.frm_daheng_camera_settings, text="Reset Zoom",
                                          command=self.reset_zoom)
        self.but_reset_daheng.grid(row=6, column=1)

        self.var_daheng_radius = tk.StringVar(self.win, value="200")
        self.ent_daheng_radius = tk.Entry(self.frm_daheng_camera_settings, textvariable=self.var_daheng_radius, width=8,
                                          validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_daheng_radius.grid(row=7, column=0)

        lbl_daheng_stage_from = tk.Label(self.frm_daheng_camera_settings, text="From")
        lbl_daheng_stage_to = tk.Label(self.frm_daheng_camera_settings, text="To")
        lbl_daheng_stage_steps = tk.Label(self.frm_daheng_camera_settings, text="Steps")
        lbl_daheng_stage_from.grid(row=8, column=0)
        lbl_daheng_stage_to.grid(row=8, column=1)
        lbl_daheng_stage_steps.grid(row=8, column=2)

        self.var_daheng_stage_from = tk.StringVar(self.win, value="6")
        self.ent_daheng_stage_from = tk.Entry(self.frm_daheng_camera_settings, textvariable=self.var_daheng_stage_from,
                                              width=5,
                                              validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_daheng_stage_from.grid(row=9, column=0)

        self.var_daheng_stage_to = tk.StringVar(self.win, value="12")
        self.ent_daheng_stage_to = tk.Entry(self.frm_daheng_camera_settings, textvariable=self.var_daheng_stage_to,
                                            width=5,
                                            validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_daheng_stage_to.grid(row=9, column=1)

        self.var_daheng_stage_steps = tk.StringVar(self.win, value="10")
        self.ent_daheng_stage_steps = tk.Entry(self.frm_daheng_camera_settings,
                                               textvariable=self.var_daheng_stage_steps,
                                               width=5,
                                               validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_daheng_stage_steps.grid(row=9, column=2)

        lbl_slm_param_scan_info = tk.Label(frm_slm_param_scan, text="Choose your fighter!")
        lbl_slm_param_scan_info.grid(row=0, column=0)
        lbl_slm_param_scan_from = tk.Label(frm_slm_param_scan, text='from:')
        lbl_slm_param_scan_from.grid(row=0, column=2)
        lbl_slm_param_scan_to = tk.Label(frm_slm_param_scan, text='to:')
        lbl_slm_param_scan_to.grid(row=0, column=3)
        lbl_slm_param_scan_steps = tk.Label(frm_slm_param_scan, text='steps:')
        lbl_slm_param_scan_steps.grid(row=0, column=4)

        self.var_scan_slm_param_option = tk.StringVar(self.win, "Nothing")
        self.rb_supergaussian_scan = tk.Radiobutton(frm_slm_param_scan, variable=self.var_scan_slm_param_option,
                                                    value="Supergaussian",
                                                    text="Supergaussian")
        self.rb_supergaussian_scan.grid(row=1, column=0)

        self.var_supergaussian_from = tk.StringVar(self.win, value="0")
        self.ent_supergaussian_from = tk.Entry(frm_slm_param_scan,
                                               textvariable=self.var_supergaussian_from, width=5,
                                               validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_supergaussian_from.grid(row=1, column=2)

        self.var_supergaussian_to = tk.StringVar(self.win, value="0.008")
        self.ent_supergaussian_to = tk.Entry(frm_slm_param_scan,
                                             textvariable=self.var_supergaussian_to,
                                             width=5,
                                             validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_supergaussian_to.grid(row=1, column=3)

        self.var_supergaussian_steps = tk.StringVar(self.win, value="10")
        self.ent_supergaussian_steps = tk.Entry(frm_slm_param_scan,
                                                textvariable=self.var_supergaussian_steps,
                                                width=5,
                                                validate='all', validatecommand=(vcmd, '%d', '%P', '%S'))
        self.ent_supergaussian_steps.grid(row=1, column=4)

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
        self.but_show_treated_image.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')

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
        self.img1r = FigureCanvasTkAgg(self.figr, frm_plt)
        self.tk_widget_figr = self.img1r.get_tk_widget()
        self.tk_widget_figr.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.img1r.draw()
        self.ax1r_blit = self.figr.canvas.copy_from_bbox(self.ax1r.bbox)
        self.ax2r_blit = self.figr.canvas.copy_from_bbox(self.ax2r.bbox)

        self.figp = Figure(figsize=(5 * sizefactor, 3 * sizefactor), dpi=100)
        self.ax1p = self.figp.add_subplot(111)
        self.phase_line, = self.ax1p.plot([], '.', ms=1)
        self.ax1p.set_xlim(0, 1000)
        self.ax1p.set_ylim([-np.pi, np.pi])
        self.ax1p.grid()
        self.figp.tight_layout()
        self.figp.canvas.draw()
        self.img1p = FigureCanvasTkAgg(self.figp, frm_plt)
        self.tk_widget_figp = self.img1p.get_tk_widget()
        self.tk_widget_figp.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.img1p.draw()
        self.ax1p_blit = self.figp.canvas.copy_from_bbox(self.ax1p.bbox)

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
        self.pid = PID(0.35, 0, 0, setpoint=0)

        self.stop_acquire = 0
        self.stop_pid = False

        self.spec_interface_initialized = False
        self.active_spec_handle = None

        self.PIKE_cam = True
        self.ANDOR_cam = False

        if self.PIKE_cam is True:
            self.name_cam = 'PIKE_cam'
        elif self.ANDOR_cam is True:
            self.name_cam = 'ANDOR_cam'

        self.cbox_mcp_cam_choice.bind("<<ComboboxSelected>>", self.change_mcp_cam)

        self.hide_frm_daheng()

    def final_image_treatment(self, im):
        bg = int(self.var_mcp_calibration_background_val.get())
        x1 = int(self.var_mcp_calibration_ROIX1_val.get())
        x2 = int(self.var_mcp_calibration_ROIX2_val.get())
        y1 = int(self.var_mcp_calibration_ROIY1_val.get())
        y2 = int(self.var_mcp_calibration_ROIY2_val.get())
        shear = float(self.var_mcp_calibration_shear_val.get())
        correct_E_axis, treated = help.treat_image_new(im, self.eaxis, x1, x2, y1, y2, bg, shear)
        return correct_E_axis, treated

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
                ignore_list = [int(x) for x in self.ent_mcp_calibration_energy_ignore_list.get().split(',') if x.strip().isdigit()]
                if ignore_list:
                    for num in ignore_list:
                        range_value = 5
                        peaks = peaks[~((num - range_value <= peaks) & (peaks <= num + range_value))]



            except:
                a=1
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
            print("Enter odd number for smooth! and something reasonable for peak prominence!")

    def update_calibration(self, var, index, mode):
        im = self.calibration_image
        try:
            im = im - int(self.var_mcp_calibration_background_val.get())
        except:
            print("Enter something reasonable for the background!!")
        try:
            im = help.shear_image(im, float(self.var_mcp_calibration_shear_val.get()), axis=1)
        except:
            print("Enter a reasonable value for the shear!")
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
            print("Enter a reasonable value for the ROI!")

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
        im_temp = self.take_image(int(self.ent_avgs.get()))
        self.calibration_image = im_temp
        self.plot_calibration_image(self.calibration_image)

    def load_test_calibration_image(self):
        file_path = askopenfilename(filetypes=[("TIFF files", "*.tiff;*.tif")])
        im_temp = np.asarray(Image.open(file_path))
        self.calibration_image = im_temp
        self.plot_calibration_image(self.calibration_image)

    def hide_frm_daheng(self):
        if self.frm_daheng_camera_settings.grid_info():
            self.frm_daheng_camera_settings.grid_remove()
            if self.frm_daheng_camera_image.grid_info():
                self.frm_daheng_camera_image.grid_remove()
        else:
            self.frm_daheng_camera_settings.grid()
            self.frm_daheng_camera_image.grid()

    def single_daheng_thread(self):
        self.daheng_thread = threading.Thread(target=self.take_single_image_daheng)
        self.daheng_thread.daemon = True
        self.daheng_thread.start()

    def scan_daheng_thread(self):
        self.daheng_thread = threading.Thread(target=self.scan_stage_daheng)
        self.daheng_thread.daemon = True
        self.daheng_thread.start()

    def enabl_scan_slm_param(self):
        self.scan_slm_param_thread = threading.Thread(target=self.measure_slm_scan)
        self.scan_slm_param_thread.daemon = True
        self.scan_slm_param_thread.start()

    def live_daheng_thread(self):
        self.daheng_is_live = not self.daheng_is_live
        self.update_daheng_live_button()

        self.daheng_thread = threading.Thread(target=self.live_daheng)
        self.daheng_thread.daemon = True
        self.daheng_thread.start()

    def update_daheng_live_button(self):
        if self.daheng_is_live == True:
            self.but_live_daheng.config(fg="green", relief='sunken')
        else:
            self.but_live_daheng.config(fg="red", relief='raised')

    def live_daheng(self):
        while self.daheng_is_live:
            # self.daheng_camera.set_exposure_gain(int(self.var_daheng_exposure.get()), int(self.var_daheng_gain.get()))
            im = self.daheng_camera.take_image(int(self.var_daheng_avg.get()))
            self.current_daheng_image = im
            self.plot_daheng(im)
            print("Focus picture taken !")

    def scan_stage_daheng(self):
        from_ = float(self.var_daheng_stage_from.get())
        to_ = float(self.var_daheng_stage_to.get())
        steps_ = int(self.var_daheng_stage_steps.get())
        stage_steps = np.linspace(from_, to_, steps_)
        if self.daheng_camera is not None:
            res = np.zeros([self.daheng_camera.imshape[0], self.daheng_camera.imshape[1], int(steps_)])
            for ind, pos in enumerate(stage_steps):
                self.strvar_WPDummy_should.set(pos)
                self.move_WPDummy()
                im = self.daheng_camera.take_image(int(self.var_daheng_avg.get()))
                self.plot_daheng(im)
                res[:, :, ind] = im
        self.save_daheng_scans(res, stage_steps)

    def take_single_image_daheng(self):
        if self.daheng_camera is not None:
            im = self.daheng_camera.take_image(int(self.var_daheng_avg.get()))
            self.current_daheng_image = im
            self.plot_daheng(im)

    def plot_daheng(self, im):

        if self.daheng_zoom is not None:
            im = im[int(self.daheng_zoom[0]):int(self.daheng_zoom[1]),
                 int(self.daheng_zoom[2]):int(self.daheng_zoom[3])]

        image = Image.fromarray(im)

        # Resize the image to fit within the specified maximum height
        max_height = 180
        new_width = int(image.width * max_height / max(image.height, max_height))
        image.thumbnail((new_width, max(max_height, image.height)))
        photo = ImageTk.PhotoImage(image)
        image_frame = self.frm_daheng_camera_image
        if hasattr(image_frame, 'image_label'):
            image_frame.image_label.config(image=photo)
            image_frame.image_label.image = photo
        else:
            image_frame.image_label = tk.Label(image_frame, image=photo)
            image_frame.image_label.pack(fill=tk.BOTH, expand=True)

    def zoom_around_com(self):
        radius = int(self.var_daheng_radius.get())
        if self.daheng_camera is not None:
            og_shape = self.daheng_camera.imshape
            if self.current_daheng_image is not None:
                x, y = self.calculate_com(self.current_daheng_image)
                x1 = x - radius
                x2 = x + radius
                y1 = y - radius
                y2 = y + radius
                if x + radius > og_shape[1]:
                    x2 = og_shape[1] - 1
                if x - radius < 0:
                    x1 = 0
                if y + radius > og_shape[0]:
                    y2 = og_shape[0] - 1
                if y - radius < 0:
                    y1 = 0
                self.daheng_zoom = [y1, y2, x1, x2]

    def reset_zoom(self):
        self.daheng_zoom = None

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

    def exp_gain_value_changed(self, event):
        if self.daheng_camera is not None:
            try:
                exposure = int(self.var_daheng_exposure.get())
                gain = int(self.var_daheng_gain.get())
                self.daheng_camera.set_exposure_gain(exposure, gain)
            except:
                print("Enter something reasonable!!")

    def initialize_daheng(self):
        device_manager = gx.DeviceManager()
        index = int(self.var_index_camera.get())
        self.daheng_camera = dh.DahengCamera(index)
        if self.daheng_camera is not None:
            self.but_initialize_daheng.config(fg="green")
            self.daheng_active = True
            self.daheng_camera.set_exposure_gain(int(self.var_daheng_exposure.get()), int(self.var_daheng_gain.get()))
            return 1
        else:
            self.but_initialize_daheng.config(fg="red")
            return 0

    def close_daheng(self):
        if self.daheng_camera is not None:
            self.but_close_daheng.config(fg="green")
            self.daheng_active = False
            self.daheng_camera = None
            return 1
        else:
            self.but_close_daheng.config(fg="red")
            return 0

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
        elif selected_value == 'Andor Camera':
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
            print("pls enter a reasonable value")

    def angle_to_power(self, angle, maxA, phase):
        power = maxA / 2 * np.cos(2 * np.pi / 90 * angle - 2 * np.pi / 90 * phase) + maxA / 2
        return power

    def power_to_angle(self, power, maxA, phase):
        A = maxA / 2
        angle = -(45 * np.arccos(power / A - 1)) / np.pi + phase
        return angle

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
            print("WPR connected")
        except:
            self.but_WPR_Ini.config(fg='red')
            print("Not able to initalize WPR")

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
            print("WPR homed!")
            self.read_WPR()
        except:
            self.but_WPR_Home.config(fg='red')
            print("Not able to home WPR")

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
            print("Impossible to read WPR position")

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
                    print("Value above maximum! Desired power set to maximum instead")
                pos = self.power_to_angle(power, float(self.ent_red_power.get()), float(self.ent_red_phase.get())) + 90
            else:
                pos = float(self.strvar_WPR_should.get())

            print("WPR is moving..")
            self.WPR.move_to(pos, True)
            print(f"WPR moved to {str(self.WPR.position)}")
            self.read_WPR()
        except Exception as e:
            print(e)
            print("Impossible to move WPR :(")

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
            print("WPG connected")
        except:
            self.but_WPG_Ini.config(fg='red')
            print("Not able to initalize WPG")

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

        """
        try:
            self.WPG.move_home(blocking=True)
            self.but_WPG_Home.config(fg='green')
            print("WPG homed!")
            self.read_WPG()
        except:
            self.but_WPG_Home.config(fg='red')
            print("Not able to home WPG")
        """

        print("Non")

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
            print("Impossible to read WPG position")

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
                    print("Value above maximum! Desired power set to maximum instead")
                pos = self.power_to_angle(power, float(self.ent_green_power.get()),
                                          float(self.ent_green_phase.get())) + 90
            else:
                pos = float(self.strvar_WPG_should.get())

            print("WPG is moving to {}".format(np.round(pos, 2)))
            self.WPG.move_to(pos, True)
            print(f"WPG moved to {str(self.WPG.position)}")
            self.read_WPG()

            # try:
            #    self.WPDummy.move_to(pos, True)
            #    print(f"Dummy moved to {str(self.WPDummy.position)}")
            #    self.read_WPDummy()
            # except Exception as e:
            #    print(e)
            #    print("Impossible to move Dummy :(")



        except Exception as e:
            print(e)
            print("Impossible to move WPG :(")
            print("Let us try to re-initialize the WPG :(")
            try:
                self.WPG.disable()
                time.sleep(1)
                self.init_WPG()
                self.WPG.move_to(pos, True)
                print(f"WPG moved to {str(self.WPG.position)}")
                self.read_WPG()
            except Exception as ee:
                print(ee)
                print("Still impossible to move WPG :(")

    def init_WPDummy(self):
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
            self.WPDummy = apt.Motor(int(self.ent_WPDummy_Nr.get()))
            self.but_WPDummy_Ini.config(fg='green')
            print("WPDummy connected")
        except:
            self.but_WPDummy_Ini.config(fg='red')
            print("Not able to initalize Dummy Waveplate")

    def home_WPDummy(self):
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
            self.WPDummy.move_home(blocking=True)
            self.but_WPDummy_Home.config(fg='green')
            print("WPDummy homed!")
            self.read_WPDummy()
        except:
            self.but_WPDummy_Home.config(fg='red')
            print("Not able to home Dummy waveplate")

        # print("Non")

    def read_WPDummy(self):
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
            pos = self.WPDummy.position
            self.strvar_WPDummy_is.set(pos)
            # self.strvar_green_current_power.set(
            #    np.round(self.angle_to_power(pos, float(self.ent_green_power.get()), float(self.ent_green_phase.get())),
            #             3))

        except:
            print("Impossible to read Dummy Waveplate position")

    def move_WPDummy(self):
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
            pos = float(self.strvar_WPDummy_should.get())
            print("WP Dummy is moving to {}".format(np.round(pos, 2)))
            self.WPDummy.move_to(pos, True)
            print(f"WPG moved to {str(self.WPDummy.position)}")
            self.read_WPDummy()
        except Exception as e:
            print(e)
            print("Impossible to move WPDummy :(")
            print("Let us try to re-initialize the WPDUMMY :(")
            try:
                self.WPDummy.disable()
                time.sleep(1)
                self.init_WPDummy()
                self.WPDummy.move_to(pos, True)
                print(f"WPG moved to {str(self.WPDummy.position)}")
                self.read_WPDummy()
            except Exception as ee:
                print(ee)
                print("Still impossible to move WPDummy :(")

    def init_Delay(self):
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
            self.Delay = apt.Motor(int(self.ent_Delay_Nr.get()))
            print("Delay connected")
            self.but_Delay_Ini.config(fg='green')
        except:
            self.but_Delay_Ini.config(fg='red')
            print("Not able to initalize Delay")

    def home_Delay(self):
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
            self.Delay.move_home(blocking=True)
            self.but_Delay_Home.config(fg='green')
            print("Delay stage homed!")
            self.read_Delay()
        except:
            self.but_Delay_Home.config(fg='red')
            print("Not able to home Delay")

    def read_Delay(self):
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
            pos = self.Delay.position
            self.strvar_Delay_is.set(pos)
        except:
            print("Impossible to read Delay position")

    def move_Delay(self):
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
            pos = float(self.strvar_Delay_should.get())
            print("Delay is moving..")
            self.Delay.move_to(pos, True)
            print(f"Delay moved to {str(self.Delay.position)}")
            self.read_Delay()
        except:
            print("Impossible to move Delay :(")

    # def scan(self):

    def disable_motors(self):
        """
        Disconnect all the motors.

        Returns
        -------
        None
        """
        if self.WPG is not None:
            self.WPG.disable()
            print('WPG disconnected')
        if self.WPR is not None:
            self.WPR.disable()
            print('WPR disconnected')
        if self.Delay is not None:
            self.Delay.disable()
            print('Delay disconnected')

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
        self.strvar_red_phase.set(str(self.calibrator.phase_red))
        self.strvar_green_phase.set(str(self.calibrator.phase_green))

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
            print(red_power[0])
            self.strvar_red_power.set(str(red_power[0]))
        except:
            print('Impossible to read red power')

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

        # To be tested
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
            print('Damn no cam')

        return image - self.background

    def save_daheng_scans(self, res, pos):
        nr = self.get_start_image_images()

        data_filename = 'C:/data/' + str(date.today()) + '/' + str(date.today()) + '-' + str(int(nr)) + '.h5'

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        try:
            if self.parent.vars_red[1].get() == 1:
                rl = np.round(float(self.lens_red.strvar_ben.get()), 3)
            else:
                rl = 0
        except:
            rl = np.nan

        try:
            if self.parent.vars_green[1].get() == 1:
                gl = np.round(float(self.lens_green.strvar_ben.get()), 3)
            else:
                gl = 0
        except:
            gl = np.nan

        log_entry = str(int(nr)) + '\t' + str(pos[0]) + '\t' + str(pos[-1]) + '\t' + str(
            int(np.size(pos))) + '\t' + str(gl) + '\t' + str(rl) + '\t' + timestamp + '\n'
        self.g.write(log_entry)
        hf = h5py.File(data_filename, 'w')
        hf.create_dataset('images', data=res)
        hf.create_dataset('positions', data=pos)
        hf.create_dataset('green_lens', data=gl)
        hf.create_dataset('red_lens', data=rl)
        hf.close()

    def save_h5(self):
        nr = self.get_start_image()
        filename = 'C:/data/' + str(date.today()) + '/' + str(date.today()) + '-' + str(int(nr)) + '.h5'

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

    def save_h5_slm_scan(self):
        nr = self.get_start_image_slm_param_scan()
        filename = 'C:/data/' + str(date.today()) + '/' + 'slm_param_scan_' + str(date.today()) + '-' + str(int(nr)) + '.h5'

        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('raw_images', data=self.measurement_array)
            hf.create_dataset('treated_images', data=self.measurement_treated_array)
            hf.create_dataset('focus_images', data=self.focus_image_array)
            hf.create_dataset('radius', data=self.pi_radius_array)
            hf.create_dataset('e_axis', data=self.eaxis_correct)
            time_stamps_array_converted = np.array(self.time_stamps_array, dtype='S')
            hf.create_dataset('timestamps', data=time_stamps_array_converted)

        log_entry = str(int(nr)) + '\n'
        self.slmps.write(log_entry)


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
        filename = 'C:/data/' + str(date.today()) + '/' + str(date.today()) + '-' + str(int(nr)) + '.tif'
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
        # self.f.close()

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
        """
        Enables the MCP measurement.

        Returns
        -------
        None
        """
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
        """
        Enables the MCP measurement.

        Returns
        -------
        None
        """

        if self.var_split_scan.get() == 1:
            if self.var_scan_wp_option.get() == "Red/Green Ratio":
                self.stop_mcp = False
                self.split = threading.Thread(target=self.split_threading)
                self.split.daemon = True
                self.split.start()

        else:
            self.stop_mcp = False
            self.mcp_thread = threading.Thread(target=self.measure_all)
            self.mcp_thread.daemon = True
            self.mcp_thread.start()

    def enabl_mcp(self):
        """
        Enables the MCP measurement.

        Returns
        -------
        None
        """
        self.stop_mcp = False
        self.mcp_thread = threading.Thread(target=self.measure)
        self.mcp_thread.daemon = True
        self.mcp_thread.start()

    def enabl_mcp_simple(self):
        """
        Enables the simple MCP measurement.

        Returns
        -------
        None
        """
        self.stop_mcp = False
        self.mcp_thread = threading.Thread(target=self.measure_simple)
        self.mcp_thread.daemon = True
        self.mcp_thread.start()

    def get_start_image_images(self):
        """
        Gets the index of the starting image.

        This method retrieves the index of the starting image from the autolog file.
        It reads the autolog file to get the latest image index, increments it by one,
        and returns the result as the starting index for the next image.

        Returns
        -------
        int
            The index of the starting image.

        Raises
        ------
        Exception
            If there is an error in retrieving the starting image index.
        """
        # self.f = open(self.autolog, "a+")
        self.g.seek(0)
        lines = np.loadtxt(self.autolog_images, comments="#", delimiter="\t", unpack=False, usecols=(0,))
        if lines.size > 0:
            try:
                start_image = lines[-1] + 1
            except:
                start_image = lines + 1
            print("The last image had index " + str(int(start_image - 1)))
        else:
            start_image = 0
        # self.f.close()
        return start_image

    def get_start_image_slm_param_scan(self):

        # self.f = open(self.autolog, "a+")
        self.slmps.seek(0)
        lines = np.loadtxt(self.autolog_slm_param_scan, comments="#", delimiter="\t", unpack=False, usecols=(0,))
        if lines.size > 0:
            try:
                start_image = lines[-1] + 1
            except:
                start_image = lines + 1
            print("The last image had index " + str(int(start_image - 1)))
        else:
            start_image = 0
        # self.f.close()
        return start_image

    def get_start_image(self):
        """
        Gets the index of the starting image.

        This method retrieves the index of the starting image from the autolog file.
        It reads the autolog file to get the latest image index, increments it by one,
        and returns the result as the starting index for the next image.

        Returns
        -------
        int
            The index of the starting image.

        Raises
        ------
        Exception
            If there is an error in retrieving the starting image index.
        """
        # self.f = open(self.autolog, "a+")
        self.f.seek(0)
        lines = np.loadtxt(self.autolog, comments="#", delimiter="\t", unpack=False, usecols=(0,))
        if lines.size > 0:
            try:
                start_image = lines[-1] + 1
            except:
                start_image = lines + 1
            print("The last image had index " + str(int(start_image - 1)))
        else:
            start_image = 0
        # self.f.close()
        return start_image

    def red_only_scan(self):
        """
        Perform a scan of the red power. It sets the 'var_wprpower' variable to 1, which indicates that we use the
        "power mode" of the waveplate. It generates a list of power values to scan and for each power value,
        it sets the 'strvar_WPR_should' variable to the current value, moves the WPR  accordingly, takes an image
        with the specified number of averages from 'ent_avgs', saves the image, and plots the MCP image.

        Returns
        -------
        None
        """
        self.var_wprpower.set(1)
        WPR_steps = int(self.ent_WPR_steps.get())
        WPR_scan_list = np.linspace(float(self.ent_WPR_from.get()), float(self.ent_WPR_to.get()), WPR_steps)
        print(WPR_scan_list)

        for i in np.arange(0, WPR_steps):
            r = WPR_scan_list[i]
            self.strvar_WPR_should.set(str(r))
            self.move_WPR()
            im = self.take_image(int(self.ent_avgs.get()))
            self.save_im(im)
            self.plot_MCP(im)

    def green_only_scan(self):
        """
        Perform a scan of the green power. It sets the 'var_wpgpower' variable to 1, which indicates that we use the
        "power mode" of the waveplate. It generates a list of power values to scan and for each power value,
        it sets the 'strvar_WPG_should' variable to the current value, moves the WPG  accordingly, takes an image
        with the specified number of averages from 'ent_avgs', saves the image, and plots the MCP image.

        ReturnsW
        -------
        None
        """
        self.var_wpgpower.set(1)
        WPG_steps = int(self.ent_WPG_steps.get())
        WPG_scan_list = np.linspace(float(self.ent_WPG_from.get()), float(self.ent_WPG_to.get()), WPG_steps)
        print(WPG_scan_list)

        for i in np.arange(0, WPG_steps):
            g = WPG_scan_list[i]
            self.strvar_WPG_should.set(str(g))
            self.move_WPG()
            im = self.take_image(int(self.ent_avgs.get()))
            self.save_im(im)
            self.plot_MCP(im)

    def red_green_ratio_scan(self):
        """
        Perform a scan of the red and green ratio.

        Returns
        -------
        None
        """
        steps = int(self.ent_ratio_steps.get())
        pr, pg = self.get_power_values_for_ratio_scan()
        self.var_wprpower.set(1)
        self.var_wpgpower.set(1)

        print("Power Values for R: ")
        print(pr)
        print("Power Values for G: ")
        print(pg)

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

    def phase_scan(self):
        """
        Scans the phase and captures images.

        The scan parameters are specified in the GUI.
        The captured images are saved to a file and the MCP signal is plotted.

        Returns
        -------
        None
        """
        start_image = self.get_start_image()
        self.phis = np.linspace(float(self.ent_from.get()), float(self.ent_to.get()), int(self.ent_steps.get()))
        print("getting to scan starting point...")
        self.strvar_setp.set(self.phis[0])
        self.set_setpoint()
        time.sleep(0.05)
        print("Ready to scan the phase!")
        for ind, phi in enumerate(self.phis):
            start_time = time.time()
            self.strvar_setp.set(phi)
            self.set_setpoint()
            t0 = time.time()
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
                print(self.measurement_counter)
                self.measurement_counter = self.measurement_counter + 1
            else:
                self.save_im(im)
            self.plot_MCP(im)
            t1 = time.time()
            print(f"Camera MCP {t1 - t0}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Imagenr ", (start_image + ind), " Phase: ", round(phi, 2), " Elapsed time: ", round(elapsed_time, 2))

    def get_power_values_for_ratio_scan(self):
        c = float(self.strvar_int_ratio_constant.get())
        x = float(self.ent_int_ratio_focus.get()) ** 2
        ratios = np.linspace(float(self.ent_ratio_from.get()), float(self.ent_ratio_to.get()),
                             int(self.ent_ratio_steps.get()))
        pr = c - c * ratios
        pg = c * ratios / x
        print(x)
        print(c)
        print(ratios)
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

    def measure_slm_scan(self):
        self.but_meas_slm_param.config(fg='red')

        status = self.var_scan_slm_param_option.get()

        if status == "Nothing":
            print("Nothing is selected")

        elif status == "Supergaussian":
            print("Supergaussian is selected")
            self.slmps.write("# Supergaussian scan, " + self.ent_comment.get() + "\n")

            pi_radius_steps = int(self.ent_supergaussian_steps.get())
            self.measurement_counter = 0
            self.measurement_array = np.zeros([pi_radius_steps]) * np.nan
            self.focus_image_array = np.zeros([pi_radius_steps]) * np.nan
            self.measurement_treated_array = np.zeros([pi_radius_steps]) * np.nan

            self.time_stamps_array = np.zeros([pi_radius_steps]) * np.nan
            self.time_stamps_array = self.time_stamps_array.astype('str')

            self.pi_radius_array = np.linspace(float(self.ent_supergaussian_from.get()),
                                                   float(self.ent_supergaussian_to.get()),
                                                   int(self.ent_supergaussian_steps.get()))
            self.aquisition_time = int(self.ent_exposure_time.get())
            self.averages = int(self.ent_avgs.get())
            self.mcp_voltage = float(self.ent_mcp.get())


            self.focus_image_array_flat = self.focus_image_array.flatten()

            self.focus_image_array_flat = np.zeros([self.focus_image_array_flat.size, 1080, 1440]) * np.nan

            self.measurement_array_flat = self.measurement_array.flatten()
            self.measurement_array_flat = np.zeros([self.measurement_array_flat.size, 512, 512]) * np.nan
            self.measurement_treated_array_flat = self.measurement_array.flatten()
            self.measurement_treated_array_flat = np.zeros([self.measurement_treated_array_flat.size, 512, 512]) * np.nan
            self.time_stamps_array_flat = self.time_stamps_array.flatten()


            self.measurement_running = 1
            self.scan_supergaussian()

        self.but_meas_slm_param.config(fg='green')
        if self.measurement_running:
            self.measurement_running = 0
            self.measurement_counter = 0


            self.measurement_array = self.measurement_array_flat.reshape(
                [self.measurement_array.shape[0], 512, 512])

            self.measurement_treated_array = self.measurement_treated_array_flat.reshape(
                [self.measurement_treated_array.shape[0], 512, 512])

            self.focus_image_array = self.focus_image_array_flat.reshape(
                [self.focus_image_array.shape[0], 1080, 1440])

            self.save_h5_slm_scan()

        # Indicate completion by setting the event
        self.scan_is_done = True
        self.scan_is_done_threading.set()

    def scan_supergaussian(self):
        # this is for IR beam only
        self.radius_steps = np.linspace(float(self.ent_supergaussian_from.get()), float(self.ent_supergaussian_to.get()), int(self.ent_supergaussian_steps.get()))

        x = np.linspace(-chip_width, chip_width, slm_size[1])
        y = np.linspace(-chip_height, chip_height, slm_size[0])
        [X, Y] = np.meshgrid(x, y)
        rho = np.sqrt(X ** 2 + Y ** 2)
        rho /= 2
        # note : make sure that there another phase on the slm, otherwise a double scan leads to an error

        if self.daheng_camera is not None:
            for ind, desired_radius in enumerate(self.radius_steps):
                # Set the new wavefront on the SLM
                indices = np.where(rho <= desired_radius)
                p1 = np.zeros_like(X)
                p1[indices] = np.pi
                phase = p1 / (2 * np.pi) * bit_depth
                phase_map = self.parent.phase_map_red + phase
                self.slm_lib.SLM_Disp_Open(int(self.parent.ent_scr_red.get()))
                self.slm_lib.SLM_Disp_Data(int(self.parent.ent_scr_red.get()), phase_map,
                                           slm_size[1], slm_size[0])
                time.sleep(2)


                im_MCP = self.take_image(int(self.ent_avgs.get()))
                im_focus = self.daheng_camera.take_image(int(self.var_daheng_avg.get()))
                #im_focus = Image.fromarray(im_focus)

                if self.measurement_running and self.var_saveh5.get():
                    self.measurement_array_flat[self.measurement_counter, :, :] = im_MCP
                    self.focus_image_array_flat[self.measurement_counter, :, :] = im_focus
                    if self.var_export_treated_image.get():
                        E_new, im_new = self.final_image_treatment(im_MCP)
                        self.measurement_treated_array_flat[self.measurement_counter, :, :] = im_new
                    self.time_stamps_array_flat[self.measurement_counter] = str(
                        datetime.datetime.now().strftime("%H:%M:%S"))
                    print(self.measurement_counter)
                    self.measurement_counter = self.measurement_counter + 1
                else:
                    self.save_im(im_MCP)

                self.plot_MCP(im_MCP)

    def phase_scan(self):
        start_image = self.get_start_image()
        self.phis = np.linspace(float(self.ent_from.get()), float(self.ent_to.get()), int(self.ent_steps.get()))
        print("getting to scan starting point...")
        self.strvar_setp.set(self.phis[0])
        self.set_setpoint()
        time.sleep(0.05)
        print("Ready to scan the phase!")
        for ind, phi in enumerate(self.phis):
            start_time = time.time()
            self.strvar_setp.set(phi)
            self.set_setpoint()
            t0 = time.time()
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
                print(self.measurement_counter)
                self.measurement_counter = self.measurement_counter + 1
            else:
                self.save_im(im)
            self.plot_MCP(im)
            t1 = time.time()
            print(f"Camera MCP {t1 - t0}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Imagenr ", (start_image + ind), " Phase: ", round(phi, 2), " Elapsed time: ", round(elapsed_time, 2))

    def measure_all(self):
        print("yay i made it into the measure all function")
        self.but_meas_all.config(fg='red')
        # self.f = open(self.autolog, "a+")

        status = self.var_scan_wp_option.get()
        print(status)

        if status == "Green Focus":
            if self.var_phasescan.get() == 1:
                if self.var_background.get() == 1:
                    self.f.write("# BACKGROUND FocusPositionScan, " + self.ent_comment.get() + "\n")
                    self.focus_position_scan_green()
                else:
                    self.f.write("# FocusPositionScan, " + self.ent_comment.get() + "\n")
                    self.focus_position_scan_green()
            else:
                print("Are you sure you do not want to scan the phase for each focus position?")
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
                print("Are you sure you do not want to scan the phase for each focus position?")
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
                print("Would you please select something to actually scan")

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
                print("Are you sure you do not want to scan the phase for each ratio?")

        elif status == "Only Red":
            self.f.write("# RedOnlyScan, " + self.ent_comment.get() + "\n")
            self.red_only_scan()
            print(status)
        elif status == "Only Green":
            self.f.write("# GreenOnlyScan, " + self.ent_comment.get() + "\n")
            self.green_only_scan()
            print(status)
        else:
            print("something fishy is going on")

        # self.f.close()
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
        """
        Performs a phase scan

        Returns
        -------
        None
        """
        self.but_meas_scan.config(fg='red')

        # if self.var_phasescan.get() == 1:
        # self.f = open(self.autolog, "a+")
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
        # self.f.close()

        self.but_meas_scan.config(fg='green')

    def measure_simple(self):
        """
        Performs a simple measurement.

        This method performs a simple measurement by capturing images and plotting the MCP signal.
        It retrieves the index of the starting image from the autolog file using the `get_start_image()`
        method, captures the image using the `take_image()` method, saves the image to a file using
        the `save_image()` method, plots the MCP signal using the `plot_MCP()` method, and then closes
        the file.

        Returns
        -------
        None
        """
        self.but_meas_simple.config(fg='red')
        # (self.autolog, "a+")

        # start_image = self.get_start_image()

        if self.var_background.get() == 1:
            self.f.write("# BACKGROUND SingleImage, " + self.ent_comment.get() + '\n')
        else:
            self.f.write("# SingleImage, " + self.ent_comment.get() + '\n')
        # info = self.ent_avgs.get() + " averages" + " comment: " + self.ent_comment.get()
        # self.save_image(im, start_image, info)
        im = self.take_image(int(self.ent_avgs.get()))
        self.save_im(im)
        self.plot_MCP(im)
        self.but_meas_simple.config(fg='green')
        # self.f.close()

    def feedback(self):
        """
        Displays a phase map on the SLM.

        The phase map is calculated based on the `phase_map` attribute of the parent object and the flat phase value specified in the GUI.
        The phase map is then displayed on the SLM.

        Returns
        -------
        None
        """
        if self.ent_flat.get() != '':
            phi = float(self.ent_flat.get())
        else:
            phi = 0
        phase_map = self.parent.phase_map_green + phi / 2 * bit_depth

        self.slm_lib.SLM_Disp_Open(int(self.parent.ent_scr_green.get()))
        self.slm_lib.SLM_Disp_Data(int(self.parent.ent_scr_green.get()), phase_map,
                                   slm_size[1], slm_size[0])

    def eval_spec(self):
        """
        Acquisition function for spectrometer.

        This function acquires raw data from a spectrometer and calculates the phase angle of the Fourier transform of the data.
        It continuously acquires data until `stop_acquire` flag is set to 1.

        Returns:
        --------
        None
        """
        while True:
            time.sleep(0.01)

            # get raw trace
            timestamp, data = avs.get_spectrum(self.active_spec_handle)
            wavelength = avs.AVS_GetLambda(self.active_spec_handle)

            start = int(self.ent_area1x.get())
            stop = int(self.ent_area1y.get())
            self.trace = data[start:stop]
            self.wavelength = wavelength[start:stop]

            # print(timestamp)

            im_fft = np.fft.fft(self.trace)
            self.abs_im_fft = np.abs(im_fft)
            self.abs_im_fft = self.abs_im_fft / np.max(self.abs_im_fft)
            ind = round(float(self.ent_indexfft.get()))
            try:
                self.im_angl = np.angle(im_fft[ind])
            except:
                self.im_angl = 0
            self.lbl_angle.config(text=np.round(self.im_angl, 6))

            # creating the phase vector
            self.im_phase[:-1] = self.im_phase[1:]
            self.im_phase[-1] = self.im_angl

            # calculating standard deviation
            mean = np.mean(self.im_phase)
            std = np.sqrt(np.sum((self.im_phase - mean) ** 2) / (len(self.im_phase) - 1))
            self.lbl_std_val.config(text=np.round(std, 4))

            if self.stop_acquire == 1:
                self.stop_acquire = 0
                break
            if self.meas_has_started:
                self.d_phase.append(self.im_angl)
                # print("phase saving should be activated")
                # g.write(str(self.im_angl)+"\n")
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

    def plot_calibration_image(self, image):
        image = np.flipud(image)
        self.axMCP_calibrate.clear()
        self.axMCP_calibrate.imshow(image.T)
        # self.axMCP.set_aspect('equal')

        self.axMCP_calibrate.set_xlabel("Energy equivalent")
        self.axMCP_calibrate.set_ylabel("Y (px)")
        self.axMCP_calibrate.set_xlim(0, 512)
        self.axMCP_calibrate.set_ylim(0, 512)

        self.axHarmonics_calibrate.clear()
        self.axHarmonics_calibrate.plot(np.arange(512), np.sum(image, 1))
        self.axHarmonics_calibrate.axhline(0, color='k', alpha = 0.5)
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
        print(maxindex)

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
        self.ax1r.draw_artist(self.trace_line)
        self.fourier_line.set_data(np.arange(50), self.abs_im_fft[:50])
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
        self.ax1p.draw_artist(self.phase_line)
        self.figp.canvas.blit(self.ax1p.bbox)
        self.figp.canvas.flush_events()
        self.win.after(50, self.plot_phase)

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
                print(str(len(speclist)) + ' spectrometer(s) found.')
                self.active_spec_handle = avs.AVS_Activate(speclist[0])
                self.ent_spc_ind.config(state='disabled')
        except:
            print('There was no spectrometer found!')

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
            print('No spectrometer found!')

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
        self.set_setpoint()
        self.set_pid_val()

        while True:
            time.sleep(0.05)
            correction = self.pid((self.im_angl - self.set_point + np.pi) % (2 * np.pi) - np.pi)
            self.strvar_flat.set(correction)
            self.feedback()
            # print(self.pid.components)
            if self.stop_pid:
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
            print(f"Spectral fringes saved to {file_path.name}")
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
            print(f"Phase stability saved to {file_path.name}")
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
        self.g.close()
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
