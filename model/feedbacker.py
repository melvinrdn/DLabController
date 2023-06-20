import os
import threading
import time
import tkinter as tk
from collections import deque
from datetime import date
import datetime

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
from model import calibrator as cal


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

        self.WPG = None
        self.WPR = None
        self.Delay = None

        self.meas_has_started = False

        # This opens the autologfile from the start! closes it on close command
        self.autolog = 'C:/data/' + str(date.today()) + '/' + str(date.today()) + '-' + 'auto-log.txt'
        # self.f = open(self.autolog, "a+")

        # creating frames
        frm_bot = tk.Frame(self.win)
        frm_plt = tk.Frame(self.win)
        frm_mcp_image = tk.Frame(self.win)
        frm_mid = tk.Frame(self.win)

        # new frame for scan related parameters
        frm_scans = tk.Frame(self.win)

        frm_spc_but = tk.Frame(self.win)
        frm_spc_but_set = tk.Frame(frm_spc_but)
        frm_plt_set = tk.LabelFrame(frm_mid, text='Plot options')

        frm_ratio = tk.LabelFrame(frm_mid, text='Phase extraction')
        frm_pid = tk.LabelFrame(frm_mid, text='PID controller')

        frm_measure = tk.LabelFrame(frm_scans, text='Measurement')
        frm_phase_scan = tk.LabelFrame(frm_scans, text='Phase Scan')
        frm_stage = tk.LabelFrame(frm_scans, text='Stage Control')
        frm_wp_power_cal = tk.LabelFrame(frm_scans, text='WP - Power calibration')
        frm_wp_scans = tk.LabelFrame(frm_scans, text='Power Scans!')

        vcmd = (self.win.register(self.parent.callback))

        # creating buttons n labels
        but_exit = tk.Button(frm_bot, text='EXIT', command=self.on_close)
        but_feedback = tk.Button(frm_bot, text='Feedback', command=self.feedback)

        lbl_spc_ind = tk.Label(frm_spc_but_set, text='Spectrometer index:')
        self.strvar_spc_ind = tk.StringVar(self.win, '1')
        self.ent_spc_ind = tk.Entry(
            frm_spc_but_set, width=9, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_spc_ind)
        lbl_spc_exp = tk.Label(frm_spc_but_set, text='Exposure time (ms):')
        self.strvar_spc_exp = tk.StringVar(self.win, '50')
        self.ent_spc_exp = tk.Entry(
            frm_spc_but_set, width=9, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_spc_exp)
        lbl_spc_gain = tk.Label(frm_spc_but_set, text='Nr. of averages:')
        self.strvar_spc_avg = tk.StringVar(self.win, '1')
        self.ent_spc_avg = tk.Entry(
            frm_spc_but_set, width=9, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_spc_avg)
        but_spc_activate = tk.Button(frm_spc_but_set, text='Activate',
                                     command=self.spec_activate, width=8)
        but_spc_deactivate = tk.Button(frm_spc_but_set, text='Deactivate',
                                       command=self.spec_deactivate, width=8)
        but_spc_start = tk.Button(frm_spc_but, text='Start\nSpectrometer',
                                  command=self.spc_img, height=2)
        but_spc_stop = tk.Button(frm_spc_but, text='Stop\nSpectrometer',
                                 command=self.stop_measure, height=2)
        but_spc_phi = tk.Button(frm_spc_but, text='fast 2pi',
                                command=self.fast_scan, height=2)
        but_auto_scale = tk.Button(frm_plt_set, text='auto-scale',
                                   command=self.auto_scale_spec_axis, width=13)
        but_bck = tk.Button(frm_plt_set, text='take background',
                            command=self.take_background, width=13)
        lbl_std = tk.Label(frm_plt_set, text='sigma:', width=6)
        self.lbl_std_val = tk.Label(frm_plt_set, text='None', width=6)
        lbl_phi = tk.Label(frm_ratio, text='Phase shift:')
        lbl_phi_2 = tk.Label(frm_ratio, text='pi')
        self.strvar_flat = tk.StringVar()
        self.ent_flat = tk.Entry(
            frm_ratio, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_flat)

        text = '17'
        self.strvar_indexfft = tk.StringVar(self.win, text)
        lbl_indexfft = tk.Label(frm_ratio, text='Index fft:')
        lbl_angle = tk.Label(frm_ratio, text='Phase:')
        self.ent_indexfft = tk.Entry(
            frm_ratio, width=11,
            textvariable=self.strvar_indexfft)
        self.lbl_angle = tk.Label(frm_ratio, text='angle')

        text = '1950'
        self.strvar_area1x = tk.StringVar(self.win, text)
        self.ent_area1x = tk.Entry(
            frm_ratio, width=11,
            textvariable=self.strvar_area1x)

        text = '2100'
        self.strvar_area1y = tk.StringVar(self.win, text)
        self.ent_area1y = tk.Entry(
            frm_ratio, width=11,
            textvariable=self.strvar_area1y)

        lbl_setp = tk.Label(frm_pid, text='Setpoint:')
        self.strvar_setp = tk.StringVar(self.win, '0')
        self.ent_setp = tk.Entry(
            frm_pid, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_setp)
        lbl_pidp = tk.Label(frm_pid, text='P-value:')
        self.strvar_pidp = tk.StringVar(self.win, '-0.2')
        self.ent_pidp = tk.Entry(
            frm_pid, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_pidp)
        lbl_pidi = tk.Label(frm_pid, text='I-value:')
        self.strvar_pidi = tk.StringVar(self.win, '-0.8')
        self.ent_pidi = tk.Entry(
            frm_pid, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_pidi)
        but_pid_setp = tk.Button(frm_pid, text='Setpoint', command=self.set_setpoint)
        but_pid_enbl = tk.Button(frm_pid, text='Start PID', command=self.enbl_pid)
        but_pid_stop = tk.Button(frm_pid, text='Stop PID', command=self.pid_stop)
        but_pid_setk = tk.Button(frm_pid, text='Set PID values', command=self.set_pid_val)

        lbl_from = tk.Label(frm_phase_scan, text='From:')
        self.strvar_from = tk.StringVar(self.win, '-3.1')
        self.ent_from = tk.Entry(
            frm_phase_scan, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_from)

        lbl_to = tk.Label(frm_phase_scan, text='To:')
        self.strvar_to = tk.StringVar(self.win, '3.1')
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

        self.var_background = tk.IntVar()
        self.cb_background = tk.Checkbutton(frm_measure, text='Background', variable=self.var_background, onvalue=1,
                                            offvalue=0,
                                            command=None)
        lbl_avgs = tk.Label(frm_measure, text='Avgs:')
        self.strvar_avgs = tk.StringVar(self.win, '20')
        self.ent_avgs = tk.Entry(
            frm_measure, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_avgs)

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
        lbl_is = tk.Label(frm_stage, text='is')
        lbl_should = tk.Label(frm_stage, text='should')

        lbl_WPR = tk.Label(frm_stage, text='WP red:')
        self.strvar_WPR_is = tk.StringVar(self.win, '')
        self.ent_WPR_is = tk.Entry(
            frm_stage, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPR_is)
        self.strvar_WPR_should = tk.StringVar(self.win, '')
        self.ent_WPR_should = tk.Entry(
            frm_stage, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPR_should)
        self.strvar_WPR_Nr = tk.StringVar(self.win, '83837724')
        self.ent_WPR_Nr = tk.Entry(
            frm_stage, width=9, validate='all',
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
            frm_stage, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPG_is)
        self.strvar_WPG_should = tk.StringVar(self.win, '')
        self.ent_WPG_should = tk.Entry(
            frm_stage, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPG_should)
        self.strvar_WPG_Nr = tk.StringVar(self.win, '83837725')
        self.ent_WPG_Nr = tk.Entry(
            frm_stage, width=9, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_WPG_Nr)
        self.but_WPG_Ini = tk.Button(frm_stage, text='Init', command=self.init_WPG)
        self.but_WPG_Home = tk.Button(frm_stage, text='Home', command=self.home_WPG)
        self.but_WPG_Read = tk.Button(frm_stage, text='Read', command=self.read_WPG)
        self.but_WPG_Move = tk.Button(frm_stage, text='Move', command=self.move_WPG)

        self.var_wpgpower = tk.IntVar()
        self.cb_wpgpower = tk.Checkbutton(frm_stage, text='Power', variable=self.var_wpgpower, onvalue=1, offvalue=0,
                                          command=None)

        lbl_Delay = tk.Label(frm_stage, text='Delay:')
        self.strvar_Delay_is = tk.StringVar(self.win, '')
        self.ent_Delay_is = tk.Entry(
            frm_stage, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_Delay_is)
        self.strvar_Delay_should = tk.StringVar(self.win, '')
        self.ent_Delay_should = tk.Entry(
            frm_stage, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_Delay_should)
        self.strvar_Delay_Nr = tk.StringVar(self.win, '83837719')
        self.ent_Delay_Nr = tk.Entry(
            frm_stage, width=9, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_Delay_Nr)
        # scan parameters
        self.strvar_Delay_from = tk.StringVar(self.win, '6.40')
        self.ent_Delay_from = tk.Entry(
            frm_stage, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_Delay_from)
        self.strvar_Delay_to = tk.StringVar(self.win, '6.45')
        self.ent_Delay_to = tk.Entry(
            frm_stage, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_Delay_to)
        self.strvar_Delay_steps = tk.StringVar(self.win, '10')
        self.ent_Delay_steps = tk.Entry(
            frm_stage, width=5, validate='all',
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
            frm_wp_power_cal, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_pharos_att)

        lbl_pharos_pp = tk.Label(frm_wp_power_cal, text='Pharos PP:')
        self.strvar_pharos_pp = tk.StringVar(self.win, '1')
        self.ent_pharos_pp = tk.Entry(
            frm_wp_power_cal, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_pharos_pp)

        self.strvar_red_power = tk.StringVar(self.win, '')
        self.ent_red_power = tk.Entry(
            frm_wp_power_cal, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_red_power)

        self.but_calibrator_open = tk.Button(frm_wp_power_cal, text='Open Calibrator!', command=self.enable_calibrator)
        # self.but_red_power = tk.Button(frm_wp_power_cal, text='Red Power :', command=self.read_red_power)

        lbl_red_power = tk.Label(frm_wp_power_cal, text='Red Max Power (W):')
        self.strvar_red_power = tk.StringVar(self.win, '4.5')
        self.ent_red_power = tk.Entry(
            frm_wp_power_cal, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_red_power)

        lbl_red_phase = tk.Label(frm_wp_power_cal, text='Red offset phase (deg):')
        self.strvar_red_phase = tk.StringVar(self.win, '-27.76')
        self.ent_red_phase = tk.Entry(
            frm_wp_power_cal, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_red_phase)

        lbl_red_current_power = tk.Label(frm_wp_power_cal, text='Red current Power (W):')
        self.strvar_red_current_power = tk.StringVar(self.win, '')
        self.ent_red_current_power = tk.Entry(
            frm_wp_power_cal, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_red_current_power)

        lbl_green_power = tk.Label(frm_wp_power_cal, text='Green Max Power (mW):')
        self.strvar_green_power = tk.StringVar(self.win, '345')
        self.ent_green_power = tk.Entry(
            frm_wp_power_cal, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_green_power)

        lbl_green_phase = tk.Label(frm_wp_power_cal, text='Green offset phase (deg):')
        self.strvar_green_phase = tk.StringVar(self.win, '44.02')
        self.ent_green_phase = tk.Entry(
            frm_wp_power_cal, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_green_phase)

        lbl_green_current_power = tk.Label(frm_wp_power_cal, text='Green current Power (mW):')
        self.strvar_green_current_power = tk.StringVar(self.win, '')
        self.ent_green_current_power = tk.Entry(
            frm_wp_power_cal, width=5, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
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
        self.strvar_ratio_to = tk.StringVar(self.win, str(np.round(x * maxG / (c - x * maxG), 3)))
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

        lbl_GFP_green = tk.Label(frm_wp_scans, text="Green Focus Position (1/m)")
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

        lbl_RFP_red = tk.Label(frm_wp_scans, text="Red Focus Position (1/m)")
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

        frm_spc_but.grid(row=0, column=0, sticky='nsew')

        frm_plt.grid(row=1, column=0, sticky='nsew')
        frm_mcp_image.grid(row=1, column=2, sticky='nsew')
        frm_scans.grid(row=1, column=1)
        frm_measure.grid(row=0, column=0, padx=5)
        frm_phase_scan.grid(row=0, column=1, padx=5)
        frm_stage.grid(row=1, column=0, padx=5)
        frm_wp_power_cal.grid(row=2, column=0, padx=5, pady=5)
        frm_wp_scans.grid(row=3, column=0, padx=5, pady=5)

        frm_mid.grid(row=2, column=0, sticky='nsew')
        frm_bot.grid(row=3, column=0)

        frm_plt_set.grid(row=0, column=0, padx=5)
        frm_ratio.grid(row=0, column=1, padx=5)
        frm_pid.grid(row=0, column=2, padx=5)

        frm_ratio.config(width=162, height=104)

        frm_ratio.grid_propagate(False)

        # setting up buttons frm_spc
        frm_spc_but_set.grid(row=0, column=0, sticky='nsew')
        but_spc_start.grid(row=0, column=1, padx=5, pady=5, ipadx=5, ipady=5)
        but_spc_stop.grid(row=0, column=2, padx=5, pady=5, ipadx=5, ipady=5)
        but_spc_phi.grid(row=0, column=3, padx=5, pady=5, ipadx=5, ipady=5)
        lbl_spc_ind.grid(row=0, column=0)
        self.ent_spc_ind.grid(row=0, column=1)
        but_spc_activate.grid(row=0, column=2, padx=(1, 5))
        lbl_spc_exp.grid(row=1, column=0)
        self.ent_spc_exp.grid(row=1, column=1)
        but_spc_deactivate.grid(row=1, column=2, padx=(1, 5))
        lbl_spc_gain.grid(row=2, column=0)
        self.ent_spc_avg.grid(row=2, column=1)

        # setting up frm_spc_set
        but_auto_scale.grid(row=0, column=0, columnspan=2, padx=5, pady=(3, 10))
        but_bck.grid(row=1, column=0, columnspan=2, padx=5)
        lbl_std.grid(row=2, column=0, pady=5)
        self.lbl_std_val.grid(row=2, column=1, pady=5)

        # setting up buttons frm_bot
        but_exit.grid(row=1, column=0, padx=5, pady=5, ipadx=5, ipady=5)
        but_feedback.grid(row=1, column=1, padx=5, pady=5, ipadx=5, ipady=5)

        # setting up frm_pid
        lbl_setp.grid(row=0, column=0)
        lbl_pidp.grid(row=1, column=0)
        lbl_pidi.grid(row=2, column=0)
        self.ent_setp.grid(row=0, column=1)
        self.ent_pidp.grid(row=1, column=1)
        self.ent_pidi.grid(row=2, column=1)
        but_pid_setp.grid(row=3, column=0)
        but_pid_setk.grid(row=3, column=1)
        but_pid_enbl.grid(row=1, column=2)
        but_pid_stop.grid(row=2, column=2)

        # setting up frm_measure
        lbl_mcp.grid(row=1, column=0, sticky='w')
        self.ent_mcp.grid(row=1, column=1, padx=5, pady=5)

        lbl_avgs.grid(row=2, column=0, sticky='w')
        self.ent_avgs.grid(row=2, column=1)

        lbl_comment.grid(row=3, column=0, sticky='w')
        self.ent_comment.grid(row=3, column=1, padx=5, pady=5)

        self.cb_background.grid(row=4, column=0, sticky='w')

        self.but_meas_all.grid(row=5, column=0)
        self.but_meas_scan.grid(row=5, column=1)
        self.but_meas_simple.grid(row=5, column=2)

        # setting up frm_phase_scan
        lbl_from.grid(row=0, column=0, sticky='w')
        lbl_to.grid(row=1, column=0, sticky='w')
        lbl_steps.grid(row=2, column=0, sticky='w')
        self.ent_from.grid(row=0, column=1)
        self.ent_to.grid(row=1, column=1)
        self.ent_steps.grid(row=2, column=1)
        self.cb_phasescan.grid(row=5, column=1)

        # setting up frm_stage
        lbl_Stage.grid(row=1, column=1)
        lbl_Nr.grid(row=1, column=2)
        lbl_is.grid(row=1, column=3)
        lbl_should.grid(row=1, column=4)

        lbl_WPR.grid(row=2, column=1)
        lbl_WPG.grid(row=3, column=1)
        lbl_Delay.grid(row=4, column=1)

        self.ent_WPR_Nr.grid(row=2, column=2)
        self.ent_WPG_Nr.grid(row=3, column=2)
        self.ent_Delay_Nr.grid(row=4, column=2)

        self.ent_WPR_is.grid(row=2, column=3)
        self.ent_WPG_is.grid(row=3, column=3)
        self.ent_Delay_is.grid(row=4, column=3)

        self.ent_WPR_should.grid(row=2, column=4)
        self.ent_WPG_should.grid(row=3, column=4)
        self.ent_Delay_should.grid(row=4, column=4)

        self.but_WPR_Ini.grid(row=2, column=5)
        self.but_WPR_Home.grid(row=2, column=6)
        self.but_WPR_Read.grid(row=2, column=7)
        self.but_WPR_Move.grid(row=2, column=8)

        self.but_WPG_Ini.grid(row=3, column=5)
        self.but_WPG_Home.grid(row=3, column=6)
        self.but_WPG_Read.grid(row=3, column=7)
        self.but_WPG_Move.grid(row=3, column=8)

        self.but_Delay_Ini.grid(row=4, column=5)
        self.but_Delay_Home.grid(row=4, column=6)
        self.but_Delay_Read.grid(row=4, column=7)
        self.but_Delay_Move.grid(row=4, column=8)

        # self.ent_WPR_from.grid(row=2, column=9)
        # self.ent_WPR_to.grid(row=2, column=10)
        # self.ent_WPR_steps.grid(row=2, column=11)

        # self.ent_WPG_from.grid(row=3, column=9)
        # self.ent_WPG_to.grid(row=3, column=10)
        # self.ent_WPG_steps.grid(row=3, column=11)

        # self.ent_Delay_from.grid(row=4, column=9)
        # self.ent_Delay_to.grid(row=4, column=10)
        # self.ent_Delay_steps.grid(row=4, column=11)

        # self.cb_wprscan.grid(row=2, column=12)
        # self.cb_wpgscan.grid(row=3, column=12)
        # self.cb_delayscan.grid(row=4, column=12)

        self.cb_wprpower.grid(row=2, column=9)
        self.cb_wpgpower.grid(row=3, column=9)

        # setting up frm_wp_power_calibration
        self.but_calibrator_open.grid(row=0, column=0)
        lbl_pharos_att.grid(row=0, column=1)
        self.ent_pharos_att.grid(row=0, column=2)
        lbl_pharos_pp.grid(row=1, column=1)
        self.ent_pharos_pp.grid(row=1, column=2)

        lbl_red_power.grid(row=0, column=5, sticky='w')
        self.ent_red_power.grid(row=0, column=6)

        lbl_red_phase.grid(row=1, column=5, sticky='w')
        self.ent_red_phase.grid(row=1, column=6)

        lbl_red_current_power.grid(row=2, column=5, sticky='w')
        self.ent_red_current_power.grid(row=2, column=6)

        lbl_green_power.grid(row=0, column=7, sticky='w')
        self.ent_green_power.grid(row=0, column=8)

        lbl_green_phase.grid(row=1, column=7, sticky='w')
        self.ent_green_phase.grid(row=1, column=8)

        lbl_green_current_power.grid(row=2, column=7, sticky='w')
        self.ent_green_current_power.grid(row=2, column=8)

        # setting up frm_wp_scans
        lbl_wp_scan_info.grid(row=0, column=0, sticky='w')
        self.rb_int_ratio.grid(row=1, column=0, sticky='w')
        self.rb_wpr.grid(row=2, column=0, sticky='w')
        self.rb_wpg.grid(row=3, column=0, sticky='w')
        self.rb_green_focus_SLM.grid(row=4, column=0, sticky='w')
        self.rb_red_focus_SLM.grid(row=5, column=0, sticky='w')
        self.rb_nothing.grid(row=6, column=0, sticky='w')

        lbl_int_ratio_focus.grid(row=1, column=1)
        self.lbl_int_ratio_constant.grid(row=1, column=3)

        lbl_stage_scan_from.grid(row=0, column=8)
        lbl_stage_scan_to.grid(row=0, column=9)
        lbl_stage_scan_steps.grid(row=0, column=10)

        self.ent_int_ratio_focus.grid(row=1, column=2, padx=3, pady=3)
        self.ent_int_ratio_constant.grid(row=1, column=4, padx=3, pady=3)

        lbl_int_green_ratio.grid(row=1, column=7, sticky='w')

        self.ent_ratio_from.grid(row=1, column=8)
        self.ent_ratio_to.grid(row=1, column=9)
        self.ent_ratio_steps.grid(row=1, column=10)

        lbl_int_red.grid(row=2, column=7, sticky='w')
        self.ent_WPR_from.grid(row=2, column=8)
        self.ent_WPR_to.grid(row=2, column=9)
        self.ent_WPR_steps.grid(row=2, column=10)

        lbl_int_green.grid(row=3, column=7, sticky='w')
        self.ent_WPG_from.grid(row=3, column=8)
        self.ent_WPG_to.grid(row=3, column=9)
        self.ent_WPG_steps.grid(row=3, column=10)

        lbl_GFP_green.grid(row=4, column=7, sticky='w')
        self.ent_GFP_from.grid(row=4, column=8)
        self.ent_GFP_to.grid(row=4, column=9)
        self.ent_GFP_steps.grid(row=4, column=10)

        lbl_RFP_red.grid(row=5, column=7, sticky='w')
        self.ent_RFP_from.grid(row=5, column=8)
        self.ent_RFP_to.grid(row=5, column=9)
        self.ent_RFP_steps.grid(row=5, column=10)

        self.figrMCP = Figure(figsize=(5, 5), dpi=100)
        self.axMCP = self.figrMCP.add_subplot(211)
        self.axHarmonics = self.figrMCP.add_subplot(212)
        self.axMCP.set_xlim(0, 1600)
        self.axMCP.set_ylim(0, 1000)
        self.axHarmonics.set_xlim(0, 1600)
        self.figrMCP.tight_layout()
        self.figrMCP.canvas.draw()
        self.imgMCP = FigureCanvasTkAgg(self.figrMCP, frm_mcp_image)
        self.tk_widget_figrMCP = self.imgMCP.get_tk_widget()
        self.tk_widget_figrMCP.grid(row=0, column=0, sticky='nsew')
        self.imgMCP.draw()

        sizefactor = 1.05

        self.figr = Figure(figsize=(5 * sizefactor, 2 * sizefactor), dpi=100)
        self.ax1r = self.figr.add_subplot(211)
        self.ax2r = self.figr.add_subplot(212)
        self.trace_line, = self.ax1r.plot([])
        self.fourier_line, = self.ax2r.plot([])
        self.fourier_indicator = self.ax2r.plot([], 'v')[0]
        self.fourier_text = self.ax2r.text(0.4, 0.5, "")
        self.ax1r.set_xlim(0, 200)
        self.ax1r.set_ylim(0, 3000)
        self.ax1r.grid()
        self.ax2r.set_xlim(0, 50)
        self.ax2r.set_ylim(0, .6)
        self.figr.tight_layout()
        self.figr.canvas.draw()
        self.img1r = FigureCanvasTkAgg(self.figr, frm_plt)
        self.tk_widget_figr = self.img1r.get_tk_widget()
        self.tk_widget_figr.grid(row=0, column=0, sticky='nsew')
        self.img1r.draw()
        self.ax1r_blit = self.figr.canvas.copy_from_bbox(self.ax1r.bbox)
        self.ax2r_blit = self.figr.canvas.copy_from_bbox(self.ax2r.bbox)

        self.figp = Figure(figsize=(5 * sizefactor, 2 * sizefactor), dpi=100)
        self.ax1p = self.figp.add_subplot(111)
        self.phase_line, = self.ax1p.plot([], '.', ms=1)
        self.ax1p.set_xlim(0, 1000)
        self.ax1p.set_ylim([-np.pi, np.pi])
        self.ax1p.grid()
        self.figp.tight_layout()
        self.figp.canvas.draw()
        self.img1p = FigureCanvasTkAgg(self.figp, frm_plt)
        self.tk_widget_figp = self.img1p.get_tk_widget()
        self.tk_widget_figp.grid(row=1, column=0, sticky='nsew')
        self.img1p.draw()
        self.ax1p_blit = self.figp.canvas.copy_from_bbox(self.ax1p.bbox)

        # setting up frm_ratio
        self.ent_area1x.grid(row=0, column=0)
        self.ent_area1y.grid(row=0, column=1)


        lbl_indexfft.grid(row=1, column=0, sticky='e')
        self.ent_indexfft.grid(row=1, column=1)
        lbl_angle.grid(row=2, column=0)
        self.lbl_angle.grid(row=2, column=1)
        lbl_phi.grid(row=3, column=0, sticky='e')
        self.ent_flat.grid(row=3, column=1)
        lbl_phi_2.grid(row=3, column=2, sticky='w')

        self.im_phase = np.zeros(1000)
        self.pid = PID(0.35, 0, 0, setpoint=0)

        self.stop_acquire = 0
        self.stop_pid = False

        self.spec_interface_initialized = False
        self.active_spec_handle = None

    def update_maxgreenratio(self, var, index, mode):
        try:
            x = float(self.ent_int_ratio_focus.get()) ** 2
            c = float(self.ent_int_ratio_constant.get())
            maxG = float(self.ent_green_power.get()) * 1e-3
            self.lbl_int_ratio_constant.config(text='Pr+{:.2f}*PG='.format(x))
            self.strvar_ratio_to.set(str(np.round(x * maxG / (c - x * maxG), 3)))

            # print(x)
            # print(c)
            # print(maxG)
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
        except:
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
        try:
            self.WPG.move_home(blocking=True)
            self.but_WPG_Home.config(fg='green')
            print("WPG homed!")
            self.read_WPG()
        except:
            self.but_WPG_Home.config(fg='red')
            print("Not able to home WPG")

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
                pos = self.power_to_angle(power, float(self.ent_green_power.get()), float(self.ent_green_phase.get()))
            else:
                pos = float(self.strvar_WPG_should.get())

            print("WPG is moving..")
            self.WPG.move_to(pos, True)
            print(f"WPG moved to {str(self.WPG.position)}")
            self.read_WPG()
        except:
            print("Impossible to move WPG :(")

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
        # except:
        #    print("Failure in opening the Calibrator")

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
        with Vimba.get_instance() as vimba:
            cams = vimba.get_all_cameras()
            image = np.zeros([1000, 1600])
            self.d_phase = deque()
            self.meas_has_started = True
            nr = avgs
            with cams[0] as cam:
                for frame in cam.get_frame_generator(limit=avgs):
                    frame = cam.get_frame()
                    frame.convert_pixel_format(PixelFormat.Mono8)
                    img = frame.as_opencv_image()
                    img = np.squeeze(frame.as_opencv_image())
                    numpy_image = img
                    image = image + numpy_image
                image = image / nr
                self.meas_has_started = False

        return image

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
        self.f = open(self.autolog, "a+")
        filename = 'C:/data/' + str(date.today()) + '/' + str(date.today()) + '-' + str(int(nr)) + '.bmp'
        cv2.imwrite(filename, image)

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = str(
            int(nr)) + '\t' + self.ent_red_current_power.get() + '\t' + self.ent_green_current_power.get() + '\t' + str(
            np.round(float(self.strvar_setp.get()), 2)) + '\t' + str(
            np.round(np.mean(np.unwrap(self.d_phase)), 2)) + '\t' + str(
            np.round(np.std(np.unwrap(self.d_phase)),
                     2)) + '\t' + self.ent_mcp.get() + '\t' + self.ent_avgs.get() + '\t' + str(
            np.round(float(self.lens_green.strvar_ben.get()), 3)) + '\t' + str(
            np.round(float(self.lens_red.strvar_ben.get()), 3)) + '\t' + timestamp + '\n'
        self.f.write(log_entry)
        self.f.close()

    def save_image(self, image, image_nr, image_info="Test"):
        """
        Saves the captured image to a file.

        Parameters
        ----------
        image : numpy.ndarray
            The captured image.
        image_nr : int
            The number of the image.
        image_info : str, optional
            Additional information about the image. The default is "Test".

        Returns
        -------
        int
            The status of the save operation (1 for success, 0 for failure).
        """
        self.f = open(self.autolog, "a+")
        filename = 'C:/data/' + str(date.today()) + '/' + str(date.today()) + '-' + str(int(image_nr)) + '.bmp'
        cv2.imwrite(filename, image)
        self.f.write(str(int(image_nr)) + "\t" + image_info + "\n")
        self.f.close()
        return 1

    def enabl_mcp_all(self):
        """
        Enables the MCP measurement.

        Returns
        -------
        None
        """
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
        self.f = open(self.autolog, "a+")
        lines = np.loadtxt(self.autolog, comments="#", delimiter="\t", unpack=False, usecols=(0,))
        if lines.size > 0:
            try:
                start_image = lines[-1] + 1
            except:
                start_image = lines + 1
            print("The last image had index " + str(int(start_image - 1)))
        else:
            start_image = 0
        self.f.close()
        return start_image

    def init_focus_cam_mono(self):

        print('Mono acquisition mode - on ')

        # create a device manager
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()

        if dev_num == 0:
            print("No connected devices")
            return

        # open the first device
        self.cam = device_manager.open_device_by_index(int(1))

        # set exposure
        self.cam.ExposureTime.set(float(1000))  # TODO exposure box

        # set gain
        self.cam.Gain.set(float(1))  # TODO gain box

        if dev_info_list[0].get("device_class") == gx.GxDeviceClassList.USB2:
            # set trigger mode
            self.cam.TriggerMode.set(gx.GxSwitchEntry.ON)
        else:
            # set trigger mode and trigger source
            self.cam.TriggerMode.set(gx.GxSwitchEntry.ON)
            self.cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)

        self.cam.stream_on()
        self.focus_cam_acq_mono(int(1))  # TODO average box
        self.cam.stream_off()
        self.cam.close_device()

        print('Mono acquisition mode - off ')

    def focus_cam_acq_mono(self, num):
        """
        acquisition function for camera in single picture mode
        """
        sum_image = None

        for i in range(num):
            self.cam.TriggerSoftware.send_command()
            self.cam.ExposureTime.set(float(10000))  # TODO gain box
            self.cam.Gain.set(float(1))  # TODO exposure box

            raw_image = self.cam.data_stream[0].get_image()
            if raw_image is None:
                print('oups')
                continue
            numpy_image = raw_image.get_numpy_array()
            if numpy_image is None:
                print('oups')
                continue

            if sum_image is None:
                sum_image = numpy_image.astype('float64')
            else:
                sum_image += numpy_image.astype('float64')

        average_image = (sum_image / num).astype('uint8')
        print(average_image)

        picture = Image.fromarray(average_image)
        picture = picture.resize((800, 600), resample=0)
        picture = ImageTk.PhotoImage(picture)

        self.focus_cam_save = True #TODO possibility to change

        if self.focus_cam_save:
            folder_path = 'C:/data/' + str(date.today()) + '/' + 'focus_camera' + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            nr = self.get_start_image()
            base_filename = 'focus_camera' + str(date.today()) + '-' + str(int(nr))

            filename = base_filename + '.bmp'

            full_path = os.path.join(folder_path, filename)

            last_image = picture

            if last_image is not None:
                pil_image = ImageTk.getimage(last_image)
                pil_image.save(full_path)
                print(f"Image saved as {full_path}")
            else:
                print("No image to save")

        print(f'Mono acquisition mode - Image taken over {num} averages')

    def focus_cam_mono_acq(self):
        self.render_thread_mono = threading.Thread(target=self.init_focus_cam_mono)
        self.render_thread_mono.daemon = True
        self.render_thread_mono.start()

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

        for i in np.arange(0, steps):
            r = pr[i]
            g = pg[i]
            self.strvar_WPR_should.set(str(r))
            self.move_WPR()
            self.strvar_WPG_should.set(str(1e3 * g))
            self.move_WPG()
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
        time.sleep(1)
        print("Ready to scan the phase!")
        for ind, phi in enumerate(self.phis):
            start_time = time.time()
            self.strvar_setp.set(phi)
            self.set_setpoint()
            im = self.take_image(int(self.ent_avgs.get()))
            self.save_im(im)
            self.plot_MCP(im)
            #self.focus_cam_mono_acq()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Imagenr ", (start_image + ind), " Phase: ", round(phi, 2), " Elapsed time: ", round(elapsed_time, 2))

    def get_power_values_for_ratio_scan(self):
        c = float(self.strvar_int_ratio_constant.get())
        x = float(self.ent_int_ratio_focus.get()) ** 2
        ratios = np.linspace(float(self.ent_ratio_from.get()), float(self.ent_ratio_to.get()),
                             int(self.ent_ratio_steps.get()))
        pr = c / (1 + ratios)
        pg = ratios * pr / x
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
            if b == 0:
                b = 0.00001
            self.lens_green.strvar_ben.set(str(b))
            self.parent.open_pub_green()
            if self.var_phasescan.get() == 1 and self.var_background.get() == 0:
                self.phase_scan()
            else:
                im = self.take_image(int(self.ent_avgs.get()))
                self.save_im(im)
                self.plot_MCP(im)
                #self.focus_cam_mono_acq()

    def focus_position_scan_red(self):
        start = float(self.ent_RFP_from.get())
        end = float(self.ent_RFP_to.get())
        steps = float(self.ent_RFP_steps.get())

        for ind, b in enumerate(np.linspace(start, end, int(steps))):
            # things go to hell if division by zero
            if b == 0:
                b = 0.00001
            self.lens_red.strvar_ben.set(str(b))
            self.parent.open_pub_red()
            if self.var_phasescan.get() == 1 and self.var_background.get() == 0:
                self.phase_scan()
            else:
                im = self.take_image(int(self.ent_avgs.get()))
                self.save_im(im)
                self.plot_MCP(im)
                #self.focus_cam_mono_acq()

    def measure_all(self):
        self.but_meas_all.config(fg='red')
        self.f = open(self.autolog, "a+")

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

        self.f.close()
        self.but_meas_all.config(fg='green')

    def measure(self):
        """
        Performs a phase scan

        Returns
        -------
        None
        """
        self.but_meas_scan.config(fg='red')

        # if self.var_phasescan.get() == 1:
        self.f = open(self.autolog, "a+")
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
        self.f.close()

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
        self.f = open(self.autolog, "a+")

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
        self.f.close()

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

            start = int(self.ent_area1x.get())
            stop = int(self.ent_area1y.get())
            self.trace = data[start:stop]

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
        self.ax1r.set_xlim(0, len(self.trace))
        self.ax1r.set_ylim(0, np.max(self.trace) * 1.2)
        self.ax1r.grid('both')
        self.figr.canvas.draw()
        self.img1r.draw()
        self.ax1r_blit = self.figr.canvas.copy_from_bbox(self.ax1r.bbox)

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
        self.axMCP.clear()
        self.axMCP.imshow(mcpimage, vmin=0, vmax=2, extent=[0, 1600, 0, 1000])
        self.axMCP.set_aspect('equal')

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
        self.ax1r.plot(self.trace)
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
        self.trace_line.set_data(np.arange(len(self.trace)), self.trace)
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

    def on_close(self):
        """
        Close the program.

        Closes the figures, cleans up the APT module, deactivates the spectrometer,
        and destroys the window.

        Returns
        -------
        None
        """
        # self.f.close()
        plt.close(self.figr)
        plt.close(self.figp)
        self.disable_motors()

        self.spec_deactivate()
        avs.AVS_Done()
        self.win.destroy()
        self.parent.feedback_win = None
        print('Feedbacker closed')
