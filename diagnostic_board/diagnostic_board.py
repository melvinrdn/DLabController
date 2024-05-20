import datetime
import threading
import time
import csv
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import h5py
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
from scipy import interpolate
import pandas as pd

import diagnostic_board.daheng_camera as dh
from diagnostic_board.beam_treatment_functions import process_image_new
from diagnostic_board.nl_pr import NLOptimizer
from drivers.thorlabs_apt_driver import core as apt
from drivers import gxipy_driver as gx


class DiagnosticBoard:
    def __init__(self, parent):
        self.cam = None
        self.roi = None
        self.wavelength = 1030e-9

        self.initial_roi = (0, 540, 0, 720)  # x1 x2 y1 y2
        self.current_roi = self.initial_roi
        self.relevant_image = None
        self.current_M2_exposures = []
        self.roi_rectangle = None
        self.default_zoom_green = (0, 720, 0, 540)
        self.default_zoom_red = (0, 720, 0, 540)

        self.parent = parent
        self.lens_green = self.parent.phase_refs_green[1]
        self.lens_red = self.parent.phase_refs_red[1]

        self.C = None

        self.abort = 0

        self.win = tk.Toplevel()
        vcmd = self.win.register(self.is_number_input)

        title = 'D-Lab Controller - Diagnostic Board'

        self.win.title(title)
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

        self.frm_cam = ttk.LabelFrame(self.win, text="Camera display")
        self.frm_main_settings = ttk.LabelFrame(self.win, text="Main settings")
        frm_wavelength = ttk.LabelFrame(self.frm_main_settings, text="Wavelength presets")
        frm_controls = ttk.LabelFrame(self.frm_main_settings, text="Camera settings")
        frm_stage = ttk.LabelFrame(self.frm_main_settings, text="Stage settings")
        self.frm_scan_settings = ttk.LabelFrame(self.frm_main_settings, text="Scan settings")

        self.frm_notebook_diagnostics = ttk.Notebook(self.win)
        self.frm_m2_diagnostics = ttk.Frame(self.frm_notebook_diagnostics)
        self.frm_phase_retrieval = ttk.Frame(self.frm_notebook_diagnostics)
        self.frm_notebook_diagnostics.add(self.frm_m2_diagnostics, text='M2 Diagnostics')
        self.frm_notebook_diagnostics.add(self.frm_phase_retrieval, text="Phase retrieval")
        self.frm_notebook_diagnostics.grid(row=0, column=3, sticky='nsew')

        self.frm_m2_parameters = ttk.LabelFrame(self.frm_m2_diagnostics, text="M2 Parameters")
        self.frm_m2_plot = ttk.LabelFrame(self.frm_m2_diagnostics, text="M2 Plot")

        self.frm_notebook_phase_retrieval = ttk.Notebook(self.frm_phase_retrieval)
        self.frm_pr_settings = ttk.Frame(self.frm_notebook_phase_retrieval)
        self.frm_notebook_phase_retrieval.add(self.frm_pr_settings, text="Nonlinear Optimization")
        self.frm_notebook_phase_retrieval.grid(row=0, column=0, sticky='nsew')

        self.frm_notebook_phase_retrieval_plots = ttk.Notebook(self.frm_phase_retrieval)
        self.frm_pr_plot_reconstruction = ttk.Frame(self.frm_notebook_phase_retrieval_plots)
        self.frm_notebook_phase_retrieval_plots.add(self.frm_pr_plot_reconstruction, text="PR - Reconstruction")
        self.frm_notebook_phase_retrieval_plots.grid(row=1, column=0, sticky='nsew')

        self.frm_cam.grid(row=0, column=0, sticky='nsew')
        self.frm_main_settings.grid(row=0, column=1, sticky='nsew')
        frm_wavelength.grid(row=0, column=0, sticky='nsew')
        frm_controls.grid(row=1, column=0, sticky='nsew')
        frm_stage.grid(row=2, column=0, sticky='nsew')
        self.frm_scan_settings.grid(row=3, column=0, sticky='nsew')

        self.frm_m2_parameters.grid(row=0, column=0, sticky='nsew')
        self.frm_m2_plot.grid(row=1, column=0, sticky='nsew')

        self.open_beam_profile = ttk.Button(self.frm_pr_settings, text='Open beam profile ',
                                            command=self.open_beam_profile)
        self.open_beam_profile.grid(row=0, column=0, sticky='nsew')
        self.take_beam_profile = ttk.Button(self.frm_pr_settings, text='Take beam profile ',
                                            command=self.open_beam_profile)
        self.take_beam_profile.grid(row=1, column=0, sticky='nsew')

        self.run_pr = ttk.Button(self.frm_pr_settings, text='Run PR', command=self.run_phase_retrieval)
        self.run_pr.grid(row=0, column=1, sticky='nsew')
        self.send_to_slm = ttk.Button(self.frm_pr_settings, text='Send to SLM ',
                                      command=self.send_correction_to_SLM)
        self.send_to_slm.grid(row=1, column=1, sticky='nsew')

        lbl_max_zn_coef = ttk.Label(self.frm_pr_settings, text="# Zn")
        self.var_max_zn_coef = tk.StringVar(self.win, value="16")
        self.ent_max_zn_coef = ttk.Entry(self.frm_pr_settings, textvariable=self.var_max_zn_coef)
        lbl_max_zn_coef.grid(row=0, column=2, sticky='nsew')
        self.ent_max_zn_coef.grid(row=0, column=3, sticky='nsew')

        self.figr_pr = Figure(figsize=(6, 4), dpi=100)
        self.axs_pr = self.figr_pr.subplots(2, 2)
        self.img_pr = FigureCanvasTkAgg(self.figr_pr, self.frm_pr_plot_reconstruction)
        self.tk_widget_pr = self.img_pr.get_tk_widget()
        self.tk_widget_pr.grid(row=0, column=0, sticky='nsew')
        self.img_pr.draw()
        self.axs_blit_pr = self.figr_pr.canvas.copy_from_bbox(self.axs_pr[0, 0].bbox)

        self.figr = Figure(figsize=(6, 4), dpi=100)
        self.ax1r = self.figr.add_subplot(111)
        self.ax1r.grid()
        self.ax2 = self.ax1r.twinx()
        self.figr.tight_layout()
        self.figr.canvas.draw()
        self.img1r = FigureCanvasTkAgg(self.figr, self.frm_m2_plot)
        self.tk_widget_figr = self.img1r.get_tk_widget()
        self.tk_widget_figr.grid(row=0, column=0, sticky='nsew')
        self.img1r.draw()
        self.ax1r_blit = self.figr.canvas.copy_from_bbox(self.ax1r.bbox)


        lbl_daheng_stage_from = ttk.Label(self.frm_scan_settings, text="From")
        lbl_daheng_stage_to = ttk.Label(self.frm_scan_settings, text="To")
        lbl_daheng_stage_steps = ttk.Label(self.frm_scan_settings, text="Steps")
        lbl_daheng_stage_from.grid(row=0, column=0)
        lbl_daheng_stage_to.grid(row=0, column=1)
        lbl_daheng_stage_steps.grid(row=0, column=2)

        lbl_comment = ttk.Label(self.frm_scan_settings, text="Comment:")
        self.strvar_comment = tk.StringVar(self.win, '')
        self.ent_comment = ttk.Entry(self.frm_scan_settings, validate='none', textvariable=self.strvar_comment,
                                     width=5)

        lbl_comment.grid(row=0, column=3, sticky='nsew')
        self.ent_comment.grid(row=1, column=3, sticky='nsew')

        self.var_daheng_stage_from = tk.StringVar(self.win, value="3")
        self.ent_daheng_stage_from = ttk.Entry(self.frm_scan_settings, textvariable=self.var_daheng_stage_from, width=5)
        self.ent_daheng_stage_from.grid(row=1, column=0, sticky='nsew')

        self.var_daheng_stage_to = tk.StringVar(self.win, value="9")
        self.ent_daheng_stage_to = ttk.Entry(self.frm_scan_settings, textvariable=self.var_daheng_stage_to,
                                             width=5)
        self.ent_daheng_stage_to.grid(row=1, column=1, sticky='nsew')

        self.var_daheng_stage_steps = tk.StringVar(self.win, value="13")
        self.ent_daheng_stage_steps = ttk.Entry(self.frm_scan_settings,
                                                textvariable=self.var_daheng_stage_steps,
                                                width=5)
        self.ent_daheng_stage_steps.grid(row=1, column=2, sticky='nsew')

        self.but_backgrounds = ttk.Button(self.frm_scan_settings, text="Get backgrounds",
                                         command=self.find_focus_thread)
        self.but_backgrounds.grid(row=1, column=4, sticky='nsew')

        self.but_scan_daheng = ttk.Button(self.frm_scan_settings, text="Get M2",
                                          command=self.scan_daheng_thread)
        self.but_scan_daheng.grid(row=2, column=4, sticky='nsew')

        self.open_h5_file = ttk.Button(self.frm_m2_parameters, text='Open h5 file', command=self.open_h5_file)
        self.open_h5_file.grid(row=0, column=0, sticky='nsew')

        frm_cam_but = ttk.Frame(frm_controls)
        frm_cam_but_set = ttk.Frame(frm_cam_but)

        self.str_wavelength = tk.StringVar(self.win, "Red")
        self.rb_red = tk.Radiobutton(frm_wavelength, variable=self.str_wavelength, value="Red", text="Red",
                                     command=self.wavelength_presets)
        self.rb_red.select()
        self.rb_green = tk.Radiobutton(frm_wavelength, variable=self.str_wavelength, value="Green", text="Green",
                                       command=self.wavelength_presets)

        self.rb_red.grid(row=0, column=0, sticky='nsew')
        self.rb_green.grid(row=0, column=1, sticky='nsew')

        self.but_cam_init = ttk.Button(frm_cam_but, text='Initialize', command=self.initialize_daheng)
        self.but_cam_disconnect = ttk.Button(frm_cam_but, text='Disconnect', command=self.close_daheng)
        self.but_cam_live = ttk.Button(frm_cam_but, text='Live', command=self.live_daheng_thread)
        self.but_cam_single = ttk.Button(frm_cam_but, text='Single', command=self.single_daheng_thread)

        lbl_cam_ind = ttk.Label(frm_cam_but_set, text='Camera index:')
        self.strvar_cam_ind = tk.StringVar(self.win, '3')
        self.ent_cam_ind = ttk.Entry(frm_cam_but_set, width=8,
                                     textvariable=self.strvar_cam_ind, validate='key',
                                     validatecommand=(vcmd, '%P'))

        lbl_cam_exp = ttk.Label(frm_cam_but_set, text='Exposure (µs):')
        self.strvar_cam_exp = tk.StringVar(self.win, '100000')
        self.ent_cam_exp = ttk.Entry(frm_cam_but_set, width=8, validate='key',
                                     validatecommand=(vcmd, '%P'),
                                     textvariable=self.strvar_cam_exp)

        self.automatic_exposure = tk.IntVar(value=0)
        self.box_auto_exp = ttk.Checkbutton(frm_cam_but_set, text='Auto exposure',
                                            variable=self.automatic_exposure,
                                            onvalue=1, offvalue=0)

        lbl_cam_gain = ttk.Label(frm_cam_but_set, text='Gain (0-24):')
        self.strvar_cam_gain = tk.StringVar(self.win, '0')
        self.ent_cam_gain = ttk.Entry(frm_cam_but_set, width=8, validate='key',
                                      validatecommand=(vcmd, '%P'),
                                      textvariable=self.strvar_cam_gain)

        lbl_cam_avg = ttk.Label(frm_cam_but_set, text='Nbr of averages :')
        self.strvar_cam_avg = tk.StringVar(self.win, '1')
        self.ent_cam_avg = ttk.Entry(frm_cam_but_set, width=8, validate='key',
                                     validatecommand=(vcmd, '%P'),
                                     textvariable=self.strvar_cam_avg)

        lbl_roi_x1 = ttk.Label(frm_cam_but_set, text='ROI x1:')
        self.strvar_roi_x1 = tk.StringVar(self.win, str(self.initial_roi[2]))
        self.ent_roi_x1 = ttk.Entry(frm_cam_but_set, width=8, validate='key',
                                    validatecommand=(vcmd, '%P'),
                                    textvariable=self.strvar_roi_x1)

        lbl_roi_x2 = ttk.Label(frm_cam_but_set, text='ROI x2:')
        self.strvar_roi_x2 = tk.StringVar(self.win, str(self.initial_roi[3]))
        self.ent_roi_x2 = ttk.Entry(frm_cam_but_set, width=8, validate='key',
                                    validatecommand=(vcmd, '%P'),
                                    textvariable=self.strvar_roi_x2)

        lbl_roi_y1 = ttk.Label(frm_cam_but_set, text='ROI y1:')
        self.strvar_roi_y1 = tk.StringVar(self.win, str(self.initial_roi[0]))
        self.ent_roi_y1 = ttk.Entry(frm_cam_but_set, width=8, validate='key',
                                    validatecommand=(vcmd, '%P'),
                                    textvariable=self.strvar_roi_y1)

        lbl_roi_y2 = ttk.Label(frm_cam_but_set, text='ROI y2:')
        self.strvar_roi_y2 = tk.StringVar(self.win, str(self.initial_roi[1]))
        self.ent_roi_y2 = ttk.Entry(frm_cam_but_set, width=8, validate='key',
                                    validatecommand=(vcmd, '%P'),
                                    textvariable=self.strvar_roi_y2)

       # lbl_upper_saturation_threshold = ttk.Label(self.frm_scan_settings, text='Up threshold:')
       # self.strvar_upper_saturation_threshold = tk.StringVar(self.win, '0.0000005')
       # self.ent_upper_saturation_threshold = ttk.Entry(self.frm_scan_settings, width=20,
        #                                                textvariable=self.strvar_upper_saturation_threshold,
        #                                                validate='key',
        #                                                validatecommand=(vcmd, '%P'))

       # lbl_lower_saturation_threshold = ttk.Label(self.frm_scan_settings, text='Lower threshold:')
       # self.strvar_lower_saturation_threshold = tk.StringVar(self.win, '0.0000001')
       # self.ent_lower_saturation_threshold = ttk.Entry(self.frm_scan_settings, width=20,
       #                                                 textvariable=self.strvar_lower_saturation_threshold,
       #                                                 validate='key',
       #                                                 validatecommand=(vcmd, '%P'))

        #lbl_increase_factor = ttk.Label(self.frm_scan_settings, text='Increase factor:')
        #self.strvar_increase_factor = tk.StringVar(self.win, '1.02')
        #self.ent_increase_factor = ttk.Entry(self.frm_scan_settings, width=20,
        #                                     textvariable=self.strvar_increase_factor, validate='key',
        #                                     validatecommand=(vcmd, '%P'))

        #lbl_decrease_factor = ttk.Label(self.frm_scan_settings, text='Decrease factor:')
        #self.strvar_decrease_factor = tk.StringVar(self.win, '0.99')
        #self.ent_decrease_factor = ttk.Entry(self.frm_scan_settings, width=20,
        #                                     textvariable=self.strvar_decrease_factor, validate='key',
        #                                     validatecommand=(vcmd, '%P'))

        #lbl_increase_factor.grid(row=2, column=0, sticky='nsew')
        #self.ent_increase_factor.grid(row=2, column=1, sticky='nsew')
        #lbl_decrease_factor.grid(row=3, column=0, sticky='nsew')
        #self.ent_decrease_factor.grid(row=3, column=1, sticky='nsew')
        #lbl_upper_saturation_threshold.grid(row=4, column=0, sticky='nsew')
        #self.ent_upper_saturation_threshold.grid(row=4, column=1, sticky='nsew')
        #lbl_lower_saturation_threshold.grid(row=5, column=0, sticky='nsew')
        #self.ent_lower_saturation_threshold.grid(row=5, column=1, sticky='nsew')
        self.but_abort = ttk.Button(self.frm_scan_settings, text="Abort scan",
                                    command=self.abort_scan)
        self.but_abort.grid(row=3, column=4, sticky='nsew')
        #self.but_abort = ttk.Button(self.frm_scan_settings, text="Stop optimization",
        #                            command=self.stop_optimization)
        #self.but_abort.grid(row=3, column=5, sticky='nsew')

        lbl_roi_x1.grid(row=2, column=0, sticky='nsew')
        self.ent_roi_x1.grid(row=2, column=1, padx=(0, 10))
        lbl_roi_x2.grid(row=2, column=2, sticky='nsew')
        self.ent_roi_x2.grid(row=2, column=3, padx=(0, 10))

        lbl_roi_y1.grid(row=3, column=0, sticky='nsew')
        self.ent_roi_y1.grid(row=3, column=1, padx=(0, 10))
        lbl_roi_y2.grid(row=3, column=2, sticky='nsew')
        self.ent_roi_y2.grid(row=3, column=3, padx=(0, 10))

        lbl_cam_ind.grid(row=0, column=0, sticky='nsew')
        self.ent_cam_ind.grid(row=0, column=1, padx=(0, 10))
        frm_cam_but_set.grid(row=0, column=0, sticky='nsew')
        frm_cam_but.grid(row=1, column=0, sticky='nsew')
        lbl_cam_exp.grid(row=6, column=0, sticky='nsew')
        self.ent_cam_exp.grid(row=6, column=1, padx=(0, 10))
        self.box_auto_exp.grid(row=6, column=2, padx=(0, 10))
        lbl_cam_gain.grid(row=7, column=0, sticky='nsew')
        self.ent_cam_gain.grid(row=7, column=1, padx=(0, 10))
        lbl_cam_avg.grid(row=8, column=0, sticky='nsew')
        self.ent_cam_avg.grid(row=8, column=1, padx=(0, 10))

        self.but_cam_init.grid(row=1, column=0, sticky='nsew')
        self.but_cam_disconnect.grid(row=2, column=0, sticky='nsew')
        self.but_cam_live.grid(row=3, column=0, sticky='nsew')
        self.but_cam_single.grid(row=4, column=0, sticky='nsew')

        lbl_Stage = ttk.Label(frm_stage, text='Stage')
        lbl_Nr = ttk.Label(frm_stage, text='#')
        lbl_is = ttk.Label(frm_stage, text='is')
        lbl_should = ttk.Label(frm_stage, text='should')

        lbl_WPcam = ttk.Label(frm_stage, text='Camera Stage:')
        self.strvar_WPcam_is = tk.StringVar(self.win, '')
        self.ent_WPcam_is = ttk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_WPcam_is)
        self.strvar_WPcam_should = tk.StringVar(self.win, '')
        self.ent_WPcam_should = ttk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_WPcam_should)
        self.strvar_WPcam_Nr = tk.StringVar(self.win, '83837725')
        self.ent_WPcam_Nr = ttk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_WPcam_Nr)

        self.but_WPcam_Ini = ttk.Button(frm_stage, text='Initialize', command=self.init_WPcam)
        self.but_WPcam_Home = ttk.Button(frm_stage, text='Home', command=self.home_WPcam)
        self.but_WPcam_Read = ttk.Button(frm_stage, text='Read', command=self.read_WPcam)
        self.but_WPcam_Move = ttk.Button(frm_stage, text='Move', command=self.move_WPcam)

        lbl_Stage.grid(row=0, column=1, pady=2, sticky='nsew')
        lbl_Nr.grid(row=0, column=2, pady=2, sticky='nsew')
        lbl_is.grid(row=0, column=3, pady=2, sticky='nsew')
        lbl_should.grid(row=0, column=4, pady=2, sticky='nsew')

        lbl_WPcam.grid(row=1, column=1, pady=2, sticky='nsew')
        self.ent_WPcam_Nr.grid(row=1, column=2, pady=2, sticky='nsew')
        self.ent_WPcam_is.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_WPcam_should.grid(row=1, column=4, padx=2, pady=2, sticky='nsew')

        self.but_WPcam_Ini.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')
        self.but_WPcam_Home.grid(row=2, column=2, padx=2, pady=2, sticky='nsew')
        self.but_WPcam_Read.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')
        self.but_WPcam_Move.grid(row=2, column=4, padx=2, pady=2, sticky='nsew')

        self.img_canvas = tk.Canvas(self.frm_cam, height=540, width=720)
        self.img_canvas.grid(row=0, sticky='nsew')
        self.img_canvas.configure(bg='grey')
        self.image = self.img_canvas.create_image(0, 0, anchor="nw")

        self.img_canvas_param = ttk.Frame(self.frm_cam)
        self.img_canvas_param.grid(row=1, sticky='nsew')
        self.pos_label = ttk.Label(self.img_canvas_param, text="Crosshair position: x=0, y=0")
        self.pos_label.grid(row=0, column=0, sticky='nsew')
        self.crosshair_status_label = ttk.Label(self.img_canvas_param, text="Crosshair: Shown")
        self.crosshair_status_label.grid(row=0, column=1, sticky='nsew')

        self.horizontal_line = self.img_canvas.create_line(0, 0, 720, 0, fill='red', dash=(4, 2))
        self.vertical_line = self.img_canvas.create_line(0, 0, 0, 540, fill='red', dash=(4, 2))

        self.output_console = ScrolledText(self.img_canvas_param, height=10, state='disabled')
        self.output_console.grid(row=1, column=0, columnspan=4, sticky='ew')

        self.img_canvas.bind('<Motion>', self.update_crosshair)
        self.img_canvas.bind('<Button-1>', self.start_roi_selection)
        self.img_canvas.bind('<B1-Motion>', self.update_roi_selection)
        self.img_canvas.bind('<ButtonRelease-1>', self.end_roi_selection)
        self.img_canvas.bind('<Button-3>', self.toggle_crosshair_lock)
        self.win.bind('<r>', self.reset_roi)
        self.win.bind('<c>', self.toggle_crosshair_visibility)

        self.crosshair_visible = True
        self.crosshair_locked = False

        frm_bottom = ttk.Frame(self.win)
        frm_bottom.grid(row=3, column=0, columnspan=2)
        but_exit = ttk.Button(frm_bottom, text='Exit', command=self.on_close)
        but_exit.grid(row=3, column=0, padx=5, pady=5, ipadx=5, ipady=5)

        self.daheng_camera = None
        self.daheng_is_live = False
        self.current_daheng_image = None
        self.WPcam = None
        self.optimized = False

        self.autolog_camera_focus = 'C:/data/' + str(datetime.date.today()) + '/' + str(
            datetime.date.today()) + '-' + 'auto-log-camera_focus.txt'
        self.autolog_cam = open(self.autolog_camera_focus, "a+")

        self.styleR = ttk.Style()
        self.styleR.configure('RED', foreground='red')
        self.styleG = ttk.Style()
        self.styleG.configure('GREEN', foreground='green')
        self.styleB =  ttk.Style()
        self.styleB.configure('BLACK', foreground='black')

    def is_number_input(self, P):
        if P.strip() == "":
            return True
        try:
            float(P)
            return True
        except ValueError:
            return False

    def insert_message(self, message):
        self.output_console.configure(state='normal')
        self.output_console.insert(tk.END, message + "\n")
        self.output_console.configure(state='disabled')
        self.output_console.see(tk.END)

    def toggle_crosshair_lock(self, event):
        self.crosshair_locked = not self.crosshair_locked
        if not self.crosshair_locked:
            self.update_crosshair_position(event.x, event.y)

    def update_crosshair(self, event):
        if not self.crosshair_locked:
            self.update_crosshair_position(event.x, event.y)

    def update_crosshair_position(self, x, y):
        self.img_canvas.coords(self.horizontal_line, 0, y, self.img_canvas.winfo_width(), y)
        self.img_canvas.coords(self.vertical_line, x, 0, x, self.img_canvas.winfo_height())
        self.pos_label.config(text=f"Position: x={x:.2f}, y={y:.2f}")

    def toggle_crosshair_visibility(self, event):
        self.crosshair_visible = not self.crosshair_visible
        if self.crosshair_visible:
            self.img_canvas.itemconfig(self.horizontal_line, state='normal')
            self.img_canvas.itemconfig(self.vertical_line, state='normal')
            self.crosshair_status_label.config(text="Crosshair: Shown")
        else:
            self.img_canvas.itemconfig(self.horizontal_line, state='hidden')
            self.img_canvas.itemconfig(self.vertical_line, state='hidden')
            self.crosshair_status_label.config(text="Crosshair: Hidden")

    def crop_array(self, C, extent_slm, xlim, ylim):
        x_proportion = (xlim[1] - xlim[0]) / (extent_slm[1] - extent_slm[0])
        y_proportion = (ylim[1] - ylim[0]) / (extent_slm[3] - extent_slm[2])
        x_elements = int(x_proportion * C.shape[1])
        y_elements = int(y_proportion * C.shape[0])
        x_start = (C.shape[1] - x_elements) // 2
        y_start = (C.shape[0] - y_elements) // 2
        cropped_C = C[y_start:y_start + y_elements, x_start:x_start + x_elements]
        return cropped_C

    def send_correction_to_SLM(self):

        C = self.crop_array(self.C, self.extent_slm, xlim=[-7.68e-3, 7.68e-3], ylim=[-4.8e-3, 4.8e-3]) # thoses limit correspond to the slm size in m
        plt.imshow(C)
        plt.show()

        x = np.arange(0, C.shape[1])
        y = np.arange(0, C.shape[0])
        x_new = np.linspace(0, C.shape[1], 1921)
        y_new = np.linspace(0, C.shape[0], 1201)
        f = interpolate.interp2d(x, y, C, kind='linear')
        C_interpolated = f(x_new, y_new)


        normalized_C = (C_interpolated - np.min(C_interpolated)) / (
                    np.max(C_interpolated) - np.min(C_interpolated)) * 1023

        normalized_C = np.clip(normalized_C, 0, 1023)

        df = pd.DataFrame(normalized_C.astype(int))
        df.to_csv('C_interpolated.csv', index=False, header=False)

    def run_phase_retrieval(self):
        self.phase_retrieval_thread = threading.Thread(target=self.open_phase_retrieval)
        self.phase_retrieval_thread.daemon = True
        self.phase_retrieval_thread.start()

    def open_phase_retrieval(self):
        print(int(self.var_max_zn_coef.get()))
        F = self.image_array
        if self.image_array is not None:
            nlopt = NLOptimizer(int(self.var_max_zn_coef.get()), F)
            nlopt.phase_retrieval()
        else:
            message = 'There is no beam profile loaded'
            self.insert_message(message)
        print('done')



    def get_M_sq(self, som_x, som_y, z, lambda_0, dx):
        def beam_quality_factor_fit(z, w0, M2, z0):
            return w0 * np.sqrt(1 + (z - z0) ** 2 * (M2 * lambda_0 / (np.pi * w0 ** 2)) ** 2)

        p0 = [20e-6, 1, 0]

        params_x, _ = curve_fit(beam_quality_factor_fit, z, som_x, p0=p0)
        w0_x_fit, M_sq_x_fit, z0_x_fit = params_x
        message = f'M_sq_x: {abs(M_sq_x_fit):.4f},w0_x: {w0_x_fit * 1e6:.4f} µm, z0_x: {z0_x_fit * 1e3:.4f} mm'
        self.insert_message(message)

        params_y, _ = curve_fit(beam_quality_factor_fit, z, som_y, p0=p0)
        w0_y_fit, M_sq_y_fit, z0_y_fit = params_y
        message = f'M_sq_y: {abs(M_sq_y_fit):.4f},w0_y: {w0_y_fit * 1e6:.4f} µm, z0_y: {z0_y_fit * 1e3:.4f} mm'
        self.insert_message(message)

        z_fit = np.linspace(z[0], z[-1], 100)

        self.ax1r.clear()
        self.ax1r.grid(True)

        self.ax1r.plot(z*1e3, som_x*1e6, linestyle='None', marker='x', color='blue')
        self.ax1r.plot(z*1e3, som_y*1e6, linestyle='None', marker='x', color='red')
        self.ax1r.plot(z_fit*1e3, beam_quality_factor_fit(z_fit, w0_y_fit, M_sq_y_fit, z0_y_fit)*1e6,
                       label=f'M_sq_y: {abs(params_y[1]):.2f}, '
                             f'w0_y: {params_y[0] * 1e6:.2f} µm, '
                             f'z0_y: {params_y[2] * 1e3:.2f} mm', color='red')
        self.ax1r.plot(z_fit*1e3, beam_quality_factor_fit(z_fit, w0_x_fit, M_sq_x_fit, z0_x_fit)*1e6,
                       label=f'M_sq_x: {abs(params_x[1]):.2f}, '
                             f'w0_x: {params_x[0] * 1e6:.2f} µm, '
                             f'z0_x: {params_x[2] * 1e3:.2f} mm', color='blue')

        self.ax1r.set_ylabel(r'$w$ (2nd order moment, μm)')
        self.ax1r.set_xlabel('Stage position (mm)')
        self.ax1r.legend()
        self.figr.tight_layout()
        self.img1r.draw()

        return params_x, params_y

    def change_method(self, event):
        selected_method = self.strvar_method_choice.get()

        if selected_method == 'Vortex':
            message = f'Method selected: {selected_method}'
            self.insert_message(message)

        elif selected_method == 'Gerchberg-Saxton':
            message = f'Method selected: {selected_method}'
            self.insert_message(message)

    def open_h5_file(self):
        filepath = tk.filedialog.askopenfilename()
        try:
            message = f'Opening {filepath}'
            self.insert_message(message)
            hfr = h5py.File(filepath, 'r')
            self.images = np.asarray(hfr.get('images'))
            self.positions = np.asarray(hfr.get('positions'))

            message = 'Successfully loaded'
            self.insert_message(message)
            processed_images, som_x, som_y = self.process_images_dict()
            zero_position = 0
            zmin = (self.positions[0] - zero_position) * 1e-3
            zmax = (self.positions[-1] - zero_position) * 1e-3  # in mm
            z = np.linspace(zmin, zmax, len(self.positions))
            if self.wavelength is not None:
                params_x, params_y = self.get_M_sq(som_x, som_y, z, float(self.wavelength), 3.45e-6)
            else:
                message = 'Please select a wavelength'
                self.insert_message(message)

        except:
            message = 'Impossible to open the file'
            self.insert_message(message)

    def open_beam_profile(self):
        filepath = tk.filedialog.askopenfilename()
        try:
            message = f'Opening {filepath}'
            self.insert_message(message)
            self.image_array = np.array(Image.open(filepath))
            message = 'Successfully loaded'
            self.insert_message(message)

        except:
            message = 'Impossible to open the file'
            self.insert_message(message)

    def process_images_dict(self):
        processed_images = {}
        dz = self.images.shape[2]
        message = f'Number of steps: {dz}'
        self.insert_message(message)
        som_x = np.zeros(dz, dtype=float)
        som_y = np.zeros(dz, dtype=float)

        for i in range(dz):
            processed_image, som_x[i], som_y[i] = process_image_new(self.images[:, :, i])
            processed_images[f'processed_image_{i}'] = processed_image

        som_y *= 3.45e-6
        som_x *= 3.45e-6
        return processed_images, som_x, som_y

    def wavelength_presets(self):
        status = self.str_wavelength.get()

        if status == "Green":
            message = f"Green (515 nm) is selected"
            self.insert_message(message)
            self.wavelength = 515e-9
            self.current_roi = self.default_zoom_green
            message = f'ROI selected :{self.current_roi}'
            self.insert_message(message)

        elif status == "Red":
            message = f"Red (1030 nm) is selected"
            self.insert_message(message)
            self.wavelength = 1030e-9
            self.current_roi = self.default_zoom_red
            message = f'ROI selected :{self.current_roi}'
            self.insert_message(message)

    def initialize_daheng(self):
        device_manager = gx.DeviceManager()
        try:
            self.daheng_camera = dh.DahengCamera(int(self.strvar_cam_ind.get()))
            #self.but_cam_init.config(style='GREEN')
        except:
            self.insert_message("Something went wrong with init of camera:(")
            #self.but_cam_init.config(style='RED')

    def close_daheng(self):
        self.daheng_camera.close_daheng()
        #self.but_cam_init.config(style='BLACK')
        #self.but_cam_disconnect.config(style='GREEN')
        self.daheng_camera = None

    def single_daheng_thread(self):
        self.daheng_thread = threading.Thread(target=self.take_single_image_daheng)
        self.daheng_thread.daemon = True
        self.daheng_thread.start()

    def live_daheng_thread(self):
        self.daheng_is_live = not self.daheng_is_live
        self.update_daheng_live_button()
        self.daheng_thread = threading.Thread(target=self.live_daheng)
        self.daheng_thread.daemon = True
        self.daheng_thread.start()

    def find_focus_thread(self):
        self.daheng_is_live = False
        self.daheng_thread = threading.Thread(target=self.get_backgrounds)
        self.daheng_thread.daemon = True
        self.daheng_thread.start()


    def get_backgrounds(self):
        self.daheng_is_live = False
        # self.but_cam_live.config(style='BLACK')
        from_ = float(self.var_daheng_stage_from.get())
        to_ = float(self.var_daheng_stage_to.get())
        steps_ = int(self.var_daheng_stage_steps.get())
        stage_steps = np.linspace(from_, to_, steps_)

        if self.WPcam is None:
            self.insert_message('Stage is not connected')
            return
        if self.daheng_camera is None:
            self.insert_message('Camera is not connected')
            return

        rel_size1 = self.relevant_image.shape[0]
        rel_size2 = self.relevant_image.shape[1]
        #        res = np.zeros([self.daheng_camera.imshape[0], self.daheng_camera.imshape[1], int(steps_)])
        res = np.zeros([rel_size1, rel_size2, int(steps_)])

        exposures = []
        self.ax1r.clear()
        # self.ax2 = self.ax1r.twinx()
        self.ax2.clear()
        self.ax1r.grid(True)

        if np.asarray(self.current_M2_exposures).size != 0:
            for ind, pos in enumerate(stage_steps):
                if self.abort == 1:
                    message = 'Aborted'
                    self.insert_message(message)
                    break
                self.strvar_WPcam_should.set(pos)
                self.move_WPcam()
                #self.adjust_exposure_new()
                #self.optimized = False
                optimized_exposure = self.current_M2_exposures[ind]
                self.insert_message(f'Position: {pos}, Exposure: {optimized_exposure} µs')
                im = self.daheng_camera.take_image(int(optimized_exposure), int(self.strvar_cam_gain.get()),
                                                   int(self.strvar_cam_avg.get()))
                self.plot_daheng(im)

                res[:, :, ind] = self.relevant_image
                #imtest, temx, temy = process_image_new(self.relevant_image)
                self.ax1r.scatter(pos, np.mean(self.relevant_image), color="blue")
                self.ax1r.scatter(pos, np.max(self.relevant_image), color="black")
                self.ax1r.set_ylabel(r'Background')
                self.ax1r.set_xlabel('Stage position (mm)')
                #self.ax1r.set_xlim([np.min(stage_steps), np.max(stage_steps)])
                #intim = self.relevant_image / np.sum(self.relevant_image)
                #maxval = np.max(intim)
                #self.ax2.scatter(pos, maxval, color="black", marker="x")
                # self.ax2.set_ylabel(r'Intensity')
                self.figr.tight_layout()
                self.img1r.draw()
                #exposures.append(optimized_exposure)
            if self.abort == 0:
                #self.current_M2_exposures = exposures
                self.save_daheng_scans(res, stage_steps, exposures, bg = 1)

    def abort_scan(self):
        if self.abort == 0:
            self.abort = 1
            message = 'self.abort=1'
            self.insert_message(message)
        else:
            self.abort = 0
            message = 'self.abort=0'
            self.insert_message(message)

    def stop_optimization(self):
        self.optimized = True
        self.insert_message("Optimization manually stopped")

    def scan_daheng_thread(self):
        self.daheng_is_live = False
        self.daheng_thread = threading.Thread(target=self.scan_stage_daheng)
        self.daheng_thread.daemon = True
        self.daheng_thread.start()

    def scan_stage_daheng(self):
        self.daheng_is_live = False
        #self.but_cam_live.config(style='BLACK')
        from_ = float(self.var_daheng_stage_from.get())
        to_ = float(self.var_daheng_stage_to.get())
        steps_ = int(self.var_daheng_stage_steps.get())
        stage_steps = np.linspace(from_, to_, steps_)

        if self.WPcam is None:
            self.insert_message('Stage is not connected')
            return
        if self.daheng_camera is None:
            self.insert_message('Camera is not connected')
            return

        rel_size1 = self.relevant_image.shape[0]
        rel_size2 = self.relevant_image.shape[1]
#        res = np.zeros([self.daheng_camera.imshape[0], self.daheng_camera.imshape[1], int(steps_)])
        res = np.zeros([rel_size1, rel_size2, int(steps_)])

        exposures = []
        self.ax1r.clear()
        #self.ax2 = self.ax1r.twinx()
        self.ax2.clear()
        self.ax1r.grid(True)

        for ind, pos in enumerate(stage_steps):
            if self.abort == 1:
                message = 'Aborted'
                self.insert_message(message)
                break
            self.strvar_WPcam_should.set(pos)
            self.move_WPcam()
            self.adjust_exposure_new()
            self.optimized = False
            optimized_exposure = float(self.strvar_cam_exp.get())
            self.insert_message(f'Position: {pos}, Exposure: {optimized_exposure} µs')
            im = self.daheng_camera.take_image(int(optimized_exposure), int(self.strvar_cam_gain.get()),
                                               int(self.strvar_cam_avg.get()))
            self.plot_daheng(im)

            res[:, :, ind] = self.relevant_image
            imtest,temx,temy = process_image_new(self.relevant_image)
            self.ax1r.scatter(pos, temx, color ="blue")
            self.ax1r.scatter(pos, temy, color="red")
            self.ax1r.set_ylabel(r'$w$ (2nd order moment, μm)')
            self.ax1r.set_xlabel('Stage position (mm)')
            self.ax1r.set_xlim([np.min(stage_steps), np.max(stage_steps)])
            intim = self.relevant_image/np.sum(self.relevant_image)
            maxval = np.max(intim)
            self.ax2.scatter(pos, maxval, color="black", marker = "x")
            #self.ax2.set_ylabel(r'Intensity')
            self.figr.tight_layout()
            self.img1r.draw()
            exposures.append(optimized_exposure)
        if self.abort == 0:
            self.current_M2_exposures = exposures
            self.save_daheng_scans(res, stage_steps, exposures)

    def save_daheng_scans(self, res, pos, exp, bg = 0):
        nr = self.get_start_image_images()
        if bg == 0:
            self.autolog_cam.write("#" + self.ent_comment.get() + "\n")
        else:
            self.autolog_cam.write("#" + " BACKGROUND for "+ self.ent_comment.get() + "\n")

        data_filename = 'C:/data/' + str(datetime.date.today()) + '/' + str(
            datetime.date.today()) + '-camera_focus-' + str(
            int(nr)) + '.h5'

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        log_entry = str(int(nr)) + '\t' + str(pos[0]) + '\t' + str(pos[-1]) + '\t' + str(
            int(np.size(pos))) + '\t' + timestamp + '\n'
        self.autolog_cam.write(log_entry)
        hf = h5py.File(data_filename, 'w')
        hf.create_dataset('images', data=res)
        hf.create_dataset('positions', data=pos)
        hf.create_dataset('exposure', data=exp)
        hf.create_dataset('gain', data=int(self.strvar_cam_gain.get()))
        hf.create_dataset('average', data=int(self.strvar_cam_avg.get()))
        hf.close()

    def get_start_image_images(self):
        self.autolog_cam.seek(0)
        lines = np.loadtxt(self.autolog_camera_focus, comments="#", delimiter="\t", unpack=False, usecols=(0,))
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

    def update_daheng_live_button(self):
        if self.daheng_is_live == True:
            message = 'Live view on'
            #self.but_cam_live.config(style='GREEN')
            self.insert_message(message)
        else:
            message = 'Live view off'
            #self.but_cam_live.config(style='BLACK')
            self.insert_message(message)


    def adjust_exposure_new(self):
        min_exposure = 20
        max_exposure = 900000
        if self.daheng_is_live == True:
            rel = self.relevant_image
            if np.max(rel) > 254:
                new_exposure = int(self.strvar_cam_exp.get())/2
                self.strvar_cam_exp.set(str(int(new_exposure)))
            else:
                new_exposure = 255/np.max(rel) * 0.8 * int(self.strvar_cam_exp.get())
                self.strvar_cam_exp.set(str(int(new_exposure)))
                self.optimized = True

        else:
            while not self.optimized:
                im = self.daheng_camera.take_image(int(self.strvar_cam_exp.get()), int(self.strvar_cam_gain.get()),
                                                   int(self.strvar_cam_avg.get()))
                self.plot_daheng(im)
                rel = self.relevant_image
                if np.max(rel) > 254:
                    new_exposure = int(self.strvar_cam_exp.get())/2
                    self.strvar_cam_exp.set(str(int(new_exposure)))
                else:
                    new_exposure = 255/np.max(rel) * 0.8 * int(self.strvar_cam_exp.get())
                    self.strvar_cam_exp.set(str(int(new_exposure)))
                    im = self.daheng_camera.take_image(int(self.strvar_cam_exp.get()), int(self.strvar_cam_gain.get()),
                                                       int(self.strvar_cam_avg.get()))
                    self.plot_daheng(im)
                    self.optimized = True
                    self.insert_message(f"Exposure is optimized at {new_exposure} µs.")

    def live_daheng(self):
        if self.daheng_camera is not None:
            while self.daheng_is_live:
                im = self.daheng_camera.take_image(int(self.strvar_cam_exp.get()), int(self.strvar_cam_gain.get()),
                                                   int(self.strvar_cam_avg.get()))
                self.current_daheng_image = im
                self.plot_daheng(im)

                if self.automatic_exposure.get():
                    #self.simple_adjust_exposure(im)
                    self.adjust_exposure_new()
        else:
            self.daheng_is_live = False
            self.insert_message('self.daheng_camera is None')


    def take_single_image_daheng(self):
        if self.daheng_camera is not None:
            im = self.daheng_camera.take_image(int(self.strvar_cam_exp.get()), int(self.strvar_cam_gain.get()),
                                               int(self.strvar_cam_avg.get()))
            message = "Single image taken"
            self.insert_message(message)
            self.current_daheng_image = im
            self.plot_daheng(im)
        else:
            message = 'self.daheng_camera is None'
            self.insert_message(message)

    def plot_daheng(self, im):
        try:
            x1 = max(0, min(im.shape[1], int(self.strvar_roi_x1.get()) * 2))
            x2 = max(0, min(im.shape[1], int(self.strvar_roi_x2.get()) * 2))
            y1 = max(0, min(im.shape[0], int(self.strvar_roi_y1.get()) * 2))
            y2 = max(0, min(im.shape[0], int(self.strvar_roi_y2.get()) * 2))

            if x1 >= x2 or y1 >= y2:
                message = "Invalid ROI coordinates. Select a zone from up to down."
                self.insert_message(message)
                return

            cropped_im = im[y1:y2, x1:x2]
            self.relevant_image = cropped_im
            if np.max(self.relevant_image)>254:
                self.frm_cam.config(text = "SATURATED!")
            elif np.max(self.relevant_image)>240:
                self.frm_cam.config(text="ALMOST saturated, max: {}".format(int(np.max(self.relevant_image))))
            else:
                self.frm_cam.config(text="Camera display, max: {}".format(int(np.max(self.relevant_image))))


            image = Image.fromarray(cropped_im)
            image_resized = image.resize((720, 540), resample=0)
            photo = ImageTk.PhotoImage(image_resized)

            self.img_canvas.itemconfig(self.image, image=photo)
            self.img_canvas.image = photo
        except ValueError as e:
            message = f"Error processing the image: {e}"
            self.insert_message(message)

    def start_roi_selection(self, event):
        if self.roi_rectangle:
            self.img_canvas.delete(self.roi_rectangle)
        self.roi_start = (event.x, event.y)
        self.roi_rectangle = self.img_canvas.create_rectangle(self.roi_start[0], self.roi_start[1],
                                                              self.roi_start[0], self.roi_start[1],
                                                              outline='blue', dash=(4, 2))

    def update_roi_selection(self, event):
        self.img_canvas.coords(self.roi_rectangle, self.roi_start[0], self.roi_start[1], event.x, event.y)

    def end_roi_selection(self, event):
        self.roi_end = (event.x, event.y)
        self.img_canvas.coords(self.roi_rectangle, self.roi_start[0], self.roi_start[1], self.roi_end[0],
                               self.roi_end[1])
        self.current_roi = (self.roi_start[1], self.roi_end[1], self.roi_start[0], self.roi_end[0])

        self.strvar_roi_x1.set(str(self.current_roi[2]))
        self.strvar_roi_x2.set(str(self.current_roi[3]))
        self.strvar_roi_y1.set(str(self.current_roi[0]))
        self.strvar_roi_y2.set(str(self.current_roi[1]))

        self.img_canvas.delete(self.roi_rectangle)
        self.roi_rectangle = None

    def reset_roi(self, event):
        self.current_roi = self.initial_roi

        self.strvar_roi_x1.set(str(self.current_roi[2]))
        self.strvar_roi_x2.set(str(self.current_roi[3]))
        self.strvar_roi_y1.set(str(self.current_roi[0]))
        self.strvar_roi_y2.set(str(self.current_roi[1]))

    def init_WPcam(self):
        try:
            self.WPcam = apt.Motor(int(self.ent_WPcam_Nr.get()))
            message = "WPcam connected"
            self.insert_message(message)
            #self.but_WPcam_Ini.config(style='GREEN')
        except:
            message = "Not able to initalize WPcam"
            #self.but_WPcam_Ini.config(style='RED')
            self.insert_message(message)

    def home_WPcam(self):

        try:
            self.WPcam.move_home(blocking=True)
            message = "WPcam homed"
            self.insert_message(message)
            self.read_WPcam()
        except:
            message = "Not able to home WPcam"
            self.insert_message(message)

    def read_WPcam(self):
        try:
            pos = self.WPcam.position
            self.strvar_WPcam_is.set(pos)
        except:
            message = "Impossible to read WPcam position"
            self.insert_message(message)

    def move_WPcam(self):
        try:
            pos = float(self.strvar_WPcam_should.get())
            message = "WPcam is moving to {}".format(np.round(pos, 4))
            self.insert_message(message)
            self.WPcam.move_to(pos, True)
            message = "WPcam moved to {}".format(np.round(self.WPcam.position, 4))
            self.insert_message(message)
            self.read_WPcam()
        except Exception as e:
            message = "Impossible to move WPcam"
            self.insert_message(message)

    def disable_motors(self):
        if self.WPcam is not None:
            self.WPcam.disable()
            message = 'WPcam disconnected'
            self.insert_message(message)

    def on_close(self):
        self.autolog_cam.close()
        if self.daheng_camera is not None:
            self.close_daheng()
        self.disable_motors()
        self.win.destroy()
