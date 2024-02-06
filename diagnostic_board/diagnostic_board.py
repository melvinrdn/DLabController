import datetime
import threading
import time
import csv
import tkinter as tk
from tkinter import ttk

import h5py
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

import diagnostic_board.focus_diagnostic as dh
from diagnostic_board.beam_treatment_functions import process_image
import diagnostic_board.phase_retrieval as pr
from drivers.thorlabs_apt_driver import core as apt
from drivers import gxipy_driver as gx

class DiagnosticBoard:
    def __init__(self, parent):
        self.cam = None
        self.roi = None
        self.wavelength = 1030e-9

        self.initial_roi = (0, 600, 0, 400)
        self.current_roi = self.initial_roi
        self.roi_rectangle = None
        self.default_zoom_green = (0, 600, 0, 400)
        self.default_zoom_red = (0, 600, 0, 400)

        self.parent = parent
        self.lens_green = self.parent.phase_refs_green[1]
        self.lens_red = self.parent.phase_refs_red[1]

        self.C = None

        self.win = tk.Toplevel()

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
        self.frm_beam_profile_settings = ttk.Frame(self.frm_notebook_phase_retrieval)
        self.frm_notebook_phase_retrieval.add(self.frm_pr_settings, text='PR settings')
        self.frm_notebook_phase_retrieval.add(self.frm_beam_profile_settings, text="Beam profile settings")
        self.frm_notebook_phase_retrieval.grid(row=0, column=0, sticky='nsew')

        self.frm_notebook_phase_retrieval_plots = ttk.Notebook(self.frm_phase_retrieval)
        self.frm_pr_plot_reconstruction = ttk.Frame(self.frm_notebook_phase_retrieval_plots)
        self.frm_pr_plot_correction = ttk.Frame(self.frm_notebook_phase_retrieval_plots)
        self.frm_notebook_phase_retrieval_plots.add(self.frm_pr_plot_reconstruction, text="PR - Reconstruction")
        self.frm_notebook_phase_retrieval_plots.add(self.frm_pr_plot_correction, text='PR - Correction')
        self.frm_notebook_phase_retrieval_plots.grid(row=1, column=0, sticky='nsew')

        self.frm_cam.grid(row=0, column=0, sticky='nsew')
        self.frm_main_settings.grid(row=0, column=1, sticky='nsew')
        frm_wavelength.grid(row=0, column=0, sticky='nsew')
        frm_controls.grid(row=1, column=0, sticky='nsew')
        frm_stage.grid(row=2, column=0, sticky='nsew')
        self.frm_scan_settings.grid(row=3, column=0, sticky='nsew')

        self.frm_m2_parameters.grid(row=0, column=0, sticky='nsew')
        self.frm_m2_plot.grid(row=1, column=0, sticky='nsew')

        lbl_method_choice = tk.Label(self.frm_pr_settings, text='Method chosen :')
        self.strvar_method_choice = tk.StringVar(self.frm_phase_retrieval, 'Gerchberg-Saxton')
        self.cbox_method_choice = ttk.Combobox(self.frm_pr_settings, textvariable=self.strvar_method_choice)
        self.cbox_method_choice.bind("<<ComboboxSelected>>", self.change_method)
        self.cbox_method_choice['values'] = ('Vortex', 'Gerchberg-Saxton')
        lbl_method_choice.grid(row=0, column=0, sticky='nsew')
        self.cbox_method_choice.grid(row=0, column=1, sticky='nsew')
        self.run_pr = tk.Button(self.frm_pr_settings, text='Run PR', command=self.run_phase_retrieval)
        self.run_pr.grid(row=0, column=2)

        self.send_to_slm = tk.Button(self.frm_pr_settings, text='Send correction to SLM ', command=self.send_correction_to_SLM)
        self.send_to_slm.grid(row=0, column=3)

        self.open_beam_profile = tk.Button(self.frm_beam_profile_settings, text='Open beam profile ', command=self.open_beam_profile)
        self.open_beam_profile.grid(row=0, column=0)

        self.figr_pr = Figure(figsize=(6, 4), dpi=100)
        self.axs_pr = self.figr_pr.subplots(2, 2)
        self.img_pr = FigureCanvasTkAgg(self.figr_pr, self.frm_pr_plot_reconstruction)
        self.tk_widget_pr = self.img_pr.get_tk_widget()
        self.tk_widget_pr.grid(row=0, column=0, sticky='nsew')
        self.img_pr.draw()
        self.axs_blit_pr = self.figr_pr.canvas.copy_from_bbox(self.axs_pr[0, 0].bbox)

        self.figr_pr_corr = Figure(figsize=(6, 4), dpi=100)
        self.axs_pr_corr = self.figr_pr_corr.subplots(1, 2)
        self.img_pr_corr = FigureCanvasTkAgg(self.figr_pr_corr, self.frm_pr_plot_correction)
        self.tk_widget_pr_corr = self.img_pr_corr.get_tk_widget()
        self.tk_widget_pr_corr.grid(row=0, column=0, sticky='nsew')
        self.img_pr_corr.draw()
        self.axs_blit_pr_corr = self.figr_pr_corr.canvas.copy_from_bbox(self.axs_pr_corr[0].bbox)

        self.figr = Figure(figsize=(6, 4), dpi=100)
        self.ax1r = self.figr.add_subplot(111)
        self.ax1r.grid()
        self.figr.tight_layout()
        self.figr.canvas.draw()
        self.img1r = FigureCanvasTkAgg(self.figr, self.frm_m2_plot)
        self.tk_widget_figr = self.img1r.get_tk_widget()
        self.tk_widget_figr.grid(row=0, column=0, sticky='nsew')
        self.img1r.draw()
        self.ax1r_blit = self.figr.canvas.copy_from_bbox(self.ax1r.bbox)

        lbl_max_iter = tk.Label(self.frm_pr_settings, text="Max iteration")
        self.var_max_iter = tk.StringVar(self.win, value="200")
        self.ent_max_iter = tk.Entry(self.frm_pr_settings, textvariable=self.var_max_iter)
        lbl_max_iter.grid(row=1, column=0)
        self.ent_max_iter.grid(row=1, column=1)

        lbl_tolerance = tk.Label(self.frm_pr_settings, text="Tolerance")
        self.var_tolerance = tk.StringVar(self.win, value="1e-7")
        self.ent_tolerance = tk.Entry(self.frm_pr_settings, textvariable=self.var_tolerance)
        lbl_tolerance.grid(row=2, column=0)
        self.ent_tolerance.grid(row=2, column=1)

        lbl_daheng_stage_from = tk.Label(self.frm_scan_settings, text="From")
        lbl_daheng_stage_to = tk.Label(self.frm_scan_settings, text="To")
        lbl_daheng_stage_steps = tk.Label(self.frm_scan_settings, text="Steps")
        lbl_daheng_stage_from.grid(row=0, column=0)
        lbl_daheng_stage_to.grid(row=0, column=1)
        lbl_daheng_stage_steps.grid(row=0, column=2)

        lbl_comment = tk.Label(self.frm_scan_settings, text="Comment:")
        self.strvar_comment = tk.StringVar(self.win, '')
        self.ent_comment = tk.Entry(self.frm_scan_settings, validate='none',textvariable=self.strvar_comment,
                                            width=5)

        lbl_comment.grid(row=0, column=3, sticky='nsew')
        self.ent_comment.grid(row=1, column=3, sticky='nsew')

        self.var_daheng_stage_from = tk.StringVar(self.win, value="0")
        self.ent_daheng_stage_from = tk.Entry(self.frm_scan_settings, textvariable=self.var_daheng_stage_from, width=5)
        self.ent_daheng_stage_from.grid(row=1, column=0, sticky='nsew')

        self.var_daheng_stage_to = tk.StringVar(self.win, value="14")
        self.ent_daheng_stage_to = tk.Entry(self.frm_scan_settings, textvariable=self.var_daheng_stage_to,
                                            width=5)
        self.ent_daheng_stage_to.grid(row=1, column=1, sticky='nsew')

        self.var_daheng_stage_steps = tk.StringVar(self.win, value="15")
        self.ent_daheng_stage_steps = tk.Entry(self.frm_scan_settings,
                                               textvariable=self.var_daheng_stage_steps,
                                               width=5)
        self.ent_daheng_stage_steps.grid(row=1, column=2, sticky='nsew')

        self.but_scan_daheng = tk.Button(self.frm_scan_settings, text="Stage scan",
                                         command=self.scan_daheng_thread)
        self.but_scan_daheng.grid(row=1, column=4, sticky='nsew')

        self.open_h5_file = tk.Button(self.frm_m2_parameters, text='Open h5 file', command=self.open_h5_file)
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

        self.but_cam_init = tk.Button(frm_cam_but, text='Initialize', command=self.initialize_daheng)
        self.but_cam_disconnect = tk.Button(frm_cam_but, text='Disconnect', command=self.close_daheng)
        self.but_cam_live = tk.Button(frm_cam_but, text='Live', command=self.live_daheng_thread)
        self.but_cam_single = tk.Button(frm_cam_but, text='Single', command=self.single_daheng_thread)

        lbl_cam_ind = ttk.Label(frm_cam_but_set, text='Camera index:')
        self.strvar_cam_ind = tk.StringVar(self.win, '2')
        self.ent_cam_ind = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                     textvariable=self.strvar_cam_ind)

        lbl_cam_exp = ttk.Label(frm_cam_but_set, text='Exposure (µs):')
        self.strvar_cam_exp = tk.StringVar(self.win, '10000')
        self.ent_cam_exp = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                     textvariable=self.strvar_cam_exp)

        lbl_cam_gain = ttk.Label(frm_cam_but_set, text='Gain (0-24):')
        self.strvar_cam_gain = tk.StringVar(self.win, '0')
        self.ent_cam_gain = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                      textvariable=self.strvar_cam_gain)

        lbl_cam_avg = ttk.Label(frm_cam_but_set, text='Nbr of averages :')
        self.strvar_cam_avg = tk.StringVar(self.win, '1')
        self.ent_cam_avg = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                     textvariable=self.strvar_cam_avg)

        lbl_roi_x1 = ttk.Label(frm_cam_but_set, text='ROI x1:')
        self.strvar_roi_x1 = tk.StringVar(self.win, str(self.initial_roi[2]))
        self.ent_roi_x1 = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                    textvariable=self.strvar_roi_x1)

        lbl_roi_x2 = ttk.Label(frm_cam_but_set, text='ROI x2:')
        self.strvar_roi_x2 = tk.StringVar(self.win, str(self.initial_roi[3]))
        self.ent_roi_x2 = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                    textvariable=self.strvar_roi_x2)

        lbl_roi_y1 = ttk.Label(frm_cam_but_set, text='ROI y1:')
        self.strvar_roi_y1 = tk.StringVar(self.win, str(self.initial_roi[0]))
        self.ent_roi_y1 = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                    textvariable=self.strvar_roi_y1)

        lbl_roi_y2 = ttk.Label(frm_cam_but_set, text='ROI y2:')
        self.strvar_roi_y2 = tk.StringVar(self.win, str(self.initial_roi[1]))
        self.ent_roi_y2 = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                    textvariable=self.strvar_roi_y2)

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
        lbl_cam_gain.grid(row=7, column=0, sticky='nsew')
        self.ent_cam_gain.grid(row=7, column=1, padx=(0, 10))
        lbl_cam_avg.grid(row=8, column=0, sticky='nsew')
        self.ent_cam_avg.grid(row=8, column=1, padx=(0, 10))

        self.but_cam_init.grid(row=1, column=0, sticky='nsew')
        self.but_cam_disconnect.grid(row=2, column=0, sticky='nsew')
        self.but_cam_live.grid(row=3, column=0, sticky='nsew')
        self.but_cam_single.grid(row=4, column=0, sticky='nsew')

        lbl_Stage = tk.Label(frm_stage, text='Stage')
        lbl_Nr = tk.Label(frm_stage, text='#')
        lbl_is = tk.Label(frm_stage, text='is')
        lbl_should = tk.Label(frm_stage, text='should')

        lbl_WPcam = tk.Label(frm_stage, text='Camera Stage:')
        self.strvar_WPcam_is = tk.StringVar(self.win, '')
        self.ent_WPcam_is = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_WPcam_is)
        self.strvar_WPcam_should = tk.StringVar(self.win, '')
        self.ent_WPcam_should = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_WPcam_should)
        self.strvar_WPcam_Nr = tk.StringVar(self.win, '83837725')
        self.ent_WPcam_Nr = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_WPcam_Nr)

        self.but_WPcam_Ini = tk.Button(frm_stage, text='Initialize', command=self.init_WPcam)
        self.but_WPcam_Home = tk.Button(frm_stage, text='Home', command=self.home_WPcam)
        self.but_WPcam_Read = tk.Button(frm_stage, text='Read', command=self.read_WPcam)
        self.but_WPcam_Move = tk.Button(frm_stage, text='Move', command=self.move_WPcam)

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

        self.img_canvas = tk.Canvas(self.frm_cam, height=400, width=600)
        self.img_canvas.grid(row=0, sticky='nsew')
        self.img_canvas.configure(bg='grey')
        self.image = self.img_canvas.create_image(0, 0, anchor="nw")

        self.img_canvas_param = tk.Frame(self.frm_cam)
        self.img_canvas_param.grid(row=1, sticky='nsew')
        self.pos_label = tk.Label(self.img_canvas_param, text="Crosshair position: x=0, y=0")
        self.pos_label.grid(row=0,column=0, sticky='nsew')
        self.crosshair_status_label = tk.Label(self.img_canvas_param, text="Crosshair: Shown")
        self.crosshair_status_label.grid(row=0,column=1, sticky='nsew')

        self.horizontal_line = self.img_canvas.create_line(0, 0, 600, 0, fill='red', dash=(4, 2))
        self.vertical_line = self.img_canvas.create_line(0, 0, 0, 400, fill='red', dash=(4, 2))

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
        self.daheng_zoom = None
        self.WPcam = None

        self.autolog_camera_focus = 'C:/data/' + str(datetime.date.today()) + '/' + str(
            datetime.date.today()) + '-' + 'auto-log-camera_focus.txt'
        self.autolog_cam = open(self.autolog_camera_focus, "a+")

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

    def send_correction_to_SLM(self):
        #TODO Recuperer C, interpoler, envoyer sur SLM. Actualiser sur l'interface principale
        # Essayer de save C as csv et interpoler apres
        filename='output_test.csv'
        with open(filename, 'w',newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(self.C)

        return 0

    def run_phase_retrieval(self):

        max_iter = int(self.ent_max_iter.get())
        tolerance = float(self.ent_tolerance.get())

        factor_plot = 2
        factor_slm = 1.6
        with_vortex = 1

        f = 25e-2
        w_L = 4.0e-3
        x_slm_max = 5 * w_L

        method = self.strvar_method_choice.get()
        if method == 'Gerchberg-Saxton':
            on = 0
        elif method == 'Vortex':
            on = 1
        else:
            print('Please select a PR method')

        if self.wavelength is None:
            print('Please select a wavelength')
        else:
            lambda_0 = self.wavelength

        beam_param = [lambda_0, f, w_L]

        y_slm_max = x_slm_max
        N_x = 2 ** 7
        N_y = N_x
        slm_param = [x_slm_max, y_slm_max, N_x, N_y]

        x_slm, y_slm, extent_slm = pr.define_slm_plan(slm_param, print_option=False)
        intensity_slm = pr.gaussian_2D(x_slm, y_slm, w_L)
        phase_vortex = pr.create_phase_vortex(x_slm, y_slm, w_L, vortex_order=1) * on
        E_slm = np.sqrt(intensity_slm) * np.exp(1j * phase_vortex)

        if on == 1:
            path = "./diagnostic_board/red_vortex_for_vgs_demo.bmp"
        else:
            path = "./diagnostic_board/red_focus_for_gs_demo.bmp"

        #image_array = np.array(Image.open(path))
        image_focus, som_x, som_y = process_image(self.image_array)
        image_focus = pr.zero_padding_to_second(image_focus, E_slm)

        dx = 3.45e-6
        x_focus = np.arange(-N_x // 2, N_x // 2) * dx
        y_focus = x_focus
        extent_focus = [x_focus[0], x_focus[-1], y_focus[0], y_focus[-1]]

        intensity_slm_zp = pr.zero_padding_2N(abs(E_slm) ** 2)

        E_focus = np.sqrt(image_focus)
        som_x_focus, som_y_focus = pr.get_som(abs(E_focus) ** 2, dx)
        som_max = np.max([som_x_focus, som_y_focus])
        E_focus_cut = pr.cut_focus(E_focus, x_focus, y_focus, som_max, factor=2)

        # Create vortex
        phase_vortex_pr = pr.create_phase_vortex(x_slm, y_slm, w_L, 1) * on
        # Zero padding
        phase_vortex_pr_zp = pr.zero_padding_2N(phase_vortex_pr)
        E_focus_cut_zp = pr.zero_padding_2N(E_focus_cut)
        # Initial guess for phase retrieval
        E_slm_pr_zp = np.sqrt(intensity_slm_zp) * np.exp(1j * phase_vortex_pr_zp)

        corr_list = []

        t_start = time.time()
        for i in np.linspace(0, max_iter - 1, max_iter):
            print(f'{((i + 1) / max_iter) * 100:.1f} %')
            # Propagation to the focus
            E_focus_pr_zp = pr.fft2c(E_slm_pr_zp)
            corr_temp = abs(np.corrcoef(E_focus_pr_zp.flatten(), E_focus_cut_zp.flatten())[0, 1])
            corr_list.append(corr_temp)
            print(f"Pearson coeff: {corr_temp}")
            # Impose amplitude of the target (ugly vortex)
            A = abs(E_focus_cut_zp) * np.exp(1j * np.angle(E_focus_pr_zp))
            # Back propagation to the slm
            B = pr.ifft2c(A)
            # Impose amplitude of the gaussian on the SLM + the phase retrieved
            E_slm_pr_zp = np.sqrt(intensity_slm_zp) * np.exp(1j * np.angle(B))

            if i > 0 and abs((corr_list[-1] - corr_list[-2]) / corr_list[-2]) < tolerance:
                print('-----')
                print(f"Converged with {int(i + 1)} iterations")
                break

        t_end = time.time()
        print(f"Loop time: {t_end - t_start:.5f} s")
        print('-----')

        # Clip the array into original form
        E_slm_pr = pr.clip_to_original_size(E_slm_pr_zp, E_focus_cut)
        E_focus_pr = pr.clip_to_original_size(E_focus_pr_zp, E_focus_cut)

        # Calculate the correction pattern
        H = np.angle(E_slm_pr)  # Distorted phase pattern on the slm
        C = H - phase_vortex_pr * with_vortex  # Correction to apply on the slm for the gaussian beam if we come with a vortex
        self.C = (C + np.pi) % (2 * np.pi) - np.pi  # Correction pattern wrapped

        E_slm_fourier_zp = pr.ifft2c(E_focus_cut_zp * np.exp(1j * np.angle(E_focus_pr_zp)))
        # Padding of the E_slm_corrected before propagation
        E_slm_corrected_zp = E_slm_fourier_zp * np.exp(-1j * pr.zero_padding_2N(C)) * np.exp(-1j * phase_vortex_pr_zp)
        # Propagation
        E_focus_corrected_zp = pr.fft2c(E_slm_corrected_zp)
        # Clip to original size
        E_focus_corrected = pr.clip_to_original_size(E_focus_corrected_zp, E_focus_cut)

        # Cutting the arrays
        C = pr.cut_slm(C, x_slm, y_slm, beam_param, factor=factor_slm)
        E_focus_cut = pr.cut_focus(E_focus, x_focus, y_focus, som_max, factor=factor_plot)
        E_focus_pr_cut = pr.cut_focus(E_focus_pr, x_focus, y_focus, som_max, factor=factor_plot)
        E_focus_corrected_cut = pr.cut_focus(E_focus_corrected, x_focus, y_focus, som_max, factor=factor_plot)

        def add_colorbar(ax, im, cmap='turbo'):
            cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, cmap=cmap)

        # TODO fix plot (figures must be cleared before plotting another one)

        im1 = self.axs_pr[0, 0].imshow(abs(E_focus_cut) ** 2, extent=extent_focus, cmap='turbo')
        self.axs_pr[0, 0].set_xlim((-factor_plot * som_max, factor_plot * som_max))
        self.axs_pr[0, 0].set_ylim((-factor_plot * som_max, factor_plot * som_max))
        self.axs_pr[0, 0].set_ylabel('y [m]')
        self.axs_pr[0, 0].set_xlabel('x [m]')
        self.axs_pr[0, 0].set_title('$|E(x,y)|^2$ Measured')
        add_colorbar(self.axs_pr[0, 0], im1, cmap='turbo')

        im2 = self.axs_pr[0, 1].imshow(abs(E_focus_pr_cut) ** 2, extent=extent_focus, cmap='turbo')
        self.axs_pr[0, 1].set_xlim((-factor_plot * som_max, factor_plot * som_max))
        self.axs_pr[0, 1].set_ylim((-factor_plot * som_max, factor_plot * som_max))
        self.axs_pr[0, 1].set_ylabel('y [m]')
        self.axs_pr[0, 1].set_xlabel('x [m]')
        self.axs_pr[0, 1].set_title('$|E(x,y)|^2$ Retrieved')
        add_colorbar(self.axs_pr[0, 1], im2, cmap='turbo')

        im3 = self.axs_pr[1, 0].imshow(np.angle(E_focus_pr_cut), extent=extent_focus, cmap='hsv')
        self.axs_pr[1, 0].set_xlim((-factor_plot * som_max, factor_plot * som_max))
        self.axs_pr[1, 0].set_ylim((-factor_plot * som_max, factor_plot * som_max))
        self.axs_pr[1, 0].set_ylabel('y [m]')
        self.axs_pr[1, 0].set_xlabel('x [m]')
        self.axs_pr[1, 0].set_title('$W(x,y)$ Retrieved')
        add_colorbar(self.axs_pr[1, 0], im3, cmap='bwr')

        self.axs_pr[1, 1].imshow(pr.complex2rgb(E_focus_pr_cut), extent=extent_focus, cmap='hsv')
        self.axs_pr[1, 1].set_xlim((-factor_plot * som_max, factor_plot * som_max))
        self.axs_pr[1, 1].set_ylim((-factor_plot * som_max, factor_plot * som_max))
        self.axs_pr[1, 1].set_ylabel('y [m]')
        self.axs_pr[1, 1].set_xlabel('x [m]')
        self.axs_pr[1, 1].set_title('$|E(x,y)|e^{iW(x,y)} $ Retrieved')
        add_colorbar(self.axs_pr[1, 1], im3, cmap='hsv')

        self.figr_pr.tight_layout()
        self.img_pr.draw()

        im4 = self.axs_pr_corr[1].imshow(C, extent=extent_slm, cmap='bwr')
        print(extent_slm)
        self.axs_pr_corr[1].set_ylim((-4.8e-3, 4.8e-3))
        self.axs_pr_corr[1].set_xlim((-7.68e-3, 7.68e-3))
        self.axs_pr_corr[1].set_ylabel('y [m]')
        self.axs_pr_corr[1].set_xlabel('x [m]')
        self.axs_pr_corr[1].set_title('Phase on SLM')
        add_colorbar(self.axs_pr_corr[1], im4, cmap='bwr')

        self.axs_pr_corr[0].imshow(pr.complex2rgb(E_focus_corrected_cut), extent=extent_focus)
        self.axs_pr_corr[0].set_xlim((-factor_plot * som_max, factor_plot * som_max))
        self.axs_pr_corr[0].set_ylim((-factor_plot * som_max, factor_plot * som_max))
        self.axs_pr_corr[0].set_ylabel('y [m]')
        self.axs_pr_corr[0].set_xlabel('x [m]')
        self.axs_pr_corr[0].set_title('$|E(x,y)|e^{iW(x,y)} $ Corrected')
        add_colorbar(self.axs_pr_corr[0], im3, cmap='hsv')

        self.figr_pr_corr.tight_layout()
        self.img_pr_corr.draw()

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

        self.ax1r.clear()
        self.ax1r.grid(True)

        self.ax1r.plot(z, som_x, linestyle='None', marker='x', color='blue')
        self.ax1r.plot(z, som_y, linestyle='None', marker='x', color='red')
        self.ax1r.plot(z_fit, beam_quality_factor_fit(z_fit, w0_y_fit, M_sq_y_fit, z0_y_fit),
                       label=f'M_sq_y: {abs(params_y[1]):.2f}, '
                             f'w0_y: {params_y[0] * 1e6:.2f} µm, '
                             f'z0_y: {params_y[2] * 1e3:.2f} mm', color='red')
        self.ax1r.plot(z_fit, beam_quality_factor_fit(z_fit, w0_x_fit, M_sq_x_fit, z0_x_fit),
                       label=f'M_sq_x: {abs(params_x[1]):.2f}, '
                             f'w0_x: {params_x[0] * 1e6:.2f} µm, '
                             f'z0_x: {params_x[2] * 1e3:.2f} mm', color='blue')

        self.ax1r.set_ylabel('z [m]')
        self.ax1r.set_xlabel('x [m]')
        self.ax1r.legend()
        self.figr.tight_layout()
        self.img1r.draw()

        return params_x, params_y

    def change_method(self, event):
        selected_method = self.strvar_method_choice.get()

        if selected_method == 'Vortex':
            print(f'Method selected: {selected_method}')

        elif selected_method == 'Gerchberg-Saxton':
            print(f'Method selected: {selected_method}')

    def open_h5_file(self):
        filepath = tk.filedialog.askopenfilename()
        try:
            print(f'Opening {filepath}')
            hfr = h5py.File(filepath, 'r')
            self.images = np.asarray(hfr.get('images'))
            self.positions = np.asarray(hfr.get('positions'))
            self.green_lens = np.asarray(hfr.get('green_lens'))
            self.red_lens = np.asarray(hfr.get('red_lens'))
            print('Successfully loaded')
            processed_images, som_x, som_y = self.process_images_dict()
            zero_position = 8
            zmin = (self.positions[0] - zero_position) * 1e-3
            zmax = (self.positions[-1] - zero_position) * 1e-3  # in mm
            z = np.linspace(zmin, zmax, len(self.positions))
            if self.wavelength is not None:
                params_x, params_y = self.get_M_sq(som_x, som_y, z, float(self.wavelength), 3.45e-6)
            else:
                print('Please select a wavelength')

        except:
            print('Impossible to open the file')

    def open_beam_profile(self):
        filepath = tk.filedialog.askopenfilename()
        try:
            print(f'Opening {filepath}')
            self.image_array = np.array(Image.open(filepath))
            print('Successfully loaded')

        except:
            print('Impossible to open the file')


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

    def wavelength_presets(self):
        status = self.str_wavelength.get()

        if status == "Green":
            print(f"Green (515 nm) is selected")
            self.wavelength = 515e-9
            self.daheng_zoom = self.default_zoom_green
            print(f'ROI selected :{self.daheng_zoom}')

        elif status == "Red":
            print(f"Red (1030 nm) is selected")
            self.wavelength = 1030e-9
            self.daheng_zoom = self.default_zoom_red
            print(f'ROI selected :{self.daheng_zoom}')

    def initialize_daheng(self):
        device_manager = gx.DeviceManager()
        self.daheng_camera = dh.DahengCamera(int(self.strvar_cam_ind.get()))

    def close_daheng(self):
        self.daheng_camera.close_daheng()
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

    def scan_daheng_thread(self):
        self.daheng_thread = threading.Thread(target=self.scan_stage_daheng)
        self.daheng_thread.daemon = True
        self.daheng_thread.start()

    def scan_stage_daheng(self):
        from_ = float(self.var_daheng_stage_from.get())
        to_ = float(self.var_daheng_stage_to.get())
        steps_ = int(self.var_daheng_stage_steps.get())
        stage_steps = np.linspace(from_, to_, steps_)
        if self.WPcam is None:
            print('Stage is not connected')
        if self.daheng_camera is None:
            print('Camera is not connected')
        if self.WPcam is not None and self.daheng_camera is not None:
            res = np.zeros([self.daheng_camera.imshape[0], self.daheng_camera.imshape[1], int(steps_)])
            for ind, pos in enumerate(stage_steps):
                self.strvar_WPcam_should.set(pos)
                self.move_WPcam()
                im = self.daheng_camera.take_image(int(self.strvar_cam_avg.get()))
                self.plot_daheng(im)
                res[:, :, ind] = im
            self.save_daheng_scans(res, stage_steps)

    def save_daheng_scans(self, res, pos):
        nr = self.get_start_image_images()

        self.autolog_cam.write("#" + self.ent_comment.get() + "\n")

        data_filename = 'C:/data/' + str(datetime.date.today()) + '/' + str(
            datetime.date.today()) + '-camera_focus-' + str(
            int(nr)) + '.h5'

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        log_entry = str(int(nr)) + '\t' + str(pos[0]) + '\t' + str(pos[-1]) + '\t' + str(
            int(np.size(pos))) + '\t' + str(int(self.strvar_cam_exp.get())) + '\t' + str(int(self.strvar_cam_gain.get())) + '\t' + str(int(self.strvar_cam_avg.get())) + '\t' + timestamp + '\n'
        self.autolog_cam.write(log_entry)
        hf = h5py.File(data_filename, 'w')
        hf.create_dataset('images', data=res)
        hf.create_dataset('positions', data=pos)
        hf.create_dataset('exposure', data=int(self.strvar_cam_exp.get()))
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
            print("The last image had index " + str(int(start_image - 1)))
        else:
            start_image = 0
        return start_image

    def update_daheng_live_button(self):
        if self.daheng_is_live == True:
            self.but_cam_live.config(relief='sunken')
            print('Live view on')
        else:
            self.but_cam_live.config(relief='raised')
            print('Live view off')

    def live_daheng(self):
        if self.daheng_camera is not None:
            while self.daheng_is_live:
                im = self.daheng_camera.take_image(int(self.strvar_cam_exp.get()), int(self.strvar_cam_gain.get()), int(self.strvar_cam_avg.get()))
                self.current_daheng_image = im
                self.plot_daheng(im)
        else:
            self.daheng_is_live == False
            print('self.daheng_camera is None')

    def take_single_image_daheng(self):
        if self.daheng_camera is not None:
            im = self.daheng_camera.take_image(int(self.strvar_cam_exp.get()), int(self.strvar_cam_gain.get()), int(self.strvar_cam_avg.get()))
            print("Single image taken")
            self.current_daheng_image = im
            self.plot_daheng(im)
        else:
            print('self.daheng_camera is None')

    def plot_daheng(self, im):
        if self.daheng_zoom is not None:
            im = im[int(self.daheng_zoom[0]):int(self.daheng_zoom[1]),
                 int(self.daheng_zoom[2]):int(self.daheng_zoom[3])]

        image = Image.fromarray(im)
        image_resized = image.resize((600, 400), resample=0)
        photo = ImageTk.PhotoImage(image_resized)
        self.img_canvas.itemconfig(self.image, image=photo)
        self.img_canvas.image = photo

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
        self.img_canvas.coords(self.roi_rectangle, self.roi_start[0], self.roi_start[1], self.roi_end[0], self.roi_end[1])
        self.current_roi = (self.roi_start[1], self.roi_end[1], self.roi_start[0], self.roi_end[0])

        self.strvar_roi_x1.set(str(self.current_roi[2]))
        self.strvar_roi_x2.set(str(self.current_roi[3]))
        self.strvar_roi_y1.set(str(self.current_roi[0]))
        self.strvar_roi_y2.set(str(self.current_roi[1]))

    def reset_roi(self, event):
        if self.roi_rectangle:
            self.img_canvas.delete(self.roi_rectangle)
            self.roi_rectangle = None
        self.current_roi = self.initial_roi

        self.strvar_roi_x1.set(str(self.current_roi[2]))
        self.strvar_roi_x2.set(str(self.current_roi[3]))
        self.strvar_roi_y1.set(str(self.current_roi[0]))
        self.strvar_roi_y2.set(str(self.current_roi[1]))


    def init_WPcam(self):
        try:
            self.WPcam = apt.Motor(int(self.ent_WPcam_Nr.get()))
            self.but_WPcam_Ini.config(fg='green')
            print("WPcam connected")
        except:
            self.but_WPcam_Ini.config(fg='red')
            print("Not able to initalize WPcam")

    def home_WPcam(self):

        try:
            self.WPcam.move_home(blocking=True)
            self.but_WPcam_Home.config(fg='green')
            print("WPcam homed!")
            self.read_WPcam()
        except:
            self.but_WPcam_Home.config(fg='red')
            print("Not able to home WPcam")

    def read_WPcam(self):
        try:
            pos = self.WPcam.position
            self.strvar_WPcam_is.set(pos)
        except:
            print("Impossible to read WPcam position")

    def move_WPcam(self):
        try:
            pos = float(self.strvar_WPcam_should.get())
            print("WPcam is moving to {}".format(np.round(pos, 4)))
            self.WPcam.move_to(pos, True)
            print("WPcam moved to {}".format(np.round(self.WPcam.position, 4)))
            self.read_WPcam()
        except Exception as e:
            print(e)
            print("Impossible to move WPcam")

    def disable_motors(self):
        if self.WPcam is not None:
            self.WPcam.disable()
            print('WPcam disconnected')

    def on_close(self):
        self.autolog_cam.close()
        self.close_daheng()
        self.disable_motors()
        self.win.destroy()
