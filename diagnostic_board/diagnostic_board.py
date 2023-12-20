import datetime
import threading
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
from drivers import gxipy_driver as gx
from drivers.thorlabs_apt_driver import core as apt


class DiagnosticBoard(object):
    def __init__(self, parent):
        self.cam = None
        self.roi = None
        self.wavelength = None

        self.initial_roi = (0, 1080, 0, 1440)
        self.default_zoom_green = (0, 1080, 0, 1440)
        self.default_zoom_red = (0, 1080, 0, 1440)

        self.parent = parent
        self.lens_green = self.parent.phase_refs_green[1]
        self.lens_red = self.parent.phase_refs_red[1]

        self.win = tk.Toplevel()

        title = 'D-Lab Controller - Diagnostic Board'

        self.win.title(title)
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

        self.frm_cam = ttk.LabelFrame(self.win, text="Camera display")
        self.frm_main_settings = ttk.LabelFrame(self.win, text="Main settings")
        frm_wavelength = ttk.LabelFrame(self.frm_main_settings, text="Wavelength presets")
        frm_controls = ttk.LabelFrame(self.frm_main_settings, text="Camera settings")
        frm_stage = ttk.LabelFrame(self.frm_main_settings, text="Stage settings")

        self.frm_notebook_diagnostics = ttk.Notebook(self.win)
        self.frm_m2_diagnostics = ttk.Frame(self.frm_notebook_diagnostics)
        self.frm_phase_retrieval = ttk.Frame(self.frm_notebook_diagnostics)
        self.frm_notebook_diagnostics.add(self.frm_m2_diagnostics, text='M2 Diagnostics')
        self.frm_notebook_diagnostics.add(self.frm_phase_retrieval, text="Phase retrieval")
        self.frm_notebook_diagnostics.grid(row=0, column=3, sticky='nsew')

        self.frm_scan_settings = ttk.LabelFrame(self.frm_m2_diagnostics, text="Scan settings")
        self.frm_m2_plot = ttk.LabelFrame(self.frm_m2_diagnostics, text="M2 Plot")

        self.frm_pr_settings = ttk.LabelFrame(self.frm_phase_retrieval, text="PR settings")

        self.frm_cam.grid(row=0, column=0, sticky='nsew')
        self.frm_main_settings.grid(row=0, column=1, sticky='nsew')
        frm_wavelength.grid(row=0, column=0, sticky='nsew')
        frm_controls.grid(row=1, column=0, sticky='nsew')
        frm_stage.grid(row=2, column=0, sticky='nsew')

        self.frm_scan_settings.grid(row=0, column=0, sticky='nsew')
        self.frm_m2_plot.grid(row=1, column=0, sticky='nsew')

        self.frm_pr_settings.grid(row=0, column=0, sticky='nsew')

        lbl_method_choice = tk.Label(self.frm_pr_settings, text='Method chosen :')
        self.strvar_method_choice = tk.StringVar(self.frm_phase_retrieval, 'Vortex')
        self.cbox_method_choice = ttk.Combobox(self.frm_pr_settings, textvariable=self.strvar_method_choice)
        self.cbox_method_choice.bind("<<ComboboxSelected>>", self.change_method)
        self.cbox_method_choice['values'] = ('Vortex', 'Gerchberg-Saxton')
        lbl_method_choice.grid(row=0, column=0, sticky='nsew')
        self.cbox_method_choice.grid(row=0, column=1, sticky='nsew')

        sizefactor = 1
        self.figr = Figure(figsize=(6 * sizefactor, 4 * sizefactor), dpi=100)
        self.ax1r = self.figr.add_subplot(111)
        self.ax1r.grid()
        self.figr.tight_layout()
        self.figr.canvas.draw()
        self.img1r = FigureCanvasTkAgg(self.figr, self.frm_m2_plot)
        self.tk_widget_figr = self.img1r.get_tk_widget()
        self.tk_widget_figr.grid(row=0, column=0, sticky='nsew')
        self.img1r.draw()
        self.ax1r_blit = self.figr.canvas.copy_from_bbox(self.ax1r.bbox)

        lbl_daheng_stage_from = tk.Label(self.frm_scan_settings, text="From")
        lbl_daheng_stage_to = tk.Label(self.frm_scan_settings, text="To")
        lbl_daheng_stage_steps = tk.Label(self.frm_scan_settings, text="Steps")
        lbl_daheng_stage_from.grid(row=0, column=0)
        lbl_daheng_stage_to.grid(row=0, column=1)
        lbl_daheng_stage_steps.grid(row=0, column=2)

        self.var_daheng_stage_from = tk.StringVar(self.win, value="6")
        self.ent_daheng_stage_from = tk.Entry(self.frm_scan_settings, textvariable=self.var_daheng_stage_from, width=5)
        self.ent_daheng_stage_from.grid(row=1, column=0)

        self.var_daheng_stage_to = tk.StringVar(self.win, value="12")
        self.ent_daheng_stage_to = tk.Entry(self.frm_scan_settings, textvariable=self.var_daheng_stage_to,
                                            width=5)
        self.ent_daheng_stage_to.grid(row=1, column=1)

        self.var_daheng_stage_steps = tk.StringVar(self.win, value="10")
        self.ent_daheng_stage_steps = tk.Entry(self.frm_scan_settings,
                                               textvariable=self.var_daheng_stage_steps,
                                               width=5)
        self.ent_daheng_stage_steps.grid(row=1, column=2)

        self.but_scan_daheng = tk.Button(self.frm_scan_settings, text="Stage scan",
                                         command=self.scan_daheng_thread)
        self.but_scan_daheng.grid(row=1, column=3)

        self.open_h5_file = tk.Button(self.frm_scan_settings, text='Open h5 file', command=self.open_h5_file)
        self.open_h5_file.grid(row=1, column=4)

        frm_cam_but = ttk.Frame(frm_controls)
        frm_cam_but_set = ttk.Frame(frm_cam_but)

        self.str_wavelength = tk.StringVar(self.win, "Nothing")
        self.rb_red = tk.Radiobutton(frm_wavelength, variable=self.str_wavelength, value="Red", text="Red",
                                     command=self.wavelength_presets)
        self.rb_green = tk.Radiobutton(frm_wavelength, variable=self.str_wavelength, value="Green", text="Green",
                                       command=self.wavelength_presets)

        self.rb_red.grid(row=0, column=0)
        self.rb_green.grid(row=0, column=1)

        self.but_cam_init = tk.Button(frm_cam_but, text='Initialize', command=self.initialize_daheng)
        self.but_cam_disconnect = tk.Button(frm_cam_but, text='Disconnect', command=self.close_daheng)
        self.but_cam_live = tk.Button(frm_cam_but, text='Live', command=self.live_daheng_thread)
        self.but_cam_single = tk.Button(frm_cam_but, text='Single', command=self.single_daheng_thread)
        but_roi_select = ttk.Button(frm_cam_but_set, text='Select ROI', command=self.select_roi)
        but_roi_reset = ttk.Button(frm_cam_but_set, text='Reset ROI', command=self.reset_roi)

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

        but_roi_select.grid(row=7, column=2, sticky='nsew')
        but_roi_reset.grid(row=7, column=3, sticky='nsew')

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

        frm_bottom = ttk.Frame(self.win)
        frm_bottom.grid(row=3, column=0, columnspan=2)
        but_exit = ttk.Button(frm_bottom, text='Exit', command=self.on_close)
        but_exit.grid(row=3, column=0, padx=5, pady=5, ipadx=5, ipady=5)

        self.reset_roi()

        self.daheng_active = False
        self.daheng_camera = None
        self.daheng_is_live = False
        self.current_daheng_image = None
        self.daheng_zoom = None
        self.WPcam = None

        self.autolog_images = 'C:/data/' + str(datetime.date.today()) + '/' + str(
            datetime.date.today()) + '-' + 'auto-log-images.txt'
        self.g = open(self.autolog_images, "a+")

    def foo(self):
        print('oui')

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
        index = int(self.strvar_cam_ind.get())
        self.daheng_camera = dh.DahengCamera(index)
        if self.daheng_camera is not None:
            self.but_cam_init.config(fg="green")
            self.daheng_active = True
            self.daheng_camera.set_exposure_gain(int(self.strvar_cam_exp.get()), int(self.strvar_cam_gain.get()))
            print('oui')
            return 1

        else:
            self.but_cam_init.config(fg="red")
            print('non')
            return 0

    def close_daheng(self):
        if self.daheng_camera is not None:
            self.but_cam_disconnect.config(fg="green")
            self.daheng_active = False
            self.daheng_camera = None
            print('Camera disconnected')
            return 1
        else:
            self.but_cam_disconnect.config(fg="red")
            print('Camera already disconnected')
            return 0

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

        data_filename = 'C:/data/' + str(datetime.date.today()) + '/' + str(
            datetime.date.today()) + '-focus-images-' + str(
            int(nr)) + '.h5'

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

    def get_start_image_images(self):
        """
        Gets the index of the starting image.

        Returns
        -------
        int
            The index of the starting image.

        Raises
        ------
        Exception
            If there is an error in retrieving the starting image index.
        """
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

    def update_daheng_live_button(self):
        if self.daheng_is_live == True:
            self.but_cam_live.config(fg="green", relief='sunken')
        else:
            self.but_cam_live.config(fg="red", relief='raised')

    def live_daheng(self):
        while self.daheng_is_live:
            im = self.daheng_camera.take_image(int(self.strvar_cam_avg.get()))
            self.current_daheng_image = im
            self.plot_daheng(im)

    def take_single_image_daheng(self):
        if self.daheng_camera is not None:
            im = self.daheng_camera.take_image(int(self.strvar_cam_avg.get()))
            self.current_daheng_image = im
            self.plot_daheng(im)
            print("Single image taken")

    def plot_daheng(self, im):

        if self.daheng_zoom is not None:
            im = im[int(self.daheng_zoom[0]):int(self.daheng_zoom[1]),
                 int(self.daheng_zoom[2]):int(self.daheng_zoom[3])]

        image = Image.fromarray(im)
        image_resized = image.resize((600, 400), resample=0)
        photo = ImageTk.PhotoImage(image_resized)
        self.img_canvas.itemconfig(self.image, image=photo)
        self.img_canvas.image = photo

    def select_roi(self):

        x1 = int(self.ent_roi_x1.get())
        x2 = int(self.ent_roi_x2.get())
        y1 = int(self.ent_roi_y1.get())
        y2 = int(self.ent_roi_y2.get())

        self.daheng_zoom = (y1, y2, x1, x2)
        print(f'ROI selected :{self.daheng_zoom}')

    def reset_roi(self):
        self.daheng_zoom = self.initial_roi
        print(f'ROI selected :{self.daheng_zoom}')

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
            print("WPcam is moving to {}".format(np.round(pos, 2)))
            self.WPcam.move_to(pos, True)
            print(f"WPG moved to {str(self.WPcam.position)}")
            self.read_WPcam()
        except Exception as e:
            print(e)
            print("Impossible to move WPcam")

    def disable_motors(self):
        if self.WPcam is not None:
            self.WPcam.disable()
            print('WPcam disconnected')


    def on_close(self):
        self.close_daheng()
        self.disable_motors()
        self.win.destroy()
