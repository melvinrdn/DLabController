import tkinter as tk
from tkinter import ttk
from drivers import gxipy_driver as gx
import diagnostic_board.focus_diagnostic as dh
from drivers.thorlabs_apt_driver import core as apt
import threading
from PIL import Image, ImageTk
import numpy as np


class DiagnosticBoard(object):
    def __init__(self):
        self.cam = None
        self.roi = None
        self.wavelength = None

        self.initial_roi = (0, 1080, 0, 1440)
        self.default_zoom_green = (0, 1080, 0, 1440)
        self.default_zoom_red = (0, 1080, 0, 1440)


        self.win = tk.Toplevel()

        title = 'D-Lab Controller - Diagnostic Board'

        self.win.title(title)
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

        self.frm_cam = ttk.LabelFrame(self.win, text="Camera display")
        self.frm_main_settings = ttk.LabelFrame(self.win, text="Main settings")
        frm_wavelength = ttk.LabelFrame(self.frm_main_settings, text="Wavelength presets")
        frm_controls = ttk.LabelFrame(self.frm_main_settings, text="Camera settings")
        frm_stage = ttk.LabelFrame(self.frm_main_settings, text="Stage settings")

        self.frm_cam.grid(row=0, column=0, sticky='nsew')
        self.frm_main_settings.grid(row=0, column=1, sticky='nsew')
        frm_wavelength.grid(row=0, column=0, sticky='nsew')
        frm_controls.grid(row=1, column=0, sticky='nsew')
        frm_stage.grid(row=2, column=0, sticky='nsew')

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

        lbl_cam_time = ttk.Label(frm_cam_but_set, text='Nbr of averages :')
        self.strvar_cam_time = tk.StringVar(self.win, '1')
        self.ent_cam_time = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                      textvariable=self.strvar_cam_time)

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
        lbl_cam_time.grid(row=8, column=0, sticky='nsew')
        self.ent_cam_time.grid(row=8, column=1, padx=(0, 10))

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

    def foo(self):
        print('ouais')

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

    def update_daheng_live_button(self):
        if self.daheng_is_live == True:
            self.but_cam_live.config(fg="green", relief='sunken')
        else:
            self.but_cam_live.config(fg="red", relief='raised')

    def live_daheng(self):
        while self.daheng_is_live:
            im = self.daheng_camera.take_image(int(self.strvar_cam_time.get()))
            self.current_daheng_image = im
            self.plot_daheng(im)

    def take_single_image_daheng(self):
        if self.daheng_camera is not None:
            im = self.daheng_camera.take_image(int(self.strvar_cam_time.get()))
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
