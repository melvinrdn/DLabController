import tkinter as tk
from tkinter import ttk
from drivers import gxipy_driver as gx
from PIL import Image, ImageTk
import time
import threading


class CameraControl(object):
    def __init__(self, parent):
        self.cam = None

        self.parent = parent
        self.win = tk.Toplevel()

        title = 'SLM Phase Control - Camera control'

        self.win.title(title)
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)
        vcmd = (self.win.register(self.parent.callback))

        frm_cam = ttk.LabelFrame(self.win, text="Camera 1")
        frm_controls = ttk.LabelFrame(self.win, text="Camera control")

        frm_cam.grid(row=0, column=0, sticky='nsew')
        frm_controls.grid(row=0, column=1, sticky='nsew')

        frm_cam_but = ttk.Frame(frm_controls)
        frm_cam_but_set = ttk.Frame(frm_cam_but)

        but_cam_init = ttk.Button(frm_cam_but, text='Start', command=self.cam_img)
        but_cam_stop = ttk.Button(frm_cam_but, text='Stop', command=self.cam_stop)

        lbl_cam_ind = ttk.Label(frm_cam_but_set, text='Camera index:')
        self.strvar_cam_ind = tk.StringVar(self.win, '1')
        self.ent_cam_ind = ttk.Entry(frm_cam_but_set, width=11, validate='all',
                                     validatecommand=(vcmd, '%d', '%P', '%S'),
                                     textvariable=self.strvar_cam_ind)

        lbl_cam_exp = ttk.Label(frm_cam_but_set, text='Exposure (µs):')
        self.strvar_cam_exp = tk.StringVar(self.win, '1000000')
        self.ent_cam_exp = ttk.Entry(frm_cam_but_set, width=11, validate='all',
                                     validatecommand=(vcmd, '%d', '%P', '%S'),
                                     textvariable=self.strvar_cam_exp)

        lbl_cam_gain = ttk.Label(frm_cam_but_set, text='Gain (0-24):')
        self.strvar_cam_gain = tk.StringVar(self.win, '24')
        self.ent_cam_gain = ttk.Entry(frm_cam_but_set, width=11, validate='all',
                                      validatecommand=(vcmd, '%d', '%P', '%S')
                                      , textvariable=self.strvar_cam_gain)

        lbl_cam_time = ttk.Label(frm_cam_but_set, text='Nbr of averages :')
        self.strvar_cam_time = tk.StringVar(self.win, '1')
        self.ent_cam_time = ttk.Entry(frm_cam_but_set, width=11, validate='all',
                                      validatecommand=(vcmd, '%d', '%P', '%S')
                                      , textvariable=self.strvar_cam_time)

        but_cam_init.grid(row=0, column=0)
        but_cam_stop.grid(row=0, column=1)

        frm_cam_but_set.grid(row=0, column=2, sticky='nsew')
        lbl_cam_ind.grid(row=0, column=0, sticky='nsew')
        self.ent_cam_ind.grid(row=0, column=1, padx=(0, 10))
        lbl_cam_exp.grid(row=1, column=0, sticky='nsew')
        self.ent_cam_exp.grid(row=1, column=1, padx=(0, 10))
        lbl_cam_gain.grid(row=2, column=0, sticky='nsew')
        self.ent_cam_gain.grid(row=2, column=1, padx=(0, 10))
        frm_cam_but.grid(row=1, column=0, sticky='nsew')
        lbl_cam_time.grid(row=3, column=0, sticky='nsew')
        self.ent_cam_time.grid(row=3, column=1, padx=(0, 10))

        self.img_canvas = tk.Canvas(frm_cam, height=300, width=450)
        self.img_canvas.grid(row=0, sticky='nsew')
        self.img_canvas.configure(bg='grey')
        self.image = self.img_canvas.create_image(0, 0, anchor="nw")

        frm_bottom = ttk.Frame(self.win)
        frm_bottom.grid(row=3, column=0, columnspan=2)
        but_exit = ttk.Button(frm_bottom, text='Exit', command=self.on_close)
        but_exit.grid(row=3, column=0, padx=5, pady=5, ipadx=5, ipady=5)

        self.cam_live = True

    def init_cam(self):
        print("Initializing...")

        self.cam_live = True

        # create a device manager
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()

        if dev_num == 0:
            print("No connected devices")
            return

        # open the first device
        self.cam = device_manager.open_device_by_index(int(self.ent_cam_ind.get()))

        # set exposure
        self.cam.ExposureTime.set(float(self.ent_cam_exp.get()))

        # set gain
        self.cam.Gain.set(float(self.ent_cam_gain.get()))

        if dev_info_list[0].get("device_class") == gx.GxDeviceClassList.USB2:
            # set trigger mode
            self.cam.TriggerMode.set(gx.GxSwitchEntry.ON)
        else:
            # set trigger mode and trigger source
            self.cam.TriggerMode.set(gx.GxSwitchEntry.ON)
            self.cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)

        self.cam.stream_on()
        print('Starting acquisition..')
        self.acq_mono()
        self.cam.stream_off()
        self.cam.close_device()

    def acq_mono(self):
        """
        acquisition function for camera
        """
        while self.cam_live:

            time.sleep(0.001)
            sum_image = None
            num = int(self.ent_cam_time.get())

            for i in range(num):
                self.cam.TriggerSoftware.send_command()
                self.cam.ExposureTime.set(float(self.ent_cam_exp.get()))
                self.cam.Gain.set(float(self.ent_cam_gain.get()))

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

            picture = Image.fromarray(average_image)
            picture = picture.resize((500, 350), resample=0)
            picture = ImageTk.PhotoImage(picture)

            self.img_canvas.itemconfig(self.image, image=picture)
            self.img_canvas.image = picture

    def cam_img(self):
        self.render_thread = threading.Thread(target=self.init_cam)
        self.render_thread.daemon = True
        self.render_thread.start()

    def cam_stop(self):
        self.cam_live = False
        print('Acquisition stopped')

    def on_close(self):
        self.win.destroy()
        self.parent.camera_win = None
