import tkinter as tk
from tkinter import ttk
from drivers import gxipy_driver as gx
from PIL import Image, ImageTk
import time
import threading
import cv2
import os
from datetime import date

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

        but_cam_init = ttk.Button(frm_cam_but, text='Live', command=self.cam_cont_acq)
        but_cam_stop = ttk.Button(frm_cam_but, text='Stop', command=self.cam_stop)
        but_cam_mono_acq = ttk.Button(frm_cam_but, text='Single image acquisition', command=self.cam_mono_acq)

        lbl_cam_ind = ttk.Label(frm_cam_but_set, text='Camera index:')
        self.strvar_cam_ind = tk.StringVar(self.win, '1')
        self.ent_cam_ind = ttk.Entry(frm_cam_but_set, width=11, validate='all',
                                     validatecommand=(vcmd, '%d', '%P', '%S'),
                                     textvariable=self.strvar_cam_ind)

        lbl_cam_exp = ttk.Label(frm_cam_but_set, text='Exposure (µs):')
        self.strvar_cam_exp = tk.StringVar(self.win, '10000')
        self.ent_cam_exp = ttk.Entry(frm_cam_but_set, width=11, validate='all',
                                     validatecommand=(vcmd, '%d', '%P', '%S'),
                                     textvariable=self.strvar_cam_exp)

        lbl_cam_gain = ttk.Label(frm_cam_but_set, text='Gain (0-24):')
        self.strvar_cam_gain = tk.StringVar(self.win, '1')
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
        but_cam_mono_acq.grid(row=1, column=0)

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

        self.img_canvas = tk.Canvas(frm_cam, height=600, width=800)
        self.img_canvas.grid(row=0, sticky='nsew')
        self.img_canvas.configure(bg='grey')
        self.image = self.img_canvas.create_image(0, 0, anchor="nw")

        frm_bottom = ttk.Frame(self.win)
        frm_bottom.grid(row=3, column=0, columnspan=2)
        but_exit = ttk.Button(frm_bottom, text='Exit', command=self.on_close)
        but_exit.grid(row=3, column=0, padx=5, pady=5, ipadx=5, ipady=5)
        but_cam_save = ttk.Button(frm_bottom, text='Save', command=self.cam_save)
        but_cam_save.grid(row=3, column=1, padx=5, pady=5, ipadx=5, ipady=5)

        self.cam_live = True

    def init_cam_cont(self):
        print('Continuous acquisition mode - on')

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
        self.acq_cont(int(self.ent_cam_time.get()))
        self.cam.stream_off()
        self.cam.close_device()

    def acq_cont(self, num):
        """
        acquisition function for camera in continuous mode
        """
        while self.cam_live:

            time.sleep(0.001)
            sum_image = None

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
            picture = picture.resize((800, 600), resample=0)
            picture = ImageTk.PhotoImage(picture)

            self.img_canvas.itemconfig(self.image, image=picture)
            self.img_canvas.image = picture

    def cam_cont_acq(self):
        self.render_thread_cont = threading.Thread(target=self.init_cam_cont)
        self.render_thread_cont.daemon = True
        self.render_thread_cont.start()

    def cam_stop(self):
        self.cam_live = False
        print('Continuous acquisition - off')

    def init_cam_mono(self):

        if self.cam_live is True:
            self.cam_live = False
            print('Continuous acquisition - off')

        print('Mono acquisition mode - on ')

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
        self.acq_mono(int(self.ent_cam_time.get()))
        self.cam.stream_off()
        self.cam.close_device()

        print('Mono acquisition mode - off ')

    def acq_mono(self, num):
        """
        acquisition function for camera in single picture mode
        """
        sum_image = None

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
        picture = picture.resize((800, 600), resample=0)
        picture = ImageTk.PhotoImage(picture)

        self.img_canvas.itemconfig(self.image, image=picture)
        self.img_canvas.image = picture

        print(f'Mono acquisition mode - Image taken over {num} averages')

    def cam_mono_acq(self):
        self.render_thread_mono = threading.Thread(target=self.init_cam_mono)
        self.render_thread_mono.daemon = True
        self.render_thread_mono.start()

    def cam_save(self):
        folder_path = 'C:/data/' + str(date.today()) + '/' + 'camera' + str(int(self.ent_cam_ind.get())) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        base_filename = str(date.today())

        filename = base_filename + '.bmp'
        i = 1
        while os.path.exists(os.path.join(folder_path, filename)):
            filename = f"{base_filename}_{i}.bmp"
            i += 1

        full_path = os.path.join(folder_path, filename)

        self.img_canvas.itemconfig(self.image)
        last_image = self.img_canvas.image

        if last_image is not None:
            pil_image = ImageTk.getimage(last_image)
            pil_image.save(full_path)
            print(f"Image saved as {full_path}")
        else:
            print("No image to save")

    def on_close(self):
        self.win.destroy()
        self.parent.camera_win = None
