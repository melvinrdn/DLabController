import tkinter as tk
from tkinter.simpledialog import askstring
from tkinter import ttk
from drivers import gxipy_driver as gx
from PIL import Image, ImageTk
import time
import threading
import numpy as np
import os


class CameraControl(object):
    def __init__(self, parent):
        self.cam1 = None
        self.cam2 = None
        self.cam3 = None

        self.parent = parent
        self.win = tk.Toplevel()

        title = 'SLM Phase Control - Camera control'

        self.win.title(title)
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)
        vcmd = (self.win.register(self.parent.callback))

        frm_cam1 = ttk.LabelFrame(self.win, text="Focus view")
        frm_controls1 = ttk.LabelFrame(self.win, text="Camera control")

        frm_cam2 = ttk.LabelFrame(self.win, text="Top nozzle view")
        frm_controls2 = ttk.LabelFrame(self.win, text="Camera control")

        frm_cam3 = ttk.LabelFrame(self.win, text="Side nozzle view")
        frm_controls3 = ttk.LabelFrame(self.win, text="Camera control")

        frm_cam1.grid(row=0, column=0, sticky='nsew')
        frm_controls1.grid(row=0, column=1, sticky='nsew')

        frm_cam2.grid(row=1, column=0, sticky='nsew')
        frm_controls2.grid(row=1, column=1, sticky='nsew')

        frm_cam3.grid(row=2, column=0, sticky='nsew')
        frm_controls3.grid(row=2, column=1, sticky='nsew')

        frm_cam_but1 = ttk.Frame(frm_controls1)
        frm_cam_but_set1 = ttk.Frame(frm_cam_but1)

        frm_cam_but2 = ttk.Frame(frm_controls2)
        frm_cam_but_set2 = ttk.Frame(frm_cam_but2)

        frm_cam_but3 = ttk.Frame(frm_controls3)
        frm_cam_but_set3 = ttk.Frame(frm_cam_but3)

        but_cam_img1 = ttk.Button(frm_cam_but1, text='Initialize camera 1', command=self.cam_img1)
        but_cam_save1 = ttk.Button(frm_cam_but1, text='Save', command=self.cam_save)

        but_cam_img2 = ttk.Button(frm_cam_but2, text='Initialize camera 2', command=self.cam_img2)
        but_cam_save2 = ttk.Button(frm_cam_but2, text='Save', command=self.cam_save)

        but_cam_img3 = ttk.Button(frm_cam_but3, text='Initialize camera 3', command=self.cam_img3)
        but_cam_save3 = ttk.Button(frm_cam_but3, text='Save', command=self.cam_save)

        lbl_cam_ind1 = ttk.Label(frm_cam_but_set1, text='Camera index:')
        self.strvar_cam_ind1 = tk.StringVar(self.win, '1')
        self.ent_cam_ind1 = ttk.Entry(frm_cam_but_set1, width=11, validate='all',
                                      validatecommand=(vcmd, '%d', '%P', '%S'),
                                      textvariable=self.strvar_cam_ind1)

        lbl_cam_ind2 = ttk.Label(frm_cam_but_set2, text='Camera index:')
        self.strvar_cam_ind2 = tk.StringVar(self.win, '2')
        self.ent_cam_ind2 = ttk.Entry(frm_cam_but_set2, width=11, validate='all',
                                      validatecommand=(vcmd, '%d', '%P', '%S'),
                                      textvariable=self.strvar_cam_ind2)

        lbl_cam_ind3 = ttk.Label(frm_cam_but_set3, text='Camera index:')
        self.strvar_cam_ind3 = tk.StringVar(self.win, '3')
        self.ent_cam_ind3 = ttk.Entry(frm_cam_but_set3, width=11, validate='all',
                                      validatecommand=(vcmd, '%d', '%P', '%S'),
                                      textvariable=self.strvar_cam_ind3)

        lbl_cam_exp1 = ttk.Label(frm_cam_but_set1, text='Camera exposure (µs):')
        self.strvar_cam_exp1 = tk.StringVar(self.win, '100000')
        self.ent_cam_exp1 = ttk.Entry(frm_cam_but_set1, width=11, validate='all',
                                      validatecommand=(vcmd, '%d', '%P', '%S'),
                                      textvariable=self.strvar_cam_exp1)

        lbl_cam_exp2 = ttk.Label(frm_cam_but_set2, text='Camera exposure (µs):')
        self.strvar_cam_exp2 = tk.StringVar(self.win, '100000')
        self.ent_cam_exp2 = ttk.Entry(frm_cam_but_set2, width=11, validate='all',
                                      validatecommand=(vcmd, '%d', '%P', '%S'),
                                      textvariable=self.strvar_cam_exp2)

        lbl_cam_exp3 = ttk.Label(frm_cam_but_set3, text='Camera exposure (µs):')
        self.strvar_cam_exp3 = tk.StringVar(self.win, '100000')
        self.ent_cam_exp3 = ttk.Entry(frm_cam_but_set3, width=11, validate='all',
                                      validatecommand=(vcmd, '%d', '%P', '%S'),
                                      textvariable=self.strvar_cam_exp3)

        lbl_cam_gain1 = ttk.Label(frm_cam_but_set1, text='Camera gain (0-24):')
        self.strvar_cam_gain1 = tk.StringVar(self.win, '24')
        self.ent_cam_gain1 = ttk.Entry(frm_cam_but_set1, width=11, validate='all',
                                       validatecommand=(vcmd, '%d', '%P', '%S')
                                       , textvariable=self.strvar_cam_gain1)

        lbl_cam_gain2 = ttk.Label(frm_cam_but_set2, text='Camera gain (0-24):')
        self.strvar_cam_gain2 = tk.StringVar(self.win, '24')
        self.ent_cam_gain2 = ttk.Entry(frm_cam_but_set2, width=11, validate='all',
                                       validatecommand=(vcmd, '%d', '%P', '%S')
                                       , textvariable=self.strvar_cam_gain2)

        lbl_cam_gain3 = ttk.Label(frm_cam_but_set3, text='Camera gain (0-24):')
        self.strvar_cam_gain3 = tk.StringVar(self.win, '24')
        self.ent_cam_gain3 = ttk.Entry(frm_cam_but_set3, width=11, validate='all',
                                       validatecommand=(vcmd, '%d', '%P', '%S')
                                       , textvariable=self.strvar_cam_gain3)

        lbl_cam_time1 = ttk.Label(frm_cam_but_set1, text='Acquisition "time" (1-inf):')
        self.strvar_cam_time1 = tk.StringVar(self.win, '1000')
        self.ent_cam_time1 = ttk.Entry(frm_cam_but_set1, width=11, validate='all',
                                       validatecommand=(vcmd, '%d', '%P', '%S')
                                       , textvariable=self.strvar_cam_time1)

        lbl_cam_time2 = ttk.Label(frm_cam_but_set2, text='Acquisition "time" (1-inf):')
        self.strvar_cam_time2 = tk.StringVar(self.win, '1000')
        self.ent_cam_time2 = ttk.Entry(frm_cam_but_set2, width=11, validate='all',
                                       validatecommand=(vcmd, '%d', '%P', '%S')
                                       , textvariable=self.strvar_cam_time2)

        lbl_cam_time3 = ttk.Label(frm_cam_but_set3, text='Acquisition "time" (1-inf):')
        self.strvar_cam_time3 = tk.StringVar(self.win, '1000')
        self.ent_cam_time3 = ttk.Entry(frm_cam_but_set3, width=11, validate='all',
                                       validatecommand=(vcmd, '%d', '%P', '%S')
                                       , textvariable=self.strvar_cam_time3)

        but_cam_img1.grid(row=0, column=0, padx=5, pady=5, ipadx=5, ipady=5)
        but_cam_save1.grid(row=0, column=1, padx=5, pady=5, ipadx=5, ipady=5)

        but_cam_img2.grid(row=1, column=0, padx=5, pady=5, ipadx=5, ipady=5)
        but_cam_save2.grid(row=1, column=1, padx=5, pady=5, ipadx=5, ipady=5)

        but_cam_img3.grid(row=2, column=0, padx=5, pady=5, ipadx=5, ipady=5)
        but_cam_save3.grid(row=2, column=1, padx=5, pady=5, ipadx=5, ipady=5)

        frm_cam_but_set1.grid(row=0, column=2, sticky='nsew')
        lbl_cam_ind1.grid(row=0, column=0)
        self.ent_cam_ind1.grid(row=0, column=1, padx=(0, 10))
        lbl_cam_exp1.grid(row=1, column=0)
        self.ent_cam_exp1.grid(row=1, column=1, padx=(0, 10))
        lbl_cam_gain1.grid(row=2, column=0)
        self.ent_cam_gain1.grid(row=2, column=1, padx=(0, 10))
        frm_cam_but1.grid(row=1, column=0, padx=5, pady=5, ipadx=5, ipady=5)
        lbl_cam_time1.grid(row=3, column=0)
        self.ent_cam_time1.grid(row=3, column=1, padx=(0, 10))

        frm_cam_but_set2.grid(row=0, column=2, sticky='nsew')
        lbl_cam_ind2.grid(row=0, column=0)
        self.ent_cam_ind2.grid(row=0, column=1, padx=(0, 10))
        lbl_cam_exp2.grid(row=1, column=0)
        self.ent_cam_exp2.grid(row=1, column=1, padx=(0, 10))
        lbl_cam_gain2.grid(row=2, column=0)
        self.ent_cam_gain2.grid(row=2, column=1, padx=(0, 10))
        frm_cam_but2.grid(row=1, column=0, padx=5, pady=5, ipadx=5, ipady=5)
        lbl_cam_time2.grid(row=3, column=0)
        self.ent_cam_time2.grid(row=3, column=1, padx=(0, 10))

        frm_cam_but_set3.grid(row=0, column=2, sticky='nsew')
        lbl_cam_ind3.grid(row=0, column=0)
        self.ent_cam_ind3.grid(row=0, column=1, padx=(0, 10))
        lbl_cam_exp3.grid(row=1, column=0)
        self.ent_cam_exp3.grid(row=1, column=1, padx=(0, 10))
        lbl_cam_gain3.grid(row=2, column=0)
        self.ent_cam_gain3.grid(row=2, column=1, padx=(0, 10))
        frm_cam_but3.grid(row=1, column=0, padx=5, pady=5, ipadx=5, ipady=5)
        lbl_cam_time3.grid(row=3, column=0)
        self.ent_cam_time3.grid(row=3, column=1, padx=(0, 10))

        self.img_canvas1 = tk.Canvas(frm_cam1, height=300, width=450)
        self.img_canvas1.grid(row=0, sticky='nsew')
        self.img_canvas1.configure(bg='grey')
        self.image1 = self.img_canvas1.create_image(0, 0, anchor="nw")

        self.img_canvas2 = tk.Canvas(frm_cam2, height=300, width=450)
        self.img_canvas2.grid(row=0, sticky='nsew')
        self.img_canvas2.configure(bg='grey')
        self.image2 = self.img_canvas2.create_image(0, 0, anchor="nw")

        self.img_canvas3 = tk.Canvas(frm_cam3, height=300, width=450)
        self.img_canvas3.grid(row=0, sticky='nsew')
        self.img_canvas3.configure(bg='grey')
        self.image3 = self.img_canvas3.create_image(0, 0, anchor="nw")

        frm_bottom = ttk.Frame(self.win)
        frm_bottom.grid(row=3, column=0, columnspan=2)
        but_exit = ttk.Button(frm_bottom, text='Exit', command=self.on_close)
        but_exit.grid(row=3, column=0, padx=5, pady=5, ipadx=5, ipady=5)

    def init_cam1(self):
        print("")
        print("Initializing...")
        print("")

        # create a device manager
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()

        if dev_num == 0:
            print("No connected devices")
            return

        # open the first device
        self.cam1 = device_manager.open_device_by_index(int(self.ent_cam_ind1.get()))

        # set exposure
        self.cam1.ExposureTime.set(float(self.ent_cam_exp1.get()))

        # set gain
        self.cam1.Gain.set(float(self.ent_cam_gain1.get()))

        if dev_info_list[0].get("device_class") == gx.GxDeviceClassList.USB2:
            # set trigger mode
            self.cam1.TriggerMode.set(gx.GxSwitchEntry.ON)
        else:
            # set trigger mode and trigger source
            self.cam1.TriggerMode.set(gx.GxSwitchEntry.ON)
            self.cam1.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)


        # start data acquisition
        self.cam1.stream_on()
        print('Streaming...')
        self.acq_mono_1(int(self.ent_cam_time1.get()))
        self.cam1.stream_off()
        self.cam1.close_device()
        print('...Re-initialisation needed')

    def acq_mono_1(self, num):
        """
        acquisition function for camera
               :brief      acquisition function of mono device
               :param      num:        number of acquisition images[int]
        """
        for i in range(num):
            time.sleep(0.001)

            # send software trigger command
            self.cam1.TriggerSoftware.send_command()

            # set exposure
            self.cam1.ExposureTime.set(float(self.ent_cam_exp1.get()))

            # set gain
            self.cam1.Gain.set(float(self.ent_cam_gain1.get()))

            # get raw image
            raw_image = self.cam1.data_stream[0].get_image()
            if raw_image is None:
                continue

            # create numpy array with data from raw image
            numpy_image = raw_image.get_numpy_array()
            if numpy_image is None:
                continue

            # Show images
            picture = Image.fromarray(numpy_image)
            picture = ImageTk.PhotoImage(picture)

            self.img_canvas1.itemconfig(self.image1, image=picture)
            self.img_canvas1.image1 = picture  # keep a reference!

    def cam_img1(self):
        self.render_thread1 = threading.Thread(target=self.init_cam1)
        self.render_thread1.daemon = True
        self.render_thread1.start()

    def init_cam2(self):
        print("")
        print("Initializing...")
        print("")

        # create a device manager
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()

        if dev_num == 0:
            print("No connected devices")
            return

        # open the first device
        self.cam2 = device_manager.open_device_by_index(int(self.ent_cam_ind2.get()))

        # set exposure
        self.cam2.ExposureTime.set(float(self.ent_cam_exp2.get()))

        # set gain
        self.cam2.Gain.set(float(self.ent_cam_gain2.get()))

        if dev_info_list[0].get("device_class") == gx.GxDeviceClassList.USB2:
            # set trigger mode
            self.cam2.TriggerMode.set(gx.GxSwitchEntry.ON)
        else:
            # set trigger mode and trigger source
            self.cam2.TriggerMode.set(gx.GxSwitchEntry.ON)
            self.cam2.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)

        # start data acquisition
        self.cam2.stream_on()
        print('Streaming...')
        self.acq_mono_2(int(self.ent_cam_time2.get()))
        self.cam2.stream_off()
        self.cam2.close_device()
        print('...Re-initialisation needed')

    def acq_mono_2(self, num):
        """
        acquisition function for camera
               :brief      acquisition function of mono device
               :param      num:        number of acquisition images[int]
        """
        for i in range(num):
            time.sleep(0.001)

            # send software trigger command
            self.cam2.TriggerSoftware.send_command()

            # set exposure
            self.cam2.ExposureTime.set(float(self.ent_cam_exp2.get()))

            # set gain
            self.cam2.Gain.set(float(self.ent_cam_gain2.get()))

            # get raw image
            raw_image = self.cam2.data_stream[0].get_image()
            if raw_image is None:
                continue

            # create numpy array with data from raw image
            numpy_image = raw_image.get_numpy_array()
            if numpy_image is None:
                continue

            # Show images
            picture = Image.fromarray(numpy_image)
            picture = ImageTk.PhotoImage(picture)

            self.img_canvas2.itemconfig(self.image2, image=picture)
            self.img_canvas2.image2 = picture  # keep a reference!

    def cam_img2(self):
        self.render_thread2 = threading.Thread(target=self.init_cam2)
        self.render_thread2.daemon = True
        self.render_thread2.start()

    def init_cam3(self):
        print("")
        print("Initializing...")
        print("")

        # create a device manager
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()

        if dev_num == 0:
            print("No connected devices")
            return

        # open the first device
        self.cam3 = device_manager.open_device_by_index(int(self.ent_cam_ind3.get()))

        # set exposure
        self.cam3.ExposureTime.set(float(self.ent_cam_exp3.get()))

        # set gain
        self.cam3.Gain.set(float(self.ent_cam_gain3.get()))

        if dev_info_list[0].get("device_class") == gx.GxDeviceClassList.USB2:
            # set trigger mode
            self.cam3.TriggerMode.set(gx.GxSwitchEntry.ON)
        else:
            # set trigger mode and trigger source
            self.cam3.TriggerMode.set(gx.GxSwitchEntry.ON)
            self.cam3.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)

        # start data acquisition
        self.cam3.stream_on()
        print('Streaming...')
        self.acq_mono_3(int(self.ent_cam_time3.get()))
        self.cam3.stream_off()
        self.cam3.close_device()
        print('...Re-initialisation needed')

    def acq_mono_3(self, num):
        """
        acquisition function for camera
               :brief      acquisition function of mono device
               :param      num:        number of acquisition images[int]
        """
        for i in range(num):
            time.sleep(0.001)

            # send software trigger command
            self.cam3.TriggerSoftware.send_command()

            # set exposure
            self.cam3.ExposureTime.set(float(self.ent_cam_exp3.get()))

            # set gain
            self.cam3.Gain.set(float(self.ent_cam_gain3.get()))

            # get raw image
            raw_image = self.cam3.data_stream[0].get_image()
            if raw_image is None:
                continue

            # create numpy array with data from raw image
            numpy_image = raw_image.get_numpy_array()
            if numpy_image is None:
                continue

            # Show images
            picture = Image.fromarray(numpy_image)
            picture = ImageTk.PhotoImage(picture)

            self.img_canvas3.itemconfig(self.image3, image=picture)
            self.img_canvas3.image3 = picture  # keep a reference!

    def cam_img3(self):
        self.render_thread3 = threading.Thread(target=self.init_cam3)
        self.render_thread3.daemon = True
        self.render_thread3.start()

    def cam_save(self):
        # send software trigger command
        self.cam.TriggerSoftware.send_command()

        # get raw image
        raw_image = self.cam.data_stream[0].get_image()
        if raw_image is None:
            print("Getting image failed.")

        # create numpy array with data from raw image
        numpy_image = raw_image.get_numpy_array()
        print(f'Value of the maximum pixel : {np.max(numpy_image)}')

        bmp_image = Image.fromarray(numpy_image)

        file_name = askstring(title="Save As", prompt="Enter a file name (without the extension) :")
        file_name += '.bmp'

        if file_name:
            folder_path = os.path.join(os.getcwd(), "beam_profiles")
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            file_path = os.path.join(folder_path, file_name)
            bmp_image.save(file_path)
            print(f"Beam profile saved as {file_path}")
        else:
            print("File save cancelled.")

    def cam_on_close1(self):
        if self.cam1 is not None:
            self.cam1.close_device()
            print('Camera 1 closed')
        else:
            pass

    def cam_on_close2(self):
        if self.cam2 is not None:
            self.cam2.close_device()
            print('Camera 2 closed')
        else:
            pass

    def cam_on_close3(self):
        if self.cam3 is not None:
            self.cam3.close_device()
            print('Camera 3 closed')
        else:
            pass

    def on_close(self):
        self.cam_on_close1()
        self.cam_on_close2()
        self.cam_on_close3()
        self.win.destroy()
        self.parent.camera_win = None
