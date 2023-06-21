import tkinter as tk
from tkinter import ttk
from drivers import gxipy_driver as gx
from PIL import Image, ImageTk
import time
import threading
import os
from datetime import date


class CameraControl(object):
    def __init__(self):
        self.cam = None
        self.roi = None

        self.initial_roi = (0, 1080, 0, 1440)

        self.win = tk.Toplevel()

        title = 'D-Lab Controller - Daheng camera'

        self.win.title(title)
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

        frm_cam = ttk.LabelFrame(self.win, text="Camera")
        frm_controls = ttk.LabelFrame(self.win, text="Camera control")

        frm_cam.grid(row=0, column=0, sticky='nsew')
        frm_controls.grid(row=0, column=1, sticky='nsew')

        frm_cam_but = ttk.Frame(frm_controls)
        frm_cam_but_set = ttk.Frame(frm_cam_but)

        but_cam_init = ttk.Button(frm_cam_but, text='Live', command=self.cam_cont_acq)
        but_cam_stop = ttk.Button(frm_cam_but, text='Stop', command=self.cam_stop)
        but_cam_mono_acq = ttk.Button(frm_cam_but, text='Single image acquisition', command=self.cam_mono_acq)
        but_roi_select = ttk.Button(frm_cam_but, text='Select ROI', command=self.select_roi)  # New button
        but_roi_reset = ttk.Button(frm_cam_but, text='Reset ROI', command=self.reset_roi)  # New button

        lbl_cam_ind = ttk.Label(frm_cam_but_set, text='Camera index:')
        self.strvar_cam_ind = tk.StringVar(self.win, '1')
        self.ent_cam_ind = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                     textvariable=self.strvar_cam_ind)

        lbl_cam_exp = ttk.Label(frm_cam_but_set, text='Exposure (µs):')
        self.strvar_cam_exp = tk.StringVar(self.win, '1000000')
        self.ent_cam_exp = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                     textvariable=self.strvar_cam_exp)

        lbl_cam_gain = ttk.Label(frm_cam_but_set, text='Gain (0-24):')
        self.strvar_cam_gain = tk.StringVar(self.win, '24')
        self.ent_cam_gain = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                      textvariable=self.strvar_cam_gain)

        lbl_cam_time = ttk.Label(frm_cam_but_set, text='Nbr of averages :')
        self.strvar_cam_time = tk.StringVar(self.win, '1')
        self.ent_cam_time = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                      textvariable=self.strvar_cam_time)

        lbl_roi_x1 = ttk.Label(frm_cam_but_set, text='ROI x1:')
        self.strvar_roi_x1 = tk.StringVar(self.win, str(self.initial_roi[0]))
        self.ent_roi_x1 = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                   textvariable=self.strvar_roi_x1)

        lbl_roi_x2 = ttk.Label(frm_cam_but_set, text='ROI x2:')
        self.strvar_roi_x2 = tk.StringVar(self.win, str(self.initial_roi[1]))
        self.ent_roi_x2 = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                   textvariable=self.strvar_roi_x2)

        lbl_roi_y1 = ttk.Label(frm_cam_but_set, text='ROI y1:')
        self.strvar_roi_y1 = tk.StringVar(self.win, str(self.initial_roi[2]))
        self.ent_roi_y1 = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                   textvariable=self.strvar_roi_y1)

        lbl_roi_y2 = ttk.Label(frm_cam_but_set, text='ROI y2:')
        self.strvar_roi_y2 = tk.StringVar(self.win, str(self.initial_roi[3]))
        self.ent_roi_y2 = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                   textvariable=self.strvar_roi_y2)

        lbl_picture_timing = ttk.Label(frm_cam_but_set, text='Picture every (µs):')
        self.strvar_picture_timing = tk.StringVar(self.win, '10000')
        self.ent_picture_timing = ttk.Entry(frm_cam_but_set, width=8, validate='all',
                                   textvariable=self.strvar_picture_timing)

        but_roi_select.grid(row=1, column=1)
        but_roi_reset.grid(row=1, column=2)
        lbl_roi_x1.grid(row=2, column=0, sticky='nsew')
        self.ent_roi_x1.grid(row=2, column=1, padx=(0, 10))
        lbl_roi_x2.grid(row=3, column=0, sticky='nsew')
        self.ent_roi_x2.grid(row=3, column=1, padx=(0, 10))
        lbl_roi_y1.grid(row=4, column=0, sticky='nsew')
        self.ent_roi_y1.grid(row=4, column=1, padx=(0, 10))
        lbl_roi_y2.grid(row=5, column=0, sticky='nsew')
        self.ent_roi_y2.grid(row=5, column=1, padx=(0, 10))

        but_cam_init.grid(row=0, column=0)
        but_cam_stop.grid(row=0, column=1)
        but_cam_mono_acq.grid(row=1, column=0)
        but_roi_select.grid(row=1, column=1)
        but_roi_reset.grid(row=1, column=2)


        lbl_cam_ind.grid(row=0, column=0, sticky='nsew')
        self.ent_cam_ind.grid(row=0, column=1, padx=(0, 10))
        frm_cam_but_set.grid(row=0, column=2, sticky='nsew')
        frm_cam_but.grid(row=1, column=0, sticky='nsew')
        lbl_cam_exp.grid(row=6, column=0, sticky='nsew')
        self.ent_cam_exp.grid(row=6, column=1, padx=(0, 10))
        lbl_cam_gain.grid(row=7, column=0, sticky='nsew')
        self.ent_cam_gain.grid(row=7, column=1, padx=(0, 10))
        lbl_cam_time.grid(row=8, column=0, sticky='nsew')
        self.ent_cam_time.grid(row=8, column=1, padx=(0, 10))
        lbl_picture_timing.grid(row=9, column=0, sticky='nsew')
        self.ent_picture_timing.grid(row=9, column=1, padx=(0, 10))

        self.img_canvas = tk.Canvas(frm_cam, height=540, width=720)
        self.img_canvas.grid(row=0, sticky='nsew')
        self.img_canvas.configure(bg='grey')
        self.image = self.img_canvas.create_image(0, 0, anchor="nw")

        frm_bottom = ttk.Frame(self.win)
        frm_bottom.grid(row=3, column=0, columnspan=2)
        but_exit = ttk.Button(frm_bottom, text='Exit', command=self.on_close)
        but_exit.grid(row=3, column=0, padx=5, pady=5, ipadx=5, ipady=5)
        but_cam_save = ttk.Button(frm_bottom, text='Save', command=self.cam_save)
        but_cam_save.grid(row=3, column=1, padx=5, pady=5, ipadx=5, ipady=5)
        but_cam_start_timer = ttk.Button(frm_bottom, text='Start auto acquisition', command=self.start_timed_mono_acq)
        but_cam_start_timer.grid(row=3, column=2, padx=5, pady=5, ipadx=5, ipady=5)
        but_cam_stop_timer = ttk.Button(frm_bottom, text='Stop auto acquisition', command=self.stop_timed_mono_acq)
        but_cam_stop_timer.grid(row=3, column=3, padx=5, pady=5, ipadx=5, ipady=5)

        #self.status_bar = ttk.Label(self.win, text="Cursor Position: (0,0) pixels", anchor=tk.W)
        #self.status_bar.grid(row=4, column=0, columnspan=2, sticky='we')
        # Bind the mouse motion event to the update_status_bar method
        #self.img_canvas.bind("<Motion>", self.update_status_bar)

        self.cam_live = True
        self.timer_running = False
        self.reset_roi()

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

            full_average_image = Image.fromarray(average_image)
            full_picture = full_average_image.resize((720, 540), resample=0)
            full_picture = ImageTk.PhotoImage(full_picture)
            self.img_canvas.full_image = full_picture

            if self.roi is not None:
                average_image = average_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

            if self.img_canvas.full_image is not None:
                picture = Image.fromarray(average_image)
                picture = picture.resize((720, 540), resample=0)
                picture = ImageTk.PhotoImage(picture)

            self.img_canvas.itemconfig(self.image, image=picture)
            self.img_canvas.image = picture

    def cam_cont_acq(self):
        self.render_thread_cont = threading.Thread(target=self.init_cam_cont)
        self.render_thread_cont.daemon = True
        self.render_thread_cont.start()

    def cam_stop(self):
        self.cam_live = False
        print('Continuous acquisition mode - off')

    def init_cam_mono(self):

        if self.cam_live is True:
            self.cam_live = False
            self.cam.stream_off()
            print('Continuous acquisition mode - off')

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

        full_average_image = Image.fromarray(average_image)
        full_picture = full_average_image.resize((720, 540), resample=0)
        full_picture = ImageTk.PhotoImage(full_picture)
        self.img_canvas.full_image = full_picture

        if self.roi is not None:
            average_image = average_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]

        if self.img_canvas.full_image is not None:
            picture = Image.fromarray(average_image)
            picture = picture.resize((720, 540), resample=0)
            picture = ImageTk.PhotoImage(picture)

            self.img_canvas.itemconfig(self.image, image=picture)
            self.img_canvas.image = picture

        print(f'Mono acquisition mode - Image taken over {num} averages')


    def cam_mono_acq(self):
        self.render_thread_mono = threading.Thread(target=self.init_cam_mono)
        self.render_thread_mono.daemon = True
        self.render_thread_mono.start()


    def start_timed_mono_acq(self):
        self.timer_running = True
        self.timed_mono_acq()

    def stop_timed_mono_acq(self):
        self.timer_running = False

    def timed_mono_acq(self):
        if self.timer_running:
            self.cam_mono_acq()
            self.cam_save()
            print('image saved')
            timing = int(self.ent_picture_timing.get())
            self.win.after(timing, self.timed_mono_acq)

    def cam_save(self):
        folder_path = 'C:/data/' + str(date.today()) + '/' + 'camera_' + str(int(self.ent_cam_ind.get())) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        base_filename = str(date.today())

        filename = base_filename + '.bmp'
        i = 1
        while os.path.exists(os.path.join(folder_path, filename)):
            filename = f"{base_filename}_{i}.bmp"
            i += 1

        full_path = os.path.join(folder_path, filename)

        last_image = self.img_canvas.full_image

        if last_image is not None:
            pil_image = ImageTk.getimage(last_image)
            pil_image.save(full_path)
            print(f"Image saved as {full_path}")
        else:
            print("No image to save")

    def select_roi(self):
        # Get the ROI coordinates from the entry fields
        x1 = int(self.ent_roi_x1.get())
        x2 = int(self.ent_roi_x2.get())
        y1 = int(self.ent_roi_y1.get())
        y2 = int(self.ent_roi_y2.get())

        self.roi = (x1, x2, y1, y2)

    def reset_roi(self):
        self.roi = self.initial_roi  # Reset ROI coordinates to initial values
        self.apply_roi()  # Apply the ROI

    def apply_roi(self):
        if self.roi is not None:
            self.ent_roi_x1.delete(0, tk.END)
            self.ent_roi_x2.delete(0, tk.END)
            self.ent_roi_y1.delete(0, tk.END)
            self.ent_roi_y2.delete(0, tk.END)
            self.ent_roi_x1.insert(0, str(self.roi[0]))
            self.ent_roi_x2.insert(0, str(self.roi[1]))
            self.ent_roi_y1.insert(0, str(self.roi[2]))
            self.ent_roi_y2.insert(0, str(self.roi[3]))

    def update_status_bar(self, event):
        x = self.img_canvas.canvasx(event.x)
        y = self.img_canvas.canvasy(event.y)

        pixel_value = self.get_pixel_value(x, y)
        self.status_bar.config(text=f"Cursor Position: ({int(2*x)},{int(2*y)}) pixels, Value: {pixel_value[0]}")

    def get_pixel_value(self, x, y):
        # Retrieve the pixel value at the given x, y coordinates from the image
        if self.img_canvas.image is not None:
            pil_image = ImageTk.getimage(self.img_canvas.image)
            try:
                if pil_image:
                    pixel_value = pil_image.getpixel((int(x), int(y)))
                    return pixel_value
            except IndexError:
                return 0
        return 0

    def on_close(self):
        self.win.destroy()
