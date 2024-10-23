import datetime
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageTk
import hardware.gxipy_driver.daheng_camera as dh
from hardware import gxipy_driver as gx


class GasDensity:
    def __init__(self):

        self.initial_roi = (0, 540, 0, 720)  # x1 x2 y1 y2
        self.current_roi = self.initial_roi
        self.relevant_image = None
        self.roi_rectangle = None

        self.win = tk.Toplevel()
        vcmd = self.win.register(self.is_number_input)

        title = 'D-Lab Controller - Gas Density'

        self.win.title(title)
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

        self.frm_cam = ttk.LabelFrame(self.win, text="Camera display")
        self.frm_main_settings = ttk.LabelFrame(self.win, text="Main settings")
        frm_controls = ttk.LabelFrame(self.frm_main_settings, text="Camera settings")

        self.frm_cam.grid(row=0, column=0, sticky='nsew')
        self.frm_main_settings.grid(row=0, column=1, sticky='nsew')
        frm_controls.grid(row=1, column=0, sticky='nsew')

        frm_cam_but = ttk.Frame(frm_controls)
        frm_cam_but_set = ttk.Frame(frm_cam_but)

        self.but_cam_init = ttk.Button(frm_cam_but, text='Initialize', command=self.initialize_daheng)
        self.but_cam_disconnect = ttk.Button(frm_cam_but, text='Disconnect', command=self.close_daheng)
        self.but_cam_live = ttk.Button(frm_cam_but, text='Live', command=self.live_daheng_thread)
        self.but_cam_single = ttk.Button(frm_cam_but, text='Single', command=self.single_daheng_thread)

        lbl_cam_ind = ttk.Label(frm_cam_but_set, text='Camera index:')
        self.strvar_cam_ind = tk.StringVar(self.win, '1')
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

        lbl_roi_x1.grid(row=2, column=0, sticky='nsew')
        self.ent_roi_x1.grid(row=2, column=1, padx=(0, 10))
        lbl_roi_x2.grid(row=2, column=2, sticky='nsew')
        self.ent_roi_x2.grid(row=2, column=3, padx=(0, 10))

        lbl_roi_y1.grid(row=3, column=0, sticky='nsew')
        self.ent_roi_y1.grid(row=3, column=1, padx=(0, 10))
        lbl_roi_y2.grid(row=3, column=2, sticky='nsew')
        self.ent_roi_y2.grid(row=3, column=3, padx=(0, 10))

        lbl_cam_ind.grid(row=0, column=0, sticky='nsew')
        self.ent_cam_ind.grid(row=0, column=1, padx=(0, 10), sticky='nsew')
        frm_cam_but_set.grid(row=0, column=0, sticky='nsew')
        frm_cam_but.grid(row=1, column=0, sticky='nsew')
        lbl_cam_exp.grid(row=6, column=0, sticky='nsew')
        self.ent_cam_exp.grid(row=6, column=1, padx=(0, 10), sticky='nsew')
        self.box_auto_exp.grid(row=6, column=2, padx=(0, 10), sticky='nsew')
        lbl_cam_gain.grid(row=7, column=0, sticky='nsew')
        self.ent_cam_gain.grid(row=7, column=1, padx=(0, 10), sticky='nsew')
        lbl_cam_avg.grid(row=8, column=0, sticky='nsew')
        self.ent_cam_avg.grid(row=8, column=1, padx=(0, 10), sticky='nsew')

        self.but_cam_init.grid(row=1, column=0, sticky='nsew')
        self.but_cam_disconnect.grid(row=2, column=0, sticky='nsew')
        self.but_cam_live.grid(row=3, column=0, sticky='nsew')
        self.but_cam_single.grid(row=4, column=0, sticky='nsew')

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

        self.img_canvas.bind('<Motion>', self.update_crosshair)
        self.img_canvas.bind('<Button-1>', self.start_roi_selection)
        self.img_canvas.bind('<B1-Motion>', self.update_roi_selection)
        self.img_canvas.bind('<ButtonRelease-1>', self.end_roi_selection)
        self.img_canvas.bind('<Button-3>', self.toggle_crosshair_lock)
        self.win.bind('<Control-r>', self.reset_roi)
        self.win.bind('<Control-c>', self.toggle_crosshair_visibility)

        self.crosshair_visible = False
        self.crosshair_locked = True

        frm_bottom = ttk.Frame(self.win)
        frm_bottom.grid(row=3, column=0, columnspan=2)
        but_exit = ttk.Button(frm_bottom, text='Exit', command=self.on_close)
        but_exit.grid(row=3, column=0, padx=5, pady=5, ipadx=5, ipady=5)

        self.daheng_camera = None
        self.daheng_is_live = False
        self.current_daheng_image = None
        self.optimized = False

        self.autolog_camera_focus = 'C:/data/' + str(datetime.date.today()) + '/' + str(
            datetime.date.today()) + '-' + 'auto-log-camera_focus.txt'
        self.autolog_cam = open(self.autolog_camera_focus, "a+")

        style = ttk.Style()
        style.configure('Green.TButton', background='green', foreground='black')
        style.configure('Red.TButton', background='red', foreground='black')

        self.initialize_daheng()
        self.live_daheng_thread()

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

    def initialize_daheng(self):
        device_manager = gx.DeviceManager()
        try:
            self.daheng_camera = dh.DahengCamera(int(self.strvar_cam_ind.get()))
        except:
            self.insert_message("Something went wrong with init of camera:(")

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
        if self.daheng_is_live:
            message = 'Live view on'

            self.but_cam_live.config(style='Green.TButton')
            self.insert_message(message)
        else:
            message = 'Live view off'
            self.but_cam_live.config(style='Red.TButton')
            self.insert_message(message)

    def adjust_exposure_new(self):
        if self.daheng_is_live:
            rel = self.relevant_image
            if np.max(rel) > 254:
                new_exposure = int(self.strvar_cam_exp.get()) / 2
                self.strvar_cam_exp.set(str(int(new_exposure)))
            else:
                new_exposure = 255 / np.max(rel) * 0.8 * int(self.strvar_cam_exp.get())
                self.strvar_cam_exp.set(str(int(new_exposure)))
                self.optimized = True

        else:
            while not self.optimized:
                im = self.daheng_camera.take_image(int(self.strvar_cam_exp.get()), int(self.strvar_cam_gain.get()),
                                                   int(self.strvar_cam_avg.get()))
                self.plot_daheng(im)
                rel = self.relevant_image
                if np.max(rel) > 254:
                    new_exposure = int(self.strvar_cam_exp.get()) / 2
                    self.strvar_cam_exp.set(str(int(new_exposure)))
                else:
                    new_exposure = 255 / np.max(rel) * 0.8 * int(self.strvar_cam_exp.get())
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
            if np.max(self.relevant_image) > 254:
                self.frm_cam.config(text="SATURATED!")
            elif np.max(self.relevant_image) > 240:
                self.frm_cam.config(text="ALMOST saturated, max: {}".format(int(np.max(self.relevant_image))))
            else:
                self.frm_cam.config(text="Camera display, max: {}".format(int(np.max(self.relevant_image))))

            colored_image_array = self.apply_colormap(cropped_im, colormap_name='turbo')
            colored_image = Image.fromarray(colored_image_array)

            image_resized = colored_image.resize((720, 540), resample=Image.NEAREST)
            photo = ImageTk.PhotoImage(image_resized)
            self.img_canvas.itemconfig(self.image, image=photo)
            self.img_canvas.image = photo
        except ValueError as e:
            message = f"Error processing the image: {e}"
            self.insert_message(message)

    def apply_colormap(self, image_array, colormap_name='viridis'):
        colormap = cm.get_cmap(colormap_name)
        colored_image = colormap(image_array / 255.0)
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
        return colored_image

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

    def on_close(self):
        self.autolog_cam.close()
        if self.daheng_camera is not None:
            self.close_daheng()
        self.win.destroy()
