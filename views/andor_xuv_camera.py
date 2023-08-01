import tkinter as tk
from tkinter import ttk
from pylablib.devices import Andor
import pylablib as pll
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import numpy as np
import threading
from matplotlib.figure import Figure
import time
import matplotlib.colors as colors


class AndorCameraViewer(object):
    def __init__(self, parent):
        self.cam = None
        self.parent = parent
        self.win = tk.Toplevel()

        title = 'D-Lab Controller - Andor camera'
        pll.par["devices/dlls/andor_sdk2"] = "drivers/andor_driver/"

        self.win.title(title)
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

        # Create a main frame to group all other frames
        self.main_frame = ttk.Frame(self.win)
        self.main_frame.grid(row=0, column=0)
        self.plot_frame = ttk.LabelFrame(self.main_frame, text="Camera display")
        self.parameters_frame = ttk.LabelFrame(self.main_frame, text="Parameters")

        self.figrMCP = Figure(figsize=(7, 7), dpi=100)
        self.axMCP = self.figrMCP.add_subplot(211)
        self.axHarmonics = self.figrMCP.add_subplot(212)
        self.axMCP.set_xlim(0, 512)
        self.axMCP.set_ylim(0, 512)
        self.axHarmonics.set_xlim(0, 512)
        self.figrMCP.tight_layout()
        self.figrMCP.canvas.draw()
        self.imgMCP = FigureCanvasTkAgg(self.figrMCP, self.plot_frame)
        self.tk_widget_figrMCP = self.imgMCP.get_tk_widget()
        self.tk_widget_figrMCP.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.selector = RectangleSelector(self.axMCP, self.on_select, useblit=True, button=[1])
        self.imgMCP.draw()

        # Create a frame for general control
        self.settings_frame = ttk.LabelFrame(self.parameters_frame, text='Control panel')
        self.init_button = ttk.Button(master=self.settings_frame, text="Live", command=self.start)
        self.init_button.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.log_button = ttk.Button(master=self.settings_frame, text="Log scale", command=self.log_scale)
        self.log_button.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        self.stop_button = ttk.Button(master=self.settings_frame, text="Stop", command=self.stop)
        self.stop_button.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')
        self.exit_button = ttk.Button(master=self.settings_frame, text="Close and exit", command=self.on_close)
        self.exit_button.grid(row=0, column=3, padx=5, pady=5, sticky='nsew')

        # Create a frame for the exposure time setting and the gain
        self.camera_settings_frame = ttk.LabelFrame(self.parameters_frame, text="Camera settings")

        lbl_exposure_time = tk.Label(self.camera_settings_frame, text='Exposure time (s) :')
        self.strvar_exposure_time = tk.StringVar(self.win, '50e-3')
        self.ent_exposure_time = tk.Entry(self.camera_settings_frame, width=10, validate='all',
                                          textvariable=self.strvar_exposure_time)
        lbl_exposure_time.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_exposure_time.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')

        lbl_EMCCD = tk.Label(self.camera_settings_frame, text='EMCCD Gain :')
        self.strvar_EMCCD = tk.StringVar(self.win, '0')
        self.ent_EMCCD = tk.Entry(self.camera_settings_frame, width=10, validate='all', textvariable=self.strvar_EMCCD)
        lbl_EMCCD.grid(row=1, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_EMCCD.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')

        lbl_avg = tk.Label(self.camera_settings_frame, text='Averages :')
        self.strvar_avg = tk.StringVar(self.win, '10')
        self.ent_avg = tk.Entry(self.camera_settings_frame, width=5, validate='all', textvariable=self.strvar_avg)
        lbl_avg.grid(row=2, column=0, padx=2, pady=2, sticky='nsew')
        self.ent_avg.grid(row=2, column=1, padx=2, pady=2, sticky='nsew')


        # Create a frame for the ROI of the image
        self.roi_frame = ttk.LabelFrame(self.parameters_frame, text="Range of interest")
        self.x_start_label = ttk.Label(master=self.roi_frame, text="from x =")
        self.x_start_label.grid(row=0, column=0, padx=5, pady=5)
        self.roi_x_start_var = tk.StringVar(value='0')
        self.x_start_entry = ttk.Entry(master=self.roi_frame, width=10, textvariable=self.roi_x_start_var)
        self.x_start_entry.grid(row=0, column=1, padx=5, pady=5)
        self.x_end_label = ttk.Label(master=self.roi_frame, text="to x =")
        self.x_end_label.grid(row=0, column=2, padx=5, pady=5)
        self.roi_x_end_var = tk.StringVar(value='512')
        self.x_end_entry = ttk.Entry(master=self.roi_frame, width=10, textvariable=self.roi_x_end_var)
        self.x_end_entry.grid(row=0, column=3, padx=5, pady=5)
        self.y_start_label = ttk.Label(master=self.roi_frame, text="from y =")
        self.y_start_label.grid(row=1, column=0, padx=5, pady=5)
        self.roi_y_start_var = tk.StringVar(value='0')
        self.y_start_entry = ttk.Entry(master=self.roi_frame, width=10, textvariable=self.roi_y_start_var)
        self.y_start_entry.grid(row=1, column=1, padx=5, pady=5)
        self.y_end_label = ttk.Label(master=self.roi_frame, text="to y =")
        self.y_end_label.grid(row=1, column=2, padx=5, pady=5)
        self.roi_y_end_var = tk.StringVar(value='512')
        self.y_end_entry = ttk.Entry(master=self.roi_frame, width=10, textvariable=self.roi_y_end_var)
        self.y_end_entry.grid(row=1, column=3, padx=5, pady=5)
        self.roi_reset_button = ttk.Button(master=self.roi_frame, text="Reset image ROI", command=self.reset_roi)
        self.roi_reset_button.grid(row=2, column=0, padx=5, pady=5)
        self.roi_set_button = ttk.Button(master=self.roi_frame, text="Set image ROI", command=self.set_roi)
        self.roi_set_button.grid(row=2, column=2, padx=5, pady=5)

        # Add all frames to the main frame
        self.plot_frame.grid(row=0, column=0, sticky="nsew")
        self.parameters_frame.grid(row=0, column=1, sticky="nsew")

        self.settings_frame.grid(row=0, column=0, sticky="nsew")
        self.roi_frame.grid(row=1, column=0, sticky="nsew")
        self.camera_settings_frame.grid(row=2, column=0, sticky="nsew")

        self.x_start = 0
        self.x_end = 512
        self.y_start = 0
        self.y_end = 512

        self.im = np.zeros([512, 512])

        self.live = False
        self.stop_live = True
        self.log_image = False

    def enable_camera(self):
        """
        Enables the MCP measurement.

        Returns
        -------
        None
        """
        self.live = True
        self.stop_live = False
        self.cam = Andor.AndorSDK2Camera(fan_mode="full")
        self.cam.set_exposure(float(self.ent_exposure_time.get()))
        self.cam.set_EMCCD_gain(float(self.ent_EMCCD.get()))
        self.cam.setup_shutter('open')
        self.camera_thread = threading.Thread(target=self.measure)
        self.camera_thread.daemon = True
        self.camera_thread.start()

    def measure(self):
        while self.live is True:
            self.im = self.take_image(int(self.ent_avg.get()))
            self.plot_MCP(self.im)
            time.sleep(0.1)
            if self.stop_live is True:
                break

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
        image = np.zeros([512, 512])
        self.cam.start_acquisition()
        for i in range(avgs):
            self.cam.wait_for_frame(timeout=20)
            frame = self.cam.read_oldest_image()
            image += frame
        image /= avgs
        self.cam.stop_acquisition()
        return image

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

        if self.log_image is True:
            image = self.axMCP.imshow(mcpimage, norm=colors.LogNorm())
        elif self.log_image is False:
            image = self.axMCP.imshow(mcpimage)

        self.axMCP.set_aspect('equal')

        self.axMCP.set_xlabel("X (px)")
        self.axMCP.set_ylabel("Y (px)")
        self.axMCP.set_xlim([self.x_start, self.x_end])
        self.axMCP.set_ylim([self.y_start, self.y_end])

        self.axHarmonics.clear()
        sums = np.sum(mcpimage[self.y_start:self.y_end, self.x_start:self.x_end], axis=0)
        self.axHarmonics.plot(np.arange(len(sums)), sums)
        self.axHarmonics.set_xlabel("X (px)")
        self.axHarmonics.set_ylabel("Counts (arb.u.)")

        self.figrMCP.tight_layout()
        self.imgMCP.draw()

    def log_scale(self):
        if self.log_image is True:
            self.log_image = False
            print('Log scale off')
        elif self.log_image is False:
            self.log_image = True
            print('Log scale on')


    def set_roi(self):
        """
        Set the Region of Interest (ROI) for the camera.

        Returns
        -------
        None

        """
        self.x_start = int(self.roi_x_start_var.get())
        self.x_end = int(self.roi_x_end_var.get())
        self.y_start = int(self.roi_y_start_var.get())
        self.y_end = int(self.roi_y_end_var.get())

    def reset_roi(self):
        """
        Reset the Region of Interest (ROI) to the default values.

        Returns
        -------
        None

        """

        self.roi_x_start_var.set(str(0))
        self.roi_x_end_var.set(str(512))
        self.roi_y_start_var.set(str(0))
        self.roi_y_end_var.set(str(512))

        self.set_roi()

    def on_select(self, e_click, e_release):
        """
        Perform actions when the user selects a region of interest (ROI) on the plot.

        Parameters
        ----------
        e_click : matplotlib.backend_bases.MouseEvent
            The mouse click event.
        e_release : matplotlib.backend_bases.MouseEvent
            The mouse release event.

        Returns
        -------
        None

        """
        x1, y1 = int(e_click.xdata), int(e_click.ydata)
        x2, y2 = int(e_release.xdata), int(e_release.ydata)

        self.roi_x_start_var.set(x1)
        self.roi_x_end_var.set(x2)
        self.roi_y_start_var.set(y1)
        self.roi_y_end_var.set(y2)

        self.set_roi()

    def start(self):
        """
        Start the acquisition process.

        Returns
        -------
        None
        """
        self.enable_camera()
        print('Acquisition started')

    def stop(self):
        """
        Stop the acquisition process.

        Returns
        -------
        None
        """
        self.stop_live = True
        print('Acquisition stopped')

    def on_close(self):
        """
        Handle the event when the window is closed.

        Returns
        -------
        None

        """
        if self.cam is None:
            self.win.destroy()
            self.parent.andor_camera = None
        else:
            self.cam.stop_acquisition()
            self.cam.close()
            self.win.destroy()
            self.parent.andor_camera = None
        print('Closing the Andor camera')
