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

        self.figrMCP = Figure(figsize=(5, 6), dpi=100)
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
        self.imgMCP.draw()

        # Create a frame for general control
        self.settings_frame = ttk.Label(self.main_frame)
        self.init_button = ttk.Button(master=self.settings_frame, text="Live", command=self.start)
        self.init_button.grid(row=0, column=0, padx=5, pady=5)
        self.stop_button = ttk.Button(master=self.settings_frame, text="Stop", command=self.stop)
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)
        self.exit_button = ttk.Button(master=self.settings_frame, text="EXIT", command=self.on_close)
        self.exit_button.grid(row=0, column=2, padx=5, pady=5)

        # Create a frame for the exposure time setting and the gain
        self.camera_settings_frame = ttk.LabelFrame(self.main_frame, text="Camera settings")
        self.exposure_label = ttk.Label(master=self.camera_settings_frame, text="Exposure time (s)")
        self.exposure_label.grid(row=0, column=0, padx=5, pady=5)
        self.exposure_time = 50E-3
        self.exposure_time_var = tk.StringVar(value=str(self.exposure_time))
        self.exposure_entry = ttk.Entry(master=self.camera_settings_frame, width=5, textvariable=self.exposure_time_var)
        self.exposure_entry.grid(row=0, column=1, padx=5, pady=5)
        self.exposure_button = ttk.Button(master=self.camera_settings_frame, text="Set exposure time",
                                          command=self.foo)
        self.exposure_button.grid(row=0, column=2, padx=5, pady=5)
        self.gain_label = ttk.Label(master=self.camera_settings_frame, text="EMCCD Gain")
        self.gain_label.grid(row=1, column=0, padx=5, pady=5)
        self.gain = 0
        self.gain_var = tk.StringVar(value=str(self.gain))
        self.gain_entry = ttk.Entry(master=self.camera_settings_frame, width=5, textvariable=self.gain_var)
        self.gain_entry.grid(row=1, column=1, padx=5, pady=5)
        self.gain_button = ttk.Button(master=self.camera_settings_frame, text="Set gain", command=self.foo)
        self.gain_button.grid(row=1, column=2, padx=5, pady=5)

        # Create a frame for the ROI of the image
        self.roi_frame = ttk.LabelFrame(self.main_frame, text="Range of interest - XUV camera")
        self.x_start_label = ttk.Label(master=self.roi_frame, text="from x =")
        self.x_start_label.grid(row=0, column=0, padx=5, pady=5)
        self.roi_x_start_var = tk.StringVar(value=str(0))
        self.x_start_entry = ttk.Entry(master=self.roi_frame, width=10, textvariable=self.roi_x_start_var)
        self.x_start_entry.grid(row=0, column=1, padx=5, pady=5)
        self.x_end_label = ttk.Label(master=self.roi_frame, text="to x =")
        self.x_end_label.grid(row=0, column=2, padx=5, pady=5)
        self.roi_x_end_var = tk.StringVar(value=str(512))
        self.x_end_entry = ttk.Entry(master=self.roi_frame, width=10, textvariable=self.roi_x_end_var)
        self.x_end_entry.grid(row=0, column=3, padx=5, pady=5)
        self.y_start_label = ttk.Label(master=self.roi_frame, text="from y =")
        self.y_start_label.grid(row=1, column=0, padx=5, pady=5)
        self.roi_y_start_var = tk.StringVar(value=str(0))
        self.y_start_entry = ttk.Entry(master=self.roi_frame, width=10, textvariable=self.roi_y_start_var)
        self.y_start_entry.grid(row=1, column=1, padx=5, pady=5)
        self.y_end_label = ttk.Label(master=self.roi_frame, text="to y =")
        self.y_end_label.grid(row=1, column=2, padx=5, pady=5)
        self.roi_y_end_var = tk.StringVar(value=str(512))
        self.y_end_entry = ttk.Entry(master=self.roi_frame, width=10, textvariable=self.roi_y_end_var)
        self.y_end_entry.grid(row=1, column=3, padx=5, pady=5)
        self.roi_reset_button = ttk.Button(master=self.roi_frame, text="Reset image ROI", command=self.foo)
        self.roi_reset_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        self.roi_set_button = ttk.Button(master=self.roi_frame, text="Set image ROI", command=self.foo)
        self.roi_set_button.grid(row=2, column=2, columnspan=2, padx=5, pady=5)

        # Create a frame for the summation settings
        self.sum_frame = tk.LabelFrame(self.main_frame, text="Range of interest - Summation plot")
        self.sum_start_var = tk.StringVar(value=str(0))
        self.sum_start_label = ttk.Label(master=self.sum_frame, text="from y =")
        self.sum_start_label.grid(row=1, column=0, padx=5, pady=5)
        self.sum_start_entry = ttk.Entry(master=self.sum_frame, width=5, textvariable=self.sum_start_var)
        self.sum_start_entry.grid(row=1, column=1, padx=5, pady=5)
        self.sum_end_label = ttk.Label(master=self.sum_frame, text="to y =")
        self.sum_end_label.grid(row=1, column=2, padx=5, pady=5)
        self.sum_end_var = tk.StringVar(value=str(512))
        self.sum_end_entry = ttk.Entry(master=self.sum_frame, width=5, textvariable=self.sum_end_var)
        self.sum_end_entry.grid(row=1, column=3, padx=5, pady=5)
        self.sum_reset_button = ttk.Button(master=self.sum_frame, text="Reset sum ROI", command=self.foo)
        self.sum_reset_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        self.sum_set_button = ttk.Button(master=self.sum_frame, text="Set sum ROI", command=self.foo)
        self.sum_set_button.grid(row=2, column=2, columnspan=2, padx=5, pady=5)

        self.average_frame = tk.LabelFrame(self.main_frame, text='Averages:')
        self.strvar_avg = tk.StringVar(self.win, '1')
        self.ent_avg = tk.Entry(self.average_frame, width=11, validate='all', textvariable=self.strvar_avg)
        self.ent_avg.grid(row=1, column=0)

        # Create a frame for the cooling settings
        self.cooling_frame = ttk.LabelFrame(self.main_frame, text="Cooling status")
        self.temp_label = ttk.Label(master=self.cooling_frame, text="Current temperature")
        self.temp_label.grid(row=0, column=0, padx=5, pady=5)
        self.camera_temp = 0
        self.camera_temp_var = tk.StringVar(value=str(self.camera_temp))
        self.temp_value_label = ttk.Label(master=self.cooling_frame, text="")
        self.temp_value_label.grid(row=0, column=1, padx=5, pady=5)

        self.temp_setpoint_label = ttk.Label(master=self.cooling_frame, text="Current setpoint")
        self.temp_setpoint_label.grid(row=1, column=0, padx=5, pady=5)
        self.camera_temp_setpoint = 0
        self.camera_temp_setpoint_var = tk.StringVar(value=str(self.camera_temp_setpoint))
        self.temp_value_setpoint_label = ttk.Label(master=self.cooling_frame, text=self.camera_temp_setpoint_var)
        self.temp_value_setpoint_label.grid(row=1, column=1, padx=5, pady=5)

        # Add all frames to the main frame
        self.plot_frame.grid(row=0, column=0, columnspan=2, rowspan=6)
        self.settings_frame.grid(row=0, column=2, sticky="ew")
        self.roi_frame.grid(row=1, column=2, sticky="ew")
        self.sum_frame.grid(row=2, column=2, sticky="ew")
        self.camera_settings_frame.grid(row=3, column=2, sticky="ew")
        self.average_frame.grid(row=4, column=2, sticky="ew")
        self.cooling_frame.grid(row=5, column=2, sticky="ew")

        # Add a status bar at the bottom
        self.status_bar = ttk.Label(self.win, text="", anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky="ew")
        self.status_bar.config(text=str('test'))

        self.sum_start_index = 0
        self.sum_end_index = 512

        self.im = np.zeros([512, 512])

        self.live = True

    def foo(self):
        if self.live is True:
            print('true')
        else:
            print('false')

    def enable_camera(self):
        """
        Enables the MCP measurement.

        Returns
        -------
        None
        """
        self.cam = Andor.AndorSDK2Camera(fan_mode="full")
        self.cam.set_exposure(50E-3)
        self.cam.start_acquisition()
        self.camera_thread = threading.Thread(target=self.measure)
        self.camera_thread.daemon = True
        self.camera_thread.start()

    def measure(self):
        while self.live is True:
            self.im = self.take_image(int(5))
            self.plot_MCP(self.im)
            print('plot plot')
            time.sleep(0.2)

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
        nr = avgs
        for i in range(nr):
            time.sleep(0.5)
            print('mimir')
            self.cam.wait_for_frame()
            frame = self.cam.read_oldest_image()
            image += frame
            print(f'{i + 1} frames on {nr}')
        image /= nr
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
        self.axMCP.imshow(mcpimage, extent=[0, 512, 0, 512])
        #self.axMCP.set_aspect('equal')

        self.axMCP.set_xlabel("X (px)")
        self.axMCP.set_ylabel("Y (px)")
        self.axMCP.set_xlim(0, 512)
        self.axMCP.set_ylim(0, 512)

        self.axHarmonics.clear()
        self.axHarmonics.plot(np.arange(512), np.sum(mcpimage, 0))
        self.axHarmonics.set_xlabel("X (px)")
        self.axHarmonics.set_ylabel("Counts (arb.u.)")

        self.axHarmonics.set_xlim(0, 512)

        self.figrMCP.tight_layout()
        self.imgMCP.draw()

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
        self.cam.close()
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
            self.cam.setup_shutter("closed")
            self.cam.close()
            self.win.destroy()
            self.parent.andor_camera = None
        print('Closing the XUV camera')
