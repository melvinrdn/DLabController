import tkinter as tk
from tkinter import ttk
from pylablib.devices import Andor
import pylablib as pll
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import numpy as np
from tkinter import filedialog


class AndorCameraViewer(object):
    def __init__(self, parent):
        self.cam = None
        self.parent = parent
        self.win = tk.Toplevel()

        title = 'SLM Phase Control - Andor camera'
        pll.par["devices/dlls/andor_sdk2"] = "drivers/andor_driver/"

        self.win.title(title)
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

        # Set up the Andor camera
        self.cam = Andor.AndorSDK2Camera(fan_mode="full")
        print('Opening the XUV camera')
        self.cam.set_exposure(50E-3)
        self.cam.set_acquisition_mode('cont')
        self.cam.setup_shutter("open")
        self.cam.start_acquisition()
        print('XUV camera ready')

        # Create a main frame to group all other frames
        self.main_frame = ttk.Frame(self.win)
        self.main_frame.grid(row=0, column=0)

        # Create a frame for the camera display
        self.plot_frame = ttk.LabelFrame(self.main_frame, text="Camera display")
        self.fig, self.ax = plt.subplots(figsize=(4.5, 4.5))
        self.ax.set_xlabel('Pixels')
        self.ax.set_ylabel('Pixels')
        self.img = self.ax.imshow(self.cam.read_oldest_image())
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=1, column=1, padx=0, pady=0, sticky=tk.NSEW)
        self.selector = RectangleSelector(self.ax, self.on_select, useblit=True, button=[1])
        self.fig.tight_layout()

        # Create a frame for the sum plot
        self.sum_plot_frame = ttk.LabelFrame(self.main_frame, text="Sum over columns")
        self.sum_fig, self.sum_ax = plt.subplots(figsize=(4.5, 2.5))
        self.sum_ax.set_xlabel('Pixels')
        self.sum_ax.set_ylabel('Counts')
        self.sum_canvas = FigureCanvasTkAgg(self.sum_fig, master=self.sum_plot_frame)
        self.sum_canvas.get_tk_widget().grid(row=0, column=1, padx=0, pady=0, sticky=tk.NSEW)
        self.sum_fig.tight_layout()

        # Create a frame for general control
        self.settings_frame = ttk.Label(self.main_frame)
        self.live_button = ttk.Button(master=self.settings_frame, text="Live", command=self.start)
        self.live_button.grid(row=0, column=0, padx=5, pady=5)
        self.stop_button = ttk.Button(master=self.settings_frame, text="Stop", command=self.stop)
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)
        self.save_button = ttk.Button(master=self.settings_frame, text="Save image", command=self.save_image)
        self.save_button.grid(row=0, column=2, padx=5, pady=5)
        self.exit_button = ttk.Button(master=self.settings_frame, text="EXIT", command=self.on_close)
        self.exit_button.grid(row=0, column=3, padx=5, pady=5)

        # Create a frame for the exposure time setting and the gain
        self.camera_settings_frame = ttk.LabelFrame(self.main_frame, text="Camera settings")
        self.exposure_label = ttk.Label(master=self.camera_settings_frame, text="Exposure time (s)")
        self.exposure_label.grid(row=0, column=0, padx=5, pady=5)
        self.exposure_time = 50E-3
        self.exposure_time_var = tk.StringVar(value=str(self.exposure_time))
        self.exposure_entry = ttk.Entry(master=self.camera_settings_frame, width=5, textvariable=self.exposure_time_var)
        self.exposure_entry.grid(row=0, column=1, padx=5, pady=5)
        self.exposure_button = ttk.Button(master=self.camera_settings_frame, text="Set exposure time",
                                         command=self.set_exposure_time)
        self.exposure_button.grid(row=0, column=2, padx=5, pady=5)
        self.gain_label = ttk.Label(master=self.camera_settings_frame, text="EMCCD Gain")
        self.gain_label.grid(row=1, column=0, padx=5, pady=5)
        self.gain = self.cam.get_EMCCD_gain()[0]
        self.gain_var = tk.StringVar(value=str(self.gain))
        self.gain_entry = ttk.Entry(master=self.camera_settings_frame, width=5, textvariable=self.gain_var)
        self.gain_entry.grid(row=1, column=1, padx=5, pady=5)
        self.gain_button = ttk.Button(master=self.camera_settings_frame, text="Set gain", command=self.set_gain)
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
        self.roi_x_end_var = tk.StringVar(value=str(self.cam.get_detector_size()[0]))
        self.x_end_entry = ttk.Entry(master=self.roi_frame, width=10, textvariable=self.roi_x_end_var)
        self.x_end_entry.grid(row=0, column=3, padx=5, pady=5)
        self.y_start_label = ttk.Label(master=self.roi_frame, text="from y =")
        self.y_start_label.grid(row=1, column=0, padx=5, pady=5)
        self.roi_y_start_var = tk.StringVar(value=str(0))
        self.y_start_entry = ttk.Entry(master=self.roi_frame, width=10, textvariable=self.roi_y_start_var)
        self.y_start_entry.grid(row=1, column=1, padx=5, pady=5)
        self.y_end_label = ttk.Label(master=self.roi_frame, text="to y =")
        self.y_end_label.grid(row=1, column=2, padx=5, pady=5)
        self.roi_y_end_var = tk.StringVar(value=str(self.cam.get_detector_size()[1]))
        self.y_end_entry = ttk.Entry(master=self.roi_frame, width=10, textvariable=self.roi_y_end_var)
        self.y_end_entry.grid(row=1, column=3, padx=5, pady=5)
        self.roi_reset_button = ttk.Button(master=self.roi_frame, text="Reset image ROI", command=self.reset_roi)
        self.roi_reset_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        self.roi_set_button = ttk.Button(master=self.roi_frame, text="Set image ROI", command=self.set_roi)
        self.roi_set_button.grid(row=2, column=2, columnspan=2, padx=5, pady=5)
        self.reset_roi()

        # Create a frame for the summation settings
        self.sum_frame = tk.LabelFrame(self.main_frame, text="Range of interest - Summation plot")
        self.sum_checkbutton = tk.Checkbutton(master=self.sum_frame, text="Show sum plot",
                                              variable=tk.BooleanVar(),
                                              command=self.toggle_sum_plot)
        self.sum_checkbutton.grid(row=0, column=0, padx=5, pady=5)
        self.sum_start_var = tk.StringVar(value=str(0))
        self.sum_start_label = ttk.Label(master=self.sum_frame, text="from y =")
        self.sum_start_label.grid(row=1, column=0, padx=5, pady=5)
        self.sum_start_entry = ttk.Entry(master=self.sum_frame, width=5, textvariable=self.sum_start_var)
        self.sum_start_entry.grid(row=1, column=1, padx=5, pady=5)
        self.sum_end_label = ttk.Label(master=self.sum_frame, text="to y =")
        self.sum_end_label.grid(row=1, column=2, padx=5, pady=5)
        self.sum_end_var = tk.StringVar(value=str(self.cam.get_detector_size()[1]))
        self.sum_end_entry = ttk.Entry(master=self.sum_frame, width=5, textvariable=self.sum_end_var)
        self.sum_end_entry.grid(row=1, column=3, padx=5, pady=5)
        self.sum_reset_button = ttk.Button(master=self.sum_frame, text="Reset sum ROI", command=self.reset_sum_roi)
        self.sum_reset_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        self.sum_set_button = ttk.Button(master=self.sum_frame, text="Set sum ROI", command=self.set_sum_roi)
        self.sum_set_button.grid(row=2, column=2, columnspan=2, padx=5, pady=5)
        self.reset_sum_roi()

        # Create a frame for the averaging settings
        self.average_frame = ttk.LabelFrame(self.main_frame, text="Averaging settings")
        self.avg_num = 10
        self.avg_checkbutton = ttk.Checkbutton(master=self.average_frame, text="Enable averaging",
                                              variable=tk.BooleanVar(), command=self.toggle_avg_mode)
        self.avg_checkbutton.grid(row=0, column=0, padx=5, pady=5)
        self.avg_label = ttk.Label(master=self.average_frame, text="Number of averages")
        self.avg_label.grid(row=1, column=0, padx=5, pady=5)
        self.avg_var = tk.StringVar(value=str(self.avg_num))
        self.avg_entry = ttk.Entry(master=self.average_frame, width=5, textvariable=self.avg_var)
        self.avg_entry.grid(row=1, column=1, padx=5, pady=5)
        self.set_avg_button = ttk.Button(master=self.average_frame, text="Set average", command=self.set_avg_num)
        self.set_avg_button.grid(row=0, column=1, padx=5, pady=5)

        # Create a frame for the cooling settings
        self.cooling_frame = ttk.LabelFrame(self.main_frame, text="Cooling status")
        self.temp_label = ttk.Label(master=self.cooling_frame, text="Current temperature")
        self.temp_label.grid(row=0, column=0, padx=5, pady=5)
        self.camera_temp = self.cam.get_temperature()
        self.camera_temp_var = tk.StringVar(value=str(self.camera_temp))
        self.temp_value_label = ttk.Label(master=self.cooling_frame, text="")
        self.temp_value_label.grid(row=0, column=1, padx=5, pady=5)

        self.temp_setpoint_label = ttk.Label(master=self.cooling_frame, text="Current setpoint")
        self.temp_setpoint_label.grid(row=1, column=0, padx=5, pady=5)
        self.camera_temp_setpoint = self.cam.get_temperature_setpoint()
        self.camera_temp_setpoint_var = tk.StringVar(value=str(self.camera_temp_setpoint))
        self.temp_value_setpoint_label = ttk.Label(master=self.cooling_frame, text=self.camera_temp_setpoint_var)
        self.temp_value_setpoint_label.grid(row=1, column=1, padx=5, pady=5)

        self.update_temperature()

        # Add all frames to the main frame
        self.plot_frame.grid(row=0, column=0, columnspan=2, rowspan=6)
        self.sum_plot_frame.grid(row=6, column=0, columnspan=2, sticky="nsew")
        self.settings_frame.grid(row=0, column=2, sticky="ew")
        self.roi_frame.grid(row=1, column=2, sticky="ew")
        self.sum_frame.grid(row=2, column=2, sticky="ew")
        self.camera_settings_frame.grid(row=3, column=2, sticky="ew")
        self.average_frame.grid(row=4, column=2, sticky="ew")
        self.cooling_frame.grid(row=5, column=2, sticky="ew")

        # Add a status bar at the bottom
        self.status_bar = ttk.Label(self.win, text="", anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky="ew")
        self.status_bar.config(text=str(self.cam.get_device_info()))

        self.sum_start_index = 0
        self.sum_end_index = self.cam.get_detector_size()[1]

        self.live = True
        self.average_mode = False
        self.show_sum_plot = False

    def update_plot(self):
        """
        Update the plot.

        Returns
        -------
        None

        Notes
        -----
        If `live` attribute is True, it waits for a frame from the camera and updates the first plot
        with the frame data. If `average_mode` attribute is True, it calculates the average frame from
        multiple frames and uses it for updating the plot.

        If `show_sum_plot` attribute is True, it calculates the sums of pixels along the vertical axis
        within the range specified by `sum_start_index` and `sum_end_index` attributes. It then updates
        the second plot with the sum data, adding horizontal lines indicating the start and end indices
        of the summed region.

        Raises
        ------
        TypeError
            If a TypeError occurs during the update process, it is caught, and an error message is printed.

        """
        if self.live:
            self.cam.wait_for_frame()
            frame = self.cam.read_oldest_image()

            try:
                # Attempt to update the first plot
                self.img.set_data(frame)

                if self.average_mode:
                    frame_sum = frame
                    for i in range(self.avg_num - 1):
                        print(f'{i + 1} frames on {self.avg_num}')
                        self.cam.wait_for_frame()
                        frame_sum += self.cam.read_oldest_image()
                    frame = frame_sum / self.avg_num

                if self.ax.images[-1].colorbar is not None:
                    self.ax.images[-1].colorbar.remove()
                    self.ax.images[-1].colorbar = None

                # Update the second plot if the checkbox is checked
                if self.show_sum_plot:
                    sums = np.sum(frame[self.sum_start_index:self.sum_end_index, :], axis=0)
                    self.sum_line.set_data(np.arange(len(sums)), sums)
                    self.sum_ax.relim()
                    self.sum_ax.autoscale_view()

                    for line in self.ax.lines:
                        if line.get_color() == 'red':
                            line.remove()

                    self.ax.axhline(self.sum_start_index, color='red')
                    self.ax.axhline(self.sum_end_index, color='red')

                # self.fig.colorbar(self.img)
                self.canvas.draw()
                self.sum_canvas.draw()

            except TypeError:
                # If a TypeError is raised, print an error message and continue with the loop
                print("Caught TypeError: skipping frame")
                pass

            # Schedule the function to run again after a delay
            self.win.update_idletasks()
            self.win.after(200, self.update_plot)

    def update_temperature(self):
        """
        Update the temperature display based on the current camera temperature.

        Returns
        -------
        None
        """
        self.camera_temp = self.cam.get_temperature()

        # Update the label with the current temperature
        self.temp_value_label.config(text="{:.2f} °C".format(self.camera_temp))

        # Schedule the next update in 5 seconds
        self.win.after(5000, self.update_temperature)

    def save_image(self):
        """
        Save the current image displayed in the plot.

        Returns
        -------
        None
        """
        filename = filedialog.asksaveasfilename(defaultextension='.bmp')
        if filename:
            image_array = self.img.get_array()
            plt.imsave(filename, image_array)
            print('Image saved')

    def start(self):
        """
        Start the acquisition process.

        Returns
        -------
        None
        """
        self.live = True
        self.cam.start_acquisition()
        self.update_plot()
        print('Acquisition started')

    def stop(self):
        """
        Stop the acquisition process.

        Returns
        -------
        None
        """
        self.live = False
        self.cam.stop_acquisition()
        print('Acquisition stopped')

    def toggle_sum_plot(self):
        """
        Toggle the display of the sum plot.

        Returns
        -------
        None
        """
        self.show_sum_plot = not self.show_sum_plot
        if not self.show_sum_plot:
            self.sum_ax.clear()
            self.canvas.draw()
        else:
            self.sum_line, = self.sum_ax.plot([], [])
            self.canvas.draw()

    def toggle_avg_mode(self):
        """
        Toggle the average mode for frame acquisition.

        Returns
        -------
        None

        """
        self.set_avg_num()
        self.average_mode = not self.average_mode
        print('Acquisition mode :', self.cam.get_acquisition_mode())

    def set_avg_num(self):
        """
        Set the number of frames used for averaging.

        Returns
        -------
        None

        """
        new_avg_num = int(self.avg_var.get())
        self.avg_num = new_avg_num
        print('Average number of frames changed')

    def set_exposure_time(self):
        """
        Set the exposure time for the camera.

        Returns
        -------
        None

        """
        self.cam.stop_acquisition()
        self.cam.set_exposure(self.exposure_entry.get())
        self.cam.start_acquisition()
        print(f'Exposure time updated, new value = {self.cam.get_exposure()}')

    def set_gain(self):
        """
        Set the gain for the camera.

        Returns
        -------
        None

        """
        self.cam.stop_acquisition()
        print(f'{self.cam.get_EMCCD_gain()}')
        self.cam.set_EMCCD_gain(self.gain_entry.get())
        self.cam.start_acquisition()
        print(f'EMCCD gain updated, new value = {self.cam.get_EMCCD_gain()}')

    def set_roi(self):
        """
        Set the Region of Interest (ROI) for the camera.

        Returns
        -------
        None

        """
        x_start = int(self.roi_x_start_var.get())
        x_end = int(self.roi_x_end_var.get())
        y_start = int(self.roi_y_start_var.get())
        y_end = int(self.roi_y_end_var.get())
        self.cam.set_roi(x_start, x_end, y_start, y_end, hbin=1, vbin=1)

        self.ax.set_xlim(x_start, x_end)
        self.ax.set_ylim(y_start, y_end)

        self.next_frame()
        print('ROI updated')

    def reset_roi(self):
        """
        Reset the Region of Interest (ROI) to the default values.

        Returns
        -------
        None

        """

        x_lim = [self.roi_x_start_var.set(str(0)), self.roi_x_end_var.set(str(self.cam.get_detector_size()[0]))]
        y_lim = [self.roi_y_start_var.set(str(0)), self.roi_y_end_var.set(str(self.cam.get_detector_size()[1]))]

        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)

        self.set_roi()
        self.next_frame()
        print('ROI set to default')

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

        self.sum_start_var.set(y1)
        self.sum_end_var.set(y2)

        self.roi_x_start_var.set(x1)
        self.roi_x_end_var.set(x2)
        self.roi_y_start_var.set(y1)
        self.roi_y_end_var.set(y2)

        self.set_sum_roi()
        self.set_roi()

    def set_sum_roi(self):
        """
        Set the start and end indices for the sum region of interest (ROI).

        Returns
        -------
        None

        """
        self.sum_start_index = int(self.sum_start_var.get())
        self.sum_end_index = int(self.sum_end_var.get())
        print('Sum ROI updated')

    def reset_sum_roi(self):
        """
        Reset the start and end indices for the sum region of interest (ROI) to default values.

        Returns
        -------
        None

        """
        self.sum_start_var.set(str(0))
        self.sum_end_var.set(self.cam.get_detector_size()[1])
        self.set_sum_roi()
        print('Sum ROI set to default')

    def next_frame(self):
        """
        Update the plot with the next frame from the camera.

        Returns
        -------
        None

        """
        self.cam.wait_for_frame()
        frame = self.cam.read_oldest_image()
        self.img.set_data(frame)
        self.canvas.draw()

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
