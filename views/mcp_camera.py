import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib


from drivers.vimba_driver import *


class Mcp(object):

    def __init__(self, parent):
        self.parent = parent
        self.win = tk.Toplevel()
        matplotlib.use("TkAgg")

        self.win.title("D-Lab Controller - MCP camera")
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)
        self.rect_id = 0
        self.ready = 0

        self.nr_averages = 1

        self.cameras = None
        vcmd = (self.win.register(self.parent.callback))

        frm_bot = tk.Frame(self.win)
        frm_plt = tk.Frame(self.win)
        frm_mid = tk.Frame(self.win)

        frm_cam = tk.Frame(self.win)
        frm_cam_but = tk.Frame(frm_cam)

        self.but_cam_init = tk.Button(frm_cam_but, text='Initialize', command=self.init_cam)
        self.but_cam_init.grid(row=0, column=0, padx=5, pady=5, ipadx=5, ipady=5)
        self.but_cam_init.config(fg='black')

        self.but_cam_img = tk.Button(frm_cam_but, text='Get image', command=self.display_image)
        self.but_cam_img.grid(row=0, column=1, padx=5, pady=5, ipadx=5, ipady=5)

        but_exit = tk.Button(frm_bot, text='EXIT', command=self.on_close)

        lbl_avg = tk.Label(frm_cam_but, text='Averages:')
        self.strvar_avg = tk.StringVar(self.win, '1')
        self.ent_avg = tk.Entry(frm_cam_but, width=11, validate='all', validatecommand=(vcmd, '%d', '%P', '%S'),
                                textvariable=self.strvar_avg)
        lbl_avg.grid(row=0, column=3)
        self.ent_avg.grid(row=0, column=4)

        # setting them up
        frm_plt.grid(row=1, column=0, sticky='nsew')
        frm_mid.grid(row=2, column=0, sticky='nsew')
        frm_bot.grid(row=3, column=0)

        frm_cam.grid(row=0, column=0, sticky='nsew')
        frm_cam_but.grid(row=1, column=0, sticky='nsew')

        self.figrMCP = Figure(figsize=(7, 7), dpi=100)
        self.axMCP = self.figrMCP.add_subplot(211)
        self.axHarmonics = self.figrMCP.add_subplot(212)
        self.axMCP.set_xlim(0, 1600)
        self.axMCP.set_ylim(0, 1000)
        self.axHarmonics.set_xlim(0, 1600)
        self.figrMCP.tight_layout()
        self.figrMCP.canvas.draw()
        self.imgMCP = FigureCanvasTkAgg(self.figrMCP, frm_cam)
        self.tk_widget_figrMCP = self.imgMCP.get_tk_widget()
        self.tk_widget_figrMCP.grid(row=0, column=0, sticky='nsew')
        self.imgMCP.draw()

        but_exit.grid(row=1, column=0, padx=5, pady=5)

    def init_cam(self):
        print("")
        print("Initializing......")
        print("")
        with Vimba.get_instance() as vimba:
            cams = vimba.get_all_cameras()
            with cams[0] as cam:
                self.cameras = cams
                self.ready = 1
                self.but_cam_init.config(fg='green')

    def get_image(self):
        """
        Takes an image from the camera.

        This method takes an image from the camera using the specified number of averages,
        and returns the captured image.

        Returns
        -------
        numpy.ndarray
            The captured image.

        """
        if self.ready:
            with Vimba.get_instance() as vimba:
                cams = vimba.get_all_cameras()
                image = np.zeros([1000, 1600])
                nr = int(self.strvar_avg.get())
                with cams[0] as cam:
                    for frame in cam.get_frame_generator(limit=nr):
                        frame = cam.get_frame()
                        frame.convert_pixel_format(PixelFormat.Mono8)
                        img = frame.as_opencv_image()
                        img = np.squeeze(frame.as_opencv_image())
                        numpy_image = img
                        image = image + numpy_image
                    image = image / nr
            return image
        else:
            print("mimimimimi")

    def display_image(self):
        numpy_image = self.get_image()
        self.plot_MCP(numpy_image)

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
        self.axMCP.imshow(mcpimage, vmin=0, vmax=2, extent=[0, 1600, 0, 1000])
        self.axMCP.set_aspect('equal')

        self.axMCP.set_xlabel("X (px)")
        self.axMCP.set_ylabel("Y (px)")
        self.axMCP.set_xlim(0, 1600)
        self.axMCP.set_ylim(0, 1000)

        self.axHarmonics.clear()
        self.axHarmonics.plot(np.arange(1600), np.sum(mcpimage, 0))
        self.axHarmonics.set_xlabel("X (px)")
        self.axHarmonics.set_ylabel("Counts (arb.u.)")

        self.axHarmonics.set_xlim(0, 1600)

        self.figrMCP.tight_layout()
        self.imgMCP.draw()

    def on_close(self):
        self.win.destroy()
        self.parent.mcp_win = None
