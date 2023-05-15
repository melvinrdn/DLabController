import tkinter as tk

import numpy as np
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

from PIL import Image, ImageTk

import time

from drivers.vimba_driver import *



class Mcp(object):

    def __init__(self, parent):
        self.parent = parent
        self.win = tk.Toplevel()
        matplotlib.use("TkAgg")

        self.win.title("MCP camera - harmonics")
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)
        self.rect_id = 0
        self.ready = 0

        self.nr_averages = 1

        self.cameras = None
        vcmd = (self.win.register(self.parent.callback))

        # creating frames
        frm_bot = tk.Frame(self.win)
        frm_plt = tk.Frame(self.win)
        frm_mid = tk.Frame(self.win)

        frm_cam = tk.Frame(self.win)
        frm_cam_but = tk.Frame(frm_cam)
        frm_cam_but_set = tk.Frame(frm_cam_but)

        frm_ratio = tk.LabelFrame(frm_mid, text='Harmonics')
        # frm_pid = tk.LabelFrame(frm_mid, text='test pid')

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

        self.img_canvas = tk.Canvas(frm_cam, height=1000 / 2, width=1600 / 2)
        self.img_canvas.grid(row=0, sticky='nsew')
        self.img_canvas.configure(bg='grey')
        self.image = self.img_canvas.create_image(0, 0, anchor="nw")

        self.profile = Figure(figsize=(8, 1.5), dpi=100)
        self.a = self.profile.add_subplot(111)
        self.trace = self.a.plot([], [])
        self.a.set_xlim(0, 1600)
        # self.a.set_ylim(0, 1)
        self.profile.canvas.draw()
        self.plot1 = FigureCanvasTkAgg(self.profile, frm_ratio)
        self.tk_widget_figp = self.plot1.get_tk_widget()
        self.tk_widget_figp.grid(row=1, column=0, sticky='nsew')
        self.plot1.draw()
        self.a_blit = self.profile.canvas.copy_from_bbox(self.a.bbox)

        # self.profile_canvas = tk.Canvas(frm_cam, height=1000/2, width=1600/2)

        frm_ratio.grid(row=0, column=0, padx=5)
        # frm_pid.grid(row=0, column=1, padx=5)
        frm_ratio.config(width=int(1600 / 2), height=300)
        frm_ratio.grid_propagate(False)

        # creating buttons n labels
        but_exit.grid(row=1, column=0, padx=5, pady=5)
        # CONTINUE HERE

    def init_cam(self):
        print("")
        print("Initializing......")
        print("")
        with Vimba.get_instance() as vimba:
            cams = vimba.get_all_cameras()

            with cams[0] as cam:
                # Aquire single frame synchronously
                self.cameras = cams
                self.ready = 1
                self.but_cam_init.config(fg='green')

    #  def cam_img(self):
    #       self.render_thread = threading.Thread(target=self.init_cam)
    #       self.render_thread.daemon = True
    #       self.render_thread.start()
    #       #self.plot_phase()

    def get_image(self):
        if self.ready:
            with Vimba.get_instance() as vimba:
                cams = vimba.get_all_cameras()
                image = np.zeros([1000, 1600])

                start_time = time.time()

                nr = int(self.strvar_avg.get())

                with cams[0] as cam:
                    for frame in cam.get_frame_generator(limit=nr):
                        frame = cam.get_frame()
                        frame.convert_pixel_format(PixelFormat.Mono8)
                        img = frame.as_opencv_image()
                        img = np.squeeze(frame.as_opencv_image())
                        numpy_image = img / 255  # this should give the image between 0 and 1, it is for later to apply the colormap
                        image = image + numpy_image
                    image = image / nr
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("Elapsed time: ", elapsed_time)
                return image
        else:
            print("somewith with the camera is wrong :(")

    # print(np.max(numpy_image))

    def display_image(self):
        numpy_image = self.get_image()
        picture = Image.fromarray(np.uint8(cm.jet(numpy_image) * 255))
        picture = picture.resize((int(1600 / 2), int(1000 / 2)), resample=0)
        picture = ImageTk.PhotoImage(picture)

        self.img_canvas.itemconfig(self.image, image=picture)
        self.img_canvas.image = picture  # keep a reference!
        self.a.clear()
        self.trace = self.a.plot(np.arange(0, 1600), np.sum(numpy_image, 0))
        self.plot1.draw()

    def on_close(self):
        # plt.close(self.figr)
        # plt.close(self.figp)

        self.win.destroy()
        self.parent.mcp_win = None
