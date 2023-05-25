from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import matplotlib
import numpy as np
from scipy.optimize import curve_fit

from tkinter import ttk
class Calibrator(object):
    def __init__(self):
        #self.parent = parent

        matplotlib.use("TkAgg")
        self.win = tk.Toplevel()
        self.win.title("Waveplate Calibrator")
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)
        self.win.geometry("1000x500")

        frm_options = tk.Frame(self.win)
        frm_plot = tk.Frame(self.win)
        frm_bot = tk.Frame(self.win)

        frm_options.grid(row=0, column=0)
        frm_plot.grid(row=0, column=1)
        frm_bot.grid(row=1, column=1)

        #vcmd = (self.win.register(self.parent.callback))

        sizefactor = 1

        self.figr = Figure(figsize=(5 * sizefactor, 3 * sizefactor), dpi=100)
        self.ax1r = self.figr.add_subplot(211)
        self.ax2r = self.figr.add_subplot(212)
        self.ax1r.set_xlim(0, 180)
        self.ax1r.grid()
        self.ax2r.set_xlim(0, 180)
        self.ax2r.grid()
        self.figr.tight_layout()
        self.figr.canvas.draw()
        self.img1r = FigureCanvasTkAgg(self.figr, frm_plot)
        self.tk_widget_figr = self.img1r.get_tk_widget()
        self.tk_widget_figr.grid(row=0, column=0, sticky='nsew')
        self.img1r.draw()
        self.ax1r_blit = self.figr.canvas.copy_from_bbox(self.ax1r.bbox)
        self.ax2r_blit = self.figr.canvas.copy_from_bbox(self.ax2r.bbox)

        self.calibration_angle_red, self.calibration_power_red = np.loadtxt('ressources/calibration/wpr_red_calib.txt', delimiter='\t', skiprows=0,
                                                                            unpack=True)
        self.calibration_angle_green, self.calibration_power_green = np.loadtxt('ressources/calibration/wpg_green_calib.txt', delimiter='\t', skiprows=0,
                                                                            unpack=True)

        self.ax1r.plot(self.calibration_angle_red, self.calibration_power_red, 'ro')
        self.ax2r.plot(self.calibration_angle_green, self.calibration_power_green, 'go')
        amplitude, phase = self.cos_fit(self.calibration_angle_red, self.calibration_power_red)
        self.max_red = 2*amplitude
        self.phase_red = phase

        self.ax1r.plot(np.linspace(0, 180, 100),
                       self.cos_func(np.linspace(0, 180, 100), amplitude, phase), 'r')
        amplitude, phase = self.cos_fit(self.calibration_angle_green, self.calibration_power_green)
        self.max_green = 2 * amplitude
        self.phase_green = phase
        self.ax2r.plot(np.linspace(0, 180, 100), self.cos_func(np.linspace(0, 180, 100), amplitude, phase),
                       'g')
        self.img1r.draw()

        self.but_calibration_power_red = tk.Button(frm_options, text='Open red Calibration file',
                                                   command=self.open_calibration_power_red)
        self.but_calibration_power_green = tk.Button(frm_options, text='Open green Calibration file',
                                                     command=self.open_calibration_power_green)

        self.but_calibration_power_red.grid(row=0, column=0, padx=5, pady=5)
        self.but_calibration_power_green.grid(row=0, column=2, padx=5, pady=5)

        lbl_red_max = tk.Label(frm_options, text='Max Red (W):')
        lbl_green_max = tk.Label(frm_options, text='Max Green (mW):')
        self.strvar_red_max = tk.StringVar(self.win, np.round(self.max_red, 2))
        self.strvar_green_max = tk.StringVar(self.win, np.round(self.max_green, 2))

        lbl_red_phase = tk.Label(frm_options, text='Red offset phase (deg):')
        lbl_green_phase = tk.Label(frm_options, text='Green offset phase (deg):')
        self.strvar_red_phase = tk.StringVar(self.win, np.round(self.phase_red, 2))
        self.strvar_green_phase = tk.StringVar(self.win, np.round(self.phase_green, 2))

        self.ent_red_max = tk.Entry(
            frm_options, width=8, validate='all',
            #validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_red_max)
        self.ent_green_max = tk.Entry(
            frm_options, width=8, validate='all',
            #validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_green_max)

        self.ent_red_phase = tk.Entry(
            frm_options, width=8, validate='all',
            # validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_red_phase)
        self.ent_green_phase = tk.Entry(
            frm_options, width=8, validate='all',
            # validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_green_phase)

        lbl_red_max.grid(row=1, column=0, padx=5, pady=5)
        lbl_green_max.grid(row=1, column=2, padx=5, pady=5)
        self.ent_red_max.grid(row=1, column=1, padx=5, pady=5)
        self.ent_green_max.grid(row=1, column=3, padx=5, pady=5)

        lbl_red_phase.grid(row=2, column=0, padx=5, pady=5)
        lbl_green_phase.grid(row=2, column=2, padx=5, pady=5)
        self.ent_red_phase.grid(row=2, column=1, padx=5, pady=5)
        self.ent_green_phase.grid(row=2, column=3, padx=5, pady=5)

        but_update = tk.Button(frm_options, text='Update calibration', command=self.update_max_val)
        but_update.grid(row=3, column=2, padx=5, pady=10)

        but_exit = tk.Button(frm_bot, text='EXIT', command=self.on_close)
        but_exit.grid(row=0, column=0, padx=5, pady=5, ipadx=5, ipady=5)

        #self.win.mainloop()


    def open_calibration_power_red(self):
        """
        Open the file where the power calibration is

        Returns
        -------
        None
        """
        try:
            filepath = tk.filedialog.askopenfilename()
            print(filepath)
            print(filepath[-4:])
            if not filepath:
                return
            if filepath[-4:] == '.txt':
                print("made it in loop")
                self.calibration_angle_red, self.calibration_power_red = np.loadtxt(filepath, delimiter='\t',skiprows=0, unpack=True)
                self.but_calibration_power_red.config(fg='green')
                self.ax1r.plot(self.calibration_angle_red, self.calibration_power_red, 'ro')
                amplitude, phase = self.cos_fit(self.calibration_angle_red, self.calibration_power_red)
                self.ax1r.plot(np.linspace(0, 180, 100),
                               self.cos_func(np.linspace(0, 180, 100), amplitude, phase), 'r')
                self.img1r.draw()
                self.max_red = 2 * amplitude
                self.phase_red = phase
        except:
            print("Impossible to read the calibration file")
            self.but_calibration_power_red.config(fg='red')

    def open_calibration_power_green(self):
        """
        Open the file where the power calibration is

        Returns
        -------
        None
        """
        try:
            filepath = tk.filedialog.askopenfilename()
            if not filepath:
                return
            if filepath[-4:] == '.txt':
                self.calibration_angle_green, self.calibration_power_green = np.loadtxt(filepath, delimiter='\t',skiprows=0, unpack=True)
                self.but_calibration_power_green.config(fg='green')
                self.ax2r.plot(self.calibration_angle_green, self.calibration_power_green, 'go')
                amplitude, phase = self.cos_fit(self.calibration_angle_green,self.calibration_power_green)
                self.ax2r.plot(np.linspace(0,180,100), self.cos_func(np.linspace(0,180,100),amplitude, phase), 'g')
                self.img1r.draw()
                self.max_green = 2 * amplitude
                self.phase_green = phase
        except:
            print("Impossible to read the calibration file")
            self.but_calibration_power_green.config(fg='red')

    def cos_func(self,x, amplitude, phase):
        return amplitude * np.cos(2 * np.pi / 90 * x - 2*np.pi/90*phase) + amplitude
    def cos_fit(self,x,y):
        initial_guess = (np.max(y)/2, 0)
        popt, pcov = curve_fit(self.cos_func, x, y, p0=initial_guess)
        amplitude, phase = popt
        print(amplitude, phase)
        return amplitude, phase

    def update_max_val(self):
        self.max_red = float(self.ent_red_max.get())
        self.max_green = float(self.ent_green_max.get())
        x = np.linspace(0,180,100)
        red = self.cos_func(x,self.max_red/2 ,self.phase_red)
        green = self.cos_func(x,self.max_green/2, self.phase_green)

        self.ax1r.clear()
        self.ax1r.plot(x, red, 'r')
        self.ax1r.set_xlim(0, 180)
        self.ax1r.grid()
        self.ax2r.clear()
        self.ax2r.plot(x, green, 'g')
        self.ax2r.grid()
        self.ax2r.set_xlim(0, 180)
        self.img1r.draw()

    def wpr_to_power(self, angle):
        power = self.cos_func(angle,self.max_red/2, 1/90, self.phase_red)
        return power

    def wpg_to_power(self, angle):
        power = self.cos_func(angle,self.max_green/2, 1/90, self.phase_green)
        return power

    def power_to_wpr(self, power):
        angle = (np.arccos(1-power/self.max_red)-self.phase_red+np.pi)/(2*np.pi/90)
        return angle

    def power_to_wpg(self, power):
        angle = (np.arccos(1-power/self.max_green)-self.phase_green+np.pi)/(2*np.pi/90)
        return angle

    def on_close(self):
        self.win.destroy()
        print('Calibrator closed')


#cal = Calibrator()
