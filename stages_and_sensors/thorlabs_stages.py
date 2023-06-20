import tkinter as tk
import matplotlib
import threading
import numpy as np
from drivers.thorlabs_apt_driver import core as apt
from stages_and_sensors import waveplate_calibrator as cal


class ThorlabsStages(object):
    def __init__(self):

        self.WPG = None
        self.WPR = None
        self.Delay = None

        matplotlib.use("TkAgg")
        self.win = tk.Toplevel()
        self.win.title("D-Lab Controller - Thorlabs stages control")
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)
        self.win.geometry("660x350")

        frm_main = tk.Frame(self.win)
        frm_bot = tk.Frame(self.win)
        frm_stage = tk.LabelFrame(frm_main, text='Thorlabs stages')
        frm_wp_power_cal = tk.LabelFrame(frm_main, text='Power calibration')

        frm_main.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        frm_stage.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        frm_wp_power_cal.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        frm_bot.grid(row=2, column=0, padx=10, pady=10, sticky='nsew')

        lbl_Stage = tk.Label(frm_stage, text='Stage')
        lbl_Nr = tk.Label(frm_stage, text='#')
        lbl_is = tk.Label(frm_stage, text='is')
        lbl_should = tk.Label(frm_stage, text='should')

        # Red
        lbl_WPR = tk.Label(frm_stage, text='WP red:')
        self.strvar_WPR_is = tk.StringVar(self.win, '')
        self.ent_WPR_is = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_WPR_is)
        self.strvar_WPR_should = tk.StringVar(self.win, '')
        self.ent_WPR_should = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_WPR_should)
        self.strvar_WPR_Nr = tk.StringVar(self.win, '83837724')
        self.ent_WPR_Nr = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_WPR_Nr)

        self.but_WPR_Ini = tk.Button(frm_stage, text='Initialize', command=self.init_WPR)
        self.but_WPR_Home = tk.Button(frm_stage, text='Home', command=self.home_WPR)
        self.but_WPR_Read = tk.Button(frm_stage, text='Read', command=self.read_WPR)
        self.but_WPR_Move = tk.Button(frm_stage, text='Move', command=self.move_WPR)

        self.var_wprpower = tk.IntVar()
        self.cb_wprpower = tk.Checkbutton(frm_stage, text='Power', variable=self.var_wprpower, onvalue=1, offvalue=0,
                                          command=None)

        # Green
        lbl_WPG = tk.Label(frm_stage, text='WP green:')
        self.strvar_WPG_is = tk.StringVar(self.win, '')
        self.ent_WPG_is = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_WPG_is)
        self.strvar_WPG_should = tk.StringVar(self.win, '')
        self.ent_WPG_should = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_WPG_should)
        self.strvar_WPG_Nr = tk.StringVar(self.win, '83837725')
        self.ent_WPG_Nr = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_WPG_Nr)
        self.but_WPG_Ini = tk.Button(frm_stage, text='Initialize', command=self.init_WPG)
        self.but_WPG_Home = tk.Button(frm_stage, text='Home', command=self.home_WPG)
        self.but_WPG_Read = tk.Button(frm_stage, text='Read', command=self.read_WPG)
        self.but_WPG_Move = tk.Button(frm_stage, text='Move', command=self.move_WPG)

        self.var_wpgpower = tk.IntVar()
        self.cb_wpgpower = tk.Checkbutton(frm_stage, text='Power', variable=self.var_wpgpower, onvalue=1, offvalue=0,
                                          command=None)

        # Delay
        lbl_Delay = tk.Label(frm_stage, text='Delay:')
        self.strvar_Delay_is = tk.StringVar(self.win, '')
        self.ent_Delay_is = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_Delay_is)
        self.strvar_Delay_should = tk.StringVar(self.win, '')
        self.ent_Delay_should = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_Delay_should)
        self.strvar_Delay_Nr = tk.StringVar(self.win, '83837719')
        self.ent_Delay_Nr = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_Delay_Nr)
        self.strvar_Delay_from = tk.StringVar(self.win, '6.40')
        self.ent_Delay_from = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_Delay_from)
        self.strvar_Delay_to = tk.StringVar(self.win, '6.45')
        self.ent_Delay_to = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_Delay_to)
        self.strvar_Delay_steps = tk.StringVar(self.win, '10')
        self.ent_Delay_steps = tk.Entry(
            frm_stage, width=10, validate='all',
            textvariable=self.strvar_Delay_steps)
        self.but_Delay_Ini = tk.Button(frm_stage, text='Initialize', command=self.init_Delay)
        self.but_Delay_Home = tk.Button(frm_stage, text='Home', command=self.home_Delay)
        self.but_Delay_Read = tk.Button(frm_stage, text='Read', command=self.read_Delay)
        self.but_Delay_Move = tk.Button(frm_stage, text='Move', command=self.move_Delay)

        # power wp calibration
        lbl_pharos_att = tk.Label(frm_wp_power_cal, text='Pharos Att:')
        self.strvar_pharos_att = tk.StringVar(self.win, '100')
        self.ent_pharos_att = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_pharos_att)

        lbl_pharos_pp = tk.Label(frm_wp_power_cal, text='Pharos PP:')
        self.strvar_pharos_pp = tk.StringVar(self.win, '1')
        self.ent_pharos_pp = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',

            textvariable=self.strvar_pharos_pp)

        self.strvar_red_power = tk.StringVar(self.win, '')
        self.ent_red_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_red_power)

        self.but_calibrator_open = tk.Button(frm_wp_power_cal, text='Open calibrator', command=self.enable_calibrator)

        lbl_red_power = tk.Label(frm_wp_power_cal, text='Red max power (W):')
        self.strvar_red_power = tk.StringVar(self.win, '4.5')
        self.ent_red_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_red_power)

        lbl_red_phase = tk.Label(frm_wp_power_cal, text='Red offset phase (deg):')
        self.strvar_red_phase = tk.StringVar(self.win, '-27.76')
        self.ent_red_phase = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_red_phase)

        lbl_red_current_power = tk.Label(frm_wp_power_cal, text='Red current power (W):')
        self.strvar_red_current_power = tk.StringVar(self.win, '')
        self.ent_red_current_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_red_current_power)

        lbl_green_power = tk.Label(frm_wp_power_cal, text='Green max power (mW):')
        self.strvar_green_power = tk.StringVar(self.win, '345')
        self.ent_green_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_green_power)

        lbl_green_phase = tk.Label(frm_wp_power_cal, text='Green offset phase (deg):')
        self.strvar_green_phase = tk.StringVar(self.win, '44.02')
        self.ent_green_phase = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_green_phase)

        lbl_green_current_power = tk.Label(frm_wp_power_cal, text='Green current power (mW):')
        self.strvar_green_current_power = tk.StringVar(self.win, '')
        self.ent_green_current_power = tk.Entry(
            frm_wp_power_cal, width=8, validate='all',
            textvariable=self.strvar_green_current_power)

        # setting up frm_stage
        lbl_Stage.grid(row=0, column=1, pady=2, sticky='nsew')
        lbl_Nr.grid(row=0, column=2, pady=2, sticky='nsew')
        lbl_is.grid(row=0, column=3, pady=2, sticky='nsew')
        lbl_should.grid(row=0, column=4, pady=2, sticky='nsew')

        lbl_WPR.grid(row=1, column=1, pady=2, sticky='nsew')
        lbl_WPG.grid(row=2, column=1, pady=2, sticky='nsew')
        lbl_Delay.grid(row=3, column=1, pady=2, sticky='nsew')

        self.ent_WPR_Nr.grid(row=1, column=2, pady=2, sticky='nsew')
        self.ent_WPG_Nr.grid(row=2, column=2, pady=2, sticky='nsew')
        self.ent_Delay_Nr.grid(row=3, column=2, pady=2, sticky='nsew')

        self.ent_WPR_is.grid(row=1, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_WPG_is.grid(row=2, column=3, padx=2, pady=2, sticky='nsew')
        self.ent_Delay_is.grid(row=3, column=3, padx=2, pady=2, sticky='nsew')

        self.ent_WPR_should.grid(row=1, column=4, padx=2, pady=2, sticky='nsew')
        self.ent_WPG_should.grid(row=2, column=4, padx=2, pady=2, sticky='nsew')
        self.ent_Delay_should.grid(row=3, column=4, padx=2, pady=2, sticky='nsew')

        self.but_WPR_Ini.grid(row=1, column=5, padx=2, pady=2, sticky='nsew')
        self.but_WPR_Home.grid(row=1, column=6, padx=2, pady=2, sticky='nsew')
        self.but_WPR_Read.grid(row=1, column=7, padx=2, pady=2, sticky='nsew')
        self.but_WPR_Move.grid(row=1, column=8, padx=2, pady=2, sticky='nsew')

        self.but_WPG_Ini.grid(row=2, column=5, padx=2, pady=2, sticky='nsew')
        self.but_WPG_Home.grid(row=2, column=6, padx=2, pady=2, sticky='nsew')
        self.but_WPG_Read.grid(row=2, column=7, padx=2, pady=2, sticky='nsew')
        self.but_WPG_Move.grid(row=2, column=8, padx=2, pady=2, sticky='nsew')

        self.but_Delay_Ini.grid(row=3, column=5, padx=2, pady=2, sticky='nsew')
        self.but_Delay_Home.grid(row=3, column=6, padx=2, pady=2, sticky='nsew')
        self.but_Delay_Read.grid(row=3, column=7, padx=2, pady=2, sticky='nsew')
        self.but_Delay_Move.grid(row=3, column=8, padx=2, pady=2, sticky='nsew')

        self.cb_wprpower.grid(row=1, column=9, padx=2, pady=2, sticky='nsew')
        self.cb_wpgpower.grid(row=2, column=9, padx=2, pady=2, sticky='nsew')

        # setting up frm_wp_power_calibration
        self.but_calibrator_open.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')

        lbl_pharos_att.grid(row=0, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_pharos_att.grid(row=0, column=2, padx=2, pady=2, sticky='nsew')
        lbl_pharos_pp.grid(row=1, column=1, padx=2, pady=2, sticky='nsew')
        self.ent_pharos_pp.grid(row=1, column=2, padx=2, pady=2, sticky='nsew')

        lbl_red_power.grid(row=0, column=5, padx=2, pady=2, sticky='nsew')
        self.ent_red_power.grid(row=0, column=6, padx=2, pady=2, sticky='nsew')
        lbl_red_phase.grid(row=1, column=5, padx=2, pady=2, sticky='nsew')
        self.ent_red_phase.grid(row=1, column=6, padx=2, pady=2, sticky='nsew')
        lbl_red_current_power.grid(row=2, column=5, padx=2, pady=2, sticky='nsew')
        self.ent_red_current_power.grid(row=2, column=6, padx=2, pady=2, sticky='nsew')

        lbl_green_power.grid(row=0, column=7, padx=2, pady=2, sticky='nsew')
        self.ent_green_power.grid(row=0, column=8, padx=2, pady=2, sticky='nsew')
        lbl_green_phase.grid(row=1, column=7, padx=2, pady=2, sticky='nsew')
        self.ent_green_phase.grid(row=1, column=8, padx=2, pady=2, sticky='nsew')
        lbl_green_current_power.grid(row=2, column=7, padx=2, pady=2, sticky='nsew')
        self.ent_green_current_power.grid(row=2, column=8, padx=2, pady=2, sticky='nsew')

        but_exit = tk.Button(frm_bot, text='Exit', command=self.on_close)
        but_exit.grid(row=0, column=0, padx=2, pady=2, sticky='nsew')

    def init_WPR(self):
        """
        Initializes the red waveplate motor object.

        Raises
        ------
        Exception
            If the motor fails to initialize.

        Returns
        -------
        None
        """
        try:
            self.WPR = apt.Motor(int(self.ent_WPR_Nr.get()))
            self.but_WPR_Ini.config(fg='green')
            print("WPR connected")
        except:
            self.but_WPR_Ini.config(fg='red')
            print("Not able to initalize WPR")

    def home_WPR(self):
        """
        Homes the red waveplate motor object.

        Raises
        ------
        Exception
            If the motor fails to home.

        Returns
        -------
        None
        """
        try:
            self.WPR.move_home(blocking=True)
            self.but_WPR_Home.config(fg='green')
            print("WPR homed!")
            self.read_WPR()
        except:
            self.but_WPR_Home.config(fg='red')
            print("Not able to home WPR")

    def read_WPR(self):
        """
        Reads the current position of the red waveplate motor.

        Raises
        ------
        Exception
            If the position cannot be read.

        Returns
        -------
        None
        """
        try:
            pos = self.WPR.position
            self.strvar_WPR_is.set(pos)
            self.strvar_red_current_power.set(
                np.round(self.angle_to_power(pos, float(self.ent_red_power.get()), float(self.ent_red_phase.get())), 3))
        except:
            print("Impossible to read WPR position")

    def move_WPR(self):
        """
        Moves the red waveplate motor to the desired position.

        Raises
        ------
        Exception
            If the motor fails to move to the desired position.

        Returns
        -------
        None
        """
        try:
            if self.var_wprpower.get() == 1:
                power = float(self.strvar_WPR_should.get())
                if power > float(self.ent_red_power.get()):
                    power = float(self.ent_red_power.get())
                    print("Value above maximum! Desired power set to maximum instead")
                pos = self.power_to_angle(power, float(self.ent_red_power.get()), float(self.ent_red_phase.get())) + 90
            else:
                pos = float(self.strvar_WPR_should.get())

            print("WPR is moving..")
            self.WPR.move_to(pos, True)
            print(f"WPR moved to {str(self.WPR.position)}")
            self.read_WPR()
        except:
            print("Impossible to move WPR :(")

    def init_WPG(self):
        """
        Initializes the green waveplate motor object.

        Raises
        ------
        Exception
            If the motor fails to initialize.

        Returns
        -------
        None
        """
        try:
            self.WPG = apt.Motor(int(self.ent_WPG_Nr.get()))
            self.but_WPG_Ini.config(fg='green')
            print("WPG connected")
        except:
            self.but_WPG_Ini.config(fg='red')
            print("Not able to initalize WPG")

    def home_WPG(self):
        """
        Homes the green waveplate motor object.
        Raises
        ------
        Exception
            If the motor fails to home.

        Returns
        -------
        None
        """
        try:
            self.WPG.move_home(blocking=True)
            self.but_WPG_Home.config(fg='green')
            print("WPG homed!")
            self.read_WPG()
        except:
            self.but_WPG_Home.config(fg='red')
            print("Not able to home WPG")

    def read_WPG(self):
        """
        Reads the current position of the green waveplate motor.

        Raises
        ------
        Exception
            If the position cannot be read.

        Returns
        -------
        None
        """
        try:
            pos = self.WPG.position
            self.strvar_WPG_is.set(pos)
            self.strvar_green_current_power.set(
                np.round(self.angle_to_power(pos, float(self.ent_green_power.get()), float(self.ent_green_phase.get())),
                         3))

        except:
            print("Impossible to read WPG position")

    def move_WPG(self):
        """
        Moves the green waveplate motor to the desired position.

        Raises
        ------
        Exception
            If the motor fails to move to the desired position.

        Returns
        -------
        None
        """
        try:
            if self.var_wpgpower.get() == 1:
                power = float(self.strvar_WPG_should.get())
                if power > float(self.ent_green_power.get()):
                    power = float(self.ent_green_power.get())
                    print("Value above maximum! Desired power set to maximum instead")
                pos = self.power_to_angle(power, float(self.ent_green_power.get()), float(self.ent_green_phase.get()))
            else:
                pos = float(self.strvar_WPG_should.get())

            print("WPG is moving..")
            self.WPG.move_to(pos, True)
            print(f"WPG moved to {str(self.WPG.position)}")
            self.read_WPG()
        except:
            print("Impossible to move WPG :(")

    def init_Delay(self):
        """
        Initializes the Delay motor object.

        Raises
        ------
        Exception
            If the motor fails to initialize.

        Returns
        -------
        None
        """
        try:
            self.Delay = apt.Motor(int(self.ent_Delay_Nr.get()))
            print("Delay connected")
            self.but_Delay_Ini.config(fg='green')
        except:
            self.but_Delay_Ini.config(fg='red')
            print("Not able to initalize Delay")

    def home_Delay(self):
        """
        Homes the delay waveplate motor object.

        Raises
        ------
        Exception
            If the motor fails to home.

        Returns
        -------
        None
        """
        try:
            self.Delay.move_home(blocking=True)
            self.but_Delay_Home.config(fg='green')
            print("Delay stage homed!")
            self.read_Delay()
        except:
            self.but_Delay_Home.config(fg='red')
            print("Not able to home Delay")

    def read_Delay(self):
        """
        Reads the current position of the Delay motor.

        Raises
        ------
        Exception
            If the position cannot be read.

        Returns
        -------
        None
        """
        try:
            pos = self.Delay.position
            self.strvar_Delay_is.set(pos)
        except:
            print("Impossible to read Delay position")

    def move_Delay(self):
        """
        Moves the Delay motor to the desired position.

        Raises
        ------
        Exception
            If the motor fails to move to the desired position.

        Returns
        -------
        None
        """
        try:
            pos = float(self.strvar_Delay_should.get())
            print("Delay is moving..")
            self.Delay.move_to(pos, True)
            print(f"Delay moved to {str(self.Delay.position)}")
            self.read_Delay()
        except:
            print("Impossible to move Delay :(")

    def angle_to_power(self, angle, maxA, phase):
        power = maxA / 2 * np.cos(2 * np.pi / 90 * angle - 2 * np.pi / 90 * phase) + maxA / 2
        return power

    def power_to_angle(self, power, maxA, phase):
        A = maxA / 2
        angle = -(45 * np.arccos(power / A - 1)) / np.pi + phase
        return angle

    def disable_motors(self):
        """
        Disconnect all the motors.

        Returns
        -------
        None
        """
        if self.WPG is not None:
            self.WPG.disable()
            print('WPG disconnected')
        if self.WPR is not None:
            self.WPR.disable()
            print('WPR disconnected')
        if self.Delay is not None:
            self.Delay.disable()
            print('Delay disconnected')

    def enable_calibrator(self):
        self.stop_calib = False
        self.calib_thread = threading.Thread(target=self.open_calibrator)
        self.calib_thread.daemon = True
        self.calib_thread.start()

    def open_calibrator(self):
        """
        Open the file where the power calibration is

        Returns
        -------
        None
        """
        self.calibrator = cal.Calibrator()
        self.strvar_red_power.set(str(self.calibrator.max_red))
        self.strvar_green_power.set(str(self.calibrator.max_green))
        self.strvar_red_phase.set(str(self.calibrator.phase_red))
        self.strvar_green_phase.set(str(self.calibrator.phase_green))

    def on_close(self):
        self.disable_motors()
        self.win.destroy()
        print('Thorlabs stages control closed')
