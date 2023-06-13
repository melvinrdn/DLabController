print('Importing the libraries...')
import json
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, asksaveasfilename

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import drivers.santec_driver._slm_py as slm
from model import phase_settings, feedbacker
from ressources.settings import slm_size, bit_depth
from views import daheng_camera, andor_xuv_camera, mcp_camera

print('Done !')


class DLabController(object):
    """
    A class for controlling the Dlab hardware, test
    """

    def __init__(self, parent):
        """
        Initializes the DLabController.
        """
        print('Initialisation of the interface..')

        self.main_win = parent
        self.main_win.protocol("WM_DELETE_WINDOW", self.exit_prog)
        self.main_win.title('D-Lab Controller - Main Interface')
        self.style = ttk.Style()
        self.style.configure('lefttab.TNotebook',
                             tabposition=tk.W + tk.N,
                             tabplacement=tk.N + tk.EW)

        self.publish_window_green = None
        self.publish_window_red = None

        self.feedback_win = None
        self.andor_camera = None
        self.camera_win = None
        self.mcp_win = None

        self.phase_map_green = np.zeros(slm_size)
        self.phase_map_red = np.zeros(slm_size)

        self.frm_top_green = ttk.LabelFrame(self.main_win, text='Green SLM interface')
        self.frm_top_b_green = ttk.LabelFrame(self.frm_top_green, text='Green SLM - Phase display')
        self.frm_mid_green = ttk.Notebook(self.main_win, style='lefttab.TNotebook')
        self.frm_bottom_green = ttk.LabelFrame(self.main_win, text='Green SLM - Options')

        self.frm_top_red = ttk.LabelFrame(self.main_win, text='Red SLM interface')
        self.frm_top_b_red = ttk.LabelFrame(self.frm_top_red, text='Red SLM - Phase display')
        self.frm_mid_red = ttk.Notebook(self.main_win, style='lefttab.TNotebook')
        self.frm_bottom_red = ttk.LabelFrame(self.main_win, text='Red SLM - Options')

        self.frm_side_panel = ttk.LabelFrame(self.main_win, text='Hardware - Cameras')
        self.frm_bottom_side_panel = ttk.Frame(self.main_win)

        but_save_green = ttk.Button(self.frm_top_b_green, text='Save green settings',
                                    command=self.save_green)  # TODO automatic saving with date, same for red
        but_load_green = ttk.Button(self.frm_top_b_green, text='Load green settings', command=self.load_green)
        but_save_green.grid(row=0, sticky='ew')
        but_load_green.grid(row=1, sticky='ew')

        but_save_red = ttk.Button(self.frm_top_b_red, text='Save red settings', command=self.save_red)
        but_load_red = ttk.Button(self.frm_top_b_red, text='Load red settings', command=self.load_red)
        but_save_red.grid(row=0, sticky='ew')
        but_load_red.grid(row=1, sticky='ew')

        lbl_screen_green = ttk.Label(self.frm_top_green, text='Display number :')
        self.strvar_green = tk.StringVar(value='3')
        self.ent_scr_green = ttk.Spinbox(self.frm_top_green, width=5, from_=1, to=5, textvariable=self.strvar_green)
        self.ent_scr_green.grid(row=0, column=1, sticky='w', padx=(0, 10))

        lbl_screen_red = ttk.Label(self.frm_top_red, text='Display number :')
        self.strvar_red = tk.StringVar(value='2')
        self.ent_scr_red = ttk.Spinbox(self.frm_top_red, width=5, from_=1, to=5, textvariable=self.strvar_red)
        self.ent_scr_red.grid(row=0, column=1, sticky='w', padx=(0, 10))

        self.setup_box_green(self.frm_top_green)
        self.setup_box_red(self.frm_top_red)

        # Set up general structure
        self.frm_top_green.grid(row=0, column=0, sticky='nsew')
        self.frm_top_b_green.grid(row=1, column=1, sticky='nsew')
        self.frm_mid_green.grid(row=2, column=0, sticky='nsew')
        self.frm_bottom_green.grid(row=3, column=0, sticky='nsew')

        self.frm_top_red.grid(row=0, column=1, sticky='nsew')
        self.frm_top_b_red.grid(row=1, column=1, sticky='nsew')
        self.frm_mid_red.grid(row=2, column=1, sticky='nsew')
        self.frm_bottom_red.grid(row=3, column=1, sticky='nsew')

        self.frm_side_panel.grid(row=0, column=2, sticky='nsew')
        self.frm_bottom_side_panel.grid(row=3, column=2, sticky='nsew')

        lbl_screen_green.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        lbl_screen_red.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        self.fig_green = Figure(figsize=(4, 3.5), dpi=110)
        self.ax_green = self.fig_green.add_subplot(111)

        self.fig_red = Figure(figsize=(4, 3.5), dpi=110)
        self.ax_red = self.fig_red.add_subplot(111)

        self.img_green = FigureCanvasTkAgg(self.fig_green, self.frm_top_b_green)
        self.tk_widget_fig_green = self.img_green.get_tk_widget()
        self.tk_widget_fig_green.grid(row=2, sticky='ew')

        self.img_red = FigureCanvasTkAgg(self.fig_red, self.frm_top_b_red)
        self.tk_widget_fig_red = self.img_red.get_tk_widget()
        self.tk_widget_fig_red.grid(row=2, sticky='ew')

        self.ax_green.axes.xaxis.set_visible(False)
        self.ax_green.axes.yaxis.set_visible(False)

        self.ax_red.axes.xaxis.set_visible(False)
        self.ax_red.axes.yaxis.set_visible(False)

        but_mcp = ttk.Button(self.frm_side_panel, text='MCP Camera', command=self.open_mcp)
        but_camera = ttk.Button(self.frm_side_panel, text='DAHENG Camera', command=self.open_camera)
        but_xuv_camera = ttk.Button(self.frm_side_panel, text='ANDOR Camera', command=self.open_xuv_camera)

        but_exit = ttk.Button(self.frm_bottom_side_panel, text='EXIT', command=self.exit_prog)

        but_mcp.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        but_xuv_camera.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        but_camera.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)

        but_exit.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        but_feedback = ttk.Button(self.frm_bottom_green, text='Feedbacker', command=self.open_feedback_window)
        but_publish_green = ttk.Button(self.frm_bottom_green, text='Publish green', command=self.open_pub_green)
        but_feedback.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        but_publish_green.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        but_publish_red = ttk.Button(self.frm_bottom_red, text='Publish red', command=self.open_pub_red)
        but_publish_red.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        print("Done !")
        print("-----------")
        print("Welcome to the D-Lab Controller")

    def open_feedback_window(self):
        """
        Open the feedback window.

        Returns
        -------
        None
        """
        self.feedback_win = feedbacker.Feedbacker(self)

    def open_camera(self):
        """
        Opens the camera window to look at the beam profile.

        Returns
        -------
        None
        """
        self.camera_win = daheng_camera.CameraControl()

    def open_xuv_camera(self):
        """
        Opens the XUV camera.

        Returns
        -------
        None
        """
        self.andor_camera = andor_xuv_camera.AndorCameraViewer(self)

    def open_mcp(self):
        """
        Open the MCP window.

        Returns
        -------
        None
        """
        self.mcp_win = mcp_camera.Mcp(self)

    def open_pub_green(self):
        """
        Open the publish display window and display the phase map.

        Returns
        -------
        None
        """
        self.ent_scr_green.config(state='disabled')
        self.phase_map_green = self.get_phase_green()

        self.publish_window_green = int(self.ent_scr_green.get())
        slm.SLM_Disp_Open(int(self.ent_scr_green.get()))
        slm.SLM_Disp_Data(int(self.ent_scr_green.get()), self.phase_map_green, slm_size[1], slm_size[0])

        self.update_phase_plot_green(self.phase_map_green)

    def open_pub_red(self):
        """
        Open the publish display window and display the phase map.

        Returns
        -------
        None
        """
        self.ent_scr_red.config(state='disabled')
        self.phase_map_red = self.get_phase_red()

        self.publish_window_red = int(self.ent_scr_red.get())
        slm.SLM_Disp_Open(int(self.ent_scr_red.get()))
        slm.SLM_Disp_Data(int(self.ent_scr_red.get()), self.phase_map_red, slm_size[1], slm_size[0])

        self.update_phase_plot_red(self.phase_map_red)

    def setup_box_green(self, frm_):
        """
        Set up a label frame containing check-buttons for enabling different types of phase.

        Parameters:
        -----------
        frm_: tkinter.Frame
            The parent frame in which the label frame and check-buttons are to be placed.
        """
        frm_box_green = ttk.LabelFrame(frm_, text='Phases enabled')
        frm_box_green.grid(column=0)
        self.types_green = phase_settings.types  # reads in different phase types
        self.vars_green = []  # init a list holding the variables from the boxes
        self.phase_refs_green = []  # init a list to hold the references to types
        self.tabs_green = []  # init a list to hold the tabs
        for ind, typ in enumerate(self.types_green):
            self.var_green_ = (tk.IntVar())
            self.vars_green.append(self.var_green_)
            self.tabs_green.append(ttk.Frame(self.frm_mid_green))
            self.frm_mid_green.add(self.tabs_green[ind], text=typ)
            self.phase_refs_green.append(phase_settings.new_type(self.tabs_green[ind],
                                                                 typ))
            self.box_green_ = ttk.Checkbutton(frm_box_green, text=typ,
                                              variable=self.vars_green[ind],
                                              onvalue=1, offvalue=0)
            self.box_green_.grid(row=ind, sticky='w')

    def setup_box_red(self, frm_):
        """
        Set up a label frame containing check-buttons for enabling different types of phase.

        Parameters:
        -----------
        frm_: tkinter.Frame
            The parent frame in which the label frame and check-buttons are to be placed.
        """
        frm_box_red = ttk.LabelFrame(frm_, text='Phases enabled')
        frm_box_red.grid(column=0)
        self.types_red = phase_settings.types  # reads in different phase types
        self.vars_red = []  # init a list holding the variables from the boxes
        self.phase_refs_red = []  # init a list to hold the references to types
        self.tabs_red = []  # init a list to hold the tabs
        for ind, typ in enumerate(self.types_red):
            self.var_red_ = (tk.IntVar())
            self.vars_red.append(self.var_red_)
            self.tabs_red.append(ttk.Frame(self.frm_mid_red))
            self.frm_mid_red.add(self.tabs_red[ind], text=typ)
            self.phase_refs_red.append(phase_settings.new_type(self.tabs_red[ind], typ))
            self.box_red_ = ttk.Checkbutton(frm_box_red, text=typ,
                                            variable=self.vars_red[ind],
                                            onvalue=1, offvalue=0)
            self.box_red_.grid(row=ind, sticky='w')

    def get_phase_green(self):
        """
        Gets the phase from the active phase types.

        Returns:
        --------
        phase: numpy.ndarray
            A 2D numpy array containing the phase values of the active phase types.
        """
        phase_green = np.zeros(slm_size)
        for ind, phase_types_green in enumerate(self.phase_refs_green):
            if self.vars_green[ind].get() == 1:
                print(phase_types_green)
                phase_green += phase_types_green.phase()
        return phase_green

    def get_phase_red(self):
        """
        Gets the phase from the active phase types.

        Returns:
        --------
        phase: numpy.ndarray
            A 2D numpy array containing the phase values of the active phase types.
        """
        phase_red = np.zeros(slm_size)
        for ind, phase_types_red in enumerate(self.phase_refs_red):
            if self.vars_red[ind].get() == 1:
                print(phase_types_red)
                phase_red += phase_types_red.phase()
        return phase_red

    def update_phase_plot_green(self, phase):
        """
        Update the phase plot of the SLM.

        This function clears the ax1, updates the phase with new values, and draws
        the img1.

        Parameters
        ----------
        phase : np.ndarray
            The new phase values to update.

        Returns
        -------
        None
        """
        self.ax_green.clear()
        self.ax_green.imshow(phase % (bit_depth + 1), cmap='RdBu',
                             interpolation='None')
        self.img_green.draw()

    def update_phase_plot_red(self, phase):
        """
        Update the phase plot of the SLM.

        This function clears the ax1, updates the phase with new values, and draws
        the img1.

        Parameters
        ----------
        phase : np.ndarray
            The new phase values to update.

        Returns
        -------
        None
        """
        self.ax_red.clear()
        self.ax_red.imshow(phase % (bit_depth + 1), cmap='RdBu',
                           interpolation='None')
        self.img_red.draw()

    def callback(self, action, P, text):
        """
        Check if the given input text is valid for insertion in an Entry widget.

        Parameters
        ----------
        action : str
            The type of action being performed on the Entry widget.
            Must be either '1' for insertion or '0' for deletion.
        P : str
            The proposed insertion position.
        text : str
            The text to be inserted.

        Returns
        -------
        bool
            True if the input text is valid for insertion, False otherwise.
        """
        if action == '1':
            if text in '0123456789.-+:':
                return True
            else:
                return False
        else:
            return True

    def save_green(self, filepath=None):
        """
        Save the current settings to a file.

        Parameters
        ----------
        filepath : str, optional
            The path to the file to save. If not specified, a dialog box will be
            displayed to prompt the user to choose a file.

        Notes
        -----
        The settings will be saved as a JSON-encoded dictionary to the specified
        file. The dictionary will contain the enabled status and parameters for
        each phase type
        """
        if filepath is None:
            filepath = asksaveasfilename(
                defaultextension='txt',
                filetypes=[('Text Files', '*.txt'), ('All Files', '*.*')]
            )
            if not filepath:
                return
        dict = {}
        with open(filepath, 'w') as f:
            for num, phase in enumerate(self.phase_refs_green):
                dict[phase.name_()] = {'Enabled': self.vars_green[num].get(),
                                       'Params': phase.save_()}
            dict['screen_pos'] = self.ent_scr_green.get()
            f.write(json.dumps(dict))

    def load_green(self, filepath=None):
        """
        Load settings from a file.

        Parameters
        ----------
        filepath : str, optional
            The path to the file to load. If not specified, a dialog box will be
            displayed to prompt the user to choose a file.

        Notes
        -----
        The settings will be loaded from a JSON-encoded dictionary in the specified
        file. The dictionary should contain the enabled status and parameters for
        each phase type.
        """
        if filepath is None:
            filepath = askopenfilename(
                filetypes=[('Text Files', '*.txt'), ('All Files', '*.*')]
            )
            if not filepath:
                return
        try:
            with open(filepath, 'r') as f:
                dics = json.loads(f.read())
            try:
                for num, phase in enumerate(self.phase_refs_green):
                    phase.load_(dics[phase.name_()]['Params'])
                    self.vars_green[num].set(dics[phase.name_()]['Enabled'])
                self.ent_scr_green.delete(0, tk.END)
                self.ent_scr_green.insert(0, dics['screen_pos'])
                print("File loaded successfully")
            except ValueError:
                print('Not able to load settings')
        except FileNotFoundError:
            print(f'No settings file found at {filepath}')

    def save_red(self, filepath=None):
        """
        Save the current settings to a file.

        Parameters
        ----------
        filepath : str, optional
            The path to the file to save. If not specified, a dialog box will be
            displayed to prompt the user to choose a file.

        Notes
        -----
        The settings will be saved as a JSON-encoded dictionary to the specified
        file. The dictionary will contain the enabled status and parameters for
        each phase type
        """
        if filepath is None:
            filepath = asksaveasfilename(
                defaultextension='txt',
                filetypes=[('Text Files', '*.txt'), ('All Files', '*.*')]
            )
            if not filepath:
                return
        dict = {}
        with open(filepath, 'w') as f:
            for num, phase in enumerate(self.phase_refs_red):
                dict[phase.name_()] = {'Enabled': self.vars_red[num].get(),
                                       'Params': phase.save_()}
            dict['screen_pos'] = self.ent_scr_red.get()
            f.write(json.dumps(dict))

    def load_red(self, filepath=None):
        """
        Load settings from a file.

        Parameters
        ----------
        filepath : str, optional
            The path to the file to load. If not specified, a dialog box will be
            displayed to prompt the user to choose a file.

        Notes
        -----
        The settings will be loaded from a JSON-encoded dictionary in the specified
        file. The dictionary should contain the enabled status and parameters for
        each phase type.
        """
        if filepath is None:
            filepath = askopenfilename(
                filetypes=[('Text Files', '*.txt'), ('All Files', '*.*')]
            )
            if not filepath:
                return
        try:
            with open(filepath, 'r') as f:
                dics = json.loads(f.read())
            try:
                for num, phase in enumerate(self.phase_refs_red):
                    phase.load_(dics[phase.name_()]['Params'])
                    self.vars_red[num].set(dics[phase.name_()]['Enabled'])
                self.ent_scr_red.delete(0, tk.END)
                self.ent_scr_red.insert(0, dics['screen_pos'])
                print("File loaded successfully")
            except ValueError:
                print('Not able to load settings')
        except FileNotFoundError:
            print(f'No settings file found at {filepath}')

    def publish_window_closed(self):
        """
        Handle the event of the publish display window being closed.

        Returns
        -------
        None
        """
        self.ent_scr_green.config(state='normal')
        slm.SLM_Disp_Close(int(self.ent_scr_green.get()))

        self.ent_scr_red.config(state='normal')
        slm.SLM_Disp_Close(int(self.ent_scr_red.get()))

    def exit_prog(self):
        """
        Exit the program.

        This function saves the last settings, closes the publication window,
        and destroys the main window.

        Returns
        -------
        None
        """
        self.publish_window_closed()
        self.feedback_win = None
        self.andor_camera = None
        self.camera_win = None
        self.mcp_win = None
        self.main_win.destroy()


root = tk.Tk()
main = DLabController(root)
root.mainloop()
