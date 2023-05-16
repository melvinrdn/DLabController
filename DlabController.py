print('Importing the libraries...')
import json
import os

import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as tkMbox
from tkinter.filedialog import askopenfilename, asksaveasfilename

import drivers.santec_driver._slm_py as slm
from model import phase_settings, feedbacker
from ressources.settings import slm_size, bit_depth
from views import preview_window, questionbox, camera_control, andor_xuv_camera, mcp

print('Done !')


class DLabController(object):
    """
    A class for controlling the Dlab hardware, test
    """

    def __init__(self, parent):
        """
        Initializes the DLabController object with its attributes.

        Parameters
        ----------
        parent : tkinter.Tk
            The tkinter parent window.

        Returns
        -------
        None
        """
        print('Interface initialization...')
        matplotlib.use("TkAgg")
        self.main_win = parent
        self.main_win.protocol("WM_DELETE_WINDOW", self.exit_prog)
        self.main_win.title('D-Lab Controller')
        self.style = ttk.Style()

        self.pub_win = None
        self.prev_win = None
        self.feedback_win = None
        self.andor_camera = None
        self.camera_win = None
        self.mcp_win = None
        self.phase_map = np.zeros(slm_size)

        # creating frames
        self.frm_top = ttk.Frame(self.main_win)
        self.frm_mid = ttk.Notebook(self.main_win)
        self.frm_bot = ttk.Frame(self.main_win)
        self.frm_topb = ttk.Frame(self.frm_top)
        self.frm_side = ttk.Frame(self.main_win)

        # Creating labels
        lbl_screen = ttk.Label(self.frm_top, text='SLM display number:')

        # Creating buttons
        # but_mcp = ttk.Button(frm_bot, text='MCP', command=self.open_mcp)
        but_camera = ttk.Button(self.frm_bot, text='Camera control', command=self.open_camera)
        but_xuv_camera = ttk.Button(self.frm_bot, text='XUV Camera', command=self.open_xuv_camera)
        but_feedback = ttk.Button(self.frm_bot, text='Feedbacker', command=self.open_feedback)
        # but_prev = ttk.Button(frm_bot, text='Preview', command=self.open_prev)
        but_pub = ttk.Button(self.frm_bot, text='Publish', command=self.open_pub)
        but_exit = ttk.Button(self.frm_bot, text='EXIT', command=self.exit_prog)
        but_save = ttk.Button(self.frm_topb, text='Save Settings', command=self.save)
        but_load = ttk.Button(self.frm_topb, text='Load Settings', command=self.load)
        # but_clean_settings = tk.Button(frm_bot, text='Clean settings file', command=self.delete_last_settings_file)

        # Creating entry
        self.ent_scr = ttk.Spinbox(self.frm_top, width=5, from_=1, to=8)

        # Setting up general structure
        self.frm_top.grid(row=2, column=0, sticky='nsew')
        self.frm_mid.grid(row=1, column=0, sticky='nsew')
        self.frm_bot.grid(row=4, column=0, sticky='nsew')
        self.frm_side.grid(row=3, column=0, sticky='nsew')

        # Setting up top frame
        lbl_screen.grid(row=0, column=0, sticky='e', padx=10, pady=10)
        self.ent_scr.grid(row=0, column=1, sticky='w', padx=(0, 10))
        self.setup_box(self.frm_top)
        self.frm_topb.grid(row=1, column=1, sticky='nsew')
        but_save.grid(row=0, sticky='ew')
        but_load.grid(row=1, sticky='ew')

        # Setting up scan and phase figure
        self.scan_options()
        self.fig = Figure(figsize=(2.5, 2), dpi=110)
        self.ax = self.fig.add_subplot(111)

        self.img = FigureCanvasTkAgg(self.fig, self.frm_topb)
        self.tk_widget_fig = self.img.get_tk_widget()
        self.tk_widget_fig.grid(row=2, sticky='ew')

        self.ax.axes.xaxis.set_visible(False)
        self.ax.axes.yaxis.set_visible(False)

        # Setting up bot frame
        # but_mcp.grid(row=0, column=0, padx=5, pady=5)
        but_xuv_camera.grid(row=0, column=0, padx=5, pady=5)
        but_camera.grid(row=0, column=1, padx=5, pady=5)
        but_feedback.grid(row=0, column=2, padx=5, pady=5)
        # but_prev.grid(row=0, column=4, padx=5, pady=5)
        but_pub.grid(row=0, column=3, padx=5, pady=5)
        but_exit.grid(row=0, column=4, padx=5, pady=5)

        # but_clean_settings.grid(row=0, column=6, padx=5, pady=5)

        print('Done !')

        # binding keys
        def left_handler(event):
            """
            Handle the 'a' key press event.

            Parameters
            ----------
            event : tkinter.Event
                The event object associated with the key press.

            Returns
            -------
            None
            """
            return self.left_arrow()

        self.main_win.bind('a', left_handler)

        def right_handler(event):
            """
            Handle the 'd' key press event.

            Parameters
            ----------
            event : tkinter.Event
                The event object associated with the key press.

            Returns
            -------
            None
            """
            return self.right_arrow()

        self.main_win.bind('d', right_handler)

        def up_handler(event):
            """
            Handle the 'w' key press event.

            Parameters
            ----------
            event : tkinter.Event
                The event object associated with the key press.

            Returns
            -------
            None
            """
            return self.up_arrow()

        self.main_win.bind('w', up_handler)

        def down_handler(event):
            """
            Handle the 's' key press event.

            Parameters
            ----------
            event : tkinter.Event
                The event object associated with the key press.

            Returns
            -------
            None
            """
            return self.down_arrow()

        self.main_win.bind('s', down_handler)

        def escape_handler(event):
            """
            Handle the 'Escape' key press event.

            Parameters
            ----------
            event : tkinter.Event
                The event object associated with the key press.

            Returns
            -------
            None
            """
            return self.escape_key()

        self.main_win.bind('<Escape>', escape_handler)

        # loading last settings
        self.load('./last_settings.txt')

    def open_feedback(self):
        """
        Open the feedback window for analyzing fringes between two beams.

        If the feedback window is not already open, a question box will pop up asking the user to choose a feedback
        method between using a camera with spatial fringes or a spectrometer with spectral fringes.

        Returns
        -------
        None
        """
        if self.feedback_win is None:
            q_str1 = 'The feedbacker needs to look at fringes between the two beams.'
            q_str2 = 'Do you want to use a camera with spatial fringes or a spectrometer with spectral fringes?'
            q_str = q_str1 + '\n' + q_str2
            questionbox.PopupQuestion(self.open_feedback_window, 'Choose feedback method',
                                      q_str, 'Open Camera', 'Open Spectrometer')

    def open_feedback_window(self, answer):
        """
        Open the feedback window using the specified feedback method.

        Parameters
        ----------
        answer : str
            The feedback method to use - 'Open Camera' for spatial fringes or 'Open Spectrometer' for spectral fringes.

        Returns
        -------
        None
        """
        self.feedback_win = feedbacker.Feedbacker(self, slm, answer)

    def open_camera(self):
        """
        Opens the camera window to look at the beam profile.

        Returns
        -------
        None
        """
        self.camera_win = camera_control.CameraControl(self)

    def open_xuv_camera(self):
        """
        Opens the XUV camera.

        Returns
        -------
        None
        """
        self.andor_camera = andor_xuv_camera.AndorCameraViewer(self)

    def open_prev(self):
        """
        Opens the preview window or updates it if it is already open.

        If the preview window is not open, it will be created. Otherwise, the update_plots() method of the existing
        preview window will be called to update the plots.

        Returns
        -------
        None
        """
        if self.prev_win is not None:
            self.prev_win.update_plots()
        else:
            self.prev_win = preview_window.PrevScreen(self)

    def prev_win_closed(self):
        """
        Handle the event of the preview window being closed.

        When the preview window is closed, this method will be called to reset the prev_win attribute to None.

        Returns
        -------
        None
        """
        print('Preview window closed')
        self.prev_win = None

    def open_pub(self):
        """
        Open the public display window and display the phase map.

        This method will first disable the entry screen, then generate a phase map using the current phase values
        and bit depth, and display it on the SLM using the SLM_Disp_Open() and SLM_Disp_Data() methods. Finally,
        the update_phase_plot() method is called to update the phase plot.

        Returns
        -------
        None
        """
        self.ent_scr.config(state='disabled')
        self.phase_map = self.get_phase()

        self.pub_win = int(self.ent_scr.get())
        slm.SLM_Disp_Open(int(self.ent_scr.get()))
        slm.SLM_Disp_Data(int(self.ent_scr.get()), self.phase_map, slm_size[1], slm_size[0])

        self.update_phase_plot(self.phase_map)

    def open_mcp(self):
        """
        Open the MCP window.

        Returns
        -------
        None
        """
        self.mcp_win = mcp.Mcp(self)

    def do_scan(self):
        """
        Perform a scan of the specified file list.

        This method loads each file in the file list, opens the public display window, and waits for the specified delay
        between files. The scan can be stopped by setting the var_stop_scan attribute to 1.

        Returns
        -------
        None
        """
        if self.strvar_delay.get() == '':
            self.strvar_delay.set('1')
        delay = float(self.strvar_delay.get())
        filelist = self.load_filelist()
        var = tk.IntVar()

        for filepath in filelist:
            if self.var_stop_scan.get():
                self.var_stop_scan.set(0)
                return
            root.after(int(delay * 1000), var.set, 1)
            self.load(filepath)

            # keeps to one window and updates for each filepath
            self.open_pub()

            self.lbl_time['text'] = delay
            self.countdown()
            root.wait_variable(var)

    def countdown(self):
        """
        Countdown the remaining time until the next file is loaded.

        This method updates the lbl_time label to display the remaining time until the next file is loaded. It is called
        by the do_scan() method.

        Returns
        -------
        None
        """
        self.lbl_time['text'] = int(self.lbl_time['text']) - 1
        if int(self.lbl_time['text']):
            self.lbl_time.after(1000, self.countdown)

    def pub_win_closed(self):
        """
        Handle the event of the public display window being closed.

        This method re-enables the entry screen and closes the public display on the SLM.

        Returns
        -------
        None
        """
        self.ent_scr.config(state='normal')
        slm.SLM_Disp_Close(int(self.ent_scr.get()))

    def setup_box(self, frm_):
        """
        Set up a label frame containing check-buttons for enabling different types of phase.

        Parameters:
        -----------
        frm_: tkinter.Frame
            The parent frame in which the label frame and check-buttons are to be placed.
        """
        frm_box = ttk.LabelFrame(frm_, text='Phases enabled')
        frm_box.grid(column=0)
        self.types = phase_settings.types  # reads in  different phase types
        self.vars = []  # init a list holding the variables from the boxes
        self.phase_refs = []  # init a list to hold the references to types
        self.tabs = []  # init a list to hold the tabs
        for ind, typ in enumerate(self.types):
            self.var_ = (tk.IntVar())
            self.vars.append(self.var_)
            self.tabs.append(ttk.Frame(self.frm_mid))
            self.frm_mid.add(self.tabs[ind], text=typ)
            self.phase_refs.append(phase_settings.new_type(self.tabs[ind],
                                                           typ))
            self.box_ = ttk.Checkbutton(frm_box, text=typ,
                                        variable=self.vars[ind],
                                        onvalue=1, offvalue=0)
            self.box_.grid(row=ind, sticky='w')

    def get_phase(self):
        """
        Gets the phase from the active phase types.

        Returns:
        --------
        phase: numpy.ndarray
            A 2D numpy array containing the phase values of the active phase types.
        """
        phase = np.zeros(slm_size)
        for ind, phase_types in enumerate(self.phase_refs):
            if self.vars[ind].get() == 1:
                print(phase_types)
                phase += phase_types.phase()
        return phase

    def save(self, filepath=None):
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
            for num, phase in enumerate(self.phase_refs):
                dict[phase.name_()] = {'Enabled': self.vars[num].get(),
                                       'Params': phase.save_()}
            dict['screen_pos'] = self.ent_scr.get()
            f.write(json.dumps(dict))

    def load(self, filepath=None):
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
                for num, phase in enumerate(self.phase_refs):
                    phase.load_(dics[phase.name_()]['Params'])
                    self.vars[num].set(dics[phase.name_()]['Enabled'])
                self.ent_scr.delete(0, tk.END)
                self.ent_scr.insert(0, dics['screen_pos'])
                print("File loaded successfully")
            except ValueError:
                print('Not able to load settings')
        except FileNotFoundError:
            print(f'No settings file found at {filepath}')

    def scan_options(self):
        """
        Creates and sets up widgets for the scan options.

        Returns
        -------
        None
        """
        self.so_frm = ttk.LabelFrame(self.frm_side, text='Scan options')
        self.so_frm.grid(row=0, sticky='nsew')

        # creating frames
        frm_file = ttk.Frame(self.so_frm)

        # creating labels
        lbl_scpar = ttk.Label(self.so_frm, text='Scan parameter')
        lbl_val = ttk.Label(self.so_frm, text='Value (strt:stop:num)')
        lbl_actf = ttk.Label(frm_file, text='Active file:')
        self.lbl_file = ttk.Label(frm_file, text='', wraplength=230,
                                  justify='left', foreground='gray')
        lbl_delay = ttk.Label(
            self.so_frm, text='Delay between each phase [s]:')
        self.lbl_time = ttk.Label(self.so_frm, text='0')

        # creating entries
        self.cbx_scpar = ttk.Combobox(
            self.so_frm, values=['Select'], postcommand=self.scan_params)
        self.cbx_scpar.current(0)
        vcmd = (self.frm_side.register(self.callback))
        self.strvar_val = tk.StringVar()
        ent_val = ttk.Entry(self.so_frm, width=10, validate='all',
                            validatecommand=(vcmd, '%d', '%P', '%S'),
                            textvariable=self.strvar_val)
        self.strvar_delay = tk.StringVar()
        ent_delay = ttk.Entry(self.so_frm, width=5, validate='all',
                              validatecommand=(vcmd, '%d', '%P', '%S'),
                              textvariable=self.strvar_delay)

        # creating buttons
        self.but_crt = ttk.Button(
            self.so_frm, text='Create loading file',
            command=self.create_loadingfile)
        but_openload = ttk.Button(
            self.so_frm, text='Open existing loading file',
            command=self.open_loadingfile)
        self.but_scan = ttk.Button(
            self.so_frm, text='Scan', command=self.do_scan)
        but_stop_scan = ttk.Button(
            self.so_frm, text='Stop scan', command=self.stop_scan)
        self.var_stop_scan = tk.IntVar(value=0)

        # setup
        frm_file.grid(row=3, sticky='w', columnspan=3)
        self.but_crt.grid(row=2, column=0, sticky='ew')
        but_openload.grid(row=2, column=1, columnspan=2, sticky='ew')

        self.but_scan.grid(row=5, column=0, padx=5, pady=5, sticky='ew')
        but_stop_scan.grid(row=5, column=1, columnspan=2, padx=5, pady=5, sticky='ew')

        lbl_scpar.grid(row=0, column=0, sticky='e')
        lbl_val.grid(row=1, column=0, sticky='e')
        self.cbx_scpar.grid(row=0, column=1, columnspan=2, sticky='w')
        ent_val.grid(row=1, column=1, columnspan=2, sticky='w')

        lbl_actf.grid(row=3, column=0)
        self.lbl_file.grid(row=3, column=1)

        lbl_delay.grid(row=4, column=0, sticky='e')
        ent_delay.grid(row=4, column=1, columnspan=2, sticky='w')
        self.lbl_time.grid(row=4, column=2, sticky='w')

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

    def scan_params(self):
        """
        Get a list of scan parameters based on the currently selected phases.

        Returns
        -------
        List[str]
            A list of scan parameters, formatted as 'phase_name:param_name'.
        """
        scparams = []
        for ind, phase in enumerate(self.phase_refs):
            if self.vars[ind].get() == 1:
                phparam = phase.save_()
                for param in phparam.keys():
                    scparams.append(phase.name_() + ':' + param)
        self.cbx_scpar['values'] = scparams
        return

    def create_loadingfile(self):
        """
        Create a loading file for phase scan files.

        Returns
        -------
        None
        """
        if self.strvar_val.get() != '':
            strval = self.strvar_val.get()
            listval = strval.split(':', 3)
            try:
                strt = float(listval[0])
                stop = float(listval[1])
                num = int(listval[2])
                val_range = np.around(np.linspace(strt, stop, num), decimals=3)
            except (ValueError, IndexError) as err:
                self.but_crt['text'] = f'Create loading file : {err}'
                return

        else:
            print('Empty value')
            return
        print(f'{strt}_{stop}_{num}')

        cwd = os.getcwd()
        print('cwd is {}'.format(cwd))
        dirstr = '\\SLM_phase_scan_files'
        if not os.path.exists(cwd + dirstr):
            os.mkdir(cwd + dirstr)
        # create folder
        scparam = self.cbx_scpar.get().split(':')
        folder_str = '\\{}_{}_{}_{}_{}'.format(
            scparam[0], scparam[1], strt, stop, num)
        cwd = cwd + dirstr + folder_str
        if not os.path.exists(cwd):
            os.mkdir(cwd)

        # create file for filepaths
        ind = self.types.index(scparam[0])
        param_dic = self.phase_refs[ind].save_()
        with open(cwd + '\\' + 'filepaths.txt', 'w') as logfile:
            for val in val_range:
                param_dic[scparam[1]] = val
                self.phase_refs[ind].load_(param_dic)
                filepath = f'{cwd}\\{val:.3f}.txt'
                print(filepath)
                self.save(filepath)
                logfile.write(filepath + '\n')
        self.lbl_file['text'] = cwd + '\\' + 'filepaths.txt'

        self.but_crt['text'] = 'Create loading file : OK'
        return

    def open_loadingfile(self):
        """
        Open a loading file for phase scan files.

        Returns
        -------
        None
        """
        filepath = askopenfilename(
            filetypes=[('Text Files', '*.txt'), ('All Files', '*.*')]
        )
        if not filepath:
            return
        self.lbl_file['text'] = f'{filepath}'
        return

    def load_filelist(self):
        """
        Load a filelist for phase scan files.

        Returns
        -------
        list
            A list of file paths in the filelist.
        """
        filelistpath = self.lbl_file['text']
        with open(filelistpath, 'r') as f:
            text = f.read()
            stringlist = text.split('\n')
            return stringlist[0:-1]

    def stop_scan(self):
        """
        Stop the scan by setting a variable to 1.

        Returns
        -------
        None
        """
        self.var_stop_scan.set(1)
        return

    def left_arrow(self):
        """
        Move the selected phase to the left.

        Returns
        -------
        None
        """
        if self.vars[2].get() == 1:
            self.phase_refs[2].left_()
            self.open_pub()
            self.main_win.after(500, self.main_win.focus_force)

    def right_arrow(self):
        """
        Move the selected phase to the right.

        Returns
        -------
        None
        """
        if self.vars[2].get() == 1:
            self.phase_refs[2].right_()
            self.open_pub()
            self.main_win.after(500, self.main_win.focus_force)

    def up_arrow(self):
        """
        Move the selected phase up.

        Returns
        -------
        None
        """
        if self.vars[2].get() == 1:
            self.phase_refs[2].up_()
            self.open_pub()
            self.main_win.after(500, self.main_win.focus_force)

    def down_arrow(self):
        """
        Move the selected phase down.

        Returns
        -------
        None
        """
        print('s pressed')
        if self.vars[2].get() == 1:
            self.phase_refs[2].down_()
            self.open_pub()
            self.main_win.after(500, self.main_win.focus_force)

    def escape_key(self):
        """
        Handle the escape key press event.

        If the publication window is open, it prompts the user to close the window.
        Otherwise, it clears the ax1 and draws the img1.

        Returns
        -------
        None
        """
        print('esc pressed')
        if self.pub_win is not None:
            q_str = 'Do you want to close the SLM Publication Window?\nThe SLM screen will instead show the desktop ' \
                    'background.'
            result = tkMbox.askquestion('Close Publication Window', q_str)
            if result == 'yes':
                self.pub_win_closed()
                self.ax.clear()
                self.img.draw()

    def update_phase_plot(self, phase):
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
        self.ax.clear()
        self.ax.imshow(phase % (bit_depth + 1), cmap='RdBu',
                       interpolation='None')
        self.img.draw()

    def exit_prog(self):
        """
        Exit the program.

        This function saves the last settings, closes the publication window,
        and destroys the main window.

        Returns
        -------
        None
        """
        self.save('./last_settings.txt')
        self.pub_win_closed()
        self.feedback_win = None
        self.prev_win = None
        self.andor_camera = None
        self.camera_win = None
        self.mcp_win = None
        self.main_win.destroy()

    # Diagnostics
    def delete_last_settings_file(self):
        """
        Deletes the last_settings.txt file if it exists.

        Returns
        -------
        None
        """
        file_path = os.path.join(os.getcwd(), 'last_settings.txt')
        if os.path.exists(file_path):
            os.remove(file_path)


root = tk.Tk()
main = DLabController(root)
root.mainloop()
