print('Importing the libraries...')
import json
import tkinter as tk
from tkinter import ttk
import os

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from hardware.SLM_driver.SpatialLightModulator import SpatialLightModulator
from hardware.SLM_driver import phase_settings

"""
Welcome to the D-lab Controller. If you are reading this this is maybe because you want to modify something is the code.
- If you want to add a button that opens a new window, directly go to setup_side_panel and add the name of your button 
and the associated function.
- If you want to add a new phase pattern, you to go to hardware>SLM_driver>phase_settings and
add a new Type. 
"""

class DLabController:
    """
    A graphical user interface (GUI) for controlling spatial light modulators (SLMs) and related hardware components
    in the D-lab.
    """
    def __init__(self, parent):
        """
        Initializes the D-Lab Controller interface and sets up the main window components.

        Parameters
        ----------
        parent : tkinter.Tk
            The root tkinter window for the application.
        """
        print('Initialisation of the interface...')
        self.main_win = parent
        self.style = ttk.Style()

        self.HHGView_win = None
        self.GasDensity_win = None
        self.FocusView_win = None

        self.SLM_green = SpatialLightModulator('green')
        self.SLM_red = SpatialLightModulator('red')

        self.create_main_window()

        self.frm_green_visible = False
        print("Loading the default parameters...")
        #self.load_default_parameters('green')
        #self.load_default_parameters('red')
        print("Welcome to the D-Lab Controller")

    ## Setting up the interface
    def create_main_window(self):
        """
        Configures the main window by setting up its properties and initiating the setup
        for the SLM windows and side panels.
        """
        self.main_win.protocol("WM_DELETE_WINDOW", self.exit_prog)
        self.main_win.title('D-Lab Controller - Main Interface')
        self.main_win.resizable(False, False)
        self.style.configure('lefttab.TNotebook', tabposition=tk.W + tk.N, tabplacement=tk.N + tk.EW)
        self.setup_slm_window('green')
        self.setup_slm_window('red')
        self.setup_side_panel()

    def setup_side_panel(self):
        """
        Creates the side panel that includes buttons for various functionalities.
        To add a panel, just add the name of button and the name of the function
        """
        self.frm_side_panel = ttk.LabelFrame(self.main_win, text='Side Panel')
        self.frm_side_panel.grid(row=0, column=2, sticky='nsew')

        buttons = [
            ('HHG View', self.open_hhg_view_win),
            ('Gas Density', self.open_gas_density_win),
            ('Focus View', self.open_focus_view_win),
            ('Hide/Show Green', self.hide_show_green_panel)
        ]
        for row, (label, cmd) in enumerate(buttons):
            button = ttk.Button(self.frm_side_panel, text=label, command=cmd)
            button.grid(row=row, column=0, sticky='nsew')

    def setup_slm_window(self, color):
        """
        Sets up the SLM window for the specified color.

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').
        """
        self.setup_frames(color)
        self.setup_save_load_buttons(color)
        self.setup_phase_tabs(color)
        self.setup_phase_display(color)
        self.setup_publish_button(color)
        self.setup_preview_button(color)
        self.setup_close_button(color)

        if color == 'red':
            getattr(self, f"frm_top_{color}").grid(row=0, column=1, sticky='nsew')
            getattr(self, f"frm_top_b_{color}").grid(row=1, column=1, sticky='nsew')
            getattr(self, f"frm_mid_{color}").grid(row=2, column=1, sticky='nsew')
            getattr(self, f"frm_bottom_{color}").grid(row=3, column=1, sticky='nsew')

    def setup_preview_button(self, color):
        """
        Sets up the preview button for the specified color.

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').
        """
        preview_button = ttk.Button(getattr(self, f"frm_bottom_{color}"), text=f'Preview {color}',
                                    command=lambda: self.get_phase(color))
        preview_button.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

    def setup_publish_button(self, color):
        """
        Sets up the publish button for the specified color.

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').
        """
        publish_button = ttk.Button(getattr(self, f"frm_bottom_{color}"), text=f'Publish {color}',
                                    command=lambda: self.open_publish_win(color))
        publish_button.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

    def setup_close_button(self, color):
        """
        Sets up the preview button for the specified color.

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').
        """
        close_button = ttk.Button(getattr(self, f"frm_bottom_{color}"), text=f'Close {color}',
                                    command=lambda: self.close_publish_win(color))
        close_button.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)

    def setup_frames(self, color):
        """
        Sets up the frames for the interface of the specified color SLM.

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').
        """
        setattr(self, f"frm_top_{color}", ttk.LabelFrame(self.main_win, text=f'{color.capitalize()} SLM interface'))
        setattr(self, f"frm_top_b_{color}",
                ttk.LabelFrame(getattr(self, f"frm_top_{color}"), text=f'{color.capitalize()} SLM - Phase display'))
        setattr(self, f"frm_mid_{color}", ttk.Notebook(self.main_win, style='lefttab.TNotebook'))
        setattr(self, f"frm_bottom_{color}", ttk.LabelFrame(self.main_win, text=f'{color.capitalize()} SLM - Options'))

    def setup_save_load_buttons(self, color):
        """
        Sets up the save and load buttons for the specified SLM.

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').
        """
        button_frame = getattr(self, f"frm_top_b_{color}")

        save_button = ttk.Button(button_frame, text=f'Save {color} settings', command=lambda: self.save_settings(color))
        load_button = ttk.Button(button_frame, text=f'Load {color} settings', command=lambda: self.load_settings(color))
        save_button.grid(row=0, sticky='ew')
        load_button.grid(row=1, sticky='ew')

        label_screen = ttk.Label(getattr(self, f"frm_top_{color}"), text='Display number:')
        setattr(self, f"strvar_{color}", tk.StringVar(value='2' if color == 'green' else '1'))
        spinbox_screen = ttk.Spinbox(getattr(self, f"frm_top_{color}"), width=8, from_=1, to=5,
                                     textvariable=getattr(self, f"strvar_{color}"))
        spinbox_screen.grid(row=0, column=1, sticky='w')
        label_screen.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        if color == 'green' or color == 'red':
            setattr(self, f'ent_scr_{color}', spinbox_screen)

    def setup_phase_tabs(self, color):
        """
        Sets up the phase tabs for the specified SLM.

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').
        """
        self.setup_box(getattr(self, f"frm_top_{color}"), color)

    def setup_box(self, frm_, color):
        """
        Sets up the box with checkboxes for enabling or disabling phase types.

        Parameters
        ----------
        frm_ : tkinter.Frame
            The parent frame where the box is placed.
        color : str
            The color of the SLM ('green' or 'red').
        """
        frm_box = ttk.LabelFrame(frm_, text='Phases enabled')
        frm_box.grid(column=0)

        types_attr = f"types_{color}"
        vars_attr = f"vars_{color}"
        phase_refs_attr = f"phase_refs_{color}"
        tabs_attr = f"tabs_{color}"
        frm_mid_attr = f"frm_mid_{color}"

        setattr(self, types_attr, phase_settings.types)
        setattr(self, vars_attr, [])
        setattr(self, phase_refs_attr, [])
        setattr(self, tabs_attr, [])

        types = getattr(self, types_attr)
        vars_list = getattr(self, vars_attr)
        phase_refs_list = getattr(self, phase_refs_attr)
        tabs_list = getattr(self, tabs_attr)
        frm_mid = getattr(self, frm_mid_attr)

        for ind, typ in enumerate(types):
            var = tk.IntVar()
            vars_list.append(var)
            tab = ttk.Frame(frm_mid)
            tabs_list.append(tab)
            frm_mid.add(tab, text=typ)
            phase_refs_list.append(phase_settings.new_type(tab, typ))
            box = ttk.Checkbutton(frm_box, text=typ,
                                  variable=vars_list[ind],
                                  onvalue=1, offvalue=0)
            box.grid(row=ind, sticky='w')

    def setup_phase_display(self, color):
        """
        Sets up the phase display using matplotlib for the specified color.

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').
        """
        figure = Figure(figsize=(3, 2))
        ax = figure.add_subplot(111)
        ax.figure.tight_layout()

        setattr(self, f"fig_{color}", figure)
        setattr(self, f"ax_{color}", ax)

        canvas = FigureCanvasTkAgg(figure, getattr(self, f"frm_top_b_{color}"))
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=2, sticky='ew')

        setattr(self, f"img_{color}", canvas)

    def hide_show_green_panel(self):
        """
        Toggles the visibility of the green SLM interface panel.
        """
        panels = ['frm_top_green', 'frm_top_b_green', 'frm_mid_green', 'frm_bottom_green']

        if self.frm_green_visible:
            for panel in panels:
                getattr(self, panel).grid_remove()
        else:
            getattr(self, 'frm_top_green').grid(row=0, column=0, sticky='nsew')
            getattr(self, 'frm_top_b_green').grid(row=1, column=1, sticky='nsew')
            getattr(self, 'frm_mid_green').grid(row=2, column=0, sticky='nsew')
            getattr(self, 'frm_bottom_green').grid(row=3, column=0, sticky='nsew')

        self.frm_green_visible = not self.frm_green_visible

    ## Phase pattern commands
    def get_phase(self, color):
        """
        Retrieves and calculates the phase for the specified SLM based on the active phase types.

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').

        Returns
        -------
        numpy.ndarray
            The phase array for the selected SLM.
        """
        slm = getattr(self, f'SLM_{color}')
        phase_refs = getattr(self, f'phase_refs_{color}')
        vars_color = getattr(self, f'vars_{color}')

        phase = np.zeros(slm.slm_size)
        slm.phase = np.zeros(slm.slm_size)
        slm.background_phase = np.zeros(slm.slm_size)

        active_phase_types = []
        for ind, phase_type in enumerate(phase_refs):
            if vars_color[ind].get() == 1:
                active_phase_types.append(phase_type.__class__.__name__)
                phase += phase_type.phase()                   # Add each phase type to the phase variable

                if phase_type.__class__.__name__ == 'TypeBackground':
                    slm.background_phase = phase_type.phase()  # Send the new background on the SLM class

        if np.all(slm.background_phase == 0):
            print(f"Warning: The background phase of the {color} SLM is an array of 0.")

        print(f"Phase(s) on the {color} SLM: {', '.join(active_phase_types)}")

        slm.phase = phase                       # Send the new phase on the SLM class
        self.update_phase_plot(slm, color)      # Update the plot on the main interface
        return phase

    def update_phase_plot(self, slm, color):
        """
        Updates the phase plot for the specified SLM. No need to specify the phase, since the get_phase function is
        logging the new phase into the slm class. One just have to call this function after setting the phase.

        Parameters
        ----------
        slm : SpatialLightModulator
            The SLM object for which the phase plot is updated.
        color : str
            The color of the SLM ('green' or 'red').
        """
        if color == 'green':
            ax = self.ax_green
            img = self.img_green
        elif color == 'red':
            ax = self.ax_red
            img = self.img_red

        ax.clear()
        ax.imshow(slm.phase-slm.background_phase, cmap='hsv', interpolation='None', extent=(
            -slm.slm_size[1] / 2, slm.slm_size[1] / 2,
            -slm.slm_size[0] / 2, slm.slm_size[0] / 2))
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.figure.tight_layout()
        img.draw()

    ##  Opening other windows
    def open_publish_win(self, color):
        """
        Publishes the current phase configuration to the specified SLM screen.

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').
        """
        slm = getattr(self, f'SLM_{color}')
        ent_scr = getattr(self, f'ent_scr_{color}')
        screen_num = int(ent_scr.get())
        phase = self.get_phase(color)

        # Check if all elements of background_phase are zero and raise an error
        if np.all(slm.background_phase == 0):
            raise ValueError(
                f"Error: The background phase of the {color} SLM is all zeros. "
                f"Please adjust the background phase before publishing.")

        slm.publish(phase, screen_num)

    def open_hhg_view_win(self):
        """
        Opens the HHG view window.
        """
        from diagnostics.HHGView import HHGView
        self.HHGView_win = HHGView.HHGView(self)

    def open_gas_density_win(self):
        """
        Opens the gas density view window.
        """
        from diagnostics.GasDensity import GasDensity
        self.GasDensity_win = GasDensity.GasDensity()

    def open_focus_view_win(self):
        """
        Opens the focus view window.
        """
        from diagnostics.FocusView import FocusView
        self.FocusView_win = FocusView.FocusView()

    ## Saving and laoding settings
    def save_settings(self, color, filepath=None):
        """
        Saves the current phase settings for the specified SLM to a file.

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').
        filepath : str, optional
            The path where the settings file is saved. If not provided, a dialog is opened.
        """
        from tkinter.filedialog import asksaveasfilename
        if filepath is None:
            initial_directory = './ressources/saved_settings'
            filepath = asksaveasfilename(initialdir=initial_directory,
                                         defaultextension='txt',
                                         filetypes=[('Text Files', '*.txt'), ('All Files', '*.*')])
            if not filepath:
                return

        dict = {}
        phase_refs = getattr(self, f"phase_refs_{color}")
        vars_ = getattr(self, f"vars_{color}")
        ent_scr = getattr(self, f"ent_scr_{color}")

        with open(filepath, 'w') as f:
            for num, phase in enumerate(phase_refs):
                dict[phase.name_()] = {'Enabled': vars_[num].get(), 'Params': phase.save_()}
            dict['screen_pos'] = ent_scr.get()
            f.write(json.dumps(dict))

    def load_settings(self, color, filepath=None):
        """
        Loads the phase settings for the specified SLM from a file.

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').
        filepath : str, optional
            The path to the settings file. If not provided, a dialog is opened.
        """
        from tkinter.filedialog import askopenfilename
        if filepath is None:
            initial_directory = './ressources/saved_settings'
            filepath = askopenfilename(initialdir=initial_directory,
                                       filetypes=[('Text Files', '*.txt'), ('All Files', '*.*')])
            if not filepath:
                return

        try:
            with open(filepath, 'r') as f:
                dics = json.loads(f.read())

            phase_refs = getattr(self, f"phase_refs_{color}")
            vars_ = getattr(self, f"vars_{color}")
            ent_scr = getattr(self, f"ent_scr_{color}")

            try:
                for num, phase in enumerate(phase_refs):
                    phase.load_(dics[phase.name_()]['Params'])
                    vars_[num].set(dics[phase.name_()]['Enabled'])
                ent_scr.delete(0, tk.END)
                ent_scr.insert(0, dics['screen_pos'])
                print(f"{color.capitalize()} settings loaded successfully")
            except ValueError:
                print(f'Not able to load {color} settings')
        except FileNotFoundError:
            print(f'No {color} settings file found at {filepath}')

    def load_default_parameters(self, color):
        """
        Loads the default parameters for the specified color SLM from the corresponding default settings file.

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').
        """
        filepath = f'./ressources/saved_settings/default_{color}_settings.txt'

        if os.path.exists(filepath):
            print(f"Loading default {color} settings from {filepath}...")
            self.load_settings(color, filepath)
        else:
            print(f"Default {color} settings file not found.")

    ## Closing and exit commands
    def close_publish_win(self,color):
        """
        Close the connection to the SLM

        Parameters
        ----------
        color : str
            The color of the SLM ('green' or 'red').
        """
        slm = getattr(self, f'SLM_{color}')
        slm.close()


    def exit_prog(self):
        """
        Closes the application and ensures the SLM connections are closed.
        """
        self.close_publish_win('red')
        self.close_publish_win('green')
        self.main_win.destroy()


root = tk.Tk()
main = DLabController(root)
root.mainloop()