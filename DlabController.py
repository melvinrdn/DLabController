print('Importing the libraries...')

import json
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, asksaveasfilename

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from model import phase_settings
import drivers.santec_driver._slm_py as slm_driver
from ressources.slm_infos import slm_size, bit_depth,chip_width,chip_height

print('Done!')


class SpatialLightModulator:
    def __init__(self, color):
        self.color = color
        self.phase_map = np.zeros(slm_size)
        self.background_phase = np.zeros(slm_size)
        self.screen_num = None

    def publish(self, screen_num):
        self.screen_num = screen_num
        phase_map = (self.phase_map % (bit_depth + 1)).astype(np.uint16)
        slm_driver.SLM_Disp_Open(self.screen_num)
        slm_driver.SLM_Disp_Data(self.screen_num, phase_map, slm_size[1], slm_size[0])

    def close(self):
        if self.screen_num is not None:
            slm_driver.SLM_Disp_Close(self.screen_num)
            self.screen_num = None


class DLabController:
    def __init__(self, parent):
        print('Initialisation of the interface...')
        self.main_win = parent
        self.style = ttk.Style()

        self.publish_win = None
        self.HHGView_win = None
        self.GasDensity_win = None
        self.FocusView_win = None

        self.configure_main_window()

        self.SLM_green = SpatialLightModulator('green')
        self.SLM_red = SpatialLightModulator('red')

        self.frm_green_visible = False
        print("Loading the default parameters...")
        print("Done! Welcome to the D-Lab Controller")

    def configure_main_window(self):
        self.main_win.protocol("WM_DELETE_WINDOW", self.exit_prog)
        self.main_win.title('D-Lab Controller - Main Interface')
        self.main_win.resizable(False, False)
        self.style.configure('lefttab.TNotebook', tabposition=tk.W + tk.N, tabplacement=tk.N + tk.EW)
        self.setup_slm_window('green')
        self.setup_slm_window('red')
        self.create_side_panel()

    def setup_slm_window(self, color):
        self.setup_frames(color)
        self.setup_save_load_buttons(color)
        self.setup_phase_tabs(color)
        self.setup_phase_display(color)
        self.setup_publish_button(color)

        if color == 'red':
            getattr(self, f"frm_top_{color}").grid(row=0, column=1, sticky='nsew')
            getattr(self, f"frm_top_b_{color}").grid(row=1, column=1, sticky='nsew')
            getattr(self, f"frm_mid_{color}").grid(row=2, column=1, sticky='nsew')
            getattr(self, f"frm_bottom_{color}").grid(row=3, column=1, sticky='nsew')

    def setup_publish_button(self, color):
        publish_button = ttk.Button(getattr(self, f"frm_bottom_{color}"), text=f'Publish {color}',
                                    command=lambda: self.open_publish_win(color))
        publish_button.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

    def open_publish_win(self, color):
        slm = getattr(self, f'SLM_{color}')
        ent_scr = getattr(self, f'ent_scr_{color}')
        screen_num = int(ent_scr.get())
        slm.phase_map = self.get_phase(color)
        self.update_phase_plot(slm.phase_map - slm.background_phase, color)
        slm.publish(screen_num)

    def close_publish_win(self):
        for color in ['green', 'red']:
            slm = getattr(self, f'SLM_{color}')
            slm.close()

    def setup_frames(self, color):
        setattr(self, f"frm_top_{color}", ttk.LabelFrame(self.main_win, text=f'{color.capitalize()} SLM interface'))
        setattr(self, f"frm_top_b_{color}",
                ttk.LabelFrame(getattr(self, f"frm_top_{color}"), text=f'{color.capitalize()} SLM - Phase display'))
        setattr(self, f"frm_mid_{color}", ttk.Notebook(self.main_win, style='lefttab.TNotebook'))
        setattr(self, f"frm_bottom_{color}", ttk.LabelFrame(self.main_win, text=f'{color.capitalize()} SLM - Options'))

    def create_side_panel(self):
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

    def setup_save_load_buttons(self, color):
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
        self.setup_box(getattr(self, f"frm_top_{color}"), color)

    def setup_box(self, frm_, color):
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

    def open_hhg_view_win(self):
        from model import HHGView
        self.HHGView_win = HHGView.HHGView(self)

    def open_gas_density_win(self):
        from diagnostic_board.GasDensity import GasDensity
        self.GasDensity_win = GasDensity.GasDensity()

    def open_focus_view_win(self):
        from diagnostic_board.FocusView import FocusView
        self.FocusView_win = FocusView.FocusView()

    def setup_phase_display(self, color):
        figure = Figure(figsize=(3, 2))
        ax = figure.add_subplot(111)
        ax.figure.tight_layout()

        setattr(self, f"fig_{color}", figure)
        setattr(self, f"ax_{color}", ax)

        canvas = FigureCanvasTkAgg(figure, getattr(self, f"frm_top_b_{color}"))
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=2, sticky='ew')

        setattr(self, f"img_{color}", canvas)

    def get_phase(self, color):
        slm = getattr(self, f'SLM_{color}')
        phase = np.zeros(slm_size)
        active_phase_types = []

        phase_refs = getattr(self, f'phase_refs_{color}')
        vars_color = getattr(self, f'vars_{color}')

        for ind, phase_type in enumerate(phase_refs):
            if vars_color[ind].get() == 1:
                active_phase_types.append(phase_type.__class__.__name__)
                phase += phase_type.phase()

                if phase_type.__class__.__name__ == 'TypeBackground':
                    slm.background_phase = phase_type.phase()

        print(f"Active phase(s) on the {color} SLM: {', '.join(active_phase_types)}")
        return phase

    def update_phase_plot(self, phase, color):

        if color == 'green':
            ax = self.ax_green
            img = self.img_green
        elif color == 'red':
            ax = self.ax_red
            img = self.img_red

        ax.clear()
        cax = ax.imshow(phase, cmap='hsv', interpolation='None', extent=(
            -chip_width *1e3/ 2, chip_width*1e3 / 2,
            -chip_height*1e3 / 2, chip_width *1e3/ 2))
        ax.set_xlabel('y (mm)')
        ax.set_ylabel('x (mm)')
        ax.figure.tight_layout()
        img.draw()

    def save_settings(self, color, filepath=None):

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

    def hide_show_green_panel(self):
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

    def exit_prog(self):
        self.close_publish_win()
        self.main_win.destroy()


root = tk.Tk()
main = DLabController(root)
root.mainloop()
