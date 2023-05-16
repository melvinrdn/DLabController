import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import matplotlib.image as mpimg
from ressources.settings import slm_size, bit_depth, chip_width, chip_height, wavelength
import model.hologram_generation as gs
import model.aberration_correction as aberration

types = ['Backgr', 'Flat', 'Tilt', 'Binary', 'Lens',
         'Multi', 'Vortex', 'Zernike', 'Image', 'Holo', 'Aberr']


def new_type(frm_mid, typ):
    """
    Factory function for creating different types of phase.

    Parameters:
    -----------
    frm_mid : numpy.ndarray
        The input numpy array.
    typ : str
        The type of numpy array to be created. Possible values are:
        'Flat', 'Tilt', 'Binary', 'Background', 'Lens', 'Multi', 'Vortex', 'Zernike', 'Image', 'Hologram'.

    Returns:
    --------
    numpy.ndarray
        The numpy array of the specified type.
    """
    if typ == 'Flat':
        return TypeFlat(frm_mid)
    elif typ == 'Tilt':
        return TypeTilt(frm_mid)
    elif typ == 'Binary':
        return TypeBinary(frm_mid)
    elif typ == 'Backgr':
        return TypeBackground(frm_mid)
    elif typ == 'Lens':
        return TypeLens(frm_mid)
    elif typ == 'Multi':
        return TypeMultibeam(frm_mid)
    elif typ == 'Vortex':
        return TypeVortex(frm_mid)
    elif typ == 'Zernike':
        return TypeZernike(frm_mid)
    elif typ == 'Image':
        return TypeImage(frm_mid)
    elif typ == 'Holo':
        return TypeHologram(frm_mid)
    elif typ == 'Aberr':
        return TypeAberration(frm_mid)


class BaseType(object):
    """
    Base class for all TypePhase classes.

    Attributes:
    -----------
    img : ndarray
        The image data loaded from the file.
    """

    def _read_file(self, filepath):
        """
        Load image data from file.

        Parameters:
        -----------
        filepath : str
            The path to the file to be loaded.

        """
        # Check if filepath is provided
        if not filepath:
            return  # Return None if filepath is not provided

        try:
            # Check if the file is a csv file
            if filepath[-4:] == '.csv':
                try:
                    # If it is a csv file, load it using numpy loadtxt function
                    # Skip the first row (header) and read all columns except the first column
                    self.img = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=np.arange(1920) + 1)
                except:
                    # If there's an error in loading the csv file, try loading it without skipping the header row
                    self.img = np.loadtxt(filepath, delimiter=',')
            else:
                # If the file is not a csv file, load it using matplotlib imread function
                self.img = mpimg.imread(filepath)
                if len(self.img.shape) == 3:  # Check if the image has 3 dimensions (multicolor image)
                    # If the image has 3 dimensions, convert it to grayscale by summing the values across the color
                    # channels
                    self.img = self.img.sum(axis=2)
        except:
            # If there's an error in loading the file, print an error message
            print('File "' + filepath + '" not found')

    def open_file(self):
        """
        Open a file dialog to choose a file and load its data.
        """
        filepath = askopenfilename(
            filetypes=[('CSV data arrays', '*.csv'), ('Image Files', '*.bmp'),
                       ('All Files', '*.*')]
        )
        self._read_file(filepath)
        self.lbl_file['text'] = f'{filepath}'

    def callback(self, action, P, text):
        """
        Callback function for text input validation.

        Parameters:
        -----------
        action : str
            The action to be performed on the text (e.g. insert, delete).
        P : str
            The current value of the text.
        text : str
            The new text to be inserted.

        Returns:
        --------
        bool
            True if the text is valid, False otherwise.
        """
        # action=1 -> insert
        if action == '1':
            if text in '0123456789.-+':
                try:
                    float(P)
                    return True
                except ValueError:
                    return False
            else:
                return False
        else:
            return True

    def name_(self):
        """
        Get the name of the object.

        Returns:
        --------
        str
            The name of the object.
        """
        return self.name

    def close_(self):
        """
        Close the object's frame.
        """
        self.frm_.destroy()


class TypeBackground(BaseType):
    """
    A class for managing the background settings for a phase calculation.

    Parameters
    ----------
    parent : Tk object
        The parent window for the frame.

    Attributes
    ----------
    name : str
        The name of the background type.
    frm_ : Tk frame object
        The frame that contains the background settings.
    lbl_file : Tk label object
        The label that displays the selected file.
    """

    def __init__(self, parent):
        """
        Initialize the TypeBackground class.

        Parameters
        ----------
        parent : Tk object
            The parent window for the frame.

        """
        self.name = 'Backgr'
        self.parent = parent
        self.frm_ = ttk.Frame(self.parent)
        self.frm_.grid(row=0, column=0, sticky='nsew')

        # Create a label frame for the background settings
        lbl_frm = ttk.LabelFrame(self.frm_, text='Background')
        lbl_frm.grid(row=0, column=0, sticky='ew')

        # Create a button for opening the background file
        btn_open = ttk.Button(lbl_frm, text='Open Background file',
                             command=self.open_file)
        btn_open.grid(row=0)

        # Create a label for displaying the selected file
        self.lbl_file = ttk.Label(lbl_frm, text='', wraplength=400,
                                 justify='left', foreground='gray')
        self.lbl_file.grid(row=1)

    def phase(self):
        """
        Return the phase data based on the selected background file.

        Returns
        -------
        phase : numpy array
            The phase data.

        """
        if self.img is not None:
            phase = self.img
        else:
            phase = np.zeros(slm_size)
        return phase

    def save_(self):
        """
        Save the current state of the TypeBackground object.

        Returns
        -------
        dict : dict
            A dictionary of the current state.

        """
        dict = {'filepath': self.lbl_file['text']}
        return dict

    def load_(self, dict):
        """
        Load a saved state for the TypeBackground object.

        Parameters
        ----------
        dict : dict
            A dictionary of the saved state.

        """
        self.lbl_file['text'] = dict['filepath']
        self._read_file(dict['filepath'])


class TypeFlat(BaseType):
    """
    A class for managing the flat settings for a phase calculation.

    Parameters
    ----------
    parent : Tk object
        The parent window for the frame.

    Attributes
    ----------
    name : str
        The name of the flat type.
    frm_ : Tk frame object
        The frame that contains the flat settings.
    ent_flat : Tk entry object
        The entry box for the phase shift value.

    """

    def __init__(self, parent):
        """
        Initialize the TypeFlat class.

        Parameters
        ----------
        parent : Tk object
            The parent window for the frame.

        """
        self.name = 'Flat'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=1, column=0, sticky='nsew')

        # Create a label frame for the flat settings
        lbl_frm = ttk.LabelFrame(self.frm_, text='Flat')
        lbl_frm.grid(row=0, column=0, sticky='ew')

        # Create a label and entry box for the phase shift value
        lbl_phi = ttk.Label(lbl_frm,
                           text='Phase shift (' + str(bit_depth) + '=2pi):')
        vcmd = (parent.register(self.callback))
        self.strvar_flat = tk.StringVar()
        self.ent_flat = tk.Entry(
            lbl_frm, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_flat)
        lbl_phi.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=5)
        self.ent_flat.grid(row=0, column=1, sticky='w', padx=(0, 10))

    def phase(self):
        """
        Return the phase data based on the selected flat settings.

        Returns
        -------
        phase : numpy array
            The phase data.

        """
        if self.ent_flat.get() != '':
            phi = float(self.ent_flat.get())
        else:
            phi = 0
        phase = np.ones(slm_size) * phi
        return phase

    def save_(self):
        """
        Save the current state of the TypeFlat object.

        Returns
        -------
        dict : dict
            A dictionary of the current state.

        """
        dict = {'flat_phase': self.ent_flat.get()}
        return dict

    def load_(self, dict):
        """
        Load a saved state for the TypeFlat object.

        Parameters
        ----------
        dict : dict
            A dictionary of the saved state.

        """
        self.strvar_flat.set(dict['flat_phase'])


class TypeTilt(BaseType):
    """
    A class to set the parameters for a tilted phase.

    Attributes
    ----------
    name : str
        The name of the type of tilt.
    frm_ : tkinter Frame
        A tkinter Frame object.
    strvar_xdir : tkinter StringVar
        A tkinter variable for the steepness along x-direction.
    strvar_ydir : tkinter StringVar
        A tkinter variable for the steepness along y-direction.
    strvar_tstep : tkinter StringVar
        A tkinter variable for the step per click.
    ent_xdir : tkinter Entry
        A tkinter Entry object for the steepness along x-direction.
    ent_ydir : tkinter Entry
        A tkinter Entry object for the steepness along y-direction.
    ent_tstep : tkinter Entry
        A tkinter Entry object for the step per click.
    """

    def __init__(self, parent):
        """
        Initialize the TypeTilt class.

        Parameters
        ----------
        parent : Tk object
            The parent window for the frame.

        """
        self.name = 'Tilt'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=2, column=0, sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Tilt')
        lbl_frm.grid(row=0, column=0, sticky='ew', padx=5, pady=10)

        # Creating objects
        lbl_xdir = ttk.Label(lbl_frm, text='Steepness along x-direction:')
        lbl_ydir = ttk.Label(lbl_frm, text='Steepness along y-direction:')
        lbl_bit = ttk.Label(lbl_frm,
                           text='(' + str(bit_depth) + ' corresponds to 2pi Rad)')
        lbl_step = ttk.Label(lbl_frm, text='(wasd) Step per click:')
        vcmd = (parent.register(self.callback))
        self.strvar_xdir = tk.StringVar()
        self.strvar_ydir = tk.StringVar()
        self.ent_xdir = tk.Entry(
            lbl_frm, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_xdir)
        self.ent_ydir = tk.Entry(
            lbl_frm, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_ydir)
        self.strvar_tstep = tk.StringVar()
        self.ent_tstep = tk.Entry(
            lbl_frm, width=11, validate='all',
            validatecommand=(vcmd, '%d', '%P', '%S'),
            textvariable=self.strvar_tstep)

        # Setting up
        lbl_xdir.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=5)
        lbl_ydir.grid(row=1, column=0, sticky='e', padx=(10, 0), pady=(0, 5))
        lbl_bit.grid(row=2, sticky='ew', padx=(10, 10), pady=(0, 5))
        self.ent_xdir.grid(row=0, column=1, sticky='w', padx=(0, 10))
        self.ent_ydir.grid(row=1, column=1, sticky='w', padx=(0, 10))
        lbl_step.grid(row=3, column=0, sticky='e', padx=(10, 0), pady=(0, 5))
        self.ent_tstep.grid(row=3, column=1, sticky='w', padx=(0, 10))

    def phase(self):
        """
        Return the phase data based on the selected tilt settings.

        Returns
        -------
        phase : numpy array
            The phase data.

        """
        x_dir = self.ent_xdir.get()
        y_dir = self.ent_ydir.get()

        if y_dir != '' and float(y_dir) != 0:
            lim = np.ones(slm_size[1]) * float(y_dir) * (slm_size[1] - 1) / 2
            phy = np.linspace(-lim, +lim, slm_size[0], axis=0)
        else:
            phy = np.zeros(slm_size)

        if x_dir != '' and float(x_dir) != 0:
            lim = np.ones(slm_size[0]) * float(x_dir) * (slm_size[0] - 1) / 2
            phx = np.linspace(-lim, +lim, slm_size[1], axis=1)
        else:
            phx = np.zeros(slm_size)

        return phx + phy

    def left_(self):
        """
        Increase the value of `strvar_xdir` by the value of `strvar_tstep`.
        """
        tmp = float(self.strvar_xdir.get()) + float(self.strvar_tstep.get())
        self.strvar_xdir.set(tmp)

    def right_(self):
        """
        Decreases the value of `strvar_xdir` by the value of `strvar_tstep`.
        """
        tmp = float(self.strvar_xdir.get()) - float(self.strvar_tstep.get())
        self.strvar_xdir.set(tmp)

    def up_(self):
        """
        Increase the value of `strvar_ydir` by the value of `strvar_tstep`.
        """
        tmp = float(self.strvar_ydir.get()) + float(self.strvar_tstep.get())
        self.strvar_ydir.set(tmp)

    def down_(self):
        """
        Decreases the value of `strvar_ydir` by the value of `strvar_tstep`.
        """
        tmp = float(self.strvar_ydir.get()) - float(self.strvar_tstep.get())
        self.strvar_ydir.set(tmp)

    def save_(self):
        """
        Save the current state of the TypeTilt object.

        Returns
        -------
        dict : dict
            A dictionary of the current state.

        """
        dict = {'ent_xdir': self.ent_xdir.get(),
                'ent_ydir': self.ent_ydir.get()}
        return dict

    def load_(self, dict):
        """
        Load a saved state for the TypeTilt object.

        Parameters
        ----------
        dict : dict
            A dictionary of the saved state.

        """
        self.strvar_xdir.set(dict['ent_xdir'])
        self.strvar_ydir.set(dict['ent_ydir'])


class TypeBinary(BaseType):
    """
    A class that represents binary settings for phase.

    Attributes
    ----------
    name : str
        The name of the class.
    frm_ : Tkinter Frame
        The frame object representing the binary settings for phase.
    cbx_dir : ttk.Combobox
        The combobox object for selecting the split direction.
    ent_area : Tkinter Spinbox
        The spinbox object for entering the area amount in percentage.
    ent_phi : Tkinter Entry
        The entry object for entering the phase change in pi.
    """

    def __init__(self, parent):
        """
        Initialize the TypeBinary class.

        Parameters
        ----------
        parent : Tk object
            The parent window for the frame.

        """
        self.name = 'Binary'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=3, column=0, sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Binary')
        lbl_frm.grid(row=0, column=0, sticky='ew', padx=5, pady=10)

        # Creating entities
        lbl_dir = ttk.Label(lbl_frm, text='Direction for split:')
        lbl_rat = ttk.Label(lbl_frm, text='Area amount (in %):')
        lbl_phi = ttk.Label(lbl_frm, text='Phase change (in pi):')
        self.cbx_dir = ttk.Combobox(
            lbl_frm,
            values=['Horizontal', 'Vertical'],
            state='readonly',
            width=10)
        self.ent_area = ttk.Spinbox(lbl_frm, width=12, from_=0, to=100)
        vcmd = (parent.register(self.callback))
        self.strvar_phi = tk.StringVar()
        self.ent_phi = ttk.Entry(lbl_frm, width=12, validate='all',
                                validatecommand=(vcmd, '%d', '%P', '%S'),
                                textvariable=self.strvar_phi)

        # Setting up
        lbl_dir.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=5)
        lbl_rat.grid(row=1, column=0, sticky='e', padx=(10, 0))
        lbl_phi.grid(row=2, column=0, sticky='e', padx=(10, 0), pady=5)
        self.cbx_dir.grid(row=0, column=1, sticky='w', padx=(0, 10))
        self.ent_area.grid(row=1, column=1, sticky='w', padx=(0, 10))
        self.ent_phi.grid(row=2, column=1, sticky='w', padx=(0, 10))

    def phase(self):
        """
        Generate the phase matrix based on the binary settings for the phase.

        Returns
        -------
        numpy.ndarray
            The phase matrix generated based on the binary settings.
        """
        direc = self.cbx_dir.get()
        if self.ent_area.get() != '':
            area = float(self.ent_area.get())
        else:
            area = 0
        if self.ent_phi.get() != '':
            tmp = float(self.ent_phi.get())
            phi = tmp * bit_depth / 2  # Converting to 0-2pi
        else:
            phi = 0

        phase_mat = np.zeros(slm_size)

        if direc == 'Horizontal':
            cutpixel = int(round(slm_size[0] * area / 100))
            tmp = np.ones([cutpixel, slm_size[1]]) * phi
            phase_mat[0:cutpixel, :] = tmp
        elif direc == 'Vertical':
            cutpixel = int(round(slm_size[1] * area / 100))
            tmp = np.ones([slm_size[0], cutpixel]) * phi
            phase_mat[:, 0:cutpixel] = tmp
        del tmp
        return phase_mat

    def save_(self):
        """
        Save the current state of the TypeBinary object.

        Returns
        -------
        dict : dict
            A dictionary of the current state.
        """
        dict = {'direc': self.cbx_dir.get(),
                'area': self.ent_area.get(),
                'phi': self.ent_phi.get()}
        return dict

    def load_(self, dict):
        """
        Load a saved state for the TypeBinary object.

        Parameters
        ----------
        dict : dict
            A dictionary of the saved state.
        """
        if dict['direc'] != 'Vertical' and dict['direc'] != 'Horizontal':
            dict['direc'] = 'Vertical'
        tmpind = self.cbx_dir['values'].index(dict['direc'])
        self.cbx_dir.current(tmpind)
        self.ent_area.delete(0, tk.END)
        self.ent_area.insert(0, dict['area'])
        self.strvar_phi.set(dict['phi'])


class TypeLens(BaseType):
    """
    A class that represents the lens settings for phase.

    Attributes
    ----------
    name : str
        The name of the class, which is "Lens".
    frm_ : tk.Frame
        A tkinter frame object.
    strvar_ben : tk.StringVar
        A tkinter variable that stores the bending strength.
    ent_ben : tk.Entry
        A tkinter entry object for bending strength input.
    """

    def __init__(self, parent):
        """
        Initialize the TypeLens class.

        Parameters
        ----------
        parent : Tk object
            The parent window for the frame.

        """
        self.name = 'Lens'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=4, column=0, sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Lens')
        lbl_frm.grid(row=0, column=0, sticky='ew')

        # creating labels
        lbl_ben = ttk.Label(lbl_frm, text='Bending strength (1/f) [1/m]:')

        # creating entries
        vcmd = (parent.register(self.callback))
        self.strvar_ben = tk.StringVar()
        self.ent_ben = ttk.Entry(lbl_frm, width=5, validate='all', validatecommand=(vcmd, '%d', '%P', '%S'),
                                textvariable=self.strvar_ben)

        # setup
        lbl_ben.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=5)
        self.ent_ben.grid(row=0, column=1, sticky='w', padx=(0, 10))

    def phase(self):
        """
        Returns the phase calculated via the hyperbolic curves.

        Returns
        -------
        Z_phi : numpy.ndarray
            An array that contains the calculated phase.
        """
        if self.ent_ben.get() != '':
            ben = float(self.ent_ben.get())
        else:
            ben = 0

        rad_sign = np.sign(ben)
        rad = 2 / np.abs(ben)  # R=2*f
        x = np.linspace(-chip_width / 2, chip_width / 2, slm_size[1])
        y = np.linspace(-chip_height / 2, chip_height / 2, slm_size[0])
        [X, Y] = np.meshgrid(x, y)
        R = np.sqrt(X ** 2 + Y ** 2)  # radius on a 2d array
        Z = rad_sign * (np.sqrt(rad ** 2 + R ** 2) - rad)
        Z_phi = Z / wavelength * bit_depth  # translating meters to wavelengths and phase
        del X, Y, R, Z

        return Z_phi

    def save_(self):
        """
        Save the current state of the TypeLens object.

        Returns
        -------
        dict : dict
            A dictionary of the current state.
        """
        dict = {'ben': self.ent_ben.get()}
        return dict

    def load_(self, dict):
        """
        Load a saved state for the TypeLens object.

        Parameters
        ----------
        dict : dict
            A dictionary of the saved state.
        """
        self.strvar_ben.set(dict['ben'])


class TypeMultibeam(BaseType):
    """
    Represents multibeam checkerboard settings for phase.

    Attributes
    ----------
    name : str
        The name of the multibeam setting.
    frm_ : tkinter.Frame
        The parent frame for the multibeam setting.
    strvar_n : tkinter.StringVar
        A string variable for the 'n' entry field.
    ent_n : tkinter.Entry
        The 'n' entry field.
    strvar_hpt : tkinter.StringVar
        A string variable for the horizontal spread entry field.
    ent_hpt : tkinter.Entry
        The horizontal spread entry field.
    strvar_vpt : tkinter.StringVar
        A string variable for the vertical spread entry field.
    ent_vpt : tkinter.Entry
        The vertical spread entry field.
    strvar_rad : tkinter.StringVar
        A string variable for the phase entry field.
    ent_rad : tkinter.Entry
        The phase entry field.
    strvar_amp : tkinter.StringVar
        A string variable for the beam selection entry field.
    ent_amp : tkinter.Entry
        The beam selection entry field.
    strvar_hit : tkinter.StringVar
        A string variable for the horizontal intensity tilt entry field.
    ent_hit : tkinter.Entry
        The horizontal intensity tilt entry field.
    strvar_vit : tkinter.StringVar
        A string variable for the vertical intensity tilt entry field.
    ent_vit : tkinter.Entry
        The vertical intensity tilt entry field.
    strvar_his : tkinter.StringVar
        A string variable for the horizontal intensity curve entry field.
    ent_his : tkinter.Entry
        The horizontal intensity curve entry field.
    strvar_vis : tkinter.StringVar
        A string variable for the vertical intensity curve entry field.
    ent_vis : tkinter.Entry
        The vertical intensity curve entry field.
    strvar_pxsiz : tkinter.StringVar
        A string variable for the pixel size entry field.
    ent_pxsiz : tkinter.Entry
        The pixel size entry field.
    """

    def __init__(self, parent):
        """
        Initialize the TypeMultibeam class.

        Parameters
        ----------
        parent : Tk object
            The parent window for the frame.

        """
        self.name = 'Multibeam'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=5, column=0, sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Multibeam')
        lbl_frm.grid(row=0, column=0, sticky='ew')

        # creating frames
        frm_n = ttk.Frame(lbl_frm)
        frm_sprrad = ttk.Frame(lbl_frm)
        frm_spr = ttk.Frame(frm_sprrad)
        frm_rad = ttk.Frame(frm_sprrad)
        frm_int = ttk.Frame(lbl_frm)
        frm_pxsiz = ttk.Frame(lbl_frm)

        # creating labels
        lbl_n = ttk.Label(frm_n, text='n^2; n=:')
        lbl_hor = ttk.Label(frm_int, text='Hor:')
        lbl_vert = ttk.Label(frm_int, text='Vert:')
        lbl_intil = ttk.Label(frm_int, text='Intensity tilt')
        lbl_insqr = ttk.Label(frm_int, text='Intensity curve')
        lbl_horspr = ttk.Label(frm_spr, text='Horizontal spread:')
        lbl_verspr = ttk.Label(frm_spr, text='Vertical spread:')
        lbl_cph = ttk.Label(frm_sprrad, text='Hyp.phase diff')
        lbl_rad = ttk.Label(frm_rad, text='Phase[' + str(bit_depth) + ']:')
        lbl_amp = ttk.Label(frm_rad, text='Choose beam:')
        lbl_pxsiz = ttk.Label(frm_pxsiz, text='pixel size:')

        # creating entries
        vcmd = (parent.register(self.callback))
        self.strvar_n = tk.StringVar()
        self.ent_n = ttk.Entry(frm_n, width=5, validate='all',
                              validatecommand=(vcmd, '%d', '%P', '%S'),
                              textvariable=self.strvar_n)
        self.strvar_hpt = tk.StringVar()
        self.ent_hpt = ttk.Entry(frm_spr, width=5, validate='all',
                                validatecommand=(vcmd, '%d', '%P', '%S'),
                                textvariable=self.strvar_hpt)
        self.strvar_vpt = tk.StringVar()
        self.ent_vpt = ttk.Entry(frm_spr, width=5, validate='all',
                                validatecommand=(vcmd, '%d', '%P', '%S'),
                                textvariable=self.strvar_vpt)
        self.strvar_rad = tk.StringVar()
        self.ent_rad = ttk.Entry(frm_rad, width=5, validate='all',
                                validatecommand=(vcmd, '%d', '%P', '%S'),
                                textvariable=self.strvar_rad)
        self.strvar_amp = tk.StringVar()
        self.ent_amp = ttk.Entry(frm_rad, width=5, validate='all',
                                validatecommand=(vcmd, '%d', '%P', '%S'),
                                textvariable=self.strvar_amp)
        self.strvar_hit = tk.StringVar()
        self.ent_hit = ttk.Entry(frm_int, width=5, validate='all',
                                validatecommand=(vcmd, '%d', '%P', '%S'),
                                textvariable=self.strvar_hit)
        self.strvar_vit = tk.StringVar()
        self.ent_vit = ttk.Entry(frm_int, width=5, validate='all',
                                validatecommand=(vcmd, '%d', '%P', '%S'),
                                textvariable=self.strvar_vit)
        self.strvar_his = tk.StringVar()
        self.ent_his = ttk.Entry(frm_int, width=5, validate='all',
                                validatecommand=(vcmd, '%d', '%P', '%S'),
                                textvariable=self.strvar_his)
        self.strvar_vis = tk.StringVar()
        self.ent_vis = ttk.Entry(frm_int, width=5, validate='all',
                                validatecommand=(vcmd, '%d', '%P', '%S'),
                                textvariable=self.strvar_vis)
        self.strvar_pxsiz = tk.StringVar()
        self.ent_pxsiz = ttk.Entry(frm_pxsiz, width=5, validate='all',
                                  validatecommand=(vcmd, '%d', '%P', '%S'),
                                  textvariable=self.strvar_pxsiz)

        # setup
        frm_n.grid(row=0, sticky='nsew')
        frm_sprrad.grid(row=1, sticky='nsew')
        frm_int.grid(row=2, sticky='nsew')
        frm_pxsiz.grid(row=3, sticky='nsew')

        frm_spr.grid(row=1, column=0, sticky='nsew')
        frm_rad.grid(row=1, column=1, sticky='ew')

        lbl_n.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=(5, 10))
        self.ent_n.grid(row=0, column=1, sticky='w',
                        padx=(0, 10), pady=(5, 10))

        lbl_horspr.grid(row=0, column=0, sticky='e', padx=(10, 0))
        lbl_verspr.grid(row=1, column=0, sticky='e', padx=(10, 0))
        self.ent_hpt.grid(row=0, column=1, sticky='w')
        self.ent_vpt.grid(row=1, column=1, sticky='w')

        lbl_cph.grid(row=0, column=1, sticky='ew')

        lbl_rad.grid(row=0, column=0, sticky='e', padx=(15, 0))
        lbl_amp.grid(row=1, column=0, sticky='e', padx=(15, 0))
        self.ent_rad.grid(row=0, column=1, sticky='w', padx=(0, 5))
        self.ent_amp.grid(row=1, column=1, sticky='w', padx=(0, 5))

        lbl_hor.grid(row=1, column=0, sticky='e', padx=(10, 0))
        lbl_vert.grid(row=2, column=0, sticky='e', padx=(10, 0))
        lbl_intil.grid(row=0, column=1, padx=5, pady=(10, 0))
        lbl_insqr.grid(row=0, column=2, padx=(0, 5), pady=(10, 0))
        self.ent_hit.grid(row=1, column=1)
        self.ent_his.grid(row=1, column=2, padx=(0, 5))
        self.ent_vit.grid(row=2, column=1, pady=(0, 5))
        self.ent_vis.grid(row=2, column=2, padx=(0, 5), pady=(0, 5))

        lbl_pxsiz.grid(row=0, column=0, sticky='e', padx=(10, 0), pady=5)
        self.ent_pxsiz.grid(row=0, column=1, sticky='w')

    def phase(self):
        """
        Computes the phase for the multibeam class.

        Returns
        -------
        numpy.ndarray
            Array of phase values computed.
        """

        if self.ent_n.get() != '':
            n = int(self.ent_n.get())
        else:
            n = 1

        # getting the different phases for the beams
        if self.ent_hpt.get() != '':
            xtilt = float(self.ent_hpt.get())
        else:
            xtilt = 0
        if self.ent_vpt.get() != '':
            ytilt = float(self.ent_vpt.get())
        else:
            ytilt = 0
        tilts = np.arange(-n + 1, n + 1, 2)  # excluding the last
        xtilts = tilts * xtilt / 2
        ytilts = tilts * ytilt / 2
        phases = np.zeros([slm_size[0], slm_size[1], n * n])
        ind = 0
        for xdir in xtilts:
            for ydir in ytilts:
                phases[:, :, ind] = self.phase_tilt(xdir, ydir)
                ind += 1

        # getting the hyperbolical curve on the phases
        if self.ent_rad.get() != '':
            tmprad = float(self.ent_rad.get())
        else:
            tmprad = 0
        if self.ent_amp.get() != '':
            amp = int(self.ent_amp.get())
        else:
            amp = 0
        if tmprad != 0:
            phases[:, :, amp] = tmprad + phases[:, :, amp]
            phases[:, :, amp + 1] = tmprad + phases[:, :, amp + 1]

        # setting up for intensity control
        if self.ent_hit.get() != '':
            xit = float(self.ent_hit.get())
        else:
            xit = 0
        if self.ent_vit.get() != '':
            yit = float(self.ent_vit.get())
        else:
            yit = 0
        if self.ent_his.get() != '':
            xis = float(self.ent_his.get())
        else:
            xis = 0
        if self.ent_vis.get() != '':
            yis = float(self.ent_vis.get())
        else:
            yis = 0
        intensities = np.ones(n ** 2)
        totnum = np.ceil((slm_size[0] * (slm_size[1] + n) / (n ** 2)))  # nbr of pixels for each phase
        #          plus n in second dimension to account for noneven placement
        phase_nbr = np.outer(np.arange(n ** 2), np.ones([int(totnum)]))

        # modifying linear intensities
        if xit != 0:
            xits = np.linspace(-((n - 1) / 2 * xit), ((n - 1) / 2 * xit), num=n)
        else:
            xits = np.zeros(n)
        if yit != 0:
            yits = np.linspace(-((n - 1) / 2 * yit), ((n - 1) / 2 * yit), num=n)
        else:
            yits = np.zeros(n)
        ii = 0
        for tmpx in xits:
            intensities[n * ii:n * (ii + 1)] = (tmpx + yits + 1)
            ii += 1

        # modifying square intensities
        spread = tilts
        xiss = -xis * (spread ** 2 - spread[0] ** 2)
        yiss = -yis * (spread ** 2 - spread[0] ** 2)
        ii = 0
        for tmpx in xiss:
            intensities[n * ii:n * (ii + 1)] += (tmpx + yiss + 1)
            ii += 1

        # creating the intensity arrays (which phase to have at which pixel)
        intensities[intensities < 0] = 0
        intensities = intensities / np.sum(intensities) * n ** 2  # normalize
        tmpint = intensities - 1
        beam = 0
        strong_beams = []
        strong_beams_int = []
        weak_beams = []
        weak_beams_int = []
        for intens in tmpint:
            if intens > 0:
                strong_beams.append(beam)
                strong_beams_int.append(intens)
            elif intens < 0:
                weak_beams.append(beam)
                weak_beams_int.append(intens)
            beam += 1

        # normalize to one
        strong_beams_int = strong_beams_int / np.sum(strong_beams_int)
        for wbeam, wbeam_int in zip(weak_beams, weak_beams_int):
            nbr_pixels = np.ceil(np.abs(wbeam_int) * totnum)  # nbrpxls to change
            nbr_each = np.ceil(nbr_pixels * strong_beams_int)
            strt = 0
            for nbr, sbeam in zip(nbr_each, strong_beams):
                phase_nbr[int(wbeam), strt:(strt + int(nbr))] = sbeam
                strt += int(nbr)
        rng = np.random.default_rng()
        rng.shuffle(phase_nbr, axis=1)  # mixing so the changed are not together
        # tic6 = time.perf_counter()
        if self.ent_pxsiz.get() != '':
            pxsiz = int(self.ent_pxsiz.get())
        else:
            pxsiz = 1

        # creating the total phase by adding the different ones
        xrange = np.arange(0, slm_size[1], 1)
        yrange = np.arange(0, slm_size[0], 1)
        tot_phase = np.zeros(slm_size)
        [X, Y] = np.meshgrid(xrange, yrange)
        ind_phase_tmp = (np.floor(X / pxsiz) % n) * n + (np.floor(Y / pxsiz) % n)
        ind_phase = ind_phase_tmp.astype(int)

        ind_phase2 = ind_phase.copy()
        for ii in range(n ** 2):
            max_nbr = np.count_nonzero(ind_phase == ii)
            if max_nbr <= phase_nbr[0, :].size:
                ind_phase2[ind_phase == ii] = phase_nbr[ii, 0:max_nbr]
            else:
                extra = ii * np.ones([max_nbr - phase_nbr[0, :].size])
                ind_phase2[ind_phase == ii] = np.append(phase_nbr[ii, :], extra)

        for ii in range(n ** 2):
            ii_phase = phases[:, :, ii]
            tot_phase[ind_phase2 == ii] = ii_phase[ind_phase2 == ii]

        return tot_phase

    def phase_tilt(self, xdir, ydir):
        """
        Calculate the phase shift due to tilting.

        Parameters
        ----------
        xdir : str
            The tilt in the x-direction in micrometers.
        ydir : str
            The tilt in the y-direction in micrometers.

        Returns
        -------
        numpy.ndarray
            The phase shift due to tilting.
        """
        if xdir != '' and float(xdir) != 0:
            lim = np.ones(slm_size[0]) * float(xdir) * (slm_size[0] - 1) / 2
            phx = np.linspace(-lim, +lim, slm_size[1], axis=1)
        else:
            phx = np.zeros(slm_size)

        if ydir != '' and float(ydir) != 0:
            lim = np.ones(slm_size[1]) * float(ydir) * (slm_size[1] - 1) / 2
            phy = np.linspace(-lim, +lim, slm_size[0], axis=0)
        else:
            phy = np.zeros(slm_size)

        return phx + phy

    def save_(self):
        """
        Save the current state of the TypeMultibeam object.

        Returns
        -------
        dict : dict
            A dictionary of the current state.
        """
        dict = {'n': self.ent_n.get(),
                'hpt': self.ent_hpt.get(),
                'rad': self.ent_rad.get(),
                'hit': self.ent_hit.get(),
                'his': self.ent_his.get(),
                'vpt': self.ent_vpt.get(),
                'amp': self.ent_amp.get(),
                'vit': self.ent_vit.get(),
                'vis': self.ent_vis.get(),
                'pxsiz': self.ent_pxsiz.get()}
        return dict

    def load_(self, dict):
        """
        Load a saved state for the TypeMultibeam object.

        Parameters
        ----------
        dict : dict
            A dictionary of the saved state.
        """
        self.strvar_n.set(dict['n'])
        self.strvar_hpt.set(dict['hpt'])
        self.strvar_rad.set(dict['rad'])
        self.strvar_hit.set(dict['hit'])
        self.strvar_his.set(dict['his'])
        self.strvar_vpt.set(dict['vpt'])
        self.strvar_amp.set(dict['amp'])
        self.strvar_vit.set(dict['vit'])
        self.strvar_vis.set(dict['vis'])
        self.strvar_pxsiz.set(dict['pxsiz'])


class TypeVortex(BaseType):
    """shows vortex settings for phase"""

    def __init__(self, parent):
        """
        Initialize the TypeVortex class.

        Parameters
        ----------
        parent : Tk object
            The parent window for the frame.

        """
        self.name = 'Vortex'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=6, column=0, sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Vortex')
        lbl_frm.grid(row=0, column=0, sticky='ew')

        lbl_texts = ['Vortex order:', 'dx [mm]:', 'dy [mm]:']
        labels = [ttk.Label(lbl_frm, text=lbl_text) for lbl_text in lbl_texts]
        vcmd = (parent.register(self.callback))
        self.strvars = [tk.StringVar() for lbl_text in lbl_texts]
        self.entries = [ttk.Entry(lbl_frm, width=11, validate='all',
                                 validatecommand=(vcmd, '%d', '%P', '%S'),
                                 textvariable=strvar)
                        for strvar in self.strvars]
        for ind, label in enumerate(labels):
            label.grid(row=ind, column=0, sticky='e', padx=(10, 0), pady=5)
        for ind, entry in enumerate(self.entries):
            entry.grid(row=ind, column=1, sticky='w', padx=(0, 10))

    def phase(self):
        coeffs = np.zeros(len(self.entries), dtype=float)
        for i, entry in enumerate(self.entries):
            if entry.get() != '':
                coeffs[i] = float(entry.get())
        vor, dx, dy = coeffs
        x = np.linspace(-chip_width * 500 + dx, chip_width * 500 + dx, slm_size[1])
        y = np.linspace(-chip_height * 500 + dy, chip_height * 500 + dy, slm_size[0])
        [X, Y] = np.meshgrid(x, y)
        theta = np.arctan2(Y, X)
        phase = theta * bit_depth / (2 * np.pi) * vor
        return phase

    def save_(self):
        """
        Save the current state of the TypeVortex object.

        Returns
        -------
        dict : dict
            A dictionary of the current state.
        """
        dict = {'vort_ord': self.entries[0].get()}
        return dict

    def load_(self, dict):
        """
        Load a saved state for the TypeVortex object.

        Parameters
        ----------
        dict : dict
            A dictionary of the saved state.
        """
        self.strvars[0].set(dict['vort_ord'])


class TypeZernike(BaseType):
    """
    Class for managing Zernike polynomials settings for a phase calculation.

    Attributes
    ----------
    name : str
        Name of the class.
    frm_ : Tkinter Frame
        The parent window for the frame.
    varnames : list of str
        List of variable names.
    strvars : list of Tkinter StringVar
        List of StringVar objects for labels.
    entries : list of Tkinter Entry
        List of Entry objects for user input.
    """

    def __init__(self, parent):
        """
        Initialize the TypeZernike class.

        Parameters
        ----------
        parent : Tk object
            The parent window for the frame.
        """
        self.name = 'Zernike'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=7, column=0, sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Zernike')
        lbl_frm.grid(row=0, column=0, sticky='ew')

        self.varnames = ['z1coef', 'z2coef', 'z3coef', 'z4coef', 'z5coef',
                         'z6coef', 'z7coef', 'z8coef', 'z9coef', 'z10coef',
                         'zsize', 'zdx', 'zdy']
        lbl_texts = ['Z_00 :', 'Z_11 :', 'Z_-11 :',
                     'Z_02 :', 'Z_22 :', 'Z_-22 :',
                     'Z_13 :', 'Z_-13 :', 'Z_33 :',
                     'Z_-33 :', 'Z size:', 'dx [mm]:', 'dy [mm]:']
        labels = [ttk.Label(lbl_frm, text=lbl_text) for lbl_text in lbl_texts]
        vcmd = (parent.register(self.callback))
        self.strvars = [tk.StringVar() for lbl_text in lbl_texts]
        self.entries = [ttk.Entry(lbl_frm, width=11, validate='all',
                                 validatecommand=(vcmd, '%d', '%P', '%S'),
                                 textvariable=strvar)
                        for strvar in self.strvars]
        for ind, label in enumerate(labels):
            label.grid(row=ind % 10, column=2 * int(ind / 10),
                       sticky='e', padx=(10, 0), pady=5)
        for ind, entry in enumerate(self.entries):
            entry.grid(row=ind % 10, column=2 * int(ind / 10) + 1,
                       sticky='w', padx=(0, 10))

    def phase(self):
        """
        Computes the phase value using Zernike polynomials.

        Returns
        -------
        numpy.ndarray
            Array of phase values computed using Zernike polynomials.
        """
        coeffs = np.zeros(len(self.entries), dtype=float)
        coeffs[10] = 1

        for i, entry in enumerate(self.entries):
            if entry.get() != '':
                coeffs[i] = float(entry.get())
        zsize, zdx, zdy = coeffs[10:]

        x = np.linspace(-chip_width * 500 + zdx, chip_width * 500 + zdx, slm_size[1])
        y = np.linspace(-chip_height * 500 + zdy, chip_height * 500 + zdy, slm_size[0])
        [X, Y] = np.meshgrid(x, y)
        theta = np.arctan2(Y, X)
        rho = np.sqrt(X ** 2 + Y ** 2) / zsize

        p1 = coeffs[0] * 1 * np.cos(0 * theta)
        p2 = coeffs[1] * rho * np.cos(1 * theta)
        p3 = coeffs[2] * rho * np.sin(1 * theta)
        p4 = coeffs[3] * (2 * rho ** 2 - 1) * np.cos(0 * theta)
        p5 = coeffs[4] * rho ** 2 * np.cos(2 * theta)
        p6 = coeffs[5] * rho ** 2 * np.sin(2 * theta)
        p7 = coeffs[6] * (3 * rho ** 3 - 2 * rho) * np.cos(1 * theta)
        p8 = coeffs[7] * (3 * rho ** 3 - 2 * rho) * np.sin(1 * theta)
        p9 = coeffs[8] * rho ** 3 * np.cos(3 * theta)
        p10 = coeffs[9] * rho ** 3 * np.sin(3 * theta)

        phase = (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10)

        return phase

    def save_(self):
        """
        Save the current state of the TypeVortex object.

        Returns
        -------
        dict : dict
            A dictionary of the current state.
        """
        dict = {varname: self.entries[i].get()
                for i, varname in enumerate(self.varnames)}
        return dict

    def load_(self, dict):
        """
        Load a saved state for the TypeZernike object.

        Parameters
        ----------
        dict : dict
            A dictionary of the saved state.
        """
        for i, varname in enumerate(self.varnames):
            self.strvars[i].set(dict[varname])


class TypeImage(TypeBackground):
    """
    A class managing image settings for a phase calculation.

   Attributes
    ----------
    name : str
        The name of the class instance.
    frm_ : Tkinter.Frame
        The Tkinter frame for the class instance.
    lbl_file : Tkinter.Label
        The Tkinter label to display the selected file.
    """

    def __init__(self, parent):
        """
        Initialize the TypeImage class.

        Parameters
        ----------
        parent : Tk object
            The parent window for the frame.

        """
        self.name = 'Image'
        self.frm_ = ttk.Frame(parent)
        self.frm_.grid(row=8, column=0)  # , sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Image')
        lbl_frm.grid(row=0, column=0, sticky='ew')

        btn_open = ttk.Button(lbl_frm, text='Open Phase Profile',
                             command=self.open_file)
        self.lbl_file = ttk.Label(lbl_frm, text='', wraplength=300, justify='left')
        btn_open.grid(row=0)
        self.lbl_file.grid(row=1)


class TypeHologram(BaseType):
    """
    A class managing hologram settings for a phase calculation.

    Attributes
    ----------
    name : str
        Name of the class.
    parent : Tk object
        The parent window for the frame.
    frm_ : tkinter Frame object
        A tkinter frame object representing the hologram.
    gen_win : object or None
        An object representing the generated hologram window.
    img : numpy.ndarray or None
        A numpy array representing the image of the hologram.
    """

    def __init__(self, parent):
        """
        Initialize the Typehologram class.

        Parameters
        ----------
        parent : Tk object
            The parent window for the frame.

        """
        self.name = 'Holo'
        self.parent = parent
        self.frm_ = ttk.Frame(self.parent)
        self.frm_.grid(row=0, column=0, sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Hologram')
        lbl_frm.grid(row=0, column=0, sticky='ew')
        self.gen_win = None
        self.img = None

        btn_open = ttk.Button(lbl_frm, text='Open generated hologram', command=self.open_file)
        self.lbl_file = ttk.Label(lbl_frm, text='', wraplength=400, justify='left', foreground='gray')
        lbl_act_file = ttk.Label(lbl_frm, text='Active Hologram file:', justify='left')
        btn_generate = ttk.Button(lbl_frm, text='Launch Hologram Generator', command=self.open_generator)

        btn_open.grid(row=0)
        lbl_act_file.grid(row=1)
        self.lbl_file.grid(row=2)
        btn_generate.grid(row=3)

    def open_generator(self):
        """
        Open the hologram generator window if it doesn't already exist.

        Returns
        -------
        None
        """
        if self.gen_win is None:
            self.gen_win = gs.GSWindow(self)

    def phase(self):
        """
        Get the current phase data.

        If an image has been loaded, use its data as the phase. Otherwise,
        create a zeros array phase.

        Returns
        -------
        phase : np.ndarray
            The phase data.
        """
        if self.img is not None:
            phase = self.img
        else:
            phase = np.zeros(slm_size)
        return phase

    def save_(self):
        """
        Save the current state of the TypeHologram object.

        Returns
        -------
        dict : dict
            A dictionary of the current state.
        """
        dict = {'filepath': self.lbl_file['text']}
        return dict

    def load_(self, dict):
        """
        Load a saved state for the TypeHologram object.

        Parameters
        ----------
        dict : dict
            A dictionary of the saved state.
        """
        self.lbl_file['text'] = dict['filepath']
        self._read_file(dict['filepath'])


class TypeAberration(BaseType):
    """
    A class managing the automatic correction of the wavefront aberration settings for a phase calculation.

    Attributes
    ----------
    name : str
        Name of the class.
    parent : Tk object
        The parent window for the frame.
    frm_ : tkinter Frame object
        A tkinter frame object representing the Aberration Correction.
    gen_win : object or None
        An object representing the generated Aberration Correction window.
    img : numpy.ndarray or None
        A numpy array representing the image of the aberration correction.
    """

    def __init__(self, parent):
        """
        Initialize the TypeAberration class.

        Parameters
        ----------
        parent : Tk object
            The parent window for the frame.

        """
        self.name = 'Aberr'
        self.parent = parent
        self.frm_ = ttk.Frame(self.parent)
        self.frm_.grid(row=0, column=0, sticky='nsew')
        lbl_frm = ttk.LabelFrame(self.frm_, text='Aberration')
        lbl_frm.grid(row=0, column=0, sticky='ew')
        self.gen_win = None
        self.img = None

        btn_open = ttk.Button(lbl_frm, text='Open generated aberration correction', command=self.open_file)
        self.lbl_file = ttk.Label(lbl_frm, text='', wraplength=400, justify='left', foreground='gray')
        lbl_act_file = ttk.Label(lbl_frm, text='Active aberration correction file:', justify='left')
        btn_generate = ttk.Button(lbl_frm, text='Launch aberration correction generator', command=self.open_generator)

        btn_open.grid(row=0)
        lbl_act_file.grid(row=1)
        self.lbl_file.grid(row=2)
        btn_generate.grid(row=3)

    def open_generator(self):
        """
        Open the aberration correction generator window if it doesn't already exist.

        Returns
        -------
        None
        """
        if self.gen_win is None:
            self.gen_win = aberration.AberrationWindow(self)

    def phase(self):
        """
        Return the phase data based on the selected background file.

        Returns
        -------
        phase : numpy array
            The phase data.

        """
        if self.img is not None:
            phase = self.img
        else:
            phase = np.zeros(slm_size)
        return phase

    def save_(self):
        """
        Save the current state of the TypeAberration object.

        Returns
        -------
        dict : dict
            A dictionary of the current state.
        """
        dict = {'filepath': self.lbl_file['text']}
        return dict

    def load_(self, dict):
        """
        Load a saved state for the TypeAberration object.

        Parameters
        ----------
        dict : dict
            A dictionary of the saved state.
        """
        self.lbl_file['text'] = dict['filepath']
        self._read_file(dict['filepath'])
