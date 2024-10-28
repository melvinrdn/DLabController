import numpy as np
from scipy import ndimage


def get_som(array, dx, center_row='None', center_col='None', centering='None'):
    """
    Calculate the Second orger moment for an array.

    Parameters:
    array (np.ndarray): The input 2D array.
    dx (float): Pixel size in the x direction.
    center_X: The center coordinates (row, col).

    Returns:
    tuple: SOM in x and y directions multiplied by dx, the pixel size.
    """
    center_row, center_col = None, None

    if centering == 'None':
        center_row, center_col = center_row, center_col

    elif centering == 'center_of_mass':
        center_of_mass = ndimage.center_of_mass(array)
        center_col, center_row = int(center_of_mass[1]), int(center_of_mass[0])

    col_deviation = np.arange(array.shape[1]) - center_col
    row_deviation = np.arange(array.shape[0]) - center_row
    col_grid, row_grid = np.meshgrid(col_deviation, row_deviation)

    som_x = np.sum(col_grid ** 2 * array)
    som_y = np.sum(row_grid ** 2 * array)
    pixel_sum = np.sum(array)

    x_som = 2 * np.sqrt(som_x / pixel_sum)
    y_som = 2 * np.sqrt(som_y / pixel_sum)

    return x_som * dx, y_som * dx


def set_outside_circle_to_zero(image, radius, centering='max', manual_center=None):
    """
    Set pixels outside a given radius to zero, with different centering options.

    Parameters:
    image (np.ndarray): The input image array.
    radius (float): The radius within which to keep the pixels.
    centering (str): The centering method ('max', 'center_of_mass', or 'manual').
    manual_center (tuple): Manual center coordinates (col, row) if centering is 'manual'.

    Returns:
    tuple: Cropped image with outside circle set to zero, and the center coordinates (row, col).
    """
    if centering == 'manual':
        if manual_center is not None:
            center_col, center_row = manual_center
        else:
            print('Please define the coordinates for manual mode')
    elif centering == 'max':
        max_coord = np.unravel_index(np.argmax(image), image.shape)
        center_col, center_row = max_coord[1], max_coord[0]
    elif centering == 'center_of_mass':
        center_of_mass = ndimage.center_of_mass(image)
        center_col, center_row = int(center_of_mass[1]), int(center_of_mass[0])
    else:
        raise ValueError("centering must be 'max', 'center_of_mass' or 'manual'")

    radius = int(radius)

    x_start = max(0, center_col - radius)
    x_end = min(image.shape[1], center_col + radius)
    y_start = max(0, center_row - radius)
    y_end = min(image.shape[0], center_row + radius)

    cropped_image = image[y_start:y_end, x_start:x_end]
    nx, ny = cropped_image.shape
    x, y = np.meshgrid(np.arange(ny), np.arange(nx))
    distance_from_center = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)
    cropped_image[distance_from_center > radius] = 0

    return cropped_image, (center_row, center_col)


def process_image(image, N=128, dx=3.45e-6, centering='center_of_mass', manual_center=None, t_coef=0):
    """
    Process a single image by applying a threshold, cropping, normalizing, and calculating the SOM.

    Parameters:
    image (np.ndarray): The input image array.
    N (int): Diameter for cropping the image.
    dx (float): Pixel size in the x direction.
    centering (str): The centering method ('max', 'center_of_mass', or 'manual').
    manual_center (tuple): Manual center coordinates (col, row) if centering is 'manual'.
    t_coef (float): coef for thresholding noise

    Returns:
    tuple: Processed image, SOM in x direction, SOM in y direction.
    """
    threshold = np.max(image) * t_coef
    image[image <= threshold] = 0  # Set pixels below threshold to zero
    new_image, center = set_outside_circle_to_zero(image, N / 2, centering, manual_center)
    new_image /= np.sum(new_image)  # Normalize image

    som_x, som_y = get_som(new_image, dx, center[0], center[1])
    return new_image, som_x, som_y


import numpy as np
from scipy.interpolate import interp1d
import cv2
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap

colors = [
    (1, 1, 1),  # white
    (0, 0, 0.5),  # dark blue
    (0, 0, 1),  # clearer blue
    (0, 1, 1),  # turquoise
    (0, 1, 0),  # green
    (1, 1, 0),  # yellow
    (1, 0.5, 0),  # orange
    (1, 0, 0),   # red
    (0.5, 0, 0)  # darker red
]

custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=512)


me = 9.1e-31
h = 6.62607015e-34
c = 299792458
qe = 1.60217662e-19
lam = 1030e-9
Eq = h * c / lam


def shear_image(image_old, val, axis=0):
    if axis == 0:  # Shear along the x-axis
        T = np.float32([[1, val / 100, 0], [0, 1, 0]])
    elif axis == 1:  # Shear along the y-axis
        T = np.float32([[1, 0, 0], [val / 100, 1, 0]])
    size_T = (image_old.shape[1], image_old.shape[0])
    image_new = cv2.warpAffine(image_old, T, size_T)
    return image_new


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def fit_energy_calibration_peaks(prof, prom=2000, roi=[0, 512], smoothing=5):
    dat = prof
    dat[0: roi[0]] = 0
    dat[roi[1]:] = 0
    calibration = savitzky_golay(prof, smoothing, 3)  # window size 51, polynomial order 3
    peak, _ = find_peaks(calibration, prominence=prom)
    return calibration, peak


def cut_image(old_image, x1, x2, y1, y2, bg):
    mask = np.zeros_like(old_image)
    mask[512 - x2:512 - x1, y1:y2] = 1
    new = old_image-bg
    new[new<0] = 0
    return new * mask


def redistribute_image(sheared_image, E_axis):
    im = sheared_image.T
    Jacobian_vect = h * c / (E_axis ** 2)
    Jacobian_vect = (E_axis ** 2)
    Jacobian_vect_norm = Jacobian_vect / np.max(Jacobian_vect)
    Jacobian_mat = np.tile(Jacobian_vect_norm, [np.shape(im)[0], 1])
    redistributed_image = np.multiply(im, Jacobian_mat)
    return redistributed_image.T


def treat_image(image_old, energy_axis, shear_parameter=-2.5):
    sheared_image = shear_image(image_old, shear_parameter)
    redistributed_image = redistribute_image(sheared_image, energy_axis)
    y_axis = np.arange(0, np.shape(sheared_image)[0])
    correct_E_axis = np.arange(energy_axis[0], energy_axis[-1],
                               abs((energy_axis[0] - energy_axis[-1]) / np.shape(redistributed_image)[1]))
    interp_func = interp1d(energy_axis, np.flip(redistributed_image, 1), axis=1, kind='linear')
    image_new = interp_func(correct_E_axis)

    return image_new


def treat_image_new(image_old, energy_axis, x1, x2, y1, y2, bg, shear_parameter=-2.5):
    sheared_image = shear_image(image_old, shear_parameter, axis=1)
    cutted_image = np.flip(cut_image(sheared_image, x1, x2, y1, y2, bg),axis = 0)
    redistributed_image = redistribute_image(cutted_image, energy_axis)
    y_axis = np.arange(0, np.shape(redistributed_image)[0])
    correct_E_axis = np.arange(energy_axis[0], energy_axis[-1],
                               abs((energy_axis[0] - energy_axis[-1]) / np.shape(redistributed_image)[1]))
    interp_func = interp1d(energy_axis, redistributed_image, axis=0, kind='linear')
    image_new = interp_func(correct_E_axis)
    #image_new = cutted_image
    return correct_E_axis, image_new


# Define the Gaussian function
def gaussian(x, A, mu, sigma, B):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + B


def fit_gaussian(x_data,y_data):
    initial_guess = [np.max(y_data), np.argmax(y_data), 20, 0]  # Initial guess for parameters [A, mu, sigma]
    params, covariance = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
    # Extract the fitted parameters
    A_fit, mu_fit, sigma_fit, B_fit = params
    return A_fit, mu_fit, sigma_fit, B_fit


