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


